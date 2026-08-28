"""Prophet baseline with XGBoost residual correction: the original headline model.

Preserved in substance, corrected in three places.

1. **Feature alignment.** The original trained XGBoost with features and target
   on the same bar, so rolling/EMA/SMA columns containing ``close[D]`` were used
   to predict ``close[D]``. Features now go through
   :func:`~btc_forecaster.features.pipeline.to_supervised`, which shifts them so
   the row predicting bar ``T`` comes from bar ``T-1``.

2. **Feature selection inside fit.** Lags, windows and spans are selected from
   the training window only, every time the model is fitted, so the walk-forward
   engine re-selects per fold instead of reusing one global choice made with
   knowledge of every fold's future.

3. **Accumulating uncertainty.** Intervals come from
   :func:`~btc_forecaster.models.volatility.monte_carlo_interval`, whose shocks
   accumulate along the path.

The XGBoost hyperparameters are carried over unchanged from the original
(themselves the frozen output of the Optuna search now in ``research/legacy/``).
They are stale -- tuned against a different cutoff, a different feature set, and
a leaky evaluation -- and re-tuning them under the walk-forward engine is
recorded as open quant debt rather than done silently here, since changing them
would confound the before/after comparison this refactor exists to support.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..features.pipeline import build_feature_frame, to_supervised
from ..features.selection import FeatureSelection, FeatureSelector
from ..timebase import HorizonSpec
from .base import ForecastModel, ForecastResult, TrainingWindow
from .prophet_model import ProphetModel
from .volatility import constant_volatility, garch_volatility, monte_carlo_interval

#: Carried over verbatim from bayesianCutoff.py. See module docstring.
LEGACY_XGB_PARAMS: dict = {
    "max_depth": 3,
    "learning_rate": 0.010349570637285655,
    "subsample": 0.8021272578985711,
    "colsample_bytree": 0.7728798862419759,
    "objective": "reg:squarederror",
    "verbosity": 0,
}

LEGACY_NUM_BOOST_ROUND = 200


class ProphetXgboostHybrid(ForecastModel):
    """Prophet trend/seasonality plus an XGBoost correction on its residuals."""

    requires = ("prophet", "xgboost")

    def __init__(
        self,
        *,
        name: str = "prophet_xgb_hybrid",
        interval_level: float = 0.95,
        selector: FeatureSelector | None = None,
        xgb_params: dict | None = None,
        num_boost_round: int = LEGACY_NUM_BOOST_ROUND,
        random_state: int = 42,
        monte_carlo_runs: int = 1000,
        use_garch: bool = True,
    ) -> None:
        super().__init__(name)
        self.interval_level = interval_level
        self.selector = selector or FeatureSelector()
        self.xgb_params = {**LEGACY_XGB_PARAMS, **(xgb_params or {})}
        self.num_boost_round = num_boost_round
        self.random_state = random_state
        self.monte_carlo_runs = monte_carlo_runs
        self.use_garch = use_garch

        self._prophet: ProphetModel | None = None
        self._booster = None
        self._selection: FeatureSelection | None = None
        self._residual_std: float = float("nan")
        self._train_residuals: pd.Series | None = None

    @property
    def min_train_bars(self) -> int:
        return 365

    def describe(self) -> dict:
        return {
            **super().describe(),
            "interval_level": self.interval_level,
            "xgb_params": self.xgb_params,
            "num_boost_round": self.num_boost_round,
            "monte_carlo_runs": self.monte_carlo_runs,
            "use_garch": self.use_garch,
            "features": self._selection.to_dict() if self._selection else None,
        }

    # -- fitting ----------------------------------------------------------

    def _fit(self, window: TrainingWindow) -> None:
        import xgboost as xgb

        # 1. Prophet baseline on the training window only.
        self._prophet = ProphetModel(
            interval_level=self.interval_level,
            name=f"{self.name}::prophet",
        )
        self._prophet.fit(window)

        baseline_log = self._prophet.predict_log(window.index)
        residual = pd.Series(
            window.log_close.to_numpy(dtype=float) - baseline_log,
            index=window.index,
            name="residual",
        )

        # 2. Features selected on this training window, then shifted so the row
        #    predicting bar T comes from bar T-1.
        self._selection = self.selector.fit(window.frame)
        features = build_feature_frame(window.frame, list(self._selection.specs))
        data = to_supervised(features, residual, step=1)

        params = {**self.xgb_params, "seed": self.random_state}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._booster = xgb.train(
                params,
                xgb.DMatrix(data.X, label=data.y),
                num_boost_round=self.num_boost_round,
                verbose_eval=False,
            )
            in_sample = self._booster.predict(xgb.DMatrix(data.X))

        self._train_residuals = pd.Series(data.y.to_numpy() - in_sample, index=data.X.index)
        self._residual_std = float(self._train_residuals.std(ddof=1))

    # -- forecasting ------------------------------------------------------

    def _recursive_residuals(self, target_bars: pd.DatetimeIndex, baseline_log: np.ndarray) -> np.ndarray:
        """Roll the model forward, one bar at a time, from the origin.

        The model maps features at bar ``T-1`` to the residual at bar ``T``. Only
        the origin bar's features are real; every later step is built from the
        model's own simulated prices, so accuracy decays with horizon. That
        decay is a property of recursive multi-step forecasting, not a bug, but
        it is why long-horizon output from this model should be read as a
        scenario rather than a prediction.

        Volume cannot be simulated and is carried forward from the last observed
        bar, which makes any volume-derived feature progressively staler. The
        original did the same without saying so.
        """
        import xgboost as xgb

        specs = list(self._selection.specs)
        history = self.window.frame.copy()
        last_volume = float(history["volume"].iloc[-1])

        residuals = np.zeros(len(target_bars), dtype=float)

        for step, bar in enumerate(target_bars):
            features = build_feature_frame(history, specs)
            row = features.iloc[[-1]]

            if row.isna().any(axis=None):
                # Not enough history for the selected windows; fall back to the
                # Prophet baseline alone rather than imputing a value.
                residuals[step] = 0.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    residuals[step] = float(self._booster.predict(xgb.DMatrix(row))[0])

            simulated_log = baseline_log[step] + residuals[step]
            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        {"close": [float(np.exp(simulated_log))], "volume": [last_volume]},
                        index=pd.DatetimeIndex([bar], name="date"),
                    ),
                ]
            )

        return residuals

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        baseline_log = self._prophet.predict_log(target_bars)
        residuals = self._recursive_residuals(target_bars, baseline_log)
        combined_log = baseline_log + residuals

        if self.use_garch:
            try:
                volatility = garch_volatility(self.window.log_returns, len(horizon))
            except Exception as exc:  # arch missing, or optimiser failure
                volatility = constant_volatility(self.window.log_returns, len(horizon))
                volatility = type(volatility)(
                    sigma=volatility.sigma,
                    model="constant (GARCH fallback)",
                    params={**volatility.params, "garch_error": str(exc)[:200]},
                )
        else:
            volatility = constant_volatility(self.window.log_returns, len(horizon))

        point, lower, upper = monte_carlo_interval(
            combined_log,
            volatility,
            level=self.interval_level,
            n_paths=self.monte_carlo_runs,
            seed=self.random_state,
        )

        return self._result(
            point,
            target_bars,
            lower=lower,
            upper=upper,
            interval_level=self.interval_level,
            metadata={
                "volatility_model": volatility.model,
                "volatility_params": volatility.params,
                "monte_carlo_runs": self.monte_carlo_runs,
                "n_features": len(self._selection.specs),
                "features": self._selection.names,
                "residual_std_in_sample": self._residual_std,
                "prophet_baseline_price": np.exp(baseline_log).tolist()[:5],
                "recursive_multi_step": True,
            },
        )


__all__ = ["LEGACY_NUM_BOOST_ROUND", "LEGACY_XGB_PARAMS", "ProphetXgboostHybrid"]
