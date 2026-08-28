"""XGBOOST_CAUSAL_RETUNED: the A2 research challenger.

A deliberately *new* model, not a retune of the legacy hybrid. The preserved
LEAKAGE_CORRECTED_REFERENCE must stay comparable to itself, so improving it in
place is forbidden (see :mod:`btc_forecaster.evidence`); building a named
challenger and scoring both through identical folds is the sanctioned route.

Three design choices distinguish it from the legacy hybrid, each following from
a Track A finding:

**No Prophet.** In the A2 baseline run Prophet alone scored MAE 9,405 against
the random walk's 3,849 -- nearly as bad as the full hybrid. That locates most
of the damage in the trend/seasonality extrapolation rather than the residual
correction: fitting yearly seasonality to a non-stationary crypto price and
extrapolating it thirty days produces large, confidently wrong forecasts. This
model drops the structural component entirely.

**Returns, not price residuals.** The target is the next bar's log return.
Log price is I(1); its residual against a trend model inherits that
non-stationarity, so a tree fitted on it must extrapolate outside its training
support the moment price makes a new high. Log returns are approximately
stationary, so "no signal" is a well-defined prediction of zero, and a tree that
learns nothing predicts approximately that -- which degrades gracefully to the
random walk instead of diverging from it.

**Hyperparameters chosen by nested inner validation.** The legacy parameters
were the frozen output of an Optuna search run against a different cutoff, a
different feature set and a leaky evaluation. Here the search runs inside each
outer fold's training history (A2.3/A2.4) on a small random budget (A2.24).

What it shares with the hybrid, honestly: multi-step forecasting is recursive,
so only step 1 uses real features and later steps compound the model's own
output. Long-horizon output is a scenario.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..features.pipeline import build_feature_frame, to_supervised
from ..features.selection import FeatureSelection, FeatureSelector
from ..features.spec import log_returns
from ..timebase import HorizonSpec
from .base import ForecastModel, ForecastResult, NotFittedError, TrainingWindow
from .tuning import SelectionRecord, sample_parameter_grid, select_by_inner_validation
from .volatility import constant_volatility, garch_volatility, monte_carlo_interval

#: Bounded, CPU-friendly search space (A2.24). Shallow trees and heavy
#: regularisation are the priors that matter on a low signal-to-noise financial
#: target: depth beyond 4 memorises the training window, and unregularised
#: boosting on daily returns fits noise almost immediately.
#:
#: ``num_boost_round`` is deliberately NOT a search dimension. It is determined
#: by early stopping on each inner validation window, which is both sounder --
#: the right number of rounds depends on the configuration, so searching them
#: jointly spends budget on pairs that differ only in where they stopped -- and
#: much cheaper, because a badly specified candidate aborts after a few dozen
#: rounds instead of running to a fixed ceiling.
DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 20],
    "reg_lambda": [1.0, 5.0, 20.0],
}

#: 12 configurations x 3 inner splits = 36 early-stopped fits per outer fold.
#: The budget encodes the trade A2.24 asks for: many honest folds beat a large
#: search evaluated on few.
DEFAULT_N_CANDIDATES = 12

#: Ceiling for early stopping, and the patience before it triggers.
MAX_BOOST_ROUNDS = 300
EARLY_STOPPING_ROUNDS = 20

#: XGBoost defaults to every core, which is slower than two threads at these
#: data sizes: per-tree parallel overhead dominates when there are only a few
#: hundred rows. Measured ~2.7x faster than the default on this workload.
DEFAULT_NTHREAD = 2


class XgboostCausalRetuned(ForecastModel):
    """Direct XGBoost on next-bar log return, tuned by nested inner validation."""

    requires = ("xgboost",)

    def __init__(
        self,
        *,
        name: str = "xgboost_causal_retuned",
        interval_level: float = 0.95,
        selector: FeatureSelector | None = None,
        search_space: dict[str, list[Any]] | None = None,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        max_boost_rounds: int = MAX_BOOST_ROUNDS,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
        nthread: int = DEFAULT_NTHREAD,
        inner_splits: int = 3,
        inner_val_bars: int = 90,
        inner_embargo: int = 5,
        inner_min_train_bars: int = 250,
        random_state: int = 42,
        monte_carlo_runs: int = 1000,
        use_garch: bool = True,
    ) -> None:
        super().__init__(name)
        self.interval_level = interval_level
        self.selector = selector or FeatureSelector()
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.n_candidates = n_candidates
        self.max_boost_rounds = max_boost_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.nthread = nthread
        self.inner_splits = inner_splits
        self.inner_val_bars = inner_val_bars
        self.inner_embargo = inner_embargo
        self.inner_min_train_bars = inner_min_train_bars
        self.random_state = random_state
        self.monte_carlo_runs = monte_carlo_runs
        self.use_garch = use_garch

        self._booster: Any | None = None
        self._selection: FeatureSelection | None = None
        self._tuning: SelectionRecord | None = None
        self._n_params: int = 0
        self._chosen_rounds: int = 0

    @property
    def min_train_bars(self) -> int:
        # inner_min_train + embargo + val, plus room for feature warm-up.
        return self.inner_min_train_bars + self.inner_embargo + self.inner_val_bars + 120

    @property
    def _fitted_booster(self) -> Any:
        if self._booster is None:
            raise NotFittedError(f"{self.name} has not been fitted")
        return self._booster

    @property
    def _fitted_selection(self) -> FeatureSelection:
        if self._selection is None:
            raise NotFittedError(f"{self.name} has not been fitted")
        return self._selection

    @property
    def tuning(self) -> SelectionRecord:
        """The per-fold selection record, for the run manifest and for audit."""
        if self._tuning is None:
            raise NotFittedError(f"{self.name} has not been fitted")
        return self._tuning

    def describe(self) -> dict:
        return {
            **super().describe(),
            "target": "next-bar log return",
            "interval_level": self.interval_level,
            "n_candidates": self.n_candidates,
            "max_boost_rounds": self.max_boost_rounds,
            "early_stopping_rounds": self.early_stopping_rounds,
            "chosen_boost_rounds": self._chosen_rounds,
            "inner_splits": self.inner_splits,
            "inner_val_bars": self.inner_val_bars,
            "inner_embargo": self.inner_embargo,
            "monte_carlo_runs": self.monte_carlo_runs,
            "search_space": {k: list(v) for k, v in self.search_space.items()},
            "tuning": self._tuning.to_dict() if self._tuning else None,
            "features": self._selection.to_dict() if self._selection else None,
            "n_boosted_trees": self._n_params,
        }

    # -- fitting ----------------------------------------------------------

    def _booster_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            **params,
            "objective": "reg:squarederror",
            "verbosity": 0,
            "seed": self.random_state,
            "nthread": self.nthread,
        }

    def _train_booster(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict[str, Any],
        *,
        rounds: int,
    ) -> Any:
        import xgboost as xgb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return xgb.train(
                self._booster_params(params),
                xgb.DMatrix(X, label=y),
                num_boost_round=rounds,
                verbose_eval=False,
            )

    def _train_early_stopped(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ) -> tuple[float, int]:
        """Fit with early stopping on the inner validation window.

        Returns ``(validation MAE, best round count)``. The round count is a
        *fitted* quantity rather than a searched one -- see DEFAULT_SEARCH_SPACE.
        """
        import xgboost as xgb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evals_result: dict = {}
            booster = xgb.train(
                {**self._booster_params(params), "eval_metric": "mae"},
                xgb.DMatrix(X_train, label=y_train),
                num_boost_round=self.max_boost_rounds,
                evals=[(xgb.DMatrix(X_val, label=y_val), "val")],
                early_stopping_rounds=self.early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=False,
            )
            best_round = int(getattr(booster, "best_iteration", 0)) + 1
            history = evals_result["val"]["mae"]
            score = float(history[min(best_round - 1, len(history) - 1)])
        return score, best_round

    def _predict_rows(self, booster: Any, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(booster.predict(xgb.DMatrix(X)), dtype=float)

    def _fit(self, window: TrainingWindow) -> None:
        # 1. Features selected on this training window only (A2.5). Refitted per
        #    outer fold because fit() is called per fold.
        self._selection = self.selector.fit(window.frame)
        features = build_feature_frame(window.frame, list(self._selection.specs))

        # 2. Target: next bar's log return, aligned so the row predicting bar T
        #    comes from bar T-1.
        target = log_returns(window.frame)
        data = to_supervised(features, target, step=1)

        # 3. Nested selection strictly inside this training history (A2.3).
        candidates = sample_parameter_grid(
            self.search_space, n_candidates=self.n_candidates, seed=self.random_state
        )

        X, y = data.X, data.y
        rounds_by_candidate: dict[tuple, list[int]] = {}

        def fit_score(params: dict[str, Any], split) -> float:
            score, best_round = self._train_early_stopped(
                X.iloc[split.train_start : split.train_end + 1],
                y.iloc[split.train_start : split.train_end + 1],
                X.iloc[split.val_start : split.val_end + 1],
                y.iloc[split.val_start : split.val_end + 1],
                params,
            )
            rounds_by_candidate.setdefault(tuple(sorted(params.items())), []).append(best_round)
            return score

        self._tuning = select_by_inner_validation(
            len(X),
            candidates,
            fit_score,
            n_splits=self.inner_splits,
            val_bars=self.inner_val_bars,
            embargo=self.inner_embargo,
            min_train_bars=self.inner_min_train_bars,
            scoring="inner-validation MAE on log return (lower is better)",
            features=tuple(data.feature_names),
        )

        # 4. Freeze the winner and refit on the whole outer training history.
        #    No held-out window remains to early-stop against, so the round count
        #    is the median of what the winner needed across the inner splits --
        #    the standard resolution, and recorded so it stays auditable.
        winner_key = tuple(sorted(self._tuning.chosen.items()))
        observed_rounds = rounds_by_candidate.get(winner_key) or [200]
        self._chosen_rounds = int(np.median(observed_rounds))

        self._booster = self._train_booster(
            X, y, self._tuning.chosen, rounds=self._chosen_rounds
        )
        self._n_params = int(len(self._fitted_booster.get_dump()))

    # -- forecasting ------------------------------------------------------

    def _recursive_log_returns(self, steps: int) -> np.ndarray:
        """Roll forward, compounding the model's own predicted returns.

        Only step 1 uses features built entirely from observed data. Later steps
        append the simulated close and recompute, so error compounds -- the same
        limitation the legacy hybrid has, stated rather than buried. Volume
        cannot be simulated and is carried forward, which makes volume-derived
        features progressively staler.
        """
        specs = list(self._fitted_selection.specs)
        history = self.window.frame.copy()
        last_volume = float(history["volume"].iloc[-1])

        predicted = np.zeros(steps, dtype=float)
        bar = history.index[-1]

        for step in range(steps):
            row = build_feature_frame(history, specs).iloc[[-1]]
            if row.isna().any(axis=None):
                predicted[step] = 0.0  # fall back to "no change", not an imputed guess
            else:
                predicted[step] = float(self._predict_rows(self._fitted_booster, row)[0])

            bar = bar + pd.Timedelta(days=1)
            next_close = float(history["close"].iloc[-1] * np.exp(predicted[step]))
            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        {"close": [next_close], "volume": [last_volume]},
                        index=pd.DatetimeIndex([bar], name="date"),
                    ),
                ]
            )
        return predicted

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        steps = len(horizon)
        step_returns = self._recursive_log_returns(steps)

        last_log_close = float(np.log(self.window.close.iloc[-1]))
        mean_log_path = last_log_close + np.cumsum(step_returns)

        if self.use_garch:
            try:
                volatility = garch_volatility(self.window.log_returns, steps)
            except Exception as exc:
                fallback = constant_volatility(self.window.log_returns, steps)
                volatility = type(fallback)(
                    sigma=fallback.sigma,
                    model="constant (GARCH fallback)",
                    params={**fallback.params, "garch_error": str(exc)[:200]},
                )
        else:
            volatility = constant_volatility(self.window.log_returns, steps)

        point, lower, upper = monte_carlo_interval(
            mean_log_path,
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
                "target": "next-bar log return",
                "volatility_model": volatility.model,
                "chosen_params": dict(self.tuning.chosen),
                "chosen_boost_rounds": self._chosen_rounds,
                "inner_validation_score": self.tuning.chosen_score,
                "search_score_spread": self.tuning.score_spread,
                "n_features": len(self._fitted_selection.specs),
                "features": self._fitted_selection.names,
                "recursive_multi_step": True,
                "mean_predicted_log_return": float(np.mean(step_returns)),
            },
        )


__all__ = [
    "DEFAULT_N_CANDIDATES",
    "DEFAULT_NTHREAD",
    "DEFAULT_SEARCH_SPACE",
    "EARLY_STOPPING_ROUNDS",
    "MAX_BOOST_ROUNDS",
    "XgboostCausalRetuned",
]
