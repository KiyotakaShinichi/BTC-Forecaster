"""Classical statistical time-series models.

These exist to prove the :class:`~btc_forecaster.models.base.ForecastModel`
contract generalises beyond naive baselines and machine-learning hybrids -- an
ARIMA, a dynamic regression and a state-space smoother have very different
internals but present the same fit/predict surface and are scored through the
same folds.

They all model **log price**. Prices are non-negative and multiplicative;
differencing log price gives log returns, which is the quantity these models'
Gaussian assumptions are least wrong about. Forecasts and intervals are
exponentiated back to price space at the boundary.

Implementation note: every model is fitted on a plain numpy array with a
positional index rather than the tz-aware DatetimeIndex. statsmodels infers
frequency from a date index and emits warnings -- or silently reindexes -- when
a series has gaps. Mapping the forecast back onto ``target_bars`` explicitly is
both more predictable and easier to test, and the time contract already
guarantees which bars those are.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..timebase import HorizonSpec
from .base import ForecastModel, ForecastResult, NotFittedError, TrainingWindow


def _fitted(model, name: str):
    """Narrow a lazily-assigned fitted model away from None.

    Subclasses assign their backend in ``_fit``. The base class guarantees
    ``_predict`` runs only after ``fit``, but a type checker cannot see that, and
    an assert would vanish under -O. This raises the same error the contract uses.
    """
    if model is None:
        raise NotFittedError(f"{name} has not been fitted")
    return model


@dataclass(frozen=True)
class OrderSelection:
    """The outcome of an AIC search over ARIMA orders, and the evidence."""

    order: tuple[int, int, int]
    aic: float
    n_candidates: int
    n_converged: int
    ranking: tuple[tuple[tuple[int, int, int], float], ...]

    def to_dict(self) -> dict:
        return {
            "order": list(self.order),
            "aic": self.aic,
            "n_candidates": self.n_candidates,
            "n_converged": self.n_converged,
            "top_5": [{"order": list(o), "aic": a} for o, a in self.ranking[:5]],
        }


DEFAULT_ORDER_GRID: tuple[tuple[int, int, int], ...] = (
    (0, 1, 0),  # random walk, as an ARIMA -- the null inside the search
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 2),
)


def select_arima_order(
    endog: np.ndarray,
    *,
    grid: tuple[tuple[int, int, int], ...] = DEFAULT_ORDER_GRID,
    trend: str | None = None,
) -> OrderSelection:
    """Pick an ARIMA order by AIC over a fixed grid.

    Called only with training data, so the walk-forward engine re-selects the
    order in every fold. Selecting once on the full sample and reusing the order
    across folds would leak, in the same way selecting features once does.

    The grid deliberately includes ``(0,1,0)`` -- the random walk. If AIC cannot
    justify anything more complex than a random walk on this data, the search
    should be allowed to say so.
    """
    from statsmodels.tsa.arima.model import ARIMA

    scored: list[tuple[tuple[int, int, int], float]] = []
    for order in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(endog, order=order, trend=trend).fit()
            aic = float(fitted.aic)
            if np.isfinite(aic):
                scored.append((order, aic))
        except Exception:
            # A non-converging order is evidence about that order, not an error
            # worth aborting the whole search for.
            continue

    if not scored:
        raise RuntimeError("no ARIMA order in the grid converged on this training window")

    scored.sort(key=lambda item: item[1])
    best_order, best_aic = scored[0]
    return OrderSelection(
        order=best_order,
        aic=best_aic,
        n_candidates=len(grid),
        n_converged=len(scored),
        ranking=tuple(scored),
    )


class ArimaModel(ForecastModel):
    """ARIMA on log price, with analytic prediction intervals.

    With ``order=None`` the order is chosen by AIC inside ``fit`` -- and
    therefore inside every walk-forward fold.
    """

    requires = ("statsmodels",)

    def __init__(
        self,
        order: tuple[int, int, int] | None = (1, 1, 1),
        *,
        trend: str | None = None,
        name: str | None = None,
        interval_level: float = 0.95,
        grid: tuple[tuple[int, int, int], ...] = DEFAULT_ORDER_GRID,
    ) -> None:
        super().__init__(name or (f"arima{order}" if order else "arima_auto"))
        self.order = order
        self.trend = trend
        self.interval_level = interval_level
        self.grid = grid
        self._fitted_model = None
        self._selection: OrderSelection | None = None
        self._chosen_order: tuple[int, int, int] | None = None

    @property
    def min_train_bars(self) -> int:
        return 60

    def describe(self) -> dict:
        return {
            **super().describe(),
            "order": list(self._chosen_order) if self._chosen_order else None,
            "requested_order": list(self.order) if self.order else "auto",
            "trend": self.trend,
            "interval_level": self.interval_level,
            "order_selection": self._selection.to_dict() if self._selection else None,
        }

    def _fit(self, window: TrainingWindow) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        endog = window.log_close.to_numpy(dtype=float)

        if self.order is None:
            self._selection = select_arima_order(endog, grid=self.grid, trend=self.trend)
            self._chosen_order = self._selection.order
        else:
            self._selection = None
            self._chosen_order = self.order

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = ARIMA(endog, order=self._chosen_order, trend=self.trend).fit()

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        fitted = _fitted(self._fitted_model, self.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = fitted.get_forecast(steps=len(horizon))
            mean_log = np.asarray(forecast.predicted_mean, dtype=float)
            conf = np.asarray(forecast.conf_int(alpha=1.0 - self.interval_level), dtype=float)

        return self._result(
            np.exp(mean_log),
            target_bars,
            lower=np.exp(conf[:, 0]),
            upper=np.exp(conf[:, 1]),
            interval_level=self.interval_level,
            metadata={
                "order": list(self._chosen_order or ()),
                "aic": float(fitted.aic),
                "order_selection": self._selection.to_dict() if self._selection else None,
            },
        )


class SarimaxModel(ForecastModel):
    """SARIMAX on log price: seasonality plus exogenous dynamic regression.

    This is the seam for external signals. ``exog_columns`` names columns that
    must be present in :attr:`TrainingWindow.exog` at fit time and in
    ``future_exog`` at predict time.

    A hard constraint follows from the time contract: exogenous regressors used
    this way must be **known in advance over the whole forecast horizon**.
    Calendar effects, scheduled protocol events and announced policy dates
    qualify. A sentiment index or a news-flow score does not -- it would have to
    be forecast itself, and feeding realised future values in would be leakage
    of the most direct kind. Track B signals therefore attach as *lagged* inputs
    within the training window, not as future regressors, unless they are
    genuinely deterministic ahead of time.
    """

    requires = ("statsmodels",)

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        *,
        exog_columns: tuple[str, ...] = (),
        trend: str | None = None,
        name: str | None = None,
        interval_level: float = 0.95,
    ) -> None:
        super().__init__(name or "sarimax")
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_columns = tuple(exog_columns)
        self.trend = trend
        self.interval_level = interval_level
        self._fitted_model = None

    @property
    def min_train_bars(self) -> int:
        return 60

    def describe(self) -> dict:
        return {
            **super().describe(),
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "exog_columns": list(self.exog_columns),
            "interval_level": self.interval_level,
        }

    def _training_exog(self, window: TrainingWindow) -> np.ndarray | None:
        if not self.exog_columns:
            return None
        if window.exog is None:
            raise ValueError(
                f"{self.name} declares exog_columns={list(self.exog_columns)} "
                "but the training window carries no exog"
            )
        missing = [c for c in self.exog_columns if c not in window.exog.columns]
        if missing:
            raise ValueError(f"training exog is missing column(s): {missing}")
        return window.exog[list(self.exog_columns)].to_numpy(dtype=float)

    def _fit(self, window: TrainingWindow) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = SARIMAX(
                window.log_close.to_numpy(dtype=float),
                exog=self._training_exog(window),
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        exog_array = None
        if self.exog_columns:
            if future_exog is None:
                raise ValueError(
                    f"{self.name} needs future values of {list(self.exog_columns)} over the "
                    "forecast horizon. Only regressors known in advance may be supplied here."
                )
            missing = [c for c in self.exog_columns if c not in future_exog.columns]
            if missing:
                raise ValueError(f"future_exog is missing column(s): {missing}")
            exog_array = future_exog[list(self.exog_columns)].to_numpy(dtype=float)

        fitted = _fitted(self._fitted_model, self.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = fitted.get_forecast(steps=len(horizon), exog=exog_array)
            mean_log = np.asarray(forecast.predicted_mean, dtype=float)
            conf = np.asarray(forecast.conf_int(alpha=1.0 - self.interval_level), dtype=float)

        return self._result(
            np.exp(mean_log),
            target_bars,
            lower=np.exp(conf[:, 0]),
            upper=np.exp(conf[:, 1]),
            interval_level=self.interval_level,
            metadata={
                "order": list(self.order),
                "seasonal_order": list(self.seasonal_order),
                "exog_columns": list(self.exog_columns),
                "aic": float(fitted.aic),
            },
        )


class EtsModel(ForecastModel):
    """Exponential smoothing / innovations state space on log price.

    A damped additive trend is the default. Undamped trends extrapolate linearly
    forever, which on a 365-day crypto horizon produces forecasts that are not
    merely wrong but unphysical; damping pulls the long-horizon forecast toward
    a constant, which is a more honest statement of what a smoother knows.
    """

    requires = ("statsmodels",)

    def __init__(
        self,
        *,
        trend: str | None = "add",
        damped_trend: bool = True,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        name: str | None = None,
        interval_level: float = 0.95,
    ) -> None:
        super().__init__(name or "ets")
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.interval_level = interval_level
        self._fitted_model = None

    @property
    def min_train_bars(self) -> int:
        return 60

    def describe(self) -> dict:
        return {
            **super().describe(),
            "trend": self.trend,
            "damped_trend": self.damped_trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "interval_level": self.interval_level,
        }

    def _fit(self, window: TrainingWindow) -> None:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        # ETSModel.get_prediction reaches for `.index` on the fitted values, so
        # unlike ARIMA/SARIMAX it cannot be fitted on a bare ndarray. A positional
        # RangeIndex keeps statsmodels away from frequency inference while still
        # giving it the pandas object it expects.
        endog = pd.Series(
            window.log_close.to_numpy(dtype=float),
            index=pd.RangeIndex(len(window)),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = ETSModel(
                endog,
                error="add",
                trend=self.trend,
                damped_trend=self.damped_trend if self.trend else False,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            ).fit(disp=False)

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        n = len(self.window)
        steps = len(horizon)

        fitted = _fitted(self._fitted_model, self.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = fitted.get_prediction(start=n, end=n + steps - 1)
            mean_log = np.asarray(prediction.predicted_mean, dtype=float)
            conf = np.asarray(
                prediction.summary_frame(alpha=1.0 - self.interval_level)[
                    ["pi_lower", "pi_upper"]
                ],
                dtype=float,
            )

        return self._result(
            np.exp(mean_log),
            target_bars,
            lower=np.exp(conf[:, 0]),
            upper=np.exp(conf[:, 1]),
            interval_level=self.interval_level,
            metadata={
                "trend": self.trend,
                "damped_trend": self.damped_trend,
                "aic": float(fitted.aic),
            },
        )


__all__ = [
    "DEFAULT_ORDER_GRID",
    "ArimaModel",
    "EtsModel",
    "OrderSelection",
    "SarimaxModel",
    "select_arima_order",
]
