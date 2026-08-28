"""Prophet adapter, preserving the original pipeline's baseline model.

Prophet is kept because it is the trend/seasonality decomposition the existing
research was built on, and dropping it would make the before/after comparison
meaningless. It is now one registered model among several rather than the
implicit centre of the pipeline, and it is scored through the same walk-forward
folds as the random walk.

Two things are corrected relative to the original usage:

* Prophet requires a timezone-naive ``ds`` column and silently misbehaves
  otherwise. The conversion happens here, once, at the boundary.
* Prophet's own ``yhat_lower``/``yhat_upper`` are surfaced as the model's
  prediction interval. The original discarded them and substituted a GARCH
  Monte Carlo band that did not accumulate along the path.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ..timebase import HorizonSpec
from .base import ForecastModel, ForecastResult, TrainingWindow


def _silence_prophet() -> None:
    """Prophet and cmdstanpy log at INFO on every fit; a backtest runs hundreds."""
    for name in ("prophet", "cmdstanpy", "prophet.models", "prophet.forecaster"):
        logging.getLogger(name).setLevel(logging.WARNING)


def to_prophet_frame(index: pd.DatetimeIndex, values: np.ndarray | None = None) -> pd.DataFrame:
    """Build Prophet's ds/y frame, stripping the timezone it cannot handle.

    Dropping the tz is safe only because the time contract guarantees the index
    is already UTC: the naive timestamps that come out are UTC by construction,
    not by luck.
    """
    ds = pd.DatetimeIndex(index).tz_convert(None) if index.tz is not None else pd.DatetimeIndex(index)
    frame = pd.DataFrame({"ds": ds})
    if values is not None:
        frame["y"] = np.asarray(values, dtype=float)
    return frame


class ProphetModel(ForecastModel):
    """Prophet trend + seasonality fitted on log price."""

    requires = ("prophet",)

    def __init__(
        self,
        *,
        name: str = "prophet",
        interval_level: float = 0.95,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        daily_seasonality: bool = False,
    ) -> None:
        super().__init__(name)
        self.interval_level = interval_level
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self._model = None

    @property
    def min_train_bars(self) -> int:
        return 90

    def describe(self) -> dict:
        return {
            **super().describe(),
            "interval_level": self.interval_level,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "daily_seasonality": self.daily_seasonality,
        }

    def _new_model(self):
        from prophet import Prophet

        _silence_prophet()
        return Prophet(
            interval_width=self.interval_level,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
        )

    def _fit(self, window: TrainingWindow) -> None:
        self._model = self._new_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(to_prophet_frame(window.index, window.log_close.to_numpy()))

    def predict_log(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Prophet's log-price prediction at arbitrary bars.

        Exposed because the hybrid needs the in-sample baseline to compute
        residuals, and re-fitting a second Prophet for that would be wasteful
        and could diverge from this one.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted = self._model.predict(to_prophet_frame(index))
        return predicted["yhat"].to_numpy(dtype=float)

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted = self._model.predict(to_prophet_frame(target_bars))

        return self._result(
            np.exp(predicted["yhat"].to_numpy(dtype=float)),
            target_bars,
            lower=np.exp(predicted["yhat_lower"].to_numpy(dtype=float)),
            upper=np.exp(predicted["yhat_upper"].to_numpy(dtype=float)),
            interval_level=self.interval_level,
            metadata={"interval_source": "prophet"},
        )


__all__ = ["ProphetModel", "to_prophet_frame"]
