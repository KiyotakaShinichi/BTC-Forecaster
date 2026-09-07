"""Naive baselines. Every sophisticated model must beat these to be interesting.

For a near-efficient asset, the random walk is not a straw man -- it is the
hypothesis to be disproved. A hybrid Prophet/XGBoost/GARCH stack that does not
beat "tomorrow's price is today's price" has produced complexity, not skill.

The pre-Track-A pipeline reported directional accuracy and a binomial p-value
against a fixed 0.5 coin, never against a baseline forecaster on the same folds.
It is possible to beat a coin and still lose to the random walk, so these
baselines are first-class models scored through the identical walk-forward
engine rather than a footnote.

All three depend only on numpy/pandas/scipy, so they always run -- there is no
environment in which the comparison can be quietly skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timebase import HorizonSpec
from .base import ForecastModel, ForecastResult, TrainingWindow, random_walk_bands


class RandomWalk(ForecastModel):
    """``P[T+h] = P[T]``. The naive forecast, flat at the last observed close.

    Under the efficient-market null this is the optimal point forecast for a
    price series, which is exactly why it is the baseline to beat.
    """

    def __init__(self, name: str = "random_walk", interval_level: float = 0.95) -> None:
        super().__init__(name)
        self.interval_level = interval_level
        self._last_log_price: float = float("nan")
        self._sigma: float = float("nan")

    @property
    def min_train_bars(self) -> int:
        return 30

    def describe(self) -> dict:
        return {**super().describe(), "interval_level": self.interval_level}

    def _fit(self, window: TrainingWindow) -> None:
        log_close = window.log_close
        self._last_log_price = float(log_close.iloc[-1])
        # Interval width comes from realised log-return volatility. The point
        # forecast does not depend on it.
        self._sigma = float(window.log_returns.dropna().std(ddof=1))

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        point, lower, upper = random_walk_bands(
            self._last_log_price,
            drift_per_step=0.0,
            sigma_per_step=self._sigma,
            steps=len(horizon),
            level=self.interval_level,
        )
        return self._result(
            point,
            target_bars,
            lower=lower,
            upper=upper,
            interval_level=self.interval_level,
            metadata={"sigma_log_return": self._sigma},
        )


class RandomWalkWithDrift(ForecastModel):
    """``P[T+h] = P[T] * exp(h * mu)``, with drift from the training endpoints.

    Uses the classical estimator ``mu = (log P_T - log P_1) / (T - 1)``, which is
    the maximum-likelihood drift of a Gaussian random walk and depends only on
    the first and last observation. On a series that ran up over the training
    window this extrapolates that run-up indefinitely -- a property worth seeing
    plainly rather than hiding, since it is what many "trend" models do.
    """

    def __init__(self, name: str = "random_walk_drift", interval_level: float = 0.95) -> None:
        super().__init__(name)
        self.interval_level = interval_level
        self._last_log_price: float = float("nan")
        self._drift: float = float("nan")
        self._sigma: float = float("nan")

    @property
    def min_train_bars(self) -> int:
        return 30

    def describe(self) -> dict:
        return {**super().describe(), "interval_level": self.interval_level}

    def _fit(self, window: TrainingWindow) -> None:
        log_close = window.log_close
        self._last_log_price = float(log_close.iloc[-1])
        self._drift = float((log_close.iloc[-1] - log_close.iloc[0]) / (len(log_close) - 1))
        # Residual sd around the fitted drift, not raw return sd.
        self._sigma = float((window.log_returns.dropna() - self._drift).std(ddof=1))

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        point, lower, upper = random_walk_bands(
            self._last_log_price,
            drift_per_step=self._drift,
            sigma_per_step=self._sigma,
            steps=len(horizon),
            level=self.interval_level,
        )
        return self._result(
            point,
            target_bars,
            lower=lower,
            upper=upper,
            interval_level=self.interval_level,
            metadata={"drift_log_per_bar": self._drift, "sigma_log_return": self._sigma},
        )


class HistoricalMeanReturn(ForecastModel):
    """``P[T+h] = P[T] * (1 + rbar)^h`` from the mean *simple* return.

    Deliberately a different estimator from :class:`RandomWalkWithDrift`, not a
    restatement of it: the arithmetic mean of simple returns over a trailing
    window, rather than the geometric mean over the whole sample. Jensen's
    inequality makes the arithmetic mean the larger of the two whenever returns
    vary, so this baseline is systematically more bullish on a volatile series --
    a useful contrast, and a reminder that "average return" is ambiguous.

    ``lookback`` defaults to 365 bars so the estimate tracks the recent regime.
    """

    def __init__(
        self,
        name: str = "historical_mean_return",
        lookback: int | None = 365,
        interval_level: float = 0.95,
    ) -> None:
        super().__init__(name)
        self.lookback = lookback
        self.interval_level = interval_level
        self._last_price: float = float("nan")
        self._mean_return: float = float("nan")
        self._sigma: float = float("nan")
        self._n_used: int = 0

    @property
    def min_train_bars(self) -> int:
        return 30

    def describe(self) -> dict:
        return {**super().describe(), "lookback": self.lookback, "interval_level": self.interval_level}

    def _fit(self, window: TrainingWindow) -> None:
        returns = window.returns.dropna()
        if self.lookback is not None:
            returns = returns.iloc[-self.lookback :]

        self._last_price = float(window.close.iloc[-1])
        self._mean_return = float(returns.mean())
        self._n_used = len(returns)

        log_returns = window.log_returns.dropna()
        if self.lookback is not None:
            log_returns = log_returns.iloc[-self.lookback :]
        self._sigma = float(log_returns.std(ddof=1))

    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult:
        steps = np.arange(1, len(horizon) + 1, dtype=float)
        point = self._last_price * np.power(1.0 + self._mean_return, steps)

        # Bands are centred on this model's own point forecast, widening with
        # sqrt(h) as a random walk's do.
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + self.interval_level / 2.0))
        sd_log = self._sigma * np.sqrt(steps)
        lower = point * np.exp(-z * sd_log)
        upper = point * np.exp(z * sd_log)

        return self._result(
            point,
            target_bars,
            lower=lower,
            upper=upper,
            interval_level=self.interval_level,
            metadata={
                "mean_simple_return": self._mean_return,
                "n_returns_used": self._n_used,
                "lookback": self.lookback,
            },
        )


def default_baselines() -> list[ForecastModel]:
    """The comparison set every candidate model is scored against."""
    return [RandomWalk(), RandomWalkWithDrift(), HistoricalMeanReturn()]


__all__ = [
    "HistoricalMeanReturn",
    "RandomWalk",
    "RandomWalkWithDrift",
    "default_baselines",
]
