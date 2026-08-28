"""Feature specifications with point-in-time semantics.

Every feature here is **causal**: the value stamped at bar ``D`` is a function of
bars ``<= D`` only. That property is what :mod:`tests.test_leakage` verifies by
mutating the future and asserting past feature values do not move.

Causal is not the same as *usable*. A causal feature at bar ``D`` still contains
information from bar ``D`` itself -- including that bar's close. Using it to
predict bar ``D`` would be circular. The alignment step in
:func:`btc_forecaster.features.pipeline.to_supervised` is what enforces the gap
between the last bar a feature may see and the bar being forecast.

This distinction is the single most important thing in the module. The
pre-Track-A pipeline built features that were causal in exactly this sense and
then predicted the *same* bar with them, which is why its holdout directional
accuracy looked strong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..timebase import ZERO_LAG, available_time


def simple_returns(frame: pd.DataFrame) -> pd.Series:
    """Close-to-close simple return. ``r[D]`` needs ``close[D]`` and ``close[D-1]``."""
    return frame["close"].pct_change().rename("return")


def log_price(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame["close"]).rename("log_close")


def log_returns(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame["close"]).diff().rename("log_return")


@dataclass(frozen=True)
class FeatureSpec:
    """A named, causal transformation of a market frame.

    ``publication_lag`` records how long after its bar closes the input becomes
    knowable. It is zero for everything derived from spot price and volume; it
    exists so that Track B's exogenous signals -- which are published late and
    revised -- can be declared honestly and checked by the same machinery.
    """

    name: str
    publication_lag: pd.Timedelta = field(default=ZERO_LAG)

    def compute(self, frame: pd.DataFrame) -> pd.Series:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def min_history(self) -> int:
        """Bars needed before this feature produces a non-NaN value."""
        return 1

    def available_at(self, bar_start: pd.Timestamp) -> pd.Timestamp:
        """When the value stamped at ``bar_start`` may first be acted on."""
        return available_time(bar_start, self.publication_lag)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


@dataclass(frozen=True)
class LagReturn(FeatureSpec):
    """Simple return from ``lag`` bars ago."""

    lag: int = 1

    def __post_init__(self) -> None:
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")

    @classmethod
    def of(cls, lag: int) -> "LagReturn":
        return cls(name=f"lag_ret_{lag}", lag=lag)

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        return simple_returns(frame).shift(self.lag).rename(self.name)

    @property
    def min_history(self) -> int:
        return self.lag + 1


@dataclass(frozen=True)
class RollingMeanReturn(FeatureSpec):
    """Mean return over a trailing window ending at the current bar."""

    window: int = 7

    @classmethod
    def of(cls, window: int) -> "RollingMeanReturn":
        return cls(name=f"roll_mean_ret_{window}", window=window)

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        return simple_returns(frame).rolling(self.window).mean().rename(self.name)

    @property
    def min_history(self) -> int:
        return self.window + 1


@dataclass(frozen=True)
class RollingStdReturn(FeatureSpec):
    """Realised volatility proxy over a trailing window."""

    window: int = 7

    @classmethod
    def of(cls, window: int) -> "RollingStdReturn":
        return cls(name=f"roll_std_ret_{window}", window=window)

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        return simple_returns(frame).rolling(self.window).std().rename(self.name)

    @property
    def min_history(self) -> int:
        return self.window + 1


@dataclass(frozen=True)
class RollingMeanVolume(FeatureSpec):
    window: int = 7

    @classmethod
    def of(cls, window: int) -> "RollingMeanVolume":
        return cls(name=f"roll_vol_{window}", window=window)

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        return frame["volume"].rolling(self.window).mean().rename(self.name)

    @property
    def min_history(self) -> int:
        return self.window


@dataclass(frozen=True)
class Ema(FeatureSpec):
    """Exponential moving average of close.

    Note a reproducibility subtlety: an EMA computed over a *slice* differs from
    the same EMA computed over the full series, because the recursion is seeded
    at the first observation. Both are causal, but they are not equal, so the
    pipeline always computes features over the longest available history rather
    than over the training window alone.
    """

    span: int = 7

    @classmethod
    def of(cls, span: int) -> "Ema":
        return cls(name=f"ema_{span}", span=span)

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        return frame["close"].ewm(span=self.span, adjust=False).mean().rename(self.name)

    @property
    def min_history(self) -> int:
        return 1


@dataclass(frozen=True)
class Sma(FeatureSpec):
    window: int = 7

    @classmethod
    def of(cls, window: int) -> "Sma":
        return cls(name=f"sma_{window}", window=window)

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        return frame["close"].rolling(self.window).mean().rename(self.name)

    @property
    def min_history(self) -> int:
        return self.window


@dataclass(frozen=True)
class PriceOverSma(FeatureSpec):
    """Close relative to its own trailing mean: a scale-free trend feature.

    Raw ``ema_*``/``sma_*`` levels are non-stationary and share the price's unit
    root, which makes them poor tree-model inputs -- a boosted tree can only
    split on levels it saw in training, so a new all-time high is out of
    support. This ratio is the stationary version.
    """

    window: int = 20

    @classmethod
    def of(cls, window: int) -> "PriceOverSma":
        return cls(name=f"close_over_sma_{window}", window=window)

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        sma = frame["close"].rolling(self.window).mean()
        return (frame["close"] / sma - 1.0).rename(self.name)

    @property
    def min_history(self) -> int:
        return self.window


def default_specs(
    lags: tuple[int, ...] = (1, 7),
    windows: tuple[int, ...] = (7, 14),
    ema_spans: tuple[int, ...] = (7,),
) -> list[FeatureSpec]:
    """A reasonable starting feature set, used when selection is disabled."""
    specs: list[FeatureSpec] = [LagReturn.of(lag) for lag in lags]
    for window in windows:
        specs.append(RollingMeanReturn.of(window))
        specs.append(RollingStdReturn.of(window))
        specs.append(RollingMeanVolume.of(window))
        specs.append(PriceOverSma.of(window))
    specs.extend(Ema.of(span) for span in ema_spans)
    return specs


__all__ = [
    "Ema",
    "FeatureSpec",
    "LagReturn",
    "PriceOverSma",
    "RollingMeanReturn",
    "RollingMeanVolume",
    "RollingStdReturn",
    "Sma",
    "default_specs",
    "log_price",
    "log_returns",
    "simple_returns",
]
