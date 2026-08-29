"""Point-in-time market regime labelling.

The trap this module exists to avoid
------------------------------------
The obvious way to label regimes is to look at the whole price series, decide
2021 was a bull market and 2022 a bear, and tag the bars accordingly. Every such
label is computed from data that did not exist at the bar it labels. Conditioning
a backtest on it -- "the model does well in bull markets" -- is look-ahead of the
purest kind: you are telling the model which regime it is in using information
from the regime's own future.

Regimes here are labelled **at the forecast origin, from a trailing window
only**. The label attached to a fold is what could have been computed at that
fold's origin, so "this model performs badly in high-volatility regimes" is a
statement someone could have acted on.

The same future-mutation test that guards features guards these:
:func:`assert_regime_labels_are_causal` perturbs the future and asserts past
labels do not move.

Thresholds
----------
Two axes, deliberately simple:

* **Volatility** -- trailing realised volatility against its own trailing
  quantile. Not a fixed number: BTC's "high volatility" in 2017 and in 2025 are
  different absolute figures, so a fixed threshold would mostly label the
  calendar. The quantile is computed over an expanding history up to the origin.
* **Trend** -- trailing return over the lookback window against a threshold in
  return space, which is already scale-free.

The legacy pipeline had a regime detector with hardcoded cutoffs
(``trend > 0.10 and volatility < 0.05``) that fed nothing but a plot title. These
labels are used to stratify results, which is what makes them worth computing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..features.spec import log_returns


class VolatilityRegime(str, Enum):
    LOW = "low_vol"
    HIGH = "high_vol"


class TrendRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass(frozen=True)
class RegimeConfig:
    """How regimes are defined. Recorded in the manifest so labels are auditable."""

    vol_window: int = 30
    vol_quantile: float = 0.5
    trend_window: int = 90
    trend_threshold: float = 0.10
    min_history: int = 365

    def to_dict(self) -> dict:
        return {
            "vol_window": self.vol_window,
            "vol_quantile": self.vol_quantile,
            "trend_window": self.trend_window,
            "trend_threshold": self.trend_threshold,
            "min_history": self.min_history,
            "definition": (
                f"volatility: {self.vol_window}-bar realised vol vs its own expanding "
                f"{self.vol_quantile:.0%} quantile; trend: {self.trend_window}-bar return "
                f"vs +/-{self.trend_threshold:.0%}. Both computed from trailing data only."
            ),
        }


def realised_volatility(frame: pd.DataFrame, window: int) -> pd.Series:
    """Trailing realised volatility of log returns, annualisation-free."""
    return log_returns(frame).rolling(window).std().rename("realised_vol")


def trailing_return(frame: pd.DataFrame, window: int) -> pd.Series:
    """Simple return over the trailing ``window`` bars."""
    close = frame["close"]
    return (close / close.shift(window) - 1.0).rename("trailing_return")


def label_regimes(frame: pd.DataFrame, config: RegimeConfig | None = None) -> pd.DataFrame:
    """Label every bar using only data available at that bar.

    Returns a frame with ``volatility_regime``, ``trend_regime``, ``regime``
    (the pair joined) and the underlying statistics. Rows before
    ``min_history`` are labelled NaN rather than guessed: an early bar has no
    trailing distribution to compare against, and inventing one would be the
    same error in miniature.
    """
    config = config or RegimeConfig()

    vol = realised_volatility(frame, config.vol_window)
    trend = trailing_return(frame, config.trend_window)

    # Expanding quantile: at bar D this uses volatility observations up to and
    # including D, never later ones. `.expanding().quantile()` is causal in
    # exactly the way `.quantile()` on the whole series would not be.
    vol_threshold = vol.expanding(min_periods=config.min_history).quantile(config.vol_quantile)

    out = pd.DataFrame(index=frame.index)
    out["realised_vol"] = vol
    out["vol_threshold"] = vol_threshold
    out["trailing_return"] = trend

    volatility_regime = pd.Series(pd.NA, index=frame.index, dtype="object")
    known_vol = vol.notna() & vol_threshold.notna()
    volatility_regime[known_vol & (vol > vol_threshold)] = VolatilityRegime.HIGH.value
    volatility_regime[known_vol & (vol <= vol_threshold)] = VolatilityRegime.LOW.value

    trend_regime = pd.Series(pd.NA, index=frame.index, dtype="object")
    known_trend = trend.notna()
    trend_regime[known_trend & (trend > config.trend_threshold)] = TrendRegime.BULL.value
    trend_regime[known_trend & (trend < -config.trend_threshold)] = TrendRegime.BEAR.value
    trend_regime[
        known_trend & (trend.abs() <= config.trend_threshold)
    ] = TrendRegime.SIDEWAYS.value

    out["volatility_regime"] = volatility_regime
    out["trend_regime"] = trend_regime
    out["regime"] = np.where(
        volatility_regime.notna() & trend_regime.notna(),
        volatility_regime.astype(str) + "/" + trend_regime.astype(str),
        None,
    )
    return out


def regime_at(frame: pd.DataFrame, origin_bar: pd.Timestamp, config: RegimeConfig | None = None) -> dict:
    """The regime label as it stood at ``origin_bar``, using bars up to it only."""
    observable = frame.loc[frame.index <= origin_bar]
    if observable.empty:
        raise ValueError(f"no observations at or before {origin_bar}")

    labels = label_regimes(observable, config)
    row = labels.iloc[-1]
    return {
        "origin": origin_bar,
        "volatility_regime": row["volatility_regime"],
        "trend_regime": row["trend_regime"],
        "regime": row["regime"],
        "realised_vol": float(row["realised_vol"]) if pd.notna(row["realised_vol"]) else None,
        "trailing_return": (
            float(row["trailing_return"]) if pd.notna(row["trailing_return"]) else None
        ),
    }


def label_fold_origins(
    frame: pd.DataFrame,
    origins: list[pd.Timestamp],
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """Regime at each forecast origin. One row per origin.

    Computed per origin from the truncated history rather than by slicing a
    whole-series labelling, so the causality is structural rather than assumed.
    """
    return pd.DataFrame([regime_at(frame, origin, config) for origin in origins])


def performance_by_regime(
    records: pd.DataFrame,
    origin_regimes: pd.DataFrame,
    *,
    by: str = "regime",
) -> pd.DataFrame:
    """Per-model, per-regime error and direction, from prediction records.

    ``records`` comes from
    :meth:`btc_forecaster.backtesting.engine.BacktestResult.prediction_records`;
    ``origin_regimes`` from :func:`label_fold_origins`.
    """
    if by not in origin_regimes.columns:
        raise ValueError(f"{by!r} is not a regime column; have {list(origin_regimes.columns)}")

    merged = records.merge(origin_regimes[["origin", by]], on="origin", how="left")
    merged = merged[merged[by].notna()]
    if merged.empty:
        return pd.DataFrame()

    merged["abs_error"] = (merged["actual"] - merged["predicted"]).abs()
    actual_return = merged["actual"] / merged["origin_close"] - 1.0
    predicted_return = merged["predicted"] / merged["origin_close"] - 1.0
    merged["direction_hit"] = (predicted_return > 0) == (actual_return > 0)

    grouped = merged.groupby(["model", by], sort=True).agg(
        n=("abs_error", "size"),
        n_origins=("origin", "nunique"),
        mae=("abs_error", "mean"),
        rmse=("abs_error", lambda s: float(np.sqrt(np.mean(s.to_numpy() ** 2)))),
        directional_accuracy=("direction_hit", "mean"),
    )
    return grouped.reset_index().set_index(["model", by])


def assert_regime_labels_are_causal(
    frame: pd.DataFrame,
    config: RegimeConfig | None = None,
    *,
    cut: int = -60,
    rng_seed: int = 0,
) -> None:
    """Perturb the future; assert past regime labels do not move.

    The same technique that guards features. It catches the specific mistake
    this module exists to prevent: computing a quantile or a threshold over the
    whole series and back-labelling the past with it.
    """
    rng = np.random.default_rng(rng_seed)
    cut_bar = frame.index[cut]

    baseline = label_regimes(frame, config).loc[:cut_bar]

    perturbed = frame.copy()
    tail = perturbed.index > cut_bar
    perturbed.loc[tail, "close"] *= 1.0 + rng.normal(scale=0.4, size=int(tail.sum()))
    perturbed.loc[tail, "close"] = perturbed.loc[tail, "close"].abs().clip(lower=1e-6)

    after = label_regimes(perturbed, config).loc[:cut_bar]

    for column in ("volatility_regime", "trend_regime", "regime"):
        left = baseline[column].fillna("<na>")
        right = after[column].fillna("<na>")
        if not left.equals(right):
            first = left.compare(right).index[0]
            raise AssertionError(
                f"regime label {column!r} is not causal: it changed at {first} when "
                f"data after {cut_bar} was modified"
            )


__all__ = [
    "RegimeConfig",
    "TrendRegime",
    "VolatilityRegime",
    "assert_regime_labels_are_causal",
    "label_fold_origins",
    "label_regimes",
    "performance_by_regime",
    "realised_volatility",
    "regime_at",
    "trailing_return",
]
