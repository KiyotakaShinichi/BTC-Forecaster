"""Train-only feature selection, as a fit/transform pair.

Selection is a model decision, so it has to be *refitted inside every
walk-forward fold*. The pre-Track-A pipeline selected lags, rolling windows and
EMA/SMA spans once, on everything except the final holdout, and then reported
walk-forward folds whose test windows lay inside that selection window. Every
fold's "out-of-sample" score was contaminated by feature choices made with
knowledge of that fold's future.

Making the selector a fitted object with an explicit ``fitted_on`` window means
the leak is structurally hard to reintroduce, and :class:`FeatureSelection`
records what it saw so an audit can check it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .spec import (
    Ema,
    FeatureSpec,
    LagReturn,
    PriceOverSma,
    RollingMeanReturn,
    RollingMeanVolume,
    RollingStdReturn,
    simple_returns,
)

DEFAULT_ROLLING_CANDIDATES: tuple[int, ...] = (3, 5, 7, 14, 21, 30, 60)
DEFAULT_EMA_CANDIDATES: tuple[int, ...] = (3, 5, 7, 14, 21, 30)


@dataclass(frozen=True)
class FeatureSelection:
    """The outcome of fitting a selector, and the evidence behind it."""

    specs: tuple[FeatureSpec, ...]
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    ema_spans: tuple[int, ...]
    fitted_start: pd.Timestamp
    fitted_end: pd.Timestamp
    n_observations: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def to_dict(self) -> dict:
        return {
            "features": self.names,
            "lags": list(self.lags),
            "rolling_windows": list(self.rolling_windows),
            "ema_spans": list(self.ema_spans),
            "fitted_on": {
                "start": self.fitted_start.isoformat(),
                "end": self.fitted_end.isoformat(),
                "n_observations": self.n_observations,
            },
            "diagnostics": self.diagnostics,
        }


def select_pacf_lags(
    returns: pd.Series,
    *,
    max_lag: int = 60,
    alpha: float = 0.05,
    max_selected: int = 10,
) -> tuple[list[int], dict]:
    """Lags whose partial autocorrelation exceeds the white-noise band.

    The band is the standard +/- z * n**-0.5 approximation. Note this is a
    multiple-comparison procedure: scanning 60 lags at a 5% band will flag ~3
    lags on pure noise by construction. ``max_selected`` caps the damage by
    keeping only the strongest, and the count of significant lags is returned so
    a caller can see whether the result is distinguishable from chance.
    """
    from scipy.stats import norm  # local import: keeps module import cheap
    from statsmodels.tsa.stattools import pacf

    series = returns.dropna()
    n = len(series)
    if n <= max_lag + 2:
        max_lag = max(1, (n // 2) - 2)

    values = pacf(series, nlags=max_lag)
    threshold = float(norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n))

    candidates = [(lag, abs(float(values[lag]))) for lag in range(1, len(values))]
    significant = [(lag, mag) for lag, mag in candidates if mag > threshold]
    significant.sort(key=lambda item: item[1], reverse=True)

    selected = sorted(lag for lag, _ in significant[:max_selected]) or [1]

    expected_by_chance = alpha * max_lag
    strongest_lag, strongest_magnitude = (
        significant[0] if significant else max(candidates, key=lambda item: item[1])
    )

    # The count of significant lags alone cannot distinguish signal from noise:
    # scanning 20 lags at alpha=0.05 flags one by chance, and a genuine AR(1)
    # may also flag exactly one. Magnitude relative to the band is the
    # discriminating quantity -- a real autoregressive term sits far above it,
    # a spurious hit sits just over the line.
    diagnostics = {
        "n_observations": n,
        "max_lag_scanned": int(max_lag),
        "significance_threshold": threshold,
        "n_significant": len(significant),
        "n_expected_by_chance": float(expected_by_chance),
        "n_significant_exceeds_chance": len(significant) > expected_by_chance,
        "strongest_lag": int(strongest_lag),
        "strongest_magnitude": float(strongest_magnitude),
        "strongest_over_threshold": float(strongest_magnitude / threshold),
        "fell_back_to_lag_1": not significant,
    }
    return selected, diagnostics


def _abs_corr_with_next_return(candidate: pd.Series, next_return: pd.Series) -> float:
    aligned = pd.concat([candidate, next_return], axis=1).dropna()
    if len(aligned) < 30 or aligned.iloc[:, 0].std() == 0:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return 0.0 if pd.isna(corr) else abs(float(corr))


def rank_rolling_windows(
    frame: pd.DataFrame,
    *,
    candidates: tuple[int, ...] = DEFAULT_ROLLING_CANDIDATES,
) -> dict[int, float]:
    """Score each rolling window by |corr| with the *next* bar's return.

    ``shift(-1)`` here is not leakage: it is how a supervised score is defined.
    The result is a ranking computed strictly inside the training window, and
    the target it correlates against is the training window's own future, never
    the evaluation period's.
    """
    returns = simple_returns(frame)
    next_return = returns.shift(-1)

    scores: dict[int, float] = {}
    for window in candidates:
        parts = [
            _abs_corr_with_next_return(returns.rolling(window).mean(), next_return),
            _abs_corr_with_next_return(returns.rolling(window).std(), next_return),
            _abs_corr_with_next_return(frame["volume"].rolling(window).mean(), next_return),
        ]
        scores[window] = float(np.mean(parts))
    return scores


def rank_ema_spans(
    frame: pd.DataFrame,
    *,
    candidates: tuple[int, ...] = DEFAULT_EMA_CANDIDATES,
) -> dict[int, float]:
    """Score EMA spans by |corr| of the price/EMA *ratio* with the next return.

    Scored on the ratio rather than the raw EMA level. A raw EMA is essentially
    the price, so its correlation with anything is dominated by the shared unit
    root and ranks spans almost arbitrarily -- the previous implementation's
    EMA/SMA choice was close to noise for this reason.
    """
    returns = simple_returns(frame)
    next_return = returns.shift(-1)

    scores: dict[int, float] = {}
    for span in candidates:
        ema = frame["close"].ewm(span=span, adjust=False).mean()
        ratio = frame["close"] / ema - 1.0
        scores[span] = _abs_corr_with_next_return(ratio, next_return)
    return scores


@dataclass
class FeatureSelector:
    """Fits a feature set on training data only.

    Instantiate once, call :meth:`fit` per walk-forward fold with that fold's
    training window, and use the returned selection for that fold alone.
    """

    max_lag: int = 60
    alpha: float = 0.05
    max_lags_selected: int = 10
    n_rolling_windows: int = 3
    n_ema_spans: int = 2
    rolling_candidates: tuple[int, ...] = DEFAULT_ROLLING_CANDIDATES
    ema_candidates: tuple[int, ...] = DEFAULT_EMA_CANDIDATES
    include_volume: bool = True

    def fit(self, train_frame: pd.DataFrame) -> FeatureSelection:
        if train_frame.empty:
            raise ValueError("cannot fit feature selection on an empty frame")

        returns = simple_returns(train_frame)
        lags, pacf_diagnostics = select_pacf_lags(
            returns,
            max_lag=self.max_lag,
            alpha=self.alpha,
            max_selected=self.max_lags_selected,
        )

        rolling_scores = rank_rolling_windows(train_frame, candidates=self.rolling_candidates)
        top_windows = sorted(
            sorted(rolling_scores, key=lambda w: rolling_scores[w], reverse=True)[
                : self.n_rolling_windows
            ]
        )

        ema_scores = rank_ema_spans(train_frame, candidates=self.ema_candidates)
        top_spans = sorted(
            sorted(ema_scores, key=lambda s: ema_scores[s], reverse=True)[: self.n_ema_spans]
        )

        specs: list[FeatureSpec] = [LagReturn.of(lag) for lag in lags]
        for window in top_windows:
            specs.append(RollingMeanReturn.of(window))
            specs.append(RollingStdReturn.of(window))
            specs.append(PriceOverSma.of(window))
            if self.include_volume:
                specs.append(RollingMeanVolume.of(window))
        specs.extend(Ema.of(span) for span in top_spans)

        return FeatureSelection(
            specs=tuple(specs),
            lags=tuple(lags),
            rolling_windows=tuple(top_windows),
            ema_spans=tuple(top_spans),
            fitted_start=train_frame.index.min(),
            fitted_end=train_frame.index.max(),
            n_observations=len(train_frame),
            diagnostics={
                "pacf": pacf_diagnostics,
                "rolling_scores": {str(k): round(v, 6) for k, v in rolling_scores.items()},
                "ema_scores": {str(k): round(v, 6) for k, v in ema_scores.items()},
            },
        )


__all__ = [
    "DEFAULT_EMA_CANDIDATES",
    "DEFAULT_ROLLING_CANDIDATES",
    "FeatureSelection",
    "FeatureSelector",
    "rank_ema_spans",
    "rank_rolling_windows",
    "select_pacf_lags",
]
