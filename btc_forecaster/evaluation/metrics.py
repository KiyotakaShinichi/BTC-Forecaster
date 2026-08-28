"""Forecast evaluation metrics.

Directional accuracy alone drove every decision in the original pipeline. It is
kept, but corrected and demoted to one column among several, because on its own
it is a poor guide: it ignores magnitude entirely, so a model can be directionally
right on ninety small moves and catastrophically wrong on the ten large ones and
still score 0.9.

The metrics here fall into four groups:

* **Magnitude** -- MAE, RMSE, MAPE, sMAPE, MASE.
* **Direction** -- directional accuracy, and the binomial test with its
  independence caveat attached rather than implied.
* **Association** -- correlation between forecast and realised returns.
* **Interval quality** -- coverage, width, and the Winkler interval score, which
  is a proper scoring rule and therefore cannot be gamed by widening the band.

Coverage matters especially here: the original's Monte Carlo bands did not widen
with horizon (see :mod:`btc_forecaster.models.volatility`), and nothing in the
pipeline measured whether its stated 95% interval contained 95% of outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPSILON = 1e-12


def _as_arrays(*series: pd.Series | np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = [np.asarray(s, dtype=float) for s in series]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"length mismatch between inputs: {sorted(lengths)}")
    return tuple(arrays)


# ---------------------------------------------------------------- magnitude


def mae(actual, predicted) -> float:
    a, p = _as_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a, p = _as_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted) -> float:
    """Mean absolute percentage error, in percent.

    Well-defined here only because BTC prices are strictly positive and far from
    zero. On a return series -- which crosses zero constantly -- MAPE is
    meaningless, so it is reported for price forecasts only.
    """
    a, p = _as_arrays(actual, predicted)
    if np.any(np.abs(a) < EPSILON):
        return float("nan")
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


def smape(actual, predicted) -> float:
    """Symmetric MAPE, in percent, bounded at 200.

    Preferred to MAPE when the two series can differ by a lot: MAPE is unbounded
    above and penalises over-forecasting far more heavily than under-forecasting.
    """
    a, p = _as_arrays(actual, predicted)
    denominator = (np.abs(a) + np.abs(p)) / 2.0
    mask = denominator > EPSILON
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(a[mask] - p[mask]) / denominator[mask]) * 100.0)


def mase(actual, predicted, *, naive_scale: float) -> float:
    """Mean absolute scaled error.

    Scaled by the in-sample mean absolute error of a one-step naive forecast, so
    ``MASE < 1`` means "better than a random walk was on the training data".
    Unit-free, defined everywhere, and not skewed by the level of the series --
    which is why it is preferred to MAPE for cross-period comparison.
    """
    if not np.isfinite(naive_scale) or naive_scale < EPSILON:
        return float("nan")
    return float(mae(actual, predicted) / naive_scale)


def naive_scale_from_training(train_prices: pd.Series | np.ndarray) -> float:
    """The MASE denominator: in-sample MAE of a one-step naive forecast."""
    values = np.asarray(train_prices, dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(values))))


# ---------------------------------------------------------------- direction


def directional_accuracy(actual, predicted, *, reference) -> float:
    """Fraction of bars whose move *away from the last known price* is called right.

    Each prediction is compared against ``reference`` -- the most recent actually
    observed close before that bar -- for both the forecast and the outcome::

        correct[t] = sign(predicted[t] - reference[t]) == sign(actual[t] - reference[t])

    This is the tradeable quantity: it answers "did the model say up, and was it
    up, relative to the price you could have transacted at".

    The original pipeline instead computed ``np.diff`` of both series, comparing
    the forecast path against *itself* rather than against the last observed
    price. See :func:`path_directional_accuracy` for that quantity and why the
    two differ.
    """
    a, p, r = _as_arrays(actual, predicted, reference)
    actual_up = (a - r) > 0
    predicted_up = (p - r) > 0
    if len(a) == 0:
        return float("nan")
    return float(np.mean(actual_up == predicted_up))


def path_directional_accuracy(actual, predicted) -> float:
    """Agreement between the *shapes* of the two paths: ``sign(diff)`` on each.

    This is what the pre-Track-A pipeline reported as "directional accuracy".
    It is a legitimate measure of whether a forecast path has the right shape,
    but it is not a measure of trading direction: a forecast that is uniformly
    wrong in level but correctly shaped scores perfectly, and no position could
    have been taken on it. Reported alongside the real thing so the historical
    numbers remain interpretable.
    """
    a, p = _as_arrays(actual, predicted)
    if len(a) < 2:
        return float("nan")
    return float(np.mean((np.diff(a) > 0) == (np.diff(p) > 0)))


@dataclass(frozen=True)
class DirectionTest:
    """A binomial test on directional hit rate, with its assumptions stated."""

    n_correct: int
    n_total: int
    accuracy: float
    p_value: float
    caveat: str = (
        "Assumes independent Bernoulli trials. Daily direction outcomes from a "
        "model with overlapping rolling features are serially dependent, so this "
        "p-value is anti-conservative. It is also uncorrected for multiple "
        "comparisons: if several models, cutoffs or feature sets were searched, "
        "the effective significance threshold is far stricter than the nominal "
        "one. Read it as a rough signal, never as a guarantee."
    )

    def to_dict(self) -> dict:
        return {
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "accuracy": self.accuracy,
            "p_value": self.p_value,
            "caveat": self.caveat,
        }


def binomial_direction_test(n_correct: int, n_total: int) -> DirectionTest:
    """One-sided binomial test of directional accuracy against a fair coin.

    Retained from the original pipeline. The independence limitation travels
    with the result rather than living in a comment, so it cannot be quoted
    without it.
    """
    from scipy.stats import binomtest

    if n_total <= 0:
        return DirectionTest(0, 0, float("nan"), float("nan"))

    result = binomtest(k=int(n_correct), n=int(n_total), p=0.5, alternative="greater")
    return DirectionTest(
        n_correct=int(n_correct),
        n_total=int(n_total),
        accuracy=float(n_correct / n_total),
        p_value=float(result.pvalue),
    )


# -------------------------------------------------------------- association


def return_forecast_correlation(actual, predicted, *, reference) -> float:
    """Pearson correlation between forecast and realised returns from ``reference``.

    Uses magnitude as well as sign, so unlike directional accuracy it rewards
    being right about big moves more than small ones -- closer to what a
    position-sized strategy would actually earn.
    """
    a, p, r = _as_arrays(actual, predicted, reference)
    actual_return = a / np.where(np.abs(r) < EPSILON, np.nan, r) - 1.0
    predicted_return = p / np.where(np.abs(r) < EPSILON, np.nan, r) - 1.0

    mask = np.isfinite(actual_return) & np.isfinite(predicted_return)
    if mask.sum() < 3:
        return float("nan")
    if np.std(actual_return[mask]) < EPSILON or np.std(predicted_return[mask]) < EPSILON:
        return float("nan")
    return float(np.corrcoef(actual_return[mask], predicted_return[mask])[0, 1])


# ----------------------------------------------------------------- interval


def interval_coverage(actual, lower, upper) -> float:
    """Fraction of outcomes inside the interval. A 95% band should score ~0.95."""
    a, lo, hi = _as_arrays(actual, lower, upper)
    if len(a) == 0:
        return float("nan")
    return float(np.mean((a >= lo) & (a <= hi)))


def mean_interval_width(lower, upper) -> float:
    lo, hi = _as_arrays(lower, upper)
    return float(np.mean(hi - lo))


def relative_interval_width(actual, lower, upper) -> float:
    """Mean interval width as a fraction of the realised price.

    Width in dollars is not comparable across periods when the price level moves
    by an order of magnitude, which BTC's does.
    """
    a, lo, hi = _as_arrays(actual, lower, upper)
    mask = np.abs(a) > EPSILON
    if not mask.any():
        return float("nan")
    return float(np.mean((hi[mask] - lo[mask]) / a[mask]))


def winkler_score(actual, lower, upper, *, level: float = 0.95) -> float:
    """Winkler interval score: a proper scoring rule. Lower is better.

    Charges the interval's width plus a penalty for each outcome that falls
    outside it, scaled by ``2/alpha``. Coverage and width cannot be traded off
    against each other to game it -- an infinitely wide band scores infinitely
    badly, and a zero-width band pays the miss penalty on every observation.
    This is why it is reported alongside raw coverage.
    """
    a, lo, hi = _as_arrays(actual, lower, upper)
    alpha = 1.0 - level
    width = hi - lo
    penalty = np.zeros_like(width)
    below = a < lo
    above = a > hi
    penalty[below] = (2.0 / alpha) * (lo[below] - a[below])
    penalty[above] = (2.0 / alpha) * (a[above] - hi[above])
    return float(np.mean(width + penalty))


def pinball_loss(actual, quantile_forecast, *, quantile: float) -> float:
    """Quantile (pinball) loss. The building block of CRPS."""
    a, q = _as_arrays(actual, quantile_forecast)
    diff = a - q
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


# ------------------------------------------------------------------ skill


def skill_score(model_error: float, baseline_error: float) -> float:
    """``1 - model/baseline``. Positive means the model beat the baseline.

    The number that decides whether a model is worth its complexity. The
    original pipeline never computed anything like it: it compared directional
    accuracy to a fixed 0.5 coin, which a model can beat while still losing to
    the random walk.
    """
    if not np.isfinite(baseline_error) or abs(baseline_error) < EPSILON:
        return float("nan")
    return float(1.0 - model_error / baseline_error)


# ----------------------------------------------------------------- bundle


@dataclass(frozen=True)
class ForecastMetrics:
    """Every metric for one forecast against one realised outcome."""

    n: int
    mae: float
    rmse: float
    mape: float
    smape: float
    mase: float
    directional_accuracy: float
    path_directional_accuracy: float
    return_correlation: float
    interval_coverage: float = float("nan")
    coverage_error: float = float("nan")
    mean_interval_width: float = float("nan")
    relative_interval_width: float = float("nan")
    winkler_score: float = float("nan")
    interval_level: float = float("nan")
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "smape": self.smape,
            "mase": self.mase,
            "directional_accuracy": self.directional_accuracy,
            "path_directional_accuracy": self.path_directional_accuracy,
            "return_correlation": self.return_correlation,
            "interval_coverage": self.interval_coverage,
            "coverage_error": self.coverage_error,
            "mean_interval_width": self.mean_interval_width,
            "relative_interval_width": self.relative_interval_width,
            "winkler_score": self.winkler_score,
            "interval_level": self.interval_level,
        }
        payload.update(self.extra)
        return payload


def evaluate_forecast(
    actual: pd.Series,
    point: pd.Series,
    *,
    reference: pd.Series | float,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    interval_level: float | None = None,
    naive_scale: float = float("nan"),
) -> ForecastMetrics:
    """Score one forecast against the realised path.

    ``reference`` is the last actually-observed close before each target bar.
    Passing a scalar -- the origin's close -- is the right choice for a
    multi-step forecast issued from a single origin, since that is genuinely the
    last price known when the forecast was made.
    """
    aligned = point.index.intersection(actual.index)
    if len(aligned) == 0:
        raise ValueError("forecast and actual series do not overlap")

    a = actual.loc[aligned]
    p = point.loc[aligned]
    r = (
        pd.Series(float(reference), index=aligned)
        if np.isscalar(reference)
        else pd.Series(reference).loc[aligned]
    )

    interval_metrics: dict = {}
    if lower is not None and upper is not None:
        lo = lower.loc[aligned]
        hi = upper.loc[aligned]
        level = interval_level if interval_level is not None else 0.95
        observed_coverage = interval_coverage(a, lo, hi)
        interval_metrics = {
            "interval_coverage": observed_coverage,
            "coverage_error": abs(observed_coverage - level),
            "mean_interval_width": mean_interval_width(lo, hi),
            "relative_interval_width": relative_interval_width(a, lo, hi),
            "winkler_score": winkler_score(a, lo, hi, level=level),
            "interval_level": float(level),
        }

    return ForecastMetrics(
        n=len(aligned),
        mae=mae(a, p),
        rmse=rmse(a, p),
        mape=mape(a, p),
        smape=smape(a, p),
        mase=mase(a, p, naive_scale=naive_scale),
        directional_accuracy=directional_accuracy(a, p, reference=r),
        path_directional_accuracy=path_directional_accuracy(a, p),
        return_correlation=return_forecast_correlation(a, p, reference=r),
        **interval_metrics,
    )


#: Metrics where a smaller value is better. Used when ranking models.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"mae", "rmse", "mape", "smape", "mase", "mean_interval_width",
     "relative_interval_width", "winkler_score", "coverage_error"}
)

#: Metrics where a larger value is better.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"directional_accuracy", "path_directional_accuracy", "return_correlation"}
)

#: Calibration metrics, where neither direction is "better": a 95% interval
#: should cover 95% of outcomes, and both 0.60 and 1.00 are miscalibrated. These
#: are reported but never used to rank models -- `coverage_error` (which IS
#: lower-is-better) is the rankable form.
CALIBRATION: frozenset[str] = frozenset({"interval_coverage", "interval_level"})

#: Everything the backtest summary aggregates across folds.
SUMMARY_METRICS: frozenset[str] = LOWER_IS_BETTER | HIGHER_IS_BETTER | CALIBRATION


__all__ = [
    "CALIBRATION",
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "SUMMARY_METRICS",
    "DirectionTest",
    "ForecastMetrics",
    "binomial_direction_test",
    "directional_accuracy",
    "evaluate_forecast",
    "interval_coverage",
    "mae",
    "mape",
    "mase",
    "mean_interval_width",
    "naive_scale_from_training",
    "path_directional_accuracy",
    "pinball_loss",
    "relative_interval_width",
    "return_forecast_correlation",
    "rmse",
    "skill_score",
    "smape",
    "winkler_score",
]
