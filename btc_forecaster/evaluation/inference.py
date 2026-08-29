"""Uncertainty for walk-forward metrics, accounting for serial dependence.

The problem with the naive procedure
------------------------------------
The legacy pipeline reported a binomial p-value on directional accuracy. That
test assumes independent Bernoulli trials, and walk-forward direction outcomes
are not independent for at least three reasons:

1. **Overlapping horizons.** An h-step forecast from origin t and one from
   origin t+1 share h-1 target bars. Their errors are mechanically correlated.
2. **Within-fold correlation.** Scoring h bars against a single origin gives h
   observations that all move together when the market trends away from that
   origin.
3. **Volatility clustering.** Errors are large in the same periods, so even
   non-overlapping forecasts are heteroskedastic and dependent.

Under positive dependence a naive test *overstates* significance, sometimes
severely: the effective sample size can be a small fraction of the nominal one.

What this module does instead
-----------------------------
* :func:`stationary_bootstrap` (Politis-Romano 1994) resamples geometric-length
  blocks, preserving short-range dependence while remaining stationary. Block
  length is the tuning knob: too short and dependence is destroyed, too long and
  the resample is nearly the original series.
* :func:`directional_accuracy_ci` gives a confidence interval for hit rate that
  a coin flip must fall outside of, which is a stronger and more honest claim
  than a p-value.
* :func:`diebold_mariano` compares two models' loss differentials with a HAC
  variance and the Harvey-Leybourne-Newbold small-sample correction.
* :func:`compare_models` runs the pairwise comparisons and applies a
  Benjamini-Hochberg correction across them, because comparing K models to a
  baseline is K-1 tests, not one.

Everything here is **exploratory unless the correction is applied**, and results
say so in their own fields.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..diagnostics.multiple_testing import benjamini_hochberg

DEFAULT_RESAMPLES = 2000


# --------------------------------------------------------------- resampling


def stationary_bootstrap_indices(
    n: int,
    *,
    expected_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Politis-Romano resample of positions ``0..n-1``.

    Blocks have geometric length with mean ``expected_block_length`` and wrap
    around the end of the series. The random block length is what makes the
    resampled series stationary, unlike a fixed-block scheme whose properties
    depend on where the blocks happen to fall.
    """
    if n < 1:
        raise ValueError("cannot bootstrap an empty sample")
    if expected_block_length < 1:
        raise ValueError("expected_block_length must be >= 1")

    p = 1.0 / expected_block_length
    indices = np.empty(n, dtype=np.int64)
    current = int(rng.integers(0, n))

    for i in range(n):
        indices[i] = current
        if rng.random() < p:
            current = int(rng.integers(0, n))
        else:
            current = (current + 1) % n
    return indices


def moving_block_indices(
    n: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One fixed-length moving-block resample. Simpler, non-stationary."""
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    block_length = min(block_length, n)

    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    indices = np.concatenate([(np.arange(block_length) + s) % n for s in starts])
    return indices[:n]


def optimal_block_length(sample: np.ndarray) -> float:
    """A pragmatic block-length rule: ``n**(1/3)``, floored at 2.

    Not the Politis-White data-driven estimator -- that requires spectral
    estimation and is fragile on the 20-80 observation samples a walk-forward
    study produces. ``n**(1/3)`` is the standard rate for block bootstrap
    consistency and is reported in every result so a reader can see what was
    assumed rather than having to infer it.
    """
    n = len(sample)
    return float(max(2.0, round(n ** (1.0 / 3.0))))


@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap confidence interval, with the assumptions that produced it."""

    statistic: float
    lower: float
    upper: float
    level: float
    n_observations: int
    n_resamples: int
    method: str
    block_length: float
    null_value: float | None = None

    @property
    def excludes_null(self) -> bool:
        """Whether the interval excludes the null. **Two-sided.**

        True for an interval entirely *below* the null as well as one entirely
        above it, so this must never be read as "beat the null" on its own --
        use :attr:`exceeds_null` for that. A flat random-walk forecast scores
        0.333 against a 0.611 base rate: its interval excludes the null while
        being significantly *worse* than it.
        """
        if self.null_value is None:
            return False
        return bool(not (self.lower <= self.null_value <= self.upper))

    @property
    def exceeds_null(self) -> bool:
        """Whether the whole interval lies above the null. One-sided."""
        if self.null_value is None:
            return False
        return bool(self.lower > self.null_value)

    @property
    def below_null(self) -> bool:
        """Whether the whole interval lies below the null. One-sided."""
        if self.null_value is None:
            return False
        return bool(self.upper < self.null_value)

    @property
    def versus_null(self) -> str:
        """``"above"``, ``"below"`` or ``"indistinguishable"``."""
        if self.null_value is None:
            return "no null"
        if self.exceeds_null:
            return "above"
        if self.below_null:
            return "below"
        return "indistinguishable"

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "level": self.level,
            "n_observations": self.n_observations,
            "n_resamples": self.n_resamples,
            "method": self.method,
            "block_length": self.block_length,
            "null_value": self.null_value,
            "excludes_null": self.excludes_null,
            "exceeds_null": self.exceeds_null,
            "below_null": self.below_null,
            "versus_null": self.versus_null,
        }

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"{self.statistic:.4f} [{self.lower:.4f}, {self.upper:.4f}] "
            f"({self.level:.0%} {self.method}, n={self.n_observations}, "
            f"block={self.block_length:g})"
        )


def stationary_bootstrap(
    sample: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: float | None = None,
    seed: int = 0,
    null_value: float | None = None,
) -> BootstrapCI:
    """Percentile confidence interval for ``statistic`` under serial dependence.

    Deterministic given ``seed`` -- a test pins that, because a bootstrap whose
    answer moves between runs cannot be cited.
    """
    values = np.asarray(sample, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        raise ValueError(f"need at least 2 finite observations, got {n}")

    block = float(block_length) if block_length is not None else optimal_block_length(values)
    rng = np.random.default_rng(seed)

    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = stationary_bootstrap_indices(n, expected_block_length=block, rng=rng)
        draws[i] = statistic(values[idx])

    tail = (1.0 - level) / 2.0
    return BootstrapCI(
        statistic=float(statistic(values)),
        lower=float(np.percentile(draws, 100.0 * tail)),
        upper=float(np.percentile(draws, 100.0 * (1.0 - tail))),
        level=level,
        n_observations=n,
        n_resamples=n_resamples,
        method="stationary bootstrap (Politis-Romano)",
        block_length=block,
        null_value=null_value,
    )


def directional_base_rate(actual, reference) -> float:
    """Share of realised up-moves from the reference price.

    Together with :func:`best_constant_accuracy` this defines the null a
    directional claim actually has to beat.
    """
    a = np.asarray(actual, dtype=float)
    r = np.asarray(reference, dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.mean((a - r) > 0))


def best_constant_accuracy(base_rate: float) -> float:
    """Accuracy of the best *constant* directional predictor.

    A model that always says "up" scores ``base_rate``; one that always says
    "down" scores ``1 - base_rate``. The better of the two is free, requires no
    model, and is what any directional claim must exceed.

    This matters enormously here and is easy to get wrong. Over 31-day windows
    in the A2 sample BTC rose 61% of the time, so a constant "up" forecast
    scores 0.611 -- comfortably "better than a coin" while containing no
    information whatsoever. In the first 36-fold run ``random_walk_drift``
    predicted up on 100% of origins and scored exactly 0.611, and the legacy
    hybrid predicted up on 67% and scored 0.667. Tested against 0.5 both look
    significant; against the base rate neither has an edge.
    """
    if not np.isfinite(base_rate):
        return float("nan")
    return float(max(base_rate, 1.0 - base_rate))


def directional_accuracy_ci(
    hits: Sequence[bool] | np.ndarray,
    *,
    level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: float | None = None,
    seed: int = 0,
    null: float = 0.5,
) -> BootstrapCI:
    """Confidence interval for hit rate, robust to serial dependence.

    ``hits`` should be the per-origin outcome series at a single forecast
    distance, from
    :func:`btc_forecaster.evaluation.targets.one_step_direction_sample`, in
    chronological order -- the bootstrap's blocks are only meaningful if
    adjacent entries are adjacent in time.

    ``null`` defaults to 0.5 but **should usually be
    :func:`best_constant_accuracy` of the sample's base rate**. A coin is only
    the right null when up and down moves are equally likely, which over a
    multi-week horizon on a trending asset they are not.

    Read :attr:`BootstrapCI.excludes_null` rather than converting to a p-value:
    an interval states the magnitude of any edge as well as its sign.
    """
    return stationary_bootstrap(
        np.asarray(hits, dtype=float),
        np.mean,
        level=level,
        n_resamples=n_resamples,
        block_length=block_length,
        seed=seed,
        null_value=null,
    )


# ------------------------------------------------------- forecast comparison


#: Pairs where the first model is a special case of the second. The standard
#: Diebold-Mariano test is invalid for nested models -- under the null the loss
#: differential is degenerate and the statistic is not asymptotically normal
#: (Clark-West 2007 is the appropriate alternative). These are flagged rather
#: than silently reported.
KNOWN_NESTED_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("random_walk", "random_walk_drift"),   # drift = 0
        ("random_walk", "arima"),               # ARIMA(0,1,0)
        ("random_walk", "arima_auto"),
        ("random_walk", "sarimax"),
        ("random_walk", "historical_mean_return"),  # mean return = 0
        ("random_walk", "ets"),                 # ETS(A,N,N) with alpha=1
        ("arima", "sarimax"),                   # no exog, no seasonal terms
    }
)


def is_nested(model_a: str, model_b: str) -> bool:
    return (model_a, model_b) in KNOWN_NESTED_PAIRS or (model_b, model_a) in KNOWN_NESTED_PAIRS


@dataclass(frozen=True)
class DieboldMarianoResult:
    """A pairwise forecast comparison, with its assumptions attached."""

    model_a: str
    model_b: str
    mean_loss_difference: float
    statistic: float
    p_value: float
    n_observations: int
    horizon: int
    hac_lags: int
    loss: str
    small_sample_corrected: bool
    nested: bool
    caveats: tuple[str, ...] = field(default=())

    @property
    def favours(self) -> str:
        """Which model had the lower loss. Not a significance claim."""
        if not np.isfinite(self.mean_loss_difference):
            return "undetermined"
        return self.model_a if self.mean_loss_difference < 0 else self.model_b

    @property
    def usable(self) -> bool:
        """Whether the p-value should be interpreted at all.

        Cast to a builtin bool deliberately: np.isfinite returns np.bool_, which
        is not JSON-serialisable and would fail when this result reaches a run
        manifest rather than when it is computed.
        """
        return bool(not self.nested and np.isfinite(self.p_value))

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "mean_loss_difference": self.mean_loss_difference,
            "dm_statistic": self.statistic,
            "p_value": self.p_value,
            "n_observations": self.n_observations,
            "horizon": self.horizon,
            "hac_lags": self.hac_lags,
            "loss": self.loss,
            "small_sample_corrected": self.small_sample_corrected,
            "nested": self.nested,
            "usable": self.usable,
            "favours": self.favours,
            "caveats": list(self.caveats),
        }


def _hac_variance(differences: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of the loss differential.

    Overlapping h-step forecasts make the differential at least MA(h-1), so the
    autocovariances up to lag h-1 must be included or the variance is
    understated and the test over-rejects. Bartlett weights keep the estimate
    non-negative.
    """
    n = len(differences)
    centred = differences - differences.mean()
    variance = float(np.dot(centred, centred) / n)

    for k in range(1, min(lags, n - 1) + 1):
        gamma = float(np.dot(centred[k:], centred[:-k]) / n)
        weight = 1.0 - k / (lags + 1.0)
        variance += 2.0 * weight * gamma

    return variance


def diebold_mariano(
    loss_a: Sequence[float] | np.ndarray,
    loss_b: Sequence[float] | np.ndarray,
    *,
    model_a: str = "a",
    model_b: str = "b",
    horizon: int = 1,
    loss: str = "absolute",
    small_sample: bool = True,
    nested: bool | None = None,
) -> DieboldMarianoResult:
    """Diebold-Mariano test of equal predictive accuracy.

    Null: the two models have equal expected loss. The statistic is the mean
    loss differential divided by its HAC standard error, referred to a
    Student-t distribution with ``n-1`` degrees of freedom when
    ``small_sample`` is set.

    Assumptions, all of which are recorded on the result:

    * The loss differential is covariance-stationary with finite variance.
    * Autocorrelation vanishes beyond lag ``horizon-1``; HAC lags are set
      accordingly. This is the standard assumption for h-step forecasts and is
      why overlapping horizons do not by themselves invalidate the test.
    * The models are **non-nested**. For nested models the differential is
      degenerate under the null and the statistic is not asymptotically normal;
      such comparisons are flagged ``nested`` and marked unusable rather than
      reported as if valid.

    ``small_sample`` applies the Harvey-Leybourne-Newbold (1997) correction,
    which matters here: a walk-forward study on daily BTC yields tens of
    origins, not thousands, and the uncorrected statistic over-rejects.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"loss series have different lengths: {len(a)} vs {len(b)}")

    mask = np.isfinite(a) & np.isfinite(b)
    differences = a[mask] - b[mask]
    n = len(differences)

    caveats: list[str] = []
    nested_flag = is_nested(model_a, model_b) if nested is None else nested
    if nested_flag:
        caveats.append(
            "Models are nested: the DM statistic is not asymptotically normal "
            "under the null. Use Clark-West or read the bootstrap interval instead."
        )
    if n < 10:
        caveats.append(f"Only {n} loss differentials; the test has very little power.")
    if horizon > 1:
        caveats.append(
            f"h={horizon} forecasts overlap; HAC variance uses {horizon - 1} lag(s). "
            "Origins spaced closer than the horizon share target bars."
        )

    if n < 3:
        return DieboldMarianoResult(
            model_a=model_a,
            model_b=model_b,
            mean_loss_difference=float(differences.mean()) if n else float("nan"),
            statistic=float("nan"),
            p_value=float("nan"),
            n_observations=n,
            horizon=horizon,
            hac_lags=max(0, horizon - 1),
            loss=loss,
            small_sample_corrected=small_sample,
            nested=nested_flag,
            caveats=(*caveats, "Too few observations to compute a statistic."),
        )

    from scipy.stats import t as student_t

    hac_lags = max(0, horizon - 1)
    variance = _hac_variance(differences, hac_lags)

    if variance <= 0:
        caveats.append("Non-positive HAC variance; the differential is degenerate.")
        return DieboldMarianoResult(
            model_a=model_a,
            model_b=model_b,
            mean_loss_difference=float(differences.mean()),
            statistic=float("nan"),
            p_value=float("nan"),
            n_observations=n,
            horizon=horizon,
            hac_lags=hac_lags,
            loss=loss,
            small_sample_corrected=small_sample,
            nested=nested_flag,
            caveats=tuple(caveats),
        )

    statistic = float(differences.mean() / np.sqrt(variance / n))

    if small_sample:
        h = horizon
        factor = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
        statistic *= float(np.sqrt(max(factor, 1e-12)))

    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))

    return DieboldMarianoResult(
        model_a=model_a,
        model_b=model_b,
        mean_loss_difference=float(differences.mean()),
        statistic=statistic,
        p_value=p_value,
        n_observations=n,
        horizon=horizon,
        hac_lags=hac_lags,
        loss=loss,
        small_sample_corrected=small_sample,
        nested=nested_flag,
        caveats=tuple(caveats),
    )


def loss_series(actual: np.ndarray, predicted: np.ndarray, *, loss: str = "absolute") -> np.ndarray:
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    if loss == "absolute":
        return np.abs(error)
    if loss == "squared":
        return error**2
    raise ValueError(f"unknown loss {loss!r}; use 'absolute' or 'squared'")


def compare_models(
    records: pd.DataFrame,
    *,
    baseline: str,
    horizon: int = 1,
    loss: str = "absolute",
    step: int | str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Diebold-Mariano of every model against ``baseline``.

    Comparing K models to one baseline is K-1 hypotheses, not one. A
    Benjamini-Hochberg correction is applied across the family and both the raw
    and adjusted p-values are reported, so a reader can see the difference.
    Nested pairs are computed but excluded from the correction and marked
    unusable -- correcting a statistic that was never valid would launder it.

    ``step`` restricts to a single horizon step -- the least-overlapping
    comparison. Pass ``"first"`` for the shortest distance actually scored,
    which under an embargo of ``e`` bars is ``e + 1``, not 1. ``None`` pools all
    scored bars, which inflates the apparent sample size; the ``horizon``
    argument then governs the HAC lag count.
    """
    required = {"model", "actual", "predicted"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records is missing column(s): {sorted(missing)}")

    if step == "first":
        from .targets import first_scored_step

        step = first_scored_step(records)

    data = records if step is None else records[records["step"] == step]
    if data.empty:
        raise ValueError(
            f"no records at step={step!r}. step counts forecast distance from the "
            "origin, so an embargoed run has no step-1 rows."
        )
    if baseline not in set(data["model"]):
        raise ValueError(f"baseline {baseline!r} is not present in the records")

    key = ["origin", "step"] if {"origin", "step"} <= set(data.columns) else ["date"]

    def losses_for(name: str) -> pd.Series:
        rows = data[data["model"] == name].sort_values(key)
        values = loss_series(
            rows["actual"].to_numpy(float), rows["predicted"].to_numpy(float), loss=loss
        )
        return pd.Series(values, index=pd.MultiIndex.from_frame(rows[key]))

    base_losses = losses_for(baseline)

    results: list[DieboldMarianoResult] = []
    for name in sorted(set(data["model"]) - {baseline}):
        challenger = losses_for(name)
        common = base_losses.index.intersection(challenger.index)
        results.append(
            diebold_mariano(
                challenger.loc[common].to_numpy(),
                base_losses.loc[common].to_numpy(),
                model_a=name,
                model_b=baseline,
                horizon=horizon,
                loss=loss,
            )
        )

    if not results:
        return pd.DataFrame()

    frame = pd.DataFrame([r.to_dict() for r in results])

    usable = frame["usable"].to_numpy(dtype=bool)
    adjusted = np.full(len(frame), np.nan)
    significant = np.zeros(len(frame), dtype=bool)
    if usable.any():
        adj, rej = benjamini_hochberg(frame.loc[usable, "p_value"].to_numpy(), alpha=alpha)
        adjusted[usable] = adj
        significant[usable] = rej

    frame["p_adjusted"] = adjusted
    frame["significant_after_correction"] = significant
    frame["n_hypotheses_corrected"] = int(usable.sum())
    frame["correction"] = "benjamini-hochberg"
    frame["alpha"] = alpha
    return frame.set_index("model_a")


#: Recorded in every benchmark manifest.
INFERENCE_NOTES = {
    "directional_uncertainty": (
        "Stationary bootstrap (Politis-Romano) over the per-origin hit series at "
        "the shortest scored horizon, block length n**(1/3). Replaces the "
        "binomial test, whose independence assumption fails under overlapping "
        "horizons, within-fold correlation and volatility clustering."
    ),
    "directional_null": (
        "The null is the accuracy of the best CONSTANT predictor, max(base_rate, "
        "1 - base_rate), not 0.5. Over 31-day windows BTC rose 61% of the time, "
        "so an always-up forecast scores 0.611 while containing no information. "
        "Testing against a coin would report several models as significant on "
        "the base rate alone."
    ),
    "forecast_comparison": (
        "Diebold-Mariano with Newey-West HAC variance (h-1 lags) and the "
        "Harvey-Leybourne-Newbold small-sample correction. Nested pairs are "
        "flagged and excluded: the statistic is not asymptotically normal for "
        "them."
    ),
    "multiple_testing": (
        "Benjamini-Hochberg across the K-1 challenger-vs-baseline comparisons. "
        "Raw and adjusted p-values are both reported. Anything not corrected is "
        "labelled exploratory."
    ),
    "residual_limitations": (
        "Block bootstrap handles short-range dependence, not structural breaks. "
        "A walk-forward study spanning multiple crypto regimes is not "
        "covariance-stationary in the strict sense the DM test assumes; treat "
        "p-values as indicative and prefer the confidence intervals."
    ),
}


__all__ = [
    "DEFAULT_RESAMPLES",
    "best_constant_accuracy",
    "directional_base_rate",
    "INFERENCE_NOTES",
    "KNOWN_NESTED_PAIRS",
    "BootstrapCI",
    "DieboldMarianoResult",
    "compare_models",
    "diebold_mariano",
    "directional_accuracy_ci",
    "is_nested",
    "loss_series",
    "moving_block_indices",
    "optimal_block_length",
    "stationary_bootstrap",
    "stationary_bootstrap_indices",
]
