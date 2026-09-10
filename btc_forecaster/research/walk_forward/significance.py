"""Is a model different from the naive forecast, is it better, and does it matter?

Three different questions, answered separately and never merged:

``statistically different``
    Diebold-Mariano on absolute loss against the naive forecast, over the same
    origins, rejects equal accuracy after Benjamini-Hochberg across the whole
    primary family. Two-sided: a model can be significantly *worse*, and at
    these sample sizes several usually are.

``statistically better``
    Different, and the difference favours the model.

``practically useful``
    Better, and by at least the preregistered skill floor. A statistically
    detectable improvement of a tenth of a percent is still a tenth of a
    percent.

The test is A2's -- :func:`btc_forecaster.evaluation.inference.diebold_mariano`,
with the Harvey-Leybourne-Newbold small-sample correction -- not a second
implementation. What A7 changes is the lag count. Walk-forward origins are one
bar apart, so an h-bar forecast overlaps its neighbours and needs at least
``h - 1`` Newey-West lags; and an absolute-loss differential on daily returns
inherits volatility clustering even at ``h = 1``, where the overlap argument
alone would allow zero. The lag count is therefore

    max(h - 1, floor(4 * (n / 100) ** (2 / 9)))

the larger of the overlap requirement and the standard Newey-West bandwidth.
More lags make the test harder to pass, which is the direction this track errs
in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ...diagnostics.multiple_testing import benjamini_hochberg, expected_false_positives
from ...evaluation.inference import diebold_mariano, is_nested, stationary_bootstrap
from .config import WalkForwardConfig
from .scoring import KEY, mae_skill, paired_with_baseline

OK = "OK"
INSUFFICIENT = "INSUFFICIENT_PAIRED_ORIGINS"
NESTED = "NESTED_NOT_TESTABLE"
DEGENERATE = "DEGENERATE_DIFFERENTIAL"


def newey_west_bandwidth(n: int) -> int:
    """The Newey-West (1994) automatic bandwidth, ``floor(4 (n/100)^(2/9))``."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if n > 0 else 0


def hac_lags_for(horizon: int, n: int) -> int:
    return max(horizon - 1, newey_west_bandwidth(n))


@dataclass(frozen=True)
class Comparison:
    """One (model, horizon, window) against the naive forecast."""

    model_id: str
    horizon: int
    window: str
    n_origins: int
    status: str
    skill: float
    mean_loss_difference: float | None = None
    statistic: float | None = None
    p_value: float | None = None
    #: One-sided: evidence that the model's loss is *lower*.
    p_value_better: float | None = None
    hac_lags: int | None = None
    skill_ci_lower: float | None = None
    skill_ci_upper: float | None = None
    #: Benjamini-Hochberg across every usable comparison in the benchmark.
    q_value: float | None = None
    #: Benjamini-Hochberg within this comparison's horizon. Reported, not gating.
    q_value_within_horizon: float | None = None

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.model_id, self.horizon, self.window)

    @property
    def testable(self) -> bool:
        return self.status == OK and self.p_value is not None and bool(np.isfinite(self.p_value))

    def statistically_different(self, alpha: float) -> bool:
        return bool(self.testable and self.q_value is not None and self.q_value < alpha)

    def statistically_better(self, alpha: float) -> bool:
        return bool(
            self.statistically_different(alpha)
            and self.mean_loss_difference is not None
            and self.mean_loss_difference < 0
        )

    def statistically_worse(self, alpha: float) -> bool:
        return bool(
            self.statistically_different(alpha)
            and self.mean_loss_difference is not None
            and self.mean_loss_difference > 0
        )

    def practically_useful(self, alpha: float, practical_min_skill: float) -> bool:
        return bool(self.statistically_better(alpha) and self.skill >= practical_min_skill)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "horizon": self.horizon,
            "window": self.window,
            "n_origins": self.n_origins,
            "status": self.status,
            "skill": self.skill,
            "skill_ci": [self.skill_ci_lower, self.skill_ci_upper],
            "mean_loss_difference": self.mean_loss_difference,
            "dm_statistic": self.statistic,
            "p_value": self.p_value,
            "p_value_better": self.p_value_better,
            "hac_lags": self.hac_lags,
            "q_value": self.q_value,
            "q_value_within_horizon": self.q_value_within_horizon,
        }


def compare(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
    *,
    key: tuple[str, int, str],
    baseline_id: str,
    config: WalkForwardConfig,
) -> Comparison:
    """One comparison, uncorrected."""
    model_id, horizon, window = key
    model_loss = np.abs(actual - predicted)
    base_loss = np.abs(actual - baseline)
    n = len(actual)
    untested = Comparison(
        model_id=model_id,
        horizon=horizon,
        window=window,
        n_origins=n,
        status=INSUFFICIENT,
        skill=mae_skill(actual - predicted, actual - baseline),
    )

    if n < config.minimum_paired_origins:
        return untested
    if is_nested(model_id, baseline_id):
        return replace(untested, status=NESTED)

    lags = hac_lags_for(horizon, n)
    result = diebold_mariano(
        model_loss,
        base_loss,
        model_a=model_id,
        model_b=baseline_id,
        horizon=horizon,
        loss="absolute",
        small_sample=True,
        nested=False,
        hac_lags=lags,
    )
    if not np.isfinite(result.statistic):
        return replace(
            untested,
            status=DEGENERATE,
            mean_loss_difference=float(result.mean_loss_difference),
            hac_lags=lags,
        )

    # Skill interval: a block bootstrap of the per-origin loss improvement,
    # expressed as a fraction of the naive MAE on the same origins.
    base_mae = float(np.mean(base_loss))
    interval = stationary_bootstrap(
        base_loss - model_loss,
        level=0.95,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
        null_value=0.0,
    )
    return replace(
        untested,
        status=OK,
        mean_loss_difference=float(result.mean_loss_difference),
        statistic=float(result.statistic),
        p_value=float(result.p_value),
        p_value_better=float(student_t.cdf(result.statistic, df=n - 1)),
        hac_lags=int(result.hac_lags),
        skill_ci_lower=interval.lower / base_mae if base_mae > 0 else None,
        skill_ci_upper=interval.upper / base_mae if base_mae > 0 else None,
    )


def compare_all(records: pd.DataFrame, config: WalkForwardConfig) -> list[Comparison]:
    """Every (model, horizon, window) except the baseline, in key order."""
    paired = paired_with_baseline(records, config.baseline)
    comparisons: list[Comparison] = []
    for key, group in paired.groupby(list(KEY), sort=True):
        if key[0] == config.baseline:
            continue
        comparisons.append(
            compare(
                group["actual"].to_numpy(dtype=float),
                group["predicted"].to_numpy(dtype=float),
                group["baseline_predicted"].to_numpy(dtype=float),
                key=(str(key[0]), int(key[1]), str(key[2])),
                baseline_id=config.baseline,
                config=config,
            )
        )
    return comparisons


def _p_value(comparison: Comparison) -> float:
    if comparison.p_value is None:
        raise ValueError(f"{comparison.key} has no p-value")
    return float(comparison.p_value)


def _bh(p_values: list[float], alpha: float) -> list[float]:
    adjusted, _rejected = benjamini_hochberg(np.asarray(p_values, dtype=float), alpha=alpha)
    return [float(q) for q in adjusted]


def correct(comparisons: list[Comparison], *, alpha: float) -> tuple[list[Comparison], dict]:
    """Attach q-values: across the primary family, and within each horizon.

    The primary family is every testable comparison in the benchmark. It is the
    one the gate reads, and it is the large one, deliberately: ~200 tests at
    alpha = 0.05 produce about ten raw rejections from nothing, and the
    correction is what stops those ten being read as ten findings.
    """
    testable = [c for c in comparisons if c.testable]
    p_values = [_p_value(c) for c in testable]
    q_primary: dict[tuple[str, int, str], float] = (
        dict(zip([c.key for c in testable], _bh(p_values, alpha), strict=True)) if testable else {}
    )

    q_within: dict[tuple[str, int, str], float] = {}
    for horizon in sorted({c.horizon for c in testable}):
        family = [c for c in testable if c.horizon == horizon]
        q_within.update(
            dict(zip([c.key for c in family], _bh([_p_value(c) for c in family], alpha), strict=True))
        )

    corrected = [
        replace(c, q_value=q_primary.get(c.key), q_value_within_horizon=q_within.get(c.key))
        for c in comparisons
    ]
    summary = {
        "family": "every testable (model, horizon, window) against the naive forecast",
        "family_size": len(testable),
        "alpha": alpha,
        "raw_significant": int(sum(1 for p in p_values if p < alpha)),
        "expected_false_positives_at_alpha": float(expected_false_positives(len(testable), alpha)),
        "significant_after_bh": int(sum(1 for c in corrected if c.statistically_different(alpha))),
        "significantly_better_after_bh": int(sum(1 for c in corrected if c.statistically_better(alpha))),
        "significantly_worse_after_bh": int(sum(1 for c in corrected if c.statistically_worse(alpha))),
        "not_testable": {
            status: int(sum(1 for c in comparisons if c.status == status))
            for status in (INSUFFICIENT, NESTED, DEGENERATE)
        },
    }
    return corrected, summary


__all__ = [
    "DEGENERATE",
    "INSUFFICIENT",
    "NESTED",
    "OK",
    "Comparison",
    "compare",
    "compare_all",
    "correct",
    "hac_lags_for",
    "newey_west_bandwidth",
]
