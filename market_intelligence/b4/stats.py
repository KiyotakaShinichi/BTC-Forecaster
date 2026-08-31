"""B4.13 / B4.15 / B4.16 / B4.17 — inference for dependent time series.

Three decisions shape everything downstream, and each is a place where the
convenient choice is the wrong one.

**Resampling is block-based, not iid.** Financial returns are serially dependent
and event windows overlap heavily; an iid bootstrap would treat 500 overlapping
72-hour windows as 500 independent observations and produce confidence intervals
several times too narrow. The moving-block and stationary bootstraps here
resample contiguous runs, so within-block dependence survives resampling. Block
length is a declared parameter, not something to tune until an interval looks
convincing.

**Every test belongs to a declared family.** A track that crosses entities by
event types by horizons by outcomes generates hundreds of tests, and at the 5%
level roughly one in twenty comes back "significant" from noise alone.
Benjamini-Hochberg is applied within families that are declared *before* results
are seen.

**Statistical significance is not the finding.** A p-value says an effect is
unlikely to be exactly zero; it says nothing about whether the effect is large
enough to matter. Practical thresholds are set from the scale of the data --
typically a fraction of the outcome's own dispersion -- and frozen in the
preregistration before any final-period result is computed.

Everything is seeded and deterministic. A bootstrap that gives a different
answer on re-run cannot be checked by anyone, including its author.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .contracts import B4DataError

BootstrapMethod = Literal["moving_block", "stationary"]


class Evidence(str, Enum):
    """B4.17. The verdict vocabulary, fixed in advance."""

    SUPPORTED = "SUPPORTED"
    WEAK = "WEAK"
    INCONCLUSIVE = "INCONCLUSIVE"
    NO_EVIDENCE = "NO_EVIDENCE"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


class BootstrapConfig(BaseModel):
    """Declared resampling configuration. Part of the preregistration."""

    model_config = ConfigDict(frozen=True)

    method: BootstrapMethod = "stationary"
    #: Expected block length in observations. For overlapping windows this
    #: should be at least the overlap length, or dependence survives into the
    #: resampled series and the interval is still too narrow.
    block_length: int = Field(default=24, ge=1)
    replicates: int = Field(default=2000, ge=100)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 20260831
    ci_method: Literal["percentile", "basic"] = "percentile"


class DescriptiveStats(BaseModel):
    """B4.13. The raw response, before any test."""

    model_config = ConfigDict(frozen=True)

    n: int
    mean: float | None
    median: float | None
    std: float | None
    positive_fraction: float | None
    quantiles: dict[str, float]

    @property
    def is_empty(self) -> bool:
        return self.n == 0


class BootstrapResult(BaseModel):
    """A dependence-aware interval and the two-sided p-value implied by it."""

    model_config = ConfigDict(frozen=True)

    point_estimate: float
    lower: float
    upper: float
    confidence: float
    method: BootstrapMethod
    block_length: int
    replicates: int
    seed: int
    ci_method: str
    #: Fraction of replicates on the opposite side of zero, doubled and clipped.
    #: A bootstrap p-value, not a parametric one -- it inherits the resampling's
    #: dependence handling instead of assuming normality.
    p_value: float
    effective_sample_size: int


class TestRecord(BaseModel):
    """One row of a multiple-testing family, before correction."""

    model_config = ConfigDict(frozen=True)

    test_id: str
    family: str
    n: int
    effect: float
    lower: float
    upper: float
    p_value: float


class CorrectedTest(BaseModel):
    """B4.16. The same row after Benjamini-Hochberg within its family."""

    model_config = ConfigDict(frozen=True)

    test_id: str
    family: str
    n: int
    effect: float
    lower: float
    upper: float
    p_value: float
    q_value: float
    family_size: int
    significant_at_q: bool


class PracticalThreshold(BaseModel):
    """B4.17. Frozen before results are seen; justified from data scale."""

    model_config = ConfigDict(frozen=True)

    outcome: str
    #: Minimum |effect| that would matter, in the outcome's own units.
    minimum_absolute_effect: float
    #: How that number was derived, so it can be argued with.
    justification: str
    minimum_sample: int = 30
    q_threshold: float = 0.10


# --------------------------------------------------------------- descriptives


def describe(values: Sequence[float]) -> DescriptiveStats:
    """Mean, median, dispersion and quantiles. Never a single p-value."""
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return DescriptiveStats(n=0, mean=None, median=None, std=None, positive_fraction=None, quantiles={})
    ordered = sorted(clean)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count if count > 1 else 0.0
    return DescriptiveStats(
        n=count,
        mean=mean,
        median=_quantile(ordered, 0.5),
        std=math.sqrt(variance),
        positive_fraction=sum(1 for value in ordered if value > 0.0) / count,
        quantiles={
            "p05": _quantile(ordered, 0.05),
            "p25": _quantile(ordered, 0.25),
            "p50": _quantile(ordered, 0.50),
            "p75": _quantile(ordered, 0.75),
            "p95": _quantile(ordered, 0.95),
        },
    )


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Linear-interpolation quantile on an already-sorted sequence."""
    if not ordered:
        raise B4DataError("cannot take a quantile of an empty sample")
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return float(ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight)


# ----------------------------------------------------------------- resampling


def _moving_block_indices(count: int, block_length: int, generator: random.Random) -> list[int]:
    """Contiguous fixed-length blocks, wrapped, until the sample is refilled."""
    indices: list[int] = []
    while len(indices) < count:
        start = generator.randrange(count)
        for offset in range(block_length):
            indices.append((start + offset) % count)
            if len(indices) == count:
                break
    return indices


def _stationary_indices(count: int, mean_block: int, generator: random.Random) -> list[int]:
    """Politis-Romano: geometric block lengths, so the resample is stationary.

    Fixed blocks make the resampled series non-stationary at block boundaries,
    which biases statistics that depend on the joins. Geometric lengths remove
    the boundary artefact at the cost of a little variance.
    """
    probability = 1.0 / max(1, mean_block)
    indices: list[int] = []
    position = generator.randrange(count)
    while len(indices) < count:
        indices.append(position)
        if generator.random() < probability:
            position = generator.randrange(count)
        else:
            position = (position + 1) % count
    return indices


def block_bootstrap(
    values: Sequence[float],
    config: BootstrapConfig,
    statistic: Callable[[Sequence[float]], float] | None = None,
) -> BootstrapResult:
    """Dependence-aware CI and p-value for a statistic of one sample.

    Determinism is a hard requirement, so the generator is seeded from the
    config alone and every replicate draws from it in a fixed order.
    """
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if len(clean) < 2:
        raise B4DataError("bootstrap needs at least two observations")

    compute = statistic or (lambda sample: sum(sample) / len(sample))
    point = compute(clean)
    generator = random.Random(config.seed)
    count = len(clean)
    block = min(config.block_length, count)

    replicates: list[float] = []
    for _ in range(config.replicates):
        if config.method == "moving_block":
            indices = _moving_block_indices(count, block, generator)
        else:
            indices = _stationary_indices(count, block, generator)
        replicates.append(compute([clean[index] for index in indices]))

    replicates.sort()
    alpha = 1.0 - config.confidence
    lower_q = _quantile(replicates, alpha / 2.0)
    upper_q = _quantile(replicates, 1.0 - alpha / 2.0)
    if config.ci_method == "basic":
        lower, upper = 2.0 * point - upper_q, 2.0 * point - lower_q
    else:
        lower, upper = lower_q, upper_q

    below = sum(1 for value in replicates if value <= 0.0)
    above = sum(1 for value in replicates if value >= 0.0)
    tail = min(below, above) / len(replicates)
    p_value = min(1.0, 2.0 * tail)

    return BootstrapResult(
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence=config.confidence,
        method=config.method,
        block_length=block,
        replicates=config.replicates,
        seed=config.seed,
        ci_method=config.ci_method,
        p_value=p_value,
        effective_sample_size=max(1, count // max(1, block)),
    )


# ----------------------------------------------------------- multiple testing


def benjamini_hochberg(tests: Sequence[TestRecord], q_threshold: float = 0.10) -> list[CorrectedTest]:
    """BH-FDR within each declared family.

    Families are corrected independently, because pooling unrelated families
    would let a large family of nulls dilute a small family's correction. Ties
    in p-value receive the same q-value, and the step-up enforcement makes
    q-values monotone -- without it a slightly larger p-value can end up with a
    smaller q, which reads as nonsense in a results table.
    """
    if not 0.0 < q_threshold < 1.0:
        raise B4DataError("q_threshold must be in (0, 1)")

    by_family: dict[str, list[TestRecord]] = {}
    for test in tests:
        by_family.setdefault(test.family, []).append(test)

    corrected: list[CorrectedTest] = []
    for family, members in by_family.items():
        ordered = sorted(members, key=lambda record: (record.p_value, record.test_id))
        size = len(ordered)
        raw_q: list[float] = []
        for rank, record in enumerate(ordered, start=1):
            raw_q.append(min(1.0, record.p_value * size / rank))
        # Step-up: enforce monotonicity from the largest p-value downwards.
        for index in range(size - 2, -1, -1):
            raw_q[index] = min(raw_q[index], raw_q[index + 1])
        for record, q_value in zip(ordered, raw_q, strict=True):
            corrected.append(
                CorrectedTest(
                    test_id=record.test_id,
                    family=family,
                    n=record.n,
                    effect=record.effect,
                    lower=record.lower,
                    upper=record.upper,
                    p_value=record.p_value,
                    q_value=q_value,
                    family_size=size,
                    significant_at_q=q_value <= q_threshold,
                )
            )
    return sorted(corrected, key=lambda record: (record.family, record.q_value, record.test_id))


# ------------------------------------------------------ practical significance


@dataclass(frozen=True)
class Classification:
    """The verdict plus the reasons behind it, so it can be argued with."""

    evidence: Evidence
    reasons: tuple[str, ...]


def classify(
    test: CorrectedTest,
    threshold: PracticalThreshold,
    *,
    stable: bool | None = None,
) -> Classification:
    """B4.17. Combine magnitude, uncertainty, sample size, FDR and stability.

    Order matters. Sample adequacy is checked first: an underpowered test that
    happens to look significant is INSUFFICIENT_SAMPLE, not SUPPORTED, and
    letting it through would be the single most likely way this track produces a
    false positive.
    """
    reasons: list[str] = []

    if test.n < threshold.minimum_sample:
        return Classification(
            Evidence.INSUFFICIENT_SAMPLE,
            (f"n={test.n} is below the declared minimum of {threshold.minimum_sample}",),
        )

    large_enough = abs(test.effect) >= threshold.minimum_absolute_effect
    excludes_zero = (test.lower > 0.0) or (test.upper < 0.0)
    passes_fdr = test.q_value <= threshold.q_threshold
    unstable = stable is False

    if not large_enough:
        reasons.append(
            f"|effect|={abs(test.effect):.5f} is below the practical threshold "
            f"{threshold.minimum_absolute_effect:.5f}"
        )
    if not excludes_zero:
        reasons.append(f"the bootstrap interval [{test.lower:.5f}, {test.upper:.5f}] contains zero")
    if not passes_fdr:
        reasons.append(f"q={test.q_value:.4f} exceeds the declared {threshold.q_threshold:.2f}")
    if unstable:
        reasons.append("the estimate does not survive leaving out one event or one period")

    if large_enough and excludes_zero and passes_fdr and not unstable:
        return Classification(
            Evidence.SUPPORTED,
            (
                f"|effect|={abs(test.effect):.5f} clears {threshold.minimum_absolute_effect:.5f}",
                f"interval [{test.lower:.5f}, {test.upper:.5f}] excludes zero",
                f"q={test.q_value:.4f} survives correction",
            )
            + (("stable to leave-one-out",) if stable else ()),
        )

    # WEAK is for a result that is real-sized and non-zero but fails exactly one
    # of the two robustness gates. Separating it from NO_EVIDENCE matters: these
    # are the candidates worth retesting out of sample, and collapsing them into
    # "no evidence" would discard the only interesting negatives.
    if large_enough and excludes_zero and (not passes_fdr or unstable):
        return Classification(Evidence.WEAK, tuple(reasons))

    if large_enough and not excludes_zero:
        return Classification(Evidence.INCONCLUSIVE, tuple(reasons))

    if not large_enough and excludes_zero and passes_fdr:
        return Classification(
            Evidence.NO_EVIDENCE,
            tuple(reasons) + ("precisely estimated, and too small to matter",),
        )

    return Classification(Evidence.NO_EVIDENCE, tuple(reasons))


def threshold_from_dispersion(
    outcome: str,
    values: Sequence[float],
    *,
    fraction: float = 0.25,
    minimum_sample: int = 30,
    q_threshold: float = 0.10,
) -> PracticalThreshold:
    """Derive a practical threshold from the outcome's own scale.

    Anchoring to a fraction of the sample standard deviation keeps the threshold
    in the data's units and makes it arguable. It must be computed on the
    development period and frozen: computing it on the data a result will be
    read from is choosing the threshold after seeing the estimate, which is the
    thing B4.17 exists to prevent.
    """
    stats = describe(values)
    if stats.std is None or stats.n < 2:
        raise B4DataError(f"cannot derive a threshold for {outcome} from fewer than two observations")
    return PracticalThreshold(
        outcome=outcome,
        minimum_absolute_effect=fraction * stats.std,
        justification=(
            f"{fraction:g} x the development-period standard deviation of {outcome} "
            f"(sd={stats.std:.6f} over n={stats.n})"
        ),
        minimum_sample=minimum_sample,
        q_threshold=q_threshold,
    )


__all__ = [
    "BootstrapConfig",
    "BootstrapMethod",
    "BootstrapResult",
    "Classification",
    "CorrectedTest",
    "DescriptiveStats",
    "Evidence",
    "PracticalThreshold",
    "TestRecord",
    "benjamini_hochberg",
    "block_bootstrap",
    "classify",
    "describe",
    "threshold_from_dispersion",
]
