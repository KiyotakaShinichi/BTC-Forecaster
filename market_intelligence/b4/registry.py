"""B4.37 / B4.38 — the candidate signal registry and its carry-forward decision.

Research-only. A registry entry records what was tested, what came back, and
whether the candidate is worth retesting out of sample. It carries **no weights**
— assigning one would be signal fusion, which belongs to a later track and to a
model that has been validated out of sample, not to a descriptive study.

The decision is a function of the frozen policy and the measured result, applied
in a fixed order, so it cannot drift with enthusiasm. Every rejection records
which clause rejected it: a registry of bare verdicts is unarguable, and an
unarguable result is not a scientific one.

`INSUFFICIENT_DATA` is a first-class outcome and is expected to dominate this
run. It is not a softer `REJECT`: a signal that could not be tested has not been
found wanting, and conflating the two would let an absence of data masquerade as
evidence of absence.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from .contracts import B4DataError, EvidenceTier
from .prereg import CarryForwardPolicy
from .stats import CorrectedTest, Evidence


class SignalDecision(str, Enum):
    """B4.37. What happens to a candidate after this run."""

    CARRY_FORWARD = "CARRY_FORWARD"
    EXPLORATORY_ONLY = "EXPLORATORY_ONLY"
    REJECT = "REJECT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SignalCandidate(BaseModel):
    """One tested candidate and everything needed to argue about it."""

    model_config = ConfigDict(frozen=True)

    signal_id: str
    definition: str
    source_domain: str
    feature_version: str
    sample_count: int
    effective_sample_count: int
    tested_horizons: tuple[str, ...]
    #: Effect estimate and interval at each tested horizon.
    effects: dict[str, float | None]
    intervals: dict[str, tuple[float, float] | None]
    q_values: dict[str, float | None]
    evidence: dict[str, Evidence]
    stability_verdict: str | None
    concentration_verdict: str | None
    largest_source_share: float | None
    pit_status: EvidenceTier | None
    multiple_testing_family: str
    decision: SignalDecision
    decision_reasons: tuple[str, ...]
    notes: str = ""


class SignalRegistry(BaseModel):
    """The full set of candidates from one run. Never overwritten in place."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at: datetime
    preregistration_hash: str
    candidates: tuple[SignalCandidate, ...]

    def by_decision(self, decision: SignalDecision) -> list[SignalCandidate]:
        return [candidate for candidate in self.candidates if candidate.decision is decision]

    def write(self, path: str | Path) -> Path:
        """Atomic write that refuses to clobber a completed run (B4.49)."""
        target = Path(path)
        if target.exists():
            raise B4DataError(
                f"{target} already exists; B4 runs are never overwritten -- write to a new run directory"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return target


def decide(
    *,
    signal_id: str,
    policy: CarryForwardPolicy,
    sample_count: int,
    effective_sample_count: int,
    pit_status: EvidenceTier | None,
    best: CorrectedTest | None,
    evidence: Evidence | None,
    stable: bool | None,
    fragile: bool | None,
    largest_source_share: float | None,
    exploratory_reason: str | None = None,
) -> tuple[SignalDecision, tuple[str, ...]]:
    """Apply the frozen policy in a fixed order and record every clause hit.

    Order matters and is deliberate. "Could not be tested" is settled before
    "was tested and failed", because the two mean opposite things to whoever
    reads the registry next.
    """
    reasons: list[str] = []

    # 1. Could this have been tested at all?
    if sample_count == 0:
        return SignalDecision.INSUFFICIENT_DATA, ("no observations exist for this signal",)
    if sample_count < policy.minimum_observations:
        return (
            SignalDecision.INSUFFICIENT_DATA,
            (f"n={sample_count} is below the preregistered minimum of {policy.minimum_observations}",),
        )
    if effective_sample_count < policy.minimum_effective_observations:
        return (
            SignalDecision.INSUFFICIENT_DATA,
            (
                f"effective (non-overlapping) n={effective_sample_count} is below the preregistered "
                f"minimum of {policy.minimum_effective_observations}",
            ),
        )
    if best is None or evidence is None:
        return SignalDecision.INSUFFICIENT_DATA, ("no horizon produced an estimable effect",)

    # 2. Is it even eligible for later forecasting use?
    if policy.require_pit_validated and pit_status is not EvidenceTier.PIT_VALIDATED:
        return (
            SignalDecision.EXPLORATORY_ONLY,
            (
                f"evidence tier is {pit_status.value if pit_status else 'unknown'}, not PIT_VALIDATED; "
                "it may describe history but may not be carried into forecasting",
            ),
        )

    # 3. The statistical bar.
    if evidence is Evidence.INSUFFICIENT_SAMPLE:
        return SignalDecision.INSUFFICIENT_DATA, ("the classifier found the sample inadequate",)

    if best.q_value > policy.maximum_q_value:
        reasons.append(f"q={best.q_value:.4f} exceeds the preregistered {policy.maximum_q_value:.2f}")
    if policy.require_interval_excludes_zero and not (best.lower > 0.0 or best.upper < 0.0):
        reasons.append(f"the interval [{best.lower:.5f}, {best.upper:.5f}] contains zero")
    if policy.require_stability and stable is False:
        reasons.append("the estimate is not stable across periods")
    if policy.require_stability and stable is None:
        reasons.append("stability could not be assessed")
    if policy.require_not_fragile and fragile:
        reasons.append("removing a single observation moves or flips the estimate")
    if (
        largest_source_share is not None
        and largest_source_share > policy.maximum_single_source_share
    ):
        reasons.append(
            f"{largest_source_share:.0%} of observations come from one source, above the "
            f"preregistered {policy.maximum_single_source_share:.0%}"
        )

    if not reasons:
        return (
            SignalDecision.CARRY_FORWARD,
            (
                f"clears every preregistered gate at {best.test_id}: "
                f"effect={best.effect:+.5f}, interval=[{best.lower:.5f}, {best.upper:.5f}], "
                f"q={best.q_value:.4f}, n={sample_count}",
            ),
        )

    # 4. Failed the bar. Exploratory carry-forward is allowed but must be argued.
    if exploratory_reason and policy.allow_exploratory and evidence in (Evidence.WEAK, Evidence.INCONCLUSIVE):
        return (
            SignalDecision.EXPLORATORY_ONLY,
            tuple(reasons) + (f"carried as exploratory only: {exploratory_reason}",),
        )

    return SignalDecision.REJECT, tuple(reasons)


def enforce_exploratory_budget(
    candidates: Sequence[SignalCandidate], policy: CarryForwardPolicy
) -> list[SignalCandidate]:
    """B4.38. Do not flood the next track with weak leads.

    Keeps the best-evidenced exploratory candidates up to the budget and demotes
    the rest to REJECT with an explicit reason, so the demotion is visible rather
    than a silent truncation of the list.
    """
    exploratory = [
        candidate for candidate in candidates if candidate.decision is SignalDecision.EXPLORATORY_ONLY
    ]
    if len(exploratory) <= policy.maximum_exploratory_candidates:
        return list(candidates)

    def strength(candidate: SignalCandidate) -> float:
        finite = [value for value in candidate.q_values.values() if value is not None]
        return min(finite) if finite else 1.0

    keep = {
        candidate.signal_id
        for candidate in sorted(exploratory, key=strength)[: policy.maximum_exploratory_candidates]
    }
    adjusted: list[SignalCandidate] = []
    for candidate in candidates:
        if candidate.decision is SignalDecision.EXPLORATORY_ONLY and candidate.signal_id not in keep:
            adjusted.append(
                candidate.model_copy(
                    update={
                        "decision": SignalDecision.REJECT,
                        "decision_reasons": candidate.decision_reasons
                        + (
                            f"demoted: the exploratory budget of "
                            f"{policy.maximum_exploratory_candidates} was already filled by "
                            "better-evidenced candidates",
                        ),
                    }
                )
            )
        else:
            adjusted.append(candidate)
    return adjusted


def registry_summary(registry: SignalRegistry) -> dict[str, int]:
    return {
        decision.value: len(registry.by_decision(decision)) for decision in SignalDecision
    }


def load_registry(path: str | Path) -> SignalRegistry:
    return SignalRegistry.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "SignalCandidate",
    "SignalDecision",
    "SignalRegistry",
    "decide",
    "enforce_exploratory_budget",
    "load_registry",
    "registry_summary",
]
