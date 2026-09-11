"""B5 — decision states, and the corpus-sufficiency policy frozen before any audit.

B5's first gate is not statistical. Before asking whether external intelligence
moves BTC, it asks whether there is a point-in-time corpus to ask it of. That
question is answered by thresholds this module does not invent.

**The adequacy thresholds are B4's.** 30 events, 20 effective (non-overlapping
at 168 hours), 3 publishers, 1 provider, a 180-day span and collection on 80% of
days were preregistered by B4 and are enforced, unchanged, by the B4.1 readiness
gate. `SufficiencyPolicy` holds B4.1's `AdequacyPolicy` object itself rather than
restating its numbers, so there is one set of thresholds and it cannot drift.

**B5 adds only clauses that are stricter.** B4's own extraction-quality floors,
point-in-time integrity, and an event time that comes from the source rather than
from our retrieval. Each can only remove events; none can make an insufficient
corpus sufficient.

**Families are judged separately, never pooled.** Each event type is its own
family, and whale transfers are split by transfer context, as B4.1.32 requires. A
family short of the bar is reported as short. Merging sparse families to reach N
is the move this gate exists to refuse.

**The thresholds are a floor, not a comfortable level.** With 20 effective events
and B4's development-period dispersion, a two-sided 5% test at 80% power can only
detect a mean effect about two and a half times B4's practical floor
(`minimum_detectable_effects`). Lowering the bar would make the study unable to
see even large effects; that is the power argument for not lowering it.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from statistics import NormalDist
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..collection.readiness import DEFAULT_POLICY, AdequacyPolicy
from ..models import EventType, TransferContext

B5_TRACK = "B5"
PREREGISTRATION_VERSION = "b5-sufficiency-v1"

#: B4's preregistration, whose thresholds this track inherits. The committed file
#: is `research/market_intelligence/b4/runs/b4-3980bac0b72aeaa0/preregistration.json`.
B4_PREREGISTRATION_HASH = "389aa76f703d125412efc29c6f678e76e3267772f8ab99e60f6c137ea3794aa8"

#: B4's practical-significance floors by horizon, copied from that preregistration
#: so the power arithmetic is traceable to a frozen record.
B4_PRACTICAL_FLOORS: dict[str, float] = {
    "1d": 0.009063248338255128,
    "3d": 0.0155129673324549,
    "7d": 0.02449621970463976,
}

#: B4 set each floor at this multiple of the development-period standard deviation.
B4_FLOOR_SD_MULTIPLE = 0.25


class Decision(str, Enum):
    """The only four ways B5 can end. None of them trades."""

    #: No defensible historical event study can yet be performed.
    INTELLIGENCE_CORPUS_INSUFFICIENT = "INTELLIGENCE_CORPUS_INSUFFICIENT"
    #: The corpus is sufficient, and no stable incremental information is detected.
    EVENT_EFFECTS_NOT_DETECTED = "EVENT_EFFECTS_NOT_DETECTED"
    #: Effects appear but fail correction, practicality, stability or diversity.
    EVENT_EFFECTS_FRAGILE = "EVENT_EFFECTS_FRAGILE"
    #: An effect survives every gate. It earns a later research question, not a trade.
    EVENT_EFFECTS_ROBUST_CANDIDATE = "EVENT_EFFECTS_ROBUST_CANDIDATE"


class SufficiencyPolicy(BaseModel):
    """Gate 1: when is there a corpus worth studying?"""

    model_config = ConfigDict(frozen=True)

    #: B4's adequacy thresholds, as the B4.1 readiness gate holds them.
    adequacy: AdequacyPolicy = DEFAULT_POLICY
    #: B4's extraction-quality filters. An event the extractor was unsure of is
    #: not evidence of an event.
    minimum_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    minimum_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Share of point-in-time-valid events whose event time is a source's own
    #: publication time. Below this, reaction windows would be aligned to our
    #: retrieval clock rather than to when the information appeared.
    minimum_event_time_coverage: float = Field(default=0.9, ge=0.0, le=1.0)
    #: Share of events whose timestamps are impossible (an event known before it
    #: happened, or before its sources were). Such events are always excluded;
    #: above this share the pipeline itself is suspect, and point-in-time
    #: integrity cannot be established for the rest.
    maximum_pit_violation_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    #: At least this many families must meet every adequacy clause on their own.
    minimum_ready_families: int = Field(default=1, ge=1)
    #: For the power arithmetic that justifies not lowering the thresholds.
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    power: float = Field(default=0.8, gt=0.0, lt=1.0)


DEFAULT_SUFFICIENCY = SufficiencyPolicy()


def study_families() -> tuple[str, ...]:
    """Every family Gate 1 judges, including those with nothing in them.

    One per event type, except whale transfers, which are judged per transfer
    context and never pooled into inflow or outflow (B4.1.32).
    """
    families = [f"event_type:{member.value}" for member in EventType if member is not EventType.WHALE_TRANSFER]
    families.extend(f"whale:{context.value}" for context in TransferContext)
    return tuple(families)


def minimum_detectable_effects(policy: SufficiencyPolicy = DEFAULT_SUFFICIENCY) -> dict[str, float]:
    """The smallest mean effect a study at the threshold could detect, by horizon.

    Two-sided test at `alpha` with `power`, on `minimum_effective_events`
    independent events, with the standard deviation B4 measured in its
    development period (each floor is 0.25 of it).
    """
    z = NormalDist().inv_cdf(1 - policy.alpha / 2) + NormalDist().inv_cdf(policy.power)
    n = policy.adequacy.minimum_effective_events
    return {
        horizon: z * (floor / B4_FLOOR_SD_MULTIPLE) / math.sqrt(n)
        for horizon, floor in sorted(B4_PRACTICAL_FLOORS.items())
    }


def canonical_json(value: Any) -> str:
    """Sorted keys, no whitespace, so equal content is equal bytes on every platform."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


STOP_RULES: tuple[str, ...] = (
    "Gate 1 fails: record INTELLIGENCE_CORPUS_INSUFFICIENT and stop. No event study is run.",
    "Point-in-time integrity cannot be established: stop.",
    "Timestamp precision is inadequate for the windows studied: stop, or restrict to the windows it supports.",
    "A result driven by one publisher, gone after deduplication or correction, practically negligible or "
    "confined to one period is fragile, and methodology is not expanded to rescue it.",
)


class Preregistration(BaseModel):
    """What Gate 1 will decide, and how, written down before it is decided."""

    model_config = ConfigDict(frozen=True)

    track: str = B5_TRACK
    version: str = PREREGISTRATION_VERSION
    question: str
    policy: SufficiencyPolicy
    b4_preregistration_hash: str = B4_PREREGISTRATION_HASH
    decision_states: tuple[str, ...]
    families: tuple[str, ...]
    practical_floors: dict[str, float]
    minimum_detectable_effects: dict[str, float]
    stop_rules: tuple[str, ...]
    declared_before: str

    def content_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()

    def as_record(self) -> dict[str, Any]:
        return {"content_hash": self.content_hash(), "plan": self.model_dump(mode="json")}

    def write(self, path: str | Path) -> Path:
        """Write the plan with its hash. A different plan already there is refused."""
        target = Path(path)
        if target.exists():
            existing = Preregistration.read(target)
            if existing.content_hash() != self.content_hash():
                raise ValueError(f"{target} already holds a different frozen plan; a preregistration is not edited")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(canonical_json(self.as_record()), encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: str | Path) -> "Preregistration":
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        plan = cls.model_validate(record["plan"])
        if record.get("content_hash") != plan.content_hash():
            raise ValueError(f"{path}: the stored hash does not match its contents; the plan was edited after it was frozen")
        return plan


def default_preregistration(policy: SufficiencyPolicy = DEFAULT_SUFFICIENCY) -> Preregistration:
    return Preregistration(
        question=(
            "Does independently timestamped external information produce measurable and stable BTC market "
            "reactions after controlling for baseline market behaviour? Gate 1: is there a point-in-time "
            "corpus sufficient to ask it?"
        ),
        policy=policy,
        decision_states=tuple(member.value for member in Decision),
        families=study_families(),
        practical_floors=dict(sorted(B4_PRACTICAL_FLOORS.items())),
        minimum_detectable_effects=minimum_detectable_effects(policy),
        stop_rules=STOP_RULES,
        declared_before="the canonical Gate 1 audit of the collected corpus",
    )


__all__ = [
    "B4_FLOOR_SD_MULTIPLE",
    "B4_PRACTICAL_FLOORS",
    "B4_PREREGISTRATION_HASH",
    "B5_TRACK",
    "DEFAULT_SUFFICIENCY",
    "PREREGISTRATION_VERSION",
    "STOP_RULES",
    "Decision",
    "Preregistration",
    "SufficiencyPolicy",
    "canonical_json",
    "default_preregistration",
    "minimum_detectable_effects",
    "study_families",
]
