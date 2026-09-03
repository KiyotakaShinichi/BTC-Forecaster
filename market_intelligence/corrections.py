"""An append-only ledger for observations a defect should never have produced.

Some records in a research corpus turn out to be artefacts of the software that
collected them rather than facts about the world. The obvious response --
delete them -- is the wrong one, and wrong in a way that is hard to see until
much later:

* A deleted row leaves no trace, so a future reader cannot tell a corpus that
  never held an observation from one that quietly dropped it. Both look like an
  absence of news.
* The manifests, snapshots and clusters that referenced the row remain, and now
  reference nothing. Integrity checks that were passing start failing, or worse,
  are relaxed until they pass.
* Deleting is itself an unrecorded edit. A corpus whose contents can change
  without leaving evidence is not evidence.

So nothing here removes or rewrites anything. A correction is a new row that
says *what is now known* about an existing observation, and the observation
stays exactly as it was written. Two views over the same records follow:

**Raw** -- everything ever collected, including what was later invalidated. This
is what audit, provenance and integrity checks read. An invalidated event is
still physically present, still points at its source document, and still
explains why a manifest from that day counted what it counted.

**Eligible** -- what research may count. Readiness, event studies and feature
generation read this, and an invalidated observation is absent from it.

The distinction is the whole design: *preserved physically, excluded
scientifically*.

Corrections are point-in-time records themselves, which raises a real question:
does applying a correction made today to a replay of last week leak information
backwards? For this class of correction, no -- and the reason is worth stating
rather than assuming. A `REDISCOVERY_DUPLICATE_PRE_FIX` correction carries no
information about the market. It says a piece of software wrote a row it should
not have written. Removing that row from an analysis does not tell the analysis
anything about prices it could not have known; it removes something that was
never an observation of the world in the first place. So the default research
mode applies every known correction regardless of when it was recorded.

`AS_BELIEVED` exists for the other question -- "what did the run on that day
actually see?" -- which is an audit question, not a research one, and there the
answer must not include corrections that had not been made yet.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import EventSignal

#: Bumped when the *rule* for deciding eligibility changes -- not when a
#: correction is added. A snapshot records which contract it was built under, so
#: a later reader can tell "this snapshot predates corrections" from "this
#: snapshot was built with corrections applied and found none".
ELIGIBILITY_CONTRACT_VERSION = "event-eligibility-v1"

#: The reason code for the defect this ledger was first built to record: events
#: manufactured by re-extracting an unchanged document, fixed in ec4d1c8.
REDISCOVERY_DUPLICATE_PRE_FIX = "REDISCOVERY_DUPLICATE_PRE_FIX"


class CorrectionStatus(str, Enum):
    """What a correction asserts about an observation."""

    #: Present in the corpus, excluded from research.
    INVALIDATED = "INVALIDATED"
    #: A previous invalidation was itself wrong. Append, never erase.
    REINSTATED = "REINSTATED"


class EligibilityMode(str, Enum):
    """Which corrections a reader applies."""

    #: Every correction on record. The default for research: a correction is a
    #: statement about our software, not about the market, so applying one
    #: retroactively introduces no lookahead.
    CORRECTED = "CORRECTED"
    #: Only corrections recorded at or before the point being reconstructed.
    #: For auditing what a past run actually counted.
    AS_BELIEVED = "AS_BELIEVED"


class EventCorrection(BaseModel):
    """One immutable statement about one observation."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(min_length=1)
    status: CorrectionStatus
    reason: str = Field(min_length=1)
    #: When the correction was *recorded*, never when the defect occurred.
    invalidated_at: datetime
    #: The version that made the correction knowable -- for the defect this was
    #: built for, the commit that stopped the events being produced.
    invalidated_by_version: str = Field(min_length=1)
    #: What produced the bad observation, in a form a human can go and read.
    source_bug: str = Field(min_length=1)
    notes: str = ""

    @field_validator("invalidated_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)

    @property
    def correction_id(self) -> str:
        """Content-addressed, so recording the same correction twice is a no-op.

        Re-running a correction script must not be able to produce a second,
        subtly different row for the same decision.
        """
        material = json.dumps(
            {
                "event_id": self.event_id,
                "status": self.status.value,
                "reason": self.reason,
                "invalidated_at": self.invalidated_at.isoformat(),
                "invalidated_by_version": self.invalidated_by_version,
                "source_bug": self.source_bug,
                "notes": self.notes,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["correction_id"] = self.correction_id
        return payload


def resolve(corrections: Iterable[EventCorrection]) -> dict[str, EventCorrection]:
    """The standing status of every corrected event.

    The ledger is append-only, so an event may carry several corrections and the
    most recent one is what stands. "Most recent" has to be a *total* order or
    the answer depends on row order: two corrections recorded in the same
    instant would otherwise leave eligibility undefined, which is exactly the
    inconsistent state an append-only design is supposed to rule out. The
    content-addressed id breaks the tie, deterministically and identically on
    every machine.
    """
    standing: dict[str, EventCorrection] = {}
    for correction in corrections:
        current = standing.get(correction.event_id)
        if current is None or (correction.invalidated_at, correction.correction_id) > (
            current.invalidated_at,
            current.correction_id,
        ):
            standing[correction.event_id] = correction
    return standing


def ineligible_ids(
    corrections: Iterable[EventCorrection],
    *,
    as_of: datetime | None = None,
    mode: EligibilityMode = EligibilityMode.CORRECTED,
) -> frozenset[str]:
    """Event ids research must not count."""
    considered = list(corrections)
    if mode is EligibilityMode.AS_BELIEVED:
        if as_of is None:
            raise ValueError("AS_BELIEVED eligibility needs the instant being reconstructed")
        considered = [item for item in considered if item.invalidated_at <= as_of]
    return frozenset(
        event_id
        for event_id, correction in resolve(considered).items()
        if correction.status is CorrectionStatus.INVALIDATED
    )


def eligible(
    events: Sequence[EventSignal],
    corrections: Iterable[EventCorrection],
    *,
    as_of: datetime | None = None,
    mode: EligibilityMode = EligibilityMode.CORRECTED,
) -> list[EventSignal]:
    """The research view: everything except what a correction excludes."""
    excluded = ineligible_ids(corrections, as_of=as_of, mode=mode)
    return [event for event in events if event.event_id not in excluded]


def load_corrections(raw: Mapping[str, Any]) -> tuple[EventCorrection, ...]:
    """Read a committed correction file.

    Corrections live in the repository rather than being generated on the host.
    The set of affected ids is written out literally, so applying it is
    reproducible and cannot widen: a rule that recomputes "which events look
    like duplicates" would keep matching new things forever, and a correction
    that can grow on its own is not a correction, it is a filter.
    """
    if "corrections" not in raw:
        raise ValueError("correction file has no 'corrections' list")
    shared = {
        key: raw[key]
        for key in ("reason", "invalidated_at", "invalidated_by_version", "source_bug", "notes")
        if key in raw
    }
    entries: list[EventCorrection] = []
    for entry in raw["corrections"]:
        merged = {**shared, **entry}
        merged.setdefault("status", CorrectionStatus.INVALIDATED.value)
        entries.append(EventCorrection.model_validate(merged))
    return tuple(entries)


__all__ = [
    "ELIGIBILITY_CONTRACT_VERSION",
    "REDISCOVERY_DUPLICATE_PRE_FIX",
    "CorrectionStatus",
    "EligibilityMode",
    "EventCorrection",
    "eligible",
    "ineligible_ids",
    "load_corrections",
    "resolve",
]
