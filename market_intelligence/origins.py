"""The canonical forecast-origin generator (B3.1.9).

One generator, so every dataset in the project agrees on what "hourly origins
from A to B" means. Three properties are non-negotiable and each is tested:

**UTC only.** A naive datetime is rejected rather than assumed. Generating
origins in local time would make a dataset depend on the machine that built it,
and the difference would show up as a silent one-hour shift twice a year.

**Deterministic and unique.** Sorted ascending, no duplicates, and the same
inputs always give the same list.

**Explicit boundary semantics.** ``start`` is inclusive, ``end`` is inclusive
when it falls exactly on the cadence. That choice is arbitrary but it has to be
written down, because "1,000 hourly origins from midnight" is ambiguous
otherwise, and an off-by-one at the end of a dataset is the kind of thing that
survives review.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum


class OriginFrequency(str, Enum):
    """Supported cadences. Each is an exact multiple of an hour."""

    HOURLY = "HOURLY"
    FOUR_HOURLY = "4H"
    DAILY = "DAILY"

    @property
    def step(self) -> timedelta:
        return {
            OriginFrequency.HOURLY: timedelta(hours=1),
            OriginFrequency.FOUR_HOURLY: timedelta(hours=4),
            OriginFrequency.DAILY: timedelta(days=1),
        }[self]


class OriginScheduleError(ValueError):
    """Raised when an origin schedule cannot be generated as requested."""


def require_utc(moment: datetime, *, label: str) -> datetime:
    """Reject naive datetimes; normalise aware ones to UTC.

    Deliberately not `assume UTC`. A naive datetime here is a caller who has not
    decided what their timestamps mean, and guessing on their behalf is how a
    dataset silently becomes machine-dependent.
    """
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise OriginScheduleError(f"{label} must be timezone-aware; origins are UTC by contract")
    return moment.astimezone(timezone.utc)


def generate_origins(
    start: datetime,
    end: datetime,
    frequency: OriginFrequency = OriginFrequency.HOURLY,
    *,
    limit: int | None = None,
) -> list[datetime]:
    """Origins from ``start`` to ``end``, inclusive of both when on cadence.

    ``limit`` caps the count and is applied from the start, so a truncated
    schedule is always a prefix of the full one -- which is what makes
    incremental extension well defined.
    """
    first = require_utc(start, label="start")
    last = require_utc(end, label="end")
    if last < first:
        raise OriginScheduleError(f"end ({last.isoformat()}) precedes start ({first.isoformat()})")
    if limit is not None and limit < 1:
        raise OriginScheduleError("limit must be >= 1")

    step = frequency.step
    origins: list[datetime] = []
    current = first
    while current <= last:
        origins.append(current)
        if limit is not None and len(origins) >= limit:
            break
        current = current + step
    return origins


def next_origin_after(origin: datetime, frequency: OriginFrequency) -> datetime:
    """The following origin on the same cadence."""
    return require_utc(origin, label="origin") + frequency.step


def validate_origin_schedule(origins: list[datetime]) -> None:
    """Assert the invariants a replay dataset depends on.

    Called before any replay so a malformed schedule fails at the boundary
    rather than producing a dataset whose rows do not mean what its manifest
    says.
    """
    if not origins:
        raise OriginScheduleError("origin schedule is empty")
    for origin in origins:
        require_utc(origin, label="origin")
    if origins != sorted(origins):
        raise OriginScheduleError("origins must be sorted ascending")
    if len(set(origins)) != len(origins):
        raise OriginScheduleError("origins must be unique")


def infer_frequency(origins: list[datetime]) -> OriginFrequency | None:
    """The cadence of an existing schedule, or None if it is irregular.

    Used by incremental extension to refuse an append whose cadence does not
    match the dataset it claims to extend.
    """
    if len(origins) < 2:
        return None
    gaps = {later - earlier for earlier, later in zip(origins, origins[1:], strict=False)}
    if len(gaps) != 1:
        return None
    gap = gaps.pop()
    for frequency in OriginFrequency:
        if frequency.step == gap:
            return frequency
    return None


def parse_origin(value: str) -> datetime:
    """Parse one forecast origin from text, in UTC, or raise.

    A naive timestamp is refused rather than assumed to be UTC. The whole
    corpus is timezone-aware and every comparison in the point-in-time contract
    is between aware instants; guessing here would make an origin mean whatever
    the machine's clock happened to be configured for, and the error would show
    up as a silent offset in a dataset rather than as a failure.

    `Z` is accepted because it is what every feed and every manifest writes;
    `datetime.fromisoformat` did not understand it before Python 3.11.
    """
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


__all__ = [
    "OriginFrequency",
    "OriginScheduleError",
    "generate_origins",
    "infer_frequency",
    "next_origin_after",
    "parse_origin",
    "require_utc",
    "validate_origin_schedule",
]
