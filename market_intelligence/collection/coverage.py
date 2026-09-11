"""B5.1 — collection coverage, computed from what collection actually did.

B4's readiness policy requires collection on 80% of the days a family spans. B5
found that nothing computed it. `corpus-status` never passed a coverage figure,
so the clause always read 0%. The readiness gate's own `coverage_fraction()`
had no callers. The collection service's `successful_run_days()` read each
watermark's *latest* retrieval and so forgot every earlier day. And
`coverage_fraction()` counted every day it was given, inside the span or not, so
a family whose span saw no collection could score 100% on runs from other months.

This module is the one definition, and `corpus-status` and B5's Gate 1 audit
both read it.

**A covered day is a UTC calendar day on which at least one provider attempt
succeeded**, read from `provider_attempts`, where every cycle records every
attempt, success or not. Run status does not decide it: `DEGRADED` means every
query succeeded and a quality check failed, which is still collection. A day
whose only attempts failed -- an outage, a provider that did not answer, a
retired feed that 404s -- is not covered.

**Days are UTC.** DuckDB returns TIMESTAMPTZ in the session's zone, so an
attempt at 23:30Z read back in Manila lands on the next day unless converted.

**Coverage is never fabricated in either direction.** A store with no
collection record is `NO_COLLECTION`, with coverage `None` rather than 0%; and a
day outside a span never counts toward it.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

#: Bumped when what counts as a covered day changes, so a readiness figure can
#: say which rule produced it.
COVERAGE_CONTRACT_VERSION = "coverage-v2-provider-success-days"

#: The collector's cadence (deploy/COLLECTION_FREEZE.md, section 12). Used only to
#: say how many cycles a span should have held. Readiness is about days.
CADENCE_HOURS = 3


class CoverageState(str, Enum):
    #: No provider attempt has ever been recorded. Coverage is not measurable; it is not zero.
    NO_COLLECTION = "NO_COLLECTION"
    MEASURED = "MEASURED"


def utc_day(moment: datetime) -> date:
    """The UTC calendar day of an instant. A naive instant is taken as UTC."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).date()


def span_coverage(success_days: Iterable[date], start: date, end: date) -> float:
    """Covered days in [start, end] over the days in [start, end].

    Both ends are inclusive, so a one-day span with collection is fully covered.
    A day outside the span never counts toward it.
    """
    if end < start:
        raise ValueError(f"a coverage span cannot end ({end}) before it starts ({start})")
    total = (end - start).days + 1
    covered = {day for day in success_days if start <= day <= end}
    return len(covered) / total


def provider_attempts(connection: Any, *, as_of: datetime) -> list[tuple[str, str, bool, datetime]]:
    """(run_id, provider_id, success, observed_at in UTC) for every attempt up to `as_of`."""
    rows = connection.execute(
        "SELECT run_id, provider_id, success, observed_at FROM provider_attempts "
        "WHERE observed_at <= ? ORDER BY observed_at, run_id, provider_id",
        [as_of],
    ).fetchall()
    return [
        (str(run_id), str(provider_id), bool(success), observed.astimezone(timezone.utc))
        for run_id, provider_id, success, observed in rows
    ]


def successful_days(connection: Any, *, as_of: datetime) -> tuple[date, ...]:
    """Every UTC day up to `as_of` on which at least one provider attempt succeeded."""
    return tuple(
        sorted({utc_day(observed) for _, _, success, observed in provider_attempts(connection, as_of=as_of) if success})
    )


class CollectionCoverage(BaseModel):
    """What collection did up to an instant, and how much of the elapsed time it covered."""

    model_config = ConfigDict(frozen=True)

    contract: str = COVERAGE_CONTRACT_VERSION
    state: CoverageState
    as_of: datetime
    first_attempt: datetime | None
    last_attempt: datetime | None
    first_success: datetime | None
    last_success: datetime | None
    #: UTC days from the first attempt's day to the instant's day, both inclusive.
    elapsed_days: int
    #: Every elapsed day was an opportunity to collect: the collector runs every
    #: three hours, so a day with no successful attempt is a missed day.
    expected_days: int
    successful_days: int
    #: successful_days / expected_days, or None when there is no collection record.
    coverage_fraction: float | None
    cadence_hours: int
    expected_cycles: int
    attempted_cycles: int
    successful_cycles: int
    failed_attempts: int
    days_since_last_success: int | None

    def describe(self) -> str:
        if self.state is CoverageState.NO_COLLECTION or self.coverage_fraction is None:
            return "no provider attempt recorded -- coverage is not yet measurable"
        last = (
            f"last success {self.last_success.date().isoformat()} ({self.days_since_last_success} day(s) before)"
            if self.last_success is not None
            else "no successful attempt yet"
        )
        return (
            f"{self.successful_days} of {self.expected_days} elapsed day(s) with a successful provider attempt "
            f"({self.coverage_fraction:.0%}); {last}; {self.successful_cycles} of {self.expected_cycles} "
            f"cycle(s) at the {self.cadence_hours} h cadence"
        )


def collection_coverage(connection: Any, *, as_of: datetime, cadence_hours: int = CADENCE_HOURS) -> CollectionCoverage:
    """The collection record up to `as_of`, measured the one way."""
    moment = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    rows = provider_attempts(connection, as_of=moment)
    if not rows:
        return CollectionCoverage(
            state=CoverageState.NO_COLLECTION,
            as_of=moment,
            first_attempt=None,
            last_attempt=None,
            first_success=None,
            last_success=None,
            elapsed_days=0,
            expected_days=0,
            successful_days=0,
            coverage_fraction=None,
            cadence_hours=cadence_hours,
            expected_cycles=0,
            attempted_cycles=0,
            successful_cycles=0,
            failed_attempts=0,
            days_since_last_success=None,
        )
    first, last = rows[0][3], rows[-1][3]
    successes = [row for row in rows if row[2]]
    first_success = min((row[3] for row in successes), default=None)
    last_success = max((row[3] for row in successes), default=None)
    start, end = utc_day(first), utc_day(moment)
    days = {utc_day(row[3]) for row in successes}
    elapsed = (end - start).days + 1
    return CollectionCoverage(
        state=CoverageState.MEASURED,
        as_of=moment,
        first_attempt=first,
        last_attempt=last,
        first_success=first_success,
        last_success=last_success,
        elapsed_days=elapsed,
        expected_days=elapsed,
        successful_days=len(days),
        coverage_fraction=span_coverage(days, start, end),
        cadence_hours=cadence_hours,
        expected_cycles=int((moment - first).total_seconds() // (cadence_hours * 3600)) + 1,
        attempted_cycles=len({row[0] for row in rows}),
        successful_cycles=len({row[0] for row in successes}),
        failed_attempts=sum(1 for row in rows if not row[2]),
        days_since_last_success=(end - utc_day(last_success)).days if last_success is not None else None,
    )


__all__ = [
    "CADENCE_HOURS",
    "COVERAGE_CONTRACT_VERSION",
    "CollectionCoverage",
    "CoverageState",
    "collection_coverage",
    "provider_attempts",
    "span_coverage",
    "successful_days",
    "utc_day",
]
