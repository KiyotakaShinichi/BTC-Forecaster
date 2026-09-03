"""Telling a quiet feed from a broken one, and both from a stale query.

Every silent-empty defect this collector has had shared one shape: a cycle
finished, reported success, and wrote nothing, and nothing in the output
distinguished that from a genuinely quiet news day. Four separate causes
produced identical green runs.

Counting is what makes them separable. A feed that returned no entries, a feed
whose entries matched no term, and a feed whose matches were all older than the
lookback are three different situations with three different remedies, and the
one thing they must never do is look alike:

* **BROKEN** -- the fetch or the parse failed. Fix the endpoint.
* **EMPTY_FEED** -- the publisher served a feed with no items. Watch it; a feed
  that stays empty for days is usually retired rather than quiet.
* **NO_TOPICAL_MATCH** -- entries arrived, none matched the query. Either the
  news genuinely is not about the watchlist, or the query no longer describes
  the source. For a stream that is not subject-bearing this is the *expected*
  state and carries no information at all.
* **HISTORICAL_MATCH_ONLY** -- entries matched, but every match predates the
  lookback window. The feed works and the query works; there is simply nothing
  new. Widening the query would fix nothing, and is the tempting wrong move.
* **COLLECTING** -- documents were admitted.

The distinction that took longest to see is the last two. A collector reporting
"0 documents" for a week looks broken and is often healthy, and looks healthy
and is sometimes broken; only the intermediate counts say which.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CollectionState(str, Enum):
    """What a feed did in one search. Stable codes, never prose."""

    COLLECTING = "COLLECTING"
    HISTORICAL_MATCH_ONLY = "HISTORICAL_MATCH_ONLY"
    NO_TOPICAL_MATCH = "NO_TOPICAL_MATCH"
    EMPTY_FEED = "EMPTY_FEED"
    BROKEN = "BROKEN"


@dataclass(frozen=True)
class FeedDiagnosis:
    """A feed's state, and whether that state is a problem."""

    feed_id: str
    state: CollectionState
    entries: int
    matched: int
    admitted: int
    #: True when this outcome is a known property of the source rather than a
    #: fault -- a non-subject-bearing stream that matches nothing, for instance.
    expected: bool
    detail: str

    @property
    def healthy(self) -> bool:
        return self.state is not CollectionState.BROKEN

    def as_dict(self) -> dict[str, object]:
        return {
            "feed_id": self.feed_id,
            "state": self.state.value,
            "entries": self.entries,
            "matched": self.matched,
            "admitted": self.admitted,
            "expected": self.expected,
            "detail": self.detail,
        }


def diagnose(feed_id: str, measured: object, *, failed: bool) -> FeedDiagnosis:
    """Classify one feed's search. `measured` is a `FeedYield`."""
    entries = int(getattr(measured, "entries", 0))
    matched = int(getattr(measured, "matched", 0))
    admitted = int(getattr(measured, "admitted", 0))
    subject_bearing = bool(getattr(measured, "subject_bearing", True))

    if failed:
        return FeedDiagnosis(
            feed_id, CollectionState.BROKEN, entries, matched, admitted, False,
            "the fetch or the parse failed; the endpoint needs attention",
        )
    if entries == 0:
        return FeedDiagnosis(
            feed_id, CollectionState.EMPTY_FEED, entries, matched, admitted, False,
            "the publisher served no items; a feed that stays empty is usually retired",
        )
    if admitted > 0:
        return FeedDiagnosis(
            feed_id, CollectionState.COLLECTING, entries, matched, admitted, True,
            f"{admitted} entries admitted",
        )
    if matched > 0:
        return FeedDiagnosis(
            feed_id, CollectionState.HISTORICAL_MATCH_ONLY, entries, matched, admitted, True,
            f"{matched} entries matched but all predate the lookback; nothing new, "
            "and widening the query would not change that",
        )
    return FeedDiagnosis(
        feed_id, CollectionState.NO_TOPICAL_MATCH, entries, matched, admitted, not subject_bearing,
        "no entry matched the query; expected for a stream whose fields carry "
        "respondent identity rather than subject matter"
        if not subject_bearing
        else "no entry matched the query; either the news is not about the "
        "watchlist, or the query no longer describes this source",
    )


__all__ = ["CollectionState", "FeedDiagnosis", "diagnose"]
