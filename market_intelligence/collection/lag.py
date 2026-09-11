"""B5.1 — collection lag: how long after publication the collector first saw a document.

    collection_lag = first_seen_at - published_at

`first_seen_at` is the store's first sighting of the document
(`document_sightings`), falling back to the document's `available_at` -- its
first retrieval, which rediscovery never moves -- when there is no sighting row.

**Descriptive, and nothing more.** It says how stale the collector's view of a
source is; at a three-hour cadence most of it is the source's own delay or a
backfill of items published before collection began. It redefines nothing: an
event's `event_time` and `available_time` are what extraction wrote, a
document's `available_at` is still its first retrieval, and every point-in-time
query still admits only what was available at or before the origin. A lag is
computed from those fields and written nowhere any of them is read.

A document with no publication time has no lag. It is counted, never guessed.
A document published after it was first seen has a negative lag -- an
impossible timestamp -- and is reported as one, never clipped to zero.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..models import Document

#: Bumped if what a lag measures ever changes.
LAG_CONTRACT_VERSION = "collection-lag-v1-first-seen-minus-published"


class LagBasis(str, Enum):
    #: From the store's sighting record.
    FIRST_SEEN = "FIRST_SEEN"
    #: No sighting row; the document's first retrieval, which is the same instant.
    AVAILABLE_AT = "AVAILABLE_AT"


class DocumentLag(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    published_at: datetime | None
    first_seen_at: datetime
    basis: LagBasis
    #: None when the document carries no publication time.
    lag_hours: float | None

    @property
    def impossible(self) -> bool:
        return self.lag_hours is not None and self.lag_hours < 0


class LagSummary(BaseModel):
    """The distribution of collection lag over a set of documents."""

    model_config = ConfigDict(frozen=True)

    contract: str = LAG_CONTRACT_VERSION
    documents: int
    measured: int
    without_publication_time: int
    #: Published after first seen. Reported, not corrected.
    impossible: int
    basis: dict[str, int]
    min_hours: float | None
    median_hours: float | None
    p90_hours: float | None
    max_hours: float | None

    def describe(self) -> str:
        if not self.documents:
            return "no documents -- collection lag is not yet measurable"
        if not self.measured:
            return f"none of {self.documents} document(s) carries a publication time -- no lag measurable"
        return (
            f"first seen - published: median {self.median_hours:.1f} h, p90 {self.p90_hours:.1f} h, "
            f"max {self.max_hours:.1f} h over {self.measured} of {self.documents} document(s); "
            f"{self.without_publication_time} without a publication time, {self.impossible} impossible"
        )


def _utc(moment: datetime) -> datetime:
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc)


def document_lag(document: Document, first_seen_at: datetime | None) -> DocumentLag:
    """One document's lag, from its sighting when there is one."""
    basis = LagBasis.FIRST_SEEN if first_seen_at is not None else LagBasis.AVAILABLE_AT
    seen = _utc(first_seen_at if first_seen_at is not None else document.available_at)
    published = _utc(document.published_at) if document.published_at is not None else None
    lag = round((seen - published).total_seconds() / 3600.0, 6) if published is not None else None
    return DocumentLag(
        document_id=document.document_id, published_at=published, first_seen_at=seen, basis=basis, lag_hours=lag
    )


def first_sightings(connection: Any) -> dict[str, datetime]:
    """Each document's first sighting, in UTC whatever the session's zone."""
    rows = connection.execute("SELECT document_id, first_seen_at FROM document_sightings").fetchall()
    return {str(document_id): _utc(seen) for document_id, seen in rows}


def summarize(lags: Iterable[DocumentLag]) -> LagSummary:
    items = list(lags)
    measured = sorted(item.lag_hours for item in items if item.lag_hours is not None)
    basis: dict[str, int] = {}
    for item in items:
        basis[item.basis.value] = basis.get(item.basis.value, 0) + 1
    return LagSummary(
        documents=len(items),
        measured=len(measured),
        without_publication_time=len(items) - len(measured),
        impossible=sum(1 for item in items if item.impossible),
        basis=dict(sorted(basis.items())),
        min_hours=measured[0] if measured else None,
        median_hours=float(statistics.median(measured)) if measured else None,
        # Nearest rank: a value that occurred, not an interpolation between two.
        p90_hours=measured[max(0, math.ceil(0.9 * len(measured)) - 1)] if measured else None,
        max_hours=measured[-1] if measured else None,
    )


def lag_summary(connection: Any, documents: Sequence[Document]) -> LagSummary:
    """The collection lag of `documents`, read against the store's sightings."""
    seen = first_sightings(connection)
    return summarize(document_lag(document, seen.get(document.document_id)) for document in documents)


__all__ = [
    "LAG_CONTRACT_VERSION",
    "DocumentLag",
    "LagBasis",
    "LagSummary",
    "document_lag",
    "first_sightings",
    "lag_summary",
    "summarize",
]
