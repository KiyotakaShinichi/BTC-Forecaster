"""B4.1.26 / B4.1.27 — the corpus catalog and its frozen read-only snapshots.

A B4 validation run must be able to say *which* corpus it analysed, and prove
later that the corpus has not moved. Forward collection means the store grows
every hour, so "the corpus" is not a thing without a name and a boundary.

A snapshot is that boundary: an as-of instant plus a frozen extractor version.
Everything available at or before the instant, extracted by that version, is in;
everything else is out. Because availability is immutable and event identity now
carries the extractor version, the same snapshot request always resolves to the
same rows — which is what makes the fingerprint mean anything.

The catalog is append-only. Re-registering identical contents is a no-op, so a
reproduced snapshot is not punished; registering different contents under an
existing id is an error, because two different corpora sharing one name would
make every result that cites it ambiguous.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from ..errors import ReplayIntegrityError
from ..models import Document, EventSignal
from .clustering import cluster_events


class CorpusSnapshot(BaseModel):
    """B4.1.27. A deterministic, read-only view of the corpus."""

    model_config = ConfigDict(frozen=True)

    corpus_id: str
    #: Everything with availability at or before this instant is included.
    as_of: datetime
    #: Only events from this extractor version. Frozen so a later extractor
    #: cannot change what a published result was computed on.
    extractor_version: str
    span_start: datetime | None
    span_end: datetime | None
    document_count: int
    event_count: int
    cluster_count: int
    provider_coverage: dict[str, int]
    entity_coverage: dict[str, int]
    event_type_coverage: dict[str, int]
    whale_context_coverage: dict[str, int]
    primary_source_documents: int
    schema_versions: tuple[str, ...]
    #: Content fingerprint over the member ids, so a moved corpus is detectable.
    membership_hash: str
    created_at: datetime

    @field_validator("as_of", "created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ReplayIntegrityError("corpus timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    def identity(self) -> dict[str, Any]:
        """The fields two snapshots sharing an id must agree on."""
        return {
            "corpus_id": self.corpus_id,
            "as_of": self.as_of.isoformat(),
            "extractor_version": self.extractor_version,
            "document_count": self.document_count,
            "event_count": self.event_count,
            "cluster_count": self.cluster_count,
            "membership_hash": self.membership_hash,
        }


def membership_hash(document_ids: Sequence[str], event_ids: Sequence[str]) -> str:
    """Order-independent over the sets, order-sensitive between them.

    Sorting inside each set means the hash does not depend on retrieval order;
    keeping the two sets separate means a document id and an event id that
    happened to collide could not silently swap.
    """
    digest = hashlib.sha256()
    digest.update(b"documents\n")
    for document_id in sorted(document_ids):
        digest.update(f"{document_id}\n".encode())
    digest.update(b"events\n")
    for event_id in sorted(event_ids):
        digest.update(f"{event_id}\n".encode())
    return digest.hexdigest()


def build_snapshot(
    documents: Sequence[Document],
    events: Sequence[EventSignal],
    *,
    as_of: datetime,
    extractor_version: str,
    created_at: datetime,
    cluster_window_hours: int = 24,
) -> CorpusSnapshot:
    """Freeze the corpus as of an instant, for one extractor version.

    Events are filtered to the frozen extractor *and* to documents inside the
    snapshot: an event whose source document is outside the boundary would give
    a study a source it cannot resolve.
    """
    # Validated here rather than only on the model: `as_of` is compared against
    # aware timestamps below, so a naive value would surface as a TypeError from
    # deep inside a sort rather than as the contract violation it is.
    as_of = _require_utc(as_of, "as_of")
    created_at = _require_utc(created_at, "created_at")

    included_documents = sorted(
        (document for document in documents if document.available_at <= as_of),
        key=lambda document: (document.available_at, document.document_id),
    )
    document_ids = {document.document_id for document in included_documents}
    included_events = sorted(
        (
            event
            for event in events
            if event.available_time <= as_of
            and event.extractor_version == extractor_version
            and set(event.source_ids) <= document_ids
        ),
        key=lambda event: (event.available_time, event.event_id),
    )
    clusters = cluster_events(included_events, included_documents, window_hours=cluster_window_hours)

    corpus_id = hashlib.sha256(
        json.dumps(
            {
                "as_of": as_of.astimezone(timezone.utc).isoformat(),
                "extractor_version": extractor_version,
                "membership": membership_hash(
                    [document.document_id for document in included_documents],
                    [event.event_id for event in included_events],
                ),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]

    return CorpusSnapshot(
        corpus_id=f"corpus-{corpus_id}",
        as_of=as_of,
        extractor_version=extractor_version,
        span_start=included_documents[0].available_at if included_documents else None,
        span_end=included_documents[-1].available_at if included_documents else None,
        document_count=len(included_documents),
        event_count=len(included_events),
        cluster_count=len(clusters),
        provider_coverage=_counted(document.provider for document in included_documents),
        entity_coverage=_counted(event.entity or "(none)" for event in included_events),
        event_type_coverage=_counted(event.event_type.value for event in included_events),
        whale_context_coverage=_counted(
            event.transfer_context.value
            for event in included_events
            if event.transfer_context is not None
        ),
        primary_source_documents=sum(
            1 for document in included_documents if document.source_metadata.primary_source
        ),
        schema_versions=tuple(
            sorted(
                {document.schema_version for document in included_documents}
                | {event.schema_version for event in included_events}
            )
        ),
        membership_hash=membership_hash(
            [document.document_id for document in included_documents],
            [event.event_id for event in included_events],
        ),
        created_at=created_at,
    )


def _require_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ReplayIntegrityError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _counted(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


class CorpusCatalog:
    """Append-only registry of corpus snapshots (B4.1.26)."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_catalog (
              corpus_id VARCHAR PRIMARY KEY,
              as_of TIMESTAMPTZ NOT NULL,
              extractor_version VARCHAR NOT NULL,
              created_at TIMESTAMPTZ NOT NULL,
              payload JSON NOT NULL
            )
            """
        )

    def register(self, snapshot: CorpusSnapshot) -> CorpusSnapshot:
        existing = self.get(snapshot.corpus_id)
        if existing is not None:
            if existing.identity() != snapshot.identity():
                raise ReplayIntegrityError(
                    f"corpus {snapshot.corpus_id} is already registered with different contents; "
                    "catalog entries are never overwritten"
                )
            return existing
        self.connection.execute(
            "INSERT INTO corpus_catalog VALUES (?, ?, ?, ?, ?)",
            [
                snapshot.corpus_id,
                snapshot.as_of,
                snapshot.extractor_version,
                snapshot.created_at,
                snapshot.model_dump_json(),
            ],
        )
        return snapshot

    def get(self, corpus_id: str) -> CorpusSnapshot | None:
        row = self.connection.execute(
            "SELECT payload FROM corpus_catalog WHERE corpus_id = ?", [corpus_id]
        ).fetchone()
        return CorpusSnapshot.model_validate(json.loads(row[0])) if row else None

    def list_snapshots(self, limit: int = 50) -> list[CorpusSnapshot]:
        rows = self.connection.execute(
            "SELECT payload FROM corpus_catalog ORDER BY as_of DESC, corpus_id LIMIT ?", [limit]
        ).fetchall()
        return [CorpusSnapshot.model_validate(json.loads(row[0])) for row in rows]

    def latest(self) -> CorpusSnapshot | None:
        found = self.list_snapshots(limit=1)
        return found[0] if found else None


__all__ = [
    "CorpusCatalog",
    "CorpusSnapshot",
    "build_snapshot",
    "membership_hash",
]
