"""O12 — verify the corpus, and fail closed when it is wrong.

Continuous collection means nobody looks at the corpus for weeks at a time. The
checks here are the ones whose violation would be silent and would invalidate
research built on the corpus later — not "is the database readable", which the
next cycle answers anyway.

The ordering rule is the load-bearing one. `available_at <= retrieved_at` and
"an event is never available before the document it came from" are the two
invariants every point-in-time claim rests on. If either is violated, a study
run on this corpus is using information from the future and will look, in every
respect, like a real finding.

Fails closed: a corrupt corpus reports `CORRUPT` and names what broke. There is
no repair mode. Repairing evidence is how a corpus stops being evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Sequence

from ..collection.corpus import CorpusCatalog, membership_hash
from ..models import Document, EventSignal
from ..storage import IntelligenceStore


class IntegrityStatus(str, Enum):
    OK = "OK"
    #: Something is off but no point-in-time claim is invalidated.
    DEGRADED = "DEGRADED"
    #: A point-in-time invariant is broken. Research on this corpus is unsafe.
    CORRUPT = "CORRUPT"


@dataclass(frozen=True)
class IntegrityFinding:
    check: str
    status: IntegrityStatus
    detail: str
    subject: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "check": self.check,
            "status": self.status.value,
            "detail": self.detail,
            "subject": self.subject,
        }


@dataclass
class IntegrityReport:
    documents: int = 0
    events: int = 0
    sightings: int = 0
    watermarks: int = 0
    snapshots: int = 0
    findings: list[IntegrityFinding] = field(default_factory=list)

    @property
    def status(self) -> IntegrityStatus:
        if any(item.status is IntegrityStatus.CORRUPT for item in self.findings):
            return IntegrityStatus.CORRUPT
        if any(item.status is IntegrityStatus.DEGRADED for item in self.findings):
            return IntegrityStatus.DEGRADED
        return IntegrityStatus.OK

    @property
    def ok(self) -> bool:
        return self.status is IntegrityStatus.OK

    def record(self, check: str, status: IntegrityStatus, detail: str, subject: str = "") -> None:
        self.findings.append(IntegrityFinding(check, status, detail, subject))

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "documents": self.documents,
            "events": self.events,
            "sightings": self.sightings,
            "watermarks": self.watermarks,
            "snapshots": self.snapshots,
            "findings": [item.as_dict() for item in self.findings],
        }

    def human_readable(self) -> str:
        lines = [
            f"corpus integrity: {self.status.value}",
            f"  documents {self.documents}  events {self.events}  sightings {self.sightings}"
            f"  watermarks {self.watermarks}  snapshots {self.snapshots}",
        ]
        for finding in self.findings:
            lines.append(f"  [{finding.status.value}] {finding.check}: {finding.detail}")
            if finding.subject:
                lines.append(f"      {finding.subject}")
        if not self.findings:
            lines.append("  no findings")
        return "\n".join(lines)


def verify(store: IntelligenceStore, *, as_of: datetime, sample_limit: int = 0) -> IntegrityReport:
    """Check the corpus. `sample_limit` bounds per-check reporting, not scanning."""
    report = IntegrityReport()
    documents = store.documents_as_of(as_of)
    events = store.signals_as_of(as_of)
    report.documents = len(documents)
    report.events = len(events)

    _check_document_identity(documents, report, sample_limit)
    _check_availability_ordering(documents, report, sample_limit)
    _check_event_sources(documents, events, report, sample_limit)
    _check_event_availability(documents, events, report, sample_limit)
    _check_extractor_versions(events, report)
    _check_sightings(store, documents, report, sample_limit)
    _check_watermarks(store, report)
    _check_snapshots(store, documents, events, report)
    return report


def _check_document_identity(
    documents: Sequence[Document], report: IntegrityReport, limit: int
) -> None:
    """A document id is derived from its URL and content hash.

    Re-deriving it catches a payload that was edited in place -- which is the
    shape a corpus takes if anything ever "fixes" a stored document.
    """
    mismatched: list[str] = []
    for document in documents:
        expected = Document.stable_id(str(document.url), document.text_hash)
        if expected != document.document_id:
            mismatched.append(document.document_id)
    if mismatched:
        report.record(
            "document_identity",
            IntegrityStatus.CORRUPT,
            f"{len(mismatched)} document(s) do not hash to their stored id",
            ", ".join(mismatched[: limit or 5]),
        )


def _check_availability_ordering(
    documents: Sequence[Document], report: IntegrityReport, limit: int
) -> None:
    violations = [
        document.document_id for document in documents if document.available_at > document.retrieved_at
    ]
    if violations:
        report.record(
            "availability_ordering",
            IntegrityStatus.CORRUPT,
            f"{len(violations)} document(s) claim availability after retrieval",
            ", ".join(violations[: limit or 5]),
        )


def _check_event_sources(
    documents: Sequence[Document], events: Sequence[EventSignal], report: IntegrityReport, limit: int
) -> None:
    known = {document.document_id for document in documents}
    orphaned = [event.event_id for event in events if not set(event.source_ids) <= known]
    if orphaned:
        report.record(
            "event_sources",
            IntegrityStatus.CORRUPT,
            f"{len(orphaned)} event(s) reference documents that are not in the corpus",
            ", ".join(orphaned[: limit or 5]),
        )


def _check_event_availability(
    documents: Sequence[Document], events: Sequence[EventSignal], report: IntegrityReport, limit: int
) -> None:
    """The invariant a study's whole validity rests on."""
    by_id = {document.document_id: document for document in documents}
    violations: list[str] = []
    for event in events:
        sources = [by_id[source] for source in event.source_ids if source in by_id]
        if sources and event.available_time < max(source.available_at for source in sources):
            violations.append(event.event_id)
    if violations:
        report.record(
            "event_availability",
            IntegrityStatus.CORRUPT,
            f"{len(violations)} event(s) are available before the document they came from; "
            "any study on this corpus would be using information from the future",
            ", ".join(violations[: limit or 5]),
        )


def _check_extractor_versions(events: Sequence[EventSignal], report: IntegrityReport) -> None:
    missing = [event.event_id for event in events if not event.extractor_version.strip()]
    if missing:
        report.record(
            "extractor_versions",
            IntegrityStatus.CORRUPT,
            f"{len(missing)} event(s) carry no extractor version, so no snapshot can freeze them",
        )
    versions = sorted({event.extractor_version for event in events if event.extractor_version})
    if len(versions) > 1:
        report.record(
            "extractor_versions",
            IntegrityStatus.OK,
            f"{len(versions)} extractor versions coexist as expected: {', '.join(versions)}",
        )


def _check_sightings(
    store: IntelligenceStore, documents: Sequence[Document], report: IntegrityReport, limit: int
) -> None:
    row = store.connection.execute("SELECT count(*) FROM document_sightings").fetchone()
    report.sightings = int(row[0]) if row else 0

    missing: list[str] = []
    late: list[str] = []
    for document in documents:
        sighting = store.sighting(document.document_id)
        if sighting is None:
            missing.append(document.document_id)
            continue
        # first_seen_at is the fact availability rests on; it must never be
        # later than the availability the document was stored with.
        if sighting.first_seen_at > document.available_at:
            late.append(document.document_id)

    if missing:
        report.record(
            "sightings",
            IntegrityStatus.DEGRADED,
            f"{len(missing)} document(s) have no sighting record; rediscovery history is incomplete "
            "but no availability claim is affected",
            ", ".join(missing[: limit or 5]),
        )
    if late:
        report.record(
            "sightings",
            IntegrityStatus.CORRUPT,
            f"{len(late)} document(s) were first seen after the availability they claim",
            ", ".join(late[: limit or 5]),
        )


def _check_watermarks(store: IntelligenceStore, report: IntegrityReport) -> None:
    rows = store.connection.execute("SELECT payload FROM watermarks").fetchall()
    report.watermarks = len(rows)
    future: list[str] = []
    for (payload,) in rows:
        record = json.loads(payload)
        available = record.get("last_successful_available_time")
        retrieval = record.get("last_retrieval_time")
        if available and retrieval and datetime.fromisoformat(available) > datetime.fromisoformat(retrieval):
            future.append(f"{record.get('provider_id')}::{record.get('query_id')}")
    if future:
        report.record(
            "watermarks",
            IntegrityStatus.CORRUPT,
            f"{len(future)} watermark(s) record availability after their own retrieval",
            ", ".join(future[:5]),
        )


def _check_snapshots(
    store: IntelligenceStore,
    documents: Sequence[Document],
    events: Sequence[EventSignal],
    report: IntegrityReport,
) -> None:
    """Re-derive each snapshot's membership hash from the live corpus.

    A snapshot is what a research result cites. If its recorded membership no
    longer matches what the store holds at that instant, the citation is broken
    -- and that is exactly the failure a catalog exists to make detectable.
    """
    catalog = CorpusCatalog(store.connection)
    snapshots = catalog.list_snapshots(limit=1000)
    report.snapshots = len(snapshots)
    for snapshot in snapshots:
        included_documents = [item for item in documents if item.available_at <= snapshot.as_of]
        document_ids = {item.document_id for item in included_documents}
        included_events = [
            item
            for item in events
            if item.available_time <= snapshot.as_of
            and item.extractor_version == snapshot.extractor_version
            and set(item.source_ids) <= document_ids
        ]
        recomputed = membership_hash(
            [item.document_id for item in included_documents],
            [item.event_id for item in included_events],
        )
        if recomputed != snapshot.membership_hash:
            report.record(
                "snapshot_membership",
                IntegrityStatus.CORRUPT,
                "a registered corpus snapshot no longer matches the store at its own as-of instant",
                snapshot.corpus_id,
            )


__all__ = [
    "IntegrityFinding",
    "IntegrityReport",
    "IntegrityStatus",
    "verify",
]
