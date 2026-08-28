from __future__ import annotations

from datetime import datetime, timezone

from .models import Document, EventSignal
from .operations import QualityIssue, QualityReport


EXPECTED_DOCUMENT_SCHEMA = "document-v2"
EXPECTED_EVENT_SCHEMA = "event-v2"


def evidence_confidence_multiplier(document: Document) -> float:
    """Operational evidence completeness only; never an ideological truth score."""
    metadata = document.source_metadata
    score = (0.25 * float(metadata.official_source) + 0.20 * float(metadata.primary_source)
             + 0.10 * float(metadata.known_publisher) + 0.25 * metadata.timestamp_quality
             + 0.20 * metadata.content_completeness)
    return min(1.0, max(0.0, score))


def evaluate_quality(documents: list[Document], events: list[EventSignal],
                     as_of: datetime | None = None) -> QualityReport:
    issues: list[QualityIssue] = []
    doc_ids, event_ids = [d.document_id for d in documents], [e.event_id for e in events]
    known = set(doc_ids)
    for duplicate in sorted({x for x in doc_ids if doc_ids.count(x) > 1}):
        issues.append(QualityIssue(code="DUPLICATE_DOCUMENT_ID", record_id=duplicate, detail="duplicate document ID"))
    for duplicate in sorted({x for x in event_ids if event_ids.count(x) > 1}):
        issues.append(QualityIssue(code="DUPLICATE_EVENT_ID", record_id=duplicate, detail="duplicate event ID"))
    for doc in documents:
        if doc.available_at.utcoffset() != timezone.utc.utcoffset(doc.available_at):
            issues.append(QualityIssue(code="NON_UTC_TIMESTAMP", record_id=doc.document_id, detail="document availability not UTC"))
        if as_of and doc.available_at > as_of:
            issues.append(QualityIssue(code="FUTURE_AVAILABILITY", record_id=doc.document_id, detail="document is future-known"))
        if doc.schema_version != EXPECTED_DOCUMENT_SCHEMA:
            issues.append(QualityIssue(code="SCHEMA_VERSION", record_id=doc.document_id, detail=doc.schema_version))
    for event in events:
        missing = set(event.source_ids) - known
        if missing:
            issues.append(QualityIssue(code="ORPHANED_EVENT", record_id=event.event_id, detail=",".join(sorted(missing))))
        if as_of and event.available_time > as_of:
            issues.append(QualityIssue(code="FUTURE_AVAILABILITY", record_id=event.event_id, detail="event is future-known"))
        if event.schema_version != EXPECTED_EVENT_SCHEMA:
            issues.append(QualityIssue(code="SCHEMA_VERSION", record_id=event.event_id, detail=event.schema_version))
    return QualityReport(valid=not issues, documents_checked=len(documents), events_checked=len(events), issues=issues)
