"""B4.1.22 – B4.1.24, B4.1.34, B4.1.42 — is collection working, and is it enough?

Two questions an operator asks daily and a researcher asks once: is the pipeline
healthy, and does the corpus yet support the studies B4 could not run.

The distinction this module exists to protect is **missingness versus zero**. A
day with no events because nothing happened and a day with no events because
every provider was down look identical in a count. They are recorded separately
here, and every report carries provider coverage alongside the counts, because
B4 ended on HOLD precisely for want of that distinction.

Storage growth is measured, not modelled. Bytes per thousand documents comes
from the bytes actually stored; the projection is linear and labelled as a
projection. Nothing is optimised on the strength of it (B4.1.42).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from ..models import Document, EventSignal, TransferContext
from .clustering import EventCluster
from .readiness import FamilyReadiness, Readiness, overall_readiness


class DailyCoverage(BaseModel):
    """B4.1.23. One day of collection, with missingness kept visible."""

    model_config = ConfigDict(frozen=True)

    day: date
    providers_enabled: int
    providers_successful: int
    documents: int
    clusters: int
    events: int
    high_relevance_events: int
    whale_observations: int
    #: True when no provider succeeded. A zero-document day with this set is a
    #: collection outage; without it, it is a quiet news day. Conflating the two
    #: is how a study mistakes an outage for an absence of events.
    collection_gap: bool

    @property
    def coverage_ratio(self) -> float:
        return self.providers_successful / self.providers_enabled if self.providers_enabled else 0.0


class StorageGrowth(BaseModel):
    """B4.1.42. Measured bytes, and a linear projection labelled as one."""

    model_config = ConfigDict(frozen=True)

    documents: int
    events: int
    document_bytes: int
    event_bytes: int
    raw_evidence_bytes: int
    total_bytes: int
    bytes_per_1k_documents: float
    bytes_per_1k_events: float
    projection_note: str = (
        "linear extrapolation from measured bytes; says nothing about whether the "
        "collection rate holds, and is not a reason to compress anything yet"
    )

    def projected_bytes(self, documents: int, events: int) -> int:
        return int(self.bytes_per_1k_documents * documents / 1000 + self.bytes_per_1k_events * events / 1000)


class CorpusStatus(BaseModel):
    """B4.1.34. The whole picture, machine-readable."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    corpus_id: str | None
    span_start: datetime | None
    span_end: datetime | None
    span_days: int
    documents: int
    events: int
    clusters: int
    primary_source_documents: int
    provider_coverage: dict[str, int]
    event_type_coverage: dict[str, int]
    entity_coverage: dict[str, int]
    whale_context_coverage: dict[str, int]
    #: Categories that were asked about and produced nothing. Listed explicitly,
    #: because an absent key in a coverage map reads as "not tracked".
    missing_event_types: tuple[str, ...]
    missing_entities: tuple[str, ...]
    missing_whale_contexts: tuple[str, ...]
    provider_states: dict[str, str]
    daily_coverage: tuple[DailyCoverage, ...]
    collection_gap_days: int
    readiness: Readiness
    families: tuple[FamilyReadiness, ...]
    storage: StorageGrowth | None = None

    def human_readable(self) -> str:
        """B4.1.34. The same facts, for a terminal."""
        lines = [
            f"corpus       {self.corpus_id or '(none registered)'}",
            f"span         {_day(self.span_start)} -> {_day(self.span_end)}  ({self.span_days} days)",
            f"documents    {self.documents}   events {self.events}   clusters {self.clusters}",
            f"primary src  {self.primary_source_documents} documents from the party the news is about",
            f"gaps         {self.collection_gap_days} day(s) with no successful provider",
            "",
            f"READINESS    {self.readiness.value}",
        ]
        for family in self.families:
            marker = "ready" if family.ready else family.readiness.value.lower()
            lines.append(f"  {family.family:<34} {family.events:>4} events  {marker}")
            for reason in family.unmet:
                lines.append(f"      - {reason}")
        if self.missing_event_types:
            lines.append("")
            lines.append(f"no evidence yet: {', '.join(self.missing_event_types)}")
        if self.missing_entities:
            lines.append(f"no entity evidence yet: {', '.join(self.missing_entities)}")
        if self.storage is not None:
            lines.append("")
            lines.append(
                f"storage      {self.storage.total_bytes:,} bytes "
                f"({self.storage.bytes_per_1k_documents:,.0f}/1k docs, "
                f"{self.storage.bytes_per_1k_events:,.0f}/1k events)"
            )
        return "\n".join(lines)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return target


def _day(moment: datetime | None) -> str:
    return moment.date().isoformat() if moment else "-"


def daily_coverage(
    documents: Sequence[Document],
    events: Sequence[EventSignal],
    clusters: Sequence[EventCluster],
    *,
    providers_enabled: int,
    successful_run_days: Sequence[datetime] = (),
    high_relevance_threshold: float = 0.75,
) -> list[DailyCoverage]:
    """One row per day in the corpus span, including empty ones.

    Days with nothing are emitted rather than skipped: a sparse series with
    holes silently omitted looks denser than it is, and the holes are the part
    a coverage-bias check needs.
    """
    if not documents:
        return []
    days_with_success = {moment.date() for moment in successful_run_days}
    by_day_documents = _bucket(documents, lambda item: item.available_at.date())
    by_day_events = _bucket(events, lambda item: item.available_time.date())
    by_day_clusters = _bucket(clusters, lambda item: item.first_available_at.date())

    start = min(document.available_at for document in documents).date()
    end = max(document.available_at for document in documents).date()
    rows: list[DailyCoverage] = []
    cursor = start
    while cursor <= end:
        day_events = by_day_events.get(cursor, [])
        day_documents = by_day_documents.get(cursor, [])
        rows.append(
            DailyCoverage(
                day=cursor,
                providers_enabled=providers_enabled,
                providers_successful=len({document.provider for document in day_documents}),
                documents=len(day_documents),
                clusters=len(by_day_clusters.get(cursor, [])),
                events=len(day_events),
                high_relevance_events=sum(
                    1 for event in day_events if event.btc_relevance >= high_relevance_threshold
                ),
                whale_observations=sum(1 for event in day_events if event.transfer_context is not None),
                collection_gap=cursor not in days_with_success if days_with_success else not day_documents,
            )
        )
        cursor += timedelta(days=1)
    return rows


def _bucket(items: Sequence[Any], key: Any) -> dict[date, list[Any]]:
    grouped: dict[date, list[Any]] = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)
    return grouped


def measure_storage(
    documents: Sequence[Document],
    events: Sequence[EventSignal],
    raw_evidence_bytes: int = 0,
) -> StorageGrowth:
    """B4.1.42. Bytes actually stored, per thousand records."""
    document_bytes = sum(len(document.model_dump_json().encode("utf-8")) for document in documents)
    event_bytes = sum(len(event.model_dump_json().encode("utf-8")) for event in events)
    return StorageGrowth(
        documents=len(documents),
        events=len(events),
        document_bytes=document_bytes,
        event_bytes=event_bytes,
        raw_evidence_bytes=raw_evidence_bytes,
        total_bytes=document_bytes + event_bytes + raw_evidence_bytes,
        bytes_per_1k_documents=(document_bytes / len(documents) * 1000) if documents else 0.0,
        bytes_per_1k_events=(event_bytes / len(events) * 1000) if events else 0.0,
    )


def build_status(
    documents: Sequence[Document],
    events: Sequence[EventSignal],
    clusters: Sequence[EventCluster],
    families: Sequence[FamilyReadiness],
    *,
    generated_at: datetime,
    corpus_id: str | None = None,
    expected_event_types: Sequence[str] = (),
    expected_entities: Sequence[str] = (),
    provider_states: dict[str, str] | None = None,
    providers_enabled: int = 0,
    successful_run_days: Sequence[datetime] = (),
    raw_evidence_bytes: int = 0,
) -> CorpusStatus:
    """Assemble the status report. Absences are listed, never implied."""
    provider_coverage = _counts(document.provider for document in documents)
    event_type_coverage = _counts(event.event_type.value for event in events)
    entity_coverage = _counts(event.entity or "(none)" for event in events)
    whale_coverage = {context.value: 0 for context in TransferContext}
    for event in events:
        if event.transfer_context is not None:
            whale_coverage[event.transfer_context.value] += 1

    rows = daily_coverage(
        documents,
        events,
        clusters,
        providers_enabled=providers_enabled,
        successful_run_days=successful_run_days,
    )
    span_start = min((document.available_at for document in documents), default=None)
    span_end = max((document.available_at for document in documents), default=None)

    return CorpusStatus(
        generated_at=generated_at.astimezone(timezone.utc),
        corpus_id=corpus_id,
        span_start=span_start,
        span_end=span_end,
        span_days=(span_end - span_start).days if span_start and span_end else 0,
        documents=len(documents),
        events=len(events),
        clusters=len(clusters),
        primary_source_documents=sum(
            1 for document in documents if document.source_metadata.primary_source
        ),
        provider_coverage=provider_coverage,
        event_type_coverage=event_type_coverage,
        entity_coverage=entity_coverage,
        whale_context_coverage=whale_coverage,
        missing_event_types=tuple(
            sorted(name for name in expected_event_types if not event_type_coverage.get(name))
        ),
        missing_entities=tuple(sorted(name for name in expected_entities if not entity_coverage.get(name))),
        missing_whale_contexts=tuple(
            sorted(name for name, count in whale_coverage.items() if count == 0)
        ),
        provider_states=dict(provider_states or {}),
        daily_coverage=tuple(rows),
        collection_gap_days=sum(1 for row in rows if row.collection_gap),
        readiness=overall_readiness(families),
        families=tuple(families),
        storage=measure_storage(documents, events, raw_evidence_bytes),
    )


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def load_status(path: str | Path) -> CorpusStatus:
    return CorpusStatus.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "CorpusStatus",
    "DailyCoverage",
    "StorageGrowth",
    "build_status",
    "daily_coverage",
    "load_status",
    "measure_storage",
]
