"""B5 Gate 1 — measure the point-in-time corpus, read-only, and decide sufficiency.

The audit answers one question: is there a corpus an event study could
defensibly use? It reads a store and never writes to it. The file is opened
read-only, and a store written under a different schema version is refused
rather than migrated, because migrating would be a write to evidence.

It counts at the level that matters. Documents are not events, and events are
not independent observations. The funnel runs from what was collected, through
what corrections leave eligible, what has resolvable sources, what has possible
timestamps, what the extractor was confident of and what can be aligned to a
publication time, down to clusters -- one per real-world occurrence -- and to
effective events that do not overlap at the longest horizon. Every event stays in
a catalog with the reasons it was or was not counted, so "why was this included"
and "what was known at the time" always have an answer.

Nothing here re-derives what the collector already defines. Eligibility is the
store's correction view, clusters are B4.1.14's, family adequacy is the readiness
gate's `assess_family` and coverage its `coverage_fraction`. B5 adds what those
do not measure -- timestamp source and precision, impossible timestamps, the
quality funnel -- and the decision.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
from pydantic import BaseModel, ConfigDict

from ..collection.clustering import EventCluster, cluster_events, effective_non_overlapping
from ..collection.corpus import membership_hash
from ..collection.readiness import FamilyReadiness, Readiness, assess_family, coverage_fraction
from ..corrections import CorrectionStatus, EventCorrection, resolve
from ..errors import StorageError
from ..models import DisclosureStream, Document, EventSignal, EventType
from ..operations import RunStatus
from ..storage import queries
from ..storage.schema import SCHEMA_VERSION, schema_version
from .contracts import DEFAULT_SUFFICIENCY, Decision, SufficiencyPolicy, canonical_json, study_families

#: A run that finished in any state but FAILED collected something that day.
COMPLETED_RUN_STATUSES = frozenset({RunStatus.SUCCESS, RunStatus.PARTIAL_SUCCESS, RunStatus.DEGRADED})


class Exclusion(str, Enum):
    """Why an event is not counted. An event can carry several."""

    #: A correction on record excludes it (the store's CORRECTED view).
    INVALIDATED = "INVALIDATED"
    #: A source document is not in the store as of the audit instant.
    UNRESOLVED_SOURCE = "UNRESOLVED_SOURCE"
    #: Its timestamps are impossible; see `PitViolation`.
    PIT_VIOLATION = "PIT_VIOLATION"
    #: Below the preregistered relevance or confidence floor.
    BELOW_QUALITY = "BELOW_QUALITY"
    #: Its event time is not any source's publication time.
    NO_PUBLICATION_TIME = "NO_PUBLICATION_TIME"


class PitViolation(str, Enum):
    #: Known to this system before it happened.
    EVENT_AFTER_AVAILABILITY = "EVENT_AFTER_AVAILABILITY"
    #: Available before the earliest of the documents it was extracted from.
    AVAILABLE_BEFORE_SOURCE = "AVAILABLE_BEFORE_SOURCE"
    #: A source claims publication after its own retrieval.
    SOURCE_PUBLISHED_AFTER_RETRIEVAL = "SOURCE_PUBLISHED_AFTER_RETRIEVAL"


class TimeSource(str, Enum):
    PUBLICATION = "PUBLICATION"
    RETRIEVAL = "RETRIEVAL"
    OTHER = "OTHER"


class TimePrecision(str, Enum):
    INTRADAY = "INTRADAY"
    DATE_ONLY = "DATE_ONLY"


class EventRecord(BaseModel):
    """One event, with everything needed to say why it was or was not counted."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    family: str
    event_type: str
    category: str
    entity: str | None
    source_ids: tuple[str, ...]
    publishers: tuple[str, ...]
    providers: tuple[str, ...]
    event_time: datetime
    available_time: datetime
    earliest_source_available_at: datetime | None
    source_published_at: tuple[datetime, ...]
    time_source: TimeSource
    time_precision: TimePrecision
    retrieval_lag_hours: float
    btc_relevance: float
    confidence: float
    novelty: float
    sentiment: float
    extraction_method: str
    extractor_version: str
    correction_reason: str | None
    pit_violations: tuple[PitViolation, ...]
    exclusions: tuple[Exclusion, ...]
    cluster_id: str | None

    @property
    def counted(self) -> bool:
        return not self.exclusions


class CorpusAudit(BaseModel):
    """Gate 1's measurements. Descriptive; the decision is `Gate1Result`."""

    model_config = ConfigDict(frozen=True)

    as_of: datetime
    schema_version: int
    content_fingerprint: str
    # documents
    documents: int
    unique_contents: int
    retrieved_copies: int
    duplicate_rate: float
    quarantined: int
    quarantine_rate: float
    publishers: dict[str, int]
    providers: dict[str, int]
    disclosure_streams: dict[str, int]
    timestamp_completeness: float
    document_span: tuple[datetime, datetime] | None
    # events
    raw_events: int
    invalidated_events: int
    invalidation_rate: float
    correction_reasons: dict[str, int]
    event_types: dict[str, int]
    categories: dict[str, int]
    entities: dict[str, int]
    events_by_month: dict[str, int]
    events_by_publisher: dict[str, int]
    event_time_span: tuple[datetime, datetime] | None
    available_time_span: tuple[datetime, datetime] | None
    median_events_per_day: float
    time_sources: dict[str, int]
    time_precision: dict[str, int]
    pit_valid_fraction: float | None
    pit_violations: dict[str, int]
    missing_fields: dict[str, float]
    retrieval_lag_hours: dict[str, float | None]
    clusters_after_corrections: int
    funnel: dict[str, int]
    # collection
    collection_runs: int
    run_statuses: dict[str, int]
    collection_days: int
    first_collection: datetime | None
    last_collection: datetime | None
    days_since_last_collection: int | None
    # adequacy, per family, including the empty ones
    families: tuple[FamilyReadiness, ...]


class Clause(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    required: str
    observed: str
    met: bool


class Gate1Result(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    #: Set only when Gate 1 ends the track. A passing gate leads to Gate 2, not
    #: to a decision.
    decision: Decision | None
    clauses: tuple[Clause, ...]
    ready_families: tuple[str, ...]


class AuditResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    audit: CorpusAudit
    catalog: tuple[EventRecord, ...]
    gate: Gate1Result


# -- opening a store without touching it ---------------------------------------


def open_read_only(path: str | Path) -> duckdb.DuckDBPyConnection:
    """A read-only connection, or a refusal. Never a migration."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no intelligence store at {target}")
    connection = duckdb.connect(str(target), read_only=True)
    try:
        version = schema_version(connection)
    except (StorageError, duckdb.Error) as exc:
        connection.close()
        raise StorageError(f"{target} is not a readable intelligence store: {exc}") from exc
    if version != SCHEMA_VERSION:
        connection.close()
        raise StorageError(
            f"{target} was written under schema {version}; this audit reads schema {SCHEMA_VERSION} and refuses to "
            "migrate a store it is auditing"
        )
    return connection


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- per-event judgement -------------------------------------------------------


def family_of(event: EventSignal) -> str:
    if event.event_type is EventType.WHALE_TRANSFER and event.transfer_context is not None:
        return f"whale:{event.transfer_context.value}"
    return f"event_type:{event.event_type.value}"


def _precision(moment: datetime) -> TimePrecision:
    if (moment.hour, moment.minute, moment.second, moment.microsecond) == (0, 0, 0, 0):
        return TimePrecision.DATE_ONLY
    return TimePrecision.INTRADAY


def judge_event(
    event: EventSignal,
    sources: Sequence[Document | None],
    correction: EventCorrection | None,
    policy: SufficiencyPolicy,
) -> EventRecord:
    """Everything Gate 1 needs to know about one event, and why it counts or not."""
    resolved = [document for document in sources if document is not None]
    earliest = min((document.available_at for document in resolved), default=None)
    published = tuple(sorted(document.published_at for document in resolved if document.published_at is not None))

    violations: list[PitViolation] = []
    if event.event_time > event.available_time:
        violations.append(PitViolation.EVENT_AFTER_AVAILABILITY)
    if earliest is not None and event.available_time < earliest:
        violations.append(PitViolation.AVAILABLE_BEFORE_SOURCE)
    if any(document.published_at is not None and document.published_at > document.available_at for document in resolved):
        violations.append(PitViolation.SOURCE_PUBLISHED_AFTER_RETRIEVAL)

    if event.event_time in published:
        time_source = TimeSource.PUBLICATION
    elif event.event_time == event.available_time or any(event.event_time == d.retrieved_at for d in resolved):
        time_source = TimeSource.RETRIEVAL
    else:
        time_source = TimeSource.OTHER

    invalidated = correction is not None and correction.status is CorrectionStatus.INVALIDATED
    exclusions: list[Exclusion] = []
    if invalidated:
        exclusions.append(Exclusion.INVALIDATED)
    if len(resolved) < len(sources) or not sources:
        exclusions.append(Exclusion.UNRESOLVED_SOURCE)
    if violations:
        exclusions.append(Exclusion.PIT_VIOLATION)
    if event.btc_relevance < policy.minimum_relevance or event.confidence < policy.minimum_confidence:
        exclusions.append(Exclusion.BELOW_QUALITY)
    if time_source is not TimeSource.PUBLICATION:
        exclusions.append(Exclusion.NO_PUBLICATION_TIME)

    return EventRecord(
        event_id=event.event_id,
        family=family_of(event),
        event_type=event.event_type.value,
        category=event.category.value,
        entity=event.entity,
        source_ids=tuple(event.source_ids),
        publishers=tuple(sorted({document.publisher for document in resolved})),
        providers=tuple(sorted({document.provider for document in resolved})),
        event_time=event.event_time,
        available_time=event.available_time,
        earliest_source_available_at=earliest,
        source_published_at=published,
        time_source=time_source,
        time_precision=_precision(event.event_time),
        retrieval_lag_hours=round((event.available_time - event.event_time).total_seconds() / 3600.0, 6),
        btc_relevance=event.btc_relevance,
        confidence=event.confidence,
        novelty=event.novelty,
        sentiment=event.sentiment,
        extraction_method=event.extraction_method.value,
        extractor_version=event.extractor_version,
        correction_reason=correction.reason if invalidated and correction is not None else None,
        pit_violations=tuple(violations),
        exclusions=tuple(exclusions),
        cluster_id=None,
    )


# -- the audit -----------------------------------------------------------------


def _utc(moment: datetime) -> datetime:
    return moment.astimezone(timezone.utc) if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _span(moments: Sequence[datetime]) -> tuple[datetime, datetime] | None:
    return (min(moments), max(moments)) if moments else None


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


def _median_per_day(moments: Sequence[datetime]) -> float:
    if not moments:
        return 0.0
    days = Counter(moment.date() for moment in moments)
    first, last = min(days), max(days)
    span = (last - first).days + 1
    return float(statistics.median(days.get(first + timedelta(days=offset), 0) for offset in range(span)))


def _family_coverage(clusters: Sequence[EventCluster], run_days: Sequence[datetime]) -> float:
    """B4.1.23, per family: days with a completed run over days in the family's span."""
    if not clusters:
        return 0.0
    starts = sorted(cluster.first_available_at for cluster in clusters)
    return coverage_fraction(run_days, starts[0], starts[-1])


def content_fingerprint(documents: Sequence[Document], events: Sequence[EventSignal], corrections: Sequence[EventCorrection], as_of: datetime) -> str:
    """What was audited, independent of file bytes and read order."""
    material = {
        "as_of": _utc(as_of).isoformat(),
        "membership": membership_hash([d.document_id for d in documents], [e.event_id for e in events]),
        "corrections": sorted(correction.correction_id for correction in corrections),
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def audit_corpus(
    connection: duckdb.DuckDBPyConnection,
    *,
    as_of: datetime,
    policy: SufficiencyPolicy = DEFAULT_SUFFICIENCY,
) -> AuditResult:
    """Measure the corpus as it stood at `as_of`, and decide Gate 1."""
    as_of = _utc(as_of)
    documents = queries.documents_as_of(connection, as_of)
    raw_events = queries.signals_as_of(connection, as_of)
    corrections = queries.corrections(connection)
    resolution = resolve(corrections)
    by_id = {document.document_id: document for document in documents}

    records = [
        judge_event(event, [by_id.get(source) for source in event.source_ids], resolution.get(event.event_id), policy)
        for event in raw_events
    ]
    judged = dict(zip((event.event_id for event in raw_events), records, strict=True))

    # Clusters over what corrections leave, for the descriptive count, and over
    # what Gate 1 counts, for the decision. B4.1.14's rule, unchanged.
    live = [event for event in raw_events if Exclusion.INVALIDATED not in judged[event.event_id].exclusions]
    counted = [event for event in raw_events if judged[event.event_id].counted]
    live_clusters = cluster_events(live, documents)
    study_clusters = cluster_events(counted, documents)
    cluster_of = {event_id: cluster.cluster_id for cluster in study_clusters for event_id in cluster.event_ids}
    catalog = tuple(
        judged[event.event_id].model_copy(update={"cluster_id": cluster_of.get(event.event_id)}) for event in raw_events
    )

    runs = [(_utc(started), manifest) for started, manifest in queries.runs_up_to(connection, as_of)]
    run_days = [started for started, manifest in runs if manifest.status in COMPLETED_RUN_STATUSES]

    family_of_cluster = {cluster.cluster_id: judged[cluster.event_ids[0]].family for cluster in study_clusters}
    families = tuple(
        assess_family(
            name,
            members,
            policy=policy.adequacy,
            coverage_fraction=_family_coverage(members, run_days),
        )
        for name in study_families()
        for members in [[c for c in study_clusters if family_of_cluster[c.cluster_id] == name]]
    )

    sightings = dict(
        connection.execute("SELECT document_id, sighting_count FROM document_sightings").fetchall()
    )
    copies = sum(int(sightings.get(document.document_id, 1)) for document in documents)
    quarantine_row = connection.execute(
        "SELECT count(*) FROM quarantine WHERE retrieval_timestamp <= ?", [as_of]
    ).fetchone()
    quarantined = int(quarantine_row[0]) if quarantine_row else 0

    resolved_live = [record for record in catalog if Exclusion.INVALIDATED not in record.exclusions and Exclusion.UNRESOLVED_SOURCE not in record.exclusions]
    pit_valid = [record for record in resolved_live if not record.pit_violations]
    lags = sorted(record.retrieval_lag_hours for record in resolved_live)
    live_records = [record for record in catalog if Exclusion.INVALIDATED not in record.exclusions]
    last_collection = max(run_days, default=None)

    audit = CorpusAudit(
        as_of=as_of,
        schema_version=schema_version(connection),
        content_fingerprint=content_fingerprint(documents, raw_events, corrections, as_of),
        documents=len(documents),
        unique_contents=len({document.text_hash for document in documents}),
        retrieved_copies=copies,
        duplicate_rate=_rate(copies - len(documents), copies),
        quarantined=quarantined,
        quarantine_rate=_rate(quarantined, quarantined + len(documents)),
        publishers=_counts(document.publisher for document in documents),
        providers=_counts(document.provider for document in documents),
        disclosure_streams=_counts(document.source_metadata.disclosure_stream.value for document in documents),
        timestamp_completeness=_rate(sum(1 for d in documents if d.published_at is not None), len(documents)),
        document_span=_span([document.available_at for document in documents]),
        raw_events=len(raw_events),
        invalidated_events=len(raw_events) - len(live),
        invalidation_rate=_rate(len(raw_events) - len(live), len(raw_events)),
        correction_reasons=_counts(record.correction_reason for record in catalog if record.correction_reason),
        event_types=_counts(record.event_type for record in live_records),
        categories=_counts(record.category for record in live_records),
        entities=_counts(record.entity or "(none)" for record in live_records),
        events_by_month=_counts(record.available_time.strftime("%Y-%m") for record in live_records),
        events_by_publisher=_counts(publisher for record in live_records for publisher in record.publishers),
        event_time_span=_span([record.event_time for record in live_records]),
        available_time_span=_span([record.available_time for record in live_records]),
        median_events_per_day=_median_per_day([record.available_time for record in live_records]),
        time_sources=_counts(record.time_source.value for record in pit_valid),
        time_precision=_counts(record.time_precision.value for record in pit_valid),
        pit_valid_fraction=_rate(len(pit_valid), len(resolved_live)) if resolved_live else None,
        pit_violations=_counts(violation.value for record in resolved_live for violation in record.pit_violations),
        missing_fields={
            "document_published_at": _rate(sum(1 for d in documents if d.published_at is None), len(documents)),
            "document_disclosure_stream": _rate(
                sum(1 for d in documents if d.source_metadata.disclosure_stream is DisclosureStream.UNCLASSIFIED),
                len(documents),
            ),
            "event_entity": _rate(sum(1 for r in live_records if r.entity is None), len(live_records)),
            "event_source": _rate(
                sum(1 for r in live_records if Exclusion.UNRESOLVED_SOURCE in r.exclusions), len(live_records)
            ),
        },
        retrieval_lag_hours={
            "min": lags[0] if lags else None,
            "median": float(statistics.median(lags)) if lags else None,
            "max": lags[-1] if lags else None,
        },
        clusters_after_corrections=len(live_clusters),
        funnel=_funnel(catalog, study_clusters, policy),
        collection_runs=len(runs),
        run_statuses=_counts(manifest.status.value for _, manifest in runs),
        collection_days=len({started.date() for started in run_days}),
        first_collection=min(run_days, default=None),
        last_collection=last_collection,
        days_since_last_collection=(as_of.date() - last_collection.date()).days if last_collection else None,
        families=families,
    )
    return AuditResult(audit=audit, catalog=catalog, gate=decide(audit, policy))


def _funnel(catalog: Sequence[EventRecord], clusters: Sequence[EventCluster], policy: SufficiencyPolicy) -> dict[str, int]:
    """Each stage keeps what the one before kept and passes its own test."""
    stages: list[tuple[str, Exclusion | None]] = [
        ("1_collected", None),
        ("2_not_invalidated", Exclusion.INVALIDATED),
        ("3_sources_resolved", Exclusion.UNRESOLVED_SOURCE),
        ("4_point_in_time_valid", Exclusion.PIT_VIOLATION),
        ("5_quality", Exclusion.BELOW_QUALITY),
        ("6_publication_time", Exclusion.NO_PUBLICATION_TIME),
    ]
    funnel: dict[str, int] = {}
    removed: set[Exclusion] = set()
    for name, exclusion in stages:
        if exclusion is not None:
            removed.add(exclusion)
        funnel[name] = sum(1 for record in catalog if not removed.intersection(record.exclusions))
    funnel["7_independent_events"] = len(clusters)
    funnel["8_effective_events"] = effective_non_overlapping(clusters, policy.adequacy.horizon_hours)
    return funnel


def decide(audit: CorpusAudit, policy: SufficiencyPolicy = DEFAULT_SUFFICIENCY) -> Gate1Result:
    """The preregistered decision. Every clause is reported, met or not."""
    ready = tuple(family.family for family in audit.families if family.readiness is Readiness.READY_FOR_VALIDATION)
    resolved = audit.funnel["3_sources_resolved"]
    valid = audit.funnel["4_point_in_time_valid"]
    violation_share = (resolved - valid) / resolved if resolved else None
    # time_sources is counted over point-in-time-valid events, so this is the
    # share the clause names.
    aligned = audit.time_sources.get(TimeSource.PUBLICATION.value, 0)
    aligned_share = aligned / valid if valid else None

    clauses = (
        Clause(
            name="ready_families",
            required=(
                f">= {policy.minimum_ready_families} family meeting every adequacy clause alone "
                f"({policy.adequacy.minimum_events} events, {policy.adequacy.minimum_effective_events} effective at "
                f"{policy.adequacy.horizon_hours}h, {policy.adequacy.minimum_publishers} publishers, "
                f"{policy.adequacy.minimum_providers} provider, {policy.adequacy.minimum_span_days} days, "
                f"{policy.adequacy.minimum_coverage_fraction:.0%} coverage)"
            ),
            observed=f"{len(ready)} of {len(audit.families)} families ready",
            met=len(ready) >= policy.minimum_ready_families,
        ),
        Clause(
            name="point_in_time_integrity",
            required=f"<= {policy.maximum_pit_violation_fraction:.0%} of resolved, uncorrected events with impossible timestamps",
            observed="no events to judge" if violation_share is None else f"{violation_share:.1%} ({resolved - valid} of {resolved})",
            met=violation_share is not None and violation_share <= policy.maximum_pit_violation_fraction,
        ),
        Clause(
            name="event_time_from_publication",
            required=f">= {policy.minimum_event_time_coverage:.0%} of point-in-time-valid events aligned to a publication time",
            observed="no events to judge" if aligned_share is None else f"{aligned_share:.1%} ({aligned} of {valid})",
            met=aligned_share is not None and aligned_share >= policy.minimum_event_time_coverage,
        ),
    )
    passed = all(clause.met for clause in clauses)
    return Gate1Result(
        passed=passed,
        decision=None if passed else Decision.INTELLIGENCE_CORPUS_INSUFFICIENT,
        clauses=clauses,
        ready_families=ready,
    )


__all__ = [
    "COMPLETED_RUN_STATUSES",
    "AuditResult",
    "Clause",
    "CorpusAudit",
    "EventRecord",
    "Exclusion",
    "Gate1Result",
    "PitViolation",
    "TimePrecision",
    "TimeSource",
    "audit_corpus",
    "content_fingerprint",
    "decide",
    "family_of",
    "file_sha256",
    "judge_event",
    "open_read_only",
]
