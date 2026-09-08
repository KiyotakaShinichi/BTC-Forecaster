"""Every read of the corpus, as functions over a connection.

The store's two halves guarantee different things, and keeping them in one file
meant neither invariant was stated anywhere.

**Writes** are about identity: first write wins, a rediscovered document keeps
the availability it was first seen with, and a correction is a new row rather
than an edit. **Reads** -- this module -- are about determinism: the same
question asked of the same corpus returns the same answer, in the same order, on
every machine.

That second property is load-bearing rather than tidy. `sum()` is not
associative, so records sharing an instant coming back in engine order moved
every float derived from them: a feature matrix built at one chunk size
disagreed with the same matrix at another in the last ULP, which made
`dataset_id` -- whose whole job is to say two datasets are identical -- depend
on an implementation detail, and let a corpus restored from a backup produce a
different id than the corpus it came from. Hence `ORDER BY <timestamp>, <id>`
throughout: availability first, id as the tiebreaker, never availability alone.

Nothing here writes. Every function takes a connection and returns a value, so
a read can be exercised against a bare in-memory database without constructing
a store, and `IntelligenceStore` delegates to these rather than reimplementing
them.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Sequence, cast

import duckdb

from ..corrections import EligibilityMode, EventCorrection, ineligible_ids
from ..errors import StorageError
from ..models import Document, EventSignal
from ..operations import (
    DocumentSighting,
    IntelligenceSnapshot,
    ProviderHealth,
    QualityScoreboard,
    QuarantineRecord,
    RunManifest,
    Watermark,
)
from .schema import SNAPSHOT_INSERT_CHUNK


def documents_as_of(connection: duckdb.DuckDBPyConnection, forecast_origin: datetime) -> list[Document]:
    """Evidence available at an origin, in a total order.

    The id is a tiebreaker, not decoration. Ordering by timestamp alone
    leaves records that share an instant in whatever order the engine
    happens to return, and every float sum downstream is order-dependent:
    `sum()` is not associative. The result was a feature matrix whose values
    changed with the chunk size used to build it -- so `dataset_id`, which
    exists to say two datasets are the same, depended on an implementation
    detail. The paginated readers already sorted this way; these two, which
    the whole replay and eligibility path goes through, did not.
    """
    rows = connection.execute(
        "SELECT payload FROM documents WHERE available_at <= ? "
        "ORDER BY available_at, document_id",
        [forecast_origin],
    ).fetchall()
    return [Document.model_validate(json.loads(row[0])) for row in rows]


def signals_as_of(connection: duckdb.DuckDBPyConnection, forecast_origin: datetime) -> list[EventSignal]:
    rows = connection.execute(
        "SELECT payload FROM signals WHERE available_time <= ? "
        "ORDER BY available_time, event_id",
        [forecast_origin],
    ).fetchall()
    return [EventSignal.model_validate(json.loads(row[0])) for row in rows]


def query_documents(
    connection: duckdb.DuckDBPyConnection,
    *,
    provider: str | None = None,
    publisher: str | None = None,
    query: str | None = None,
    published_from: datetime | None = None,
    published_to: datetime | None = None,
    available_from: datetime | None = None,
    available_to: datetime | None = None,
    forecast_origin: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Document]:
    clauses: list[str] = []
    values: list[object] = []
    json_filters = (("provider", provider), ("publisher", publisher), ("query", query))
    for field, filter_value in json_filters:
        if filter_value is not None:
            clauses.append(f"json_extract_string(payload, '$.{field}') = ?")
            values.append(filter_value)
    for expression, boundary_value in (
        ("CAST(json_extract_string(payload, '$.published_at') AS TIMESTAMPTZ) >= ?", published_from),
        ("CAST(json_extract_string(payload, '$.published_at') AS TIMESTAMPTZ) <= ?", published_to),
        ("available_at >= ?", available_from),
        ("available_at <= ?", available_to),
        ("available_at <= ?", forecast_origin),
    ):
        if boundary_value is not None:
            clauses.append(expression)
            values.append(boundary_value)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = connection.execute(
        f"SELECT payload FROM documents{where} ORDER BY available_at, document_id LIMIT ? OFFSET ?",
        [*values, limit, offset],
    ).fetchall()
    return [Document.model_validate(json.loads(row[0])) for row in rows]


def get_document(connection: duckdb.DuckDBPyConnection, document_id: str) -> Document | None:
    row = connection.execute("SELECT payload FROM documents WHERE document_id=?", [document_id]).fetchone()
    return Document.model_validate(json.loads(row[0])) if row else None


def query_events(
    connection: duckdb.DuckDBPyConnection,
    *,
    entity: str | None = None,
    event_type: str | None = None,
    direction: str | None = None,
    minimum_relevance: float | None = None,
    minimum_confidence: float | None = None,
    available_from: datetime | None = None,
    available_to: datetime | None = None,
    forecast_origin: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[EventSignal]:
    clauses: list[str] = []
    values: list[object] = []
    for field, filter_value in (("entity", entity), ("event_type", event_type), ("direction", direction)):
        if filter_value is not None:
            clauses.append(f"json_extract_string(payload, '$.{field}') = ?")
            values.append(filter_value)
    for expression, boundary_value in (
        ("CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE) >= ?", minimum_relevance),
        ("CAST(json_extract_string(payload, '$.confidence') AS DOUBLE) >= ?", minimum_confidence),
        ("available_time >= ?", available_from),
        ("available_time <= ?", available_to),
        ("available_time <= ?", forecast_origin),
    ):
        if boundary_value is not None:
            clauses.append(expression)
            values.append(boundary_value)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = connection.execute(
        f"SELECT payload FROM signals{where} ORDER BY available_time, event_id LIMIT ? OFFSET ?",
        [*values, limit, offset],
    ).fetchall()
    return [EventSignal.model_validate(json.loads(row[0])) for row in rows]


def get_event(connection: duckdb.DuckDBPyConnection, event_id: str) -> EventSignal | None:
    row = connection.execute("SELECT payload FROM signals WHERE event_id=?", [event_id]).fetchone()
    return EventSignal.model_validate(json.loads(row[0])) if row else None


def event_sources(connection: duckdb.DuckDBPyConnection, event_id: str) -> list[Document]:
    event = get_event(connection, event_id)
    if event is None:
        return []
    return [document for source_id in event.source_ids if (document := get_document(connection, source_id)) is not None]


def query_runs(
    connection: duckdb.DuckDBPyConnection,
    *,
    status: str | None = None,
    provider: str | None = None,
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[RunManifest]:
    clauses: list[str] = []
    values: list[object] = []
    if status is not None:
        clauses.append("status = ?")
        values.append(status)
    if provider is not None:
        clauses.append("list_contains(provider_ids, ?)")
        values.append(provider)
    if started_from is not None:
        clauses.append("started_at >= ?")
        values.append(started_from)
    if started_to is not None:
        clauses.append("started_at <= ?")
        values.append(started_to)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = connection.execute(
        f"SELECT payload FROM runs{where} ORDER BY started_at DESC, run_id LIMIT ? OFFSET ?",
        [*values, limit, offset],
    ).fetchall()
    return [RunManifest.model_validate(json.loads(row[0])) for row in rows]


def get_run(connection: duckdb.DuckDBPyConnection, run_id: str) -> RunManifest | None:
    row = connection.execute("SELECT payload FROM runs WHERE run_id=?", [run_id]).fetchone()
    return RunManifest.model_validate(json.loads(row[0])) if row else None


def latest_run_as_of(connection: duckdb.DuckDBPyConnection, origin: datetime) -> RunManifest | None:
    row = connection.execute(
        "SELECT payload FROM runs WHERE finished_at <= ? ORDER BY finished_at DESC LIMIT 1", [origin]
    ).fetchone()
    return RunManifest.model_validate(json.loads(row[0])) if row else None


def runs_up_to(connection: duckdb.DuckDBPyConnection, origin: datetime) -> list[tuple[datetime, RunManifest]]:
    """All runs finished at or before ``origin``, ascending.

    Batched counterpart to :meth:`latest_run_as_of`. A DuckDB statement costs
    ~4 ms of parse and plan time even against an empty table, so asking once
    per origin costs ~4 s per 1,000 origins for information that one ordered
    read plus a binary search already contains.
    """
    rows = connection.execute(
        "SELECT finished_at, payload FROM runs WHERE finished_at <= ? ORDER BY finished_at", [origin]
    ).fetchall()
    return [(row[0], RunManifest.model_validate(json.loads(row[1]))) for row in rows]


def list_snapshots(connection: duckdb.DuckDBPyConnection, limit: int = 100, offset: int = 0) -> list[IntelligenceSnapshot]:
    rows = connection.execute(
        "SELECT payload FROM snapshots ORDER BY forecast_origin DESC LIMIT ? OFFSET ?", [limit, offset]
    ).fetchall()
    return [IntelligenceSnapshot.model_validate(json.loads(row[0])) for row in rows]


def get_snapshot(connection: duckdb.DuckDBPyConnection, snapshot_id: str) -> IntelligenceSnapshot | None:
    row = connection.execute("SELECT payload FROM snapshots WHERE snapshot_id=?", [snapshot_id]).fetchone()
    return IntelligenceSnapshot.model_validate(json.loads(row[0])) if row else None


def extractor_versions_for(connection: duckdb.DuckDBPyConnection, snapshot_ids: Sequence[str]) -> tuple[str, ...]:
    """Union of extractor versions across many snapshots, in batches.

    The obvious spelling -- ``get_snapshot`` in a loop -- is one round trip
    *and* one full payload parse per origin, which is how a long historical
    build quietly reacquires the per-origin query pattern bulk replay was
    built to remove. Extracting just the versions array server-side keeps the
    parsed JSON small, and batching keeps the statement count sub-proportional.

    Missing ids are skipped rather than raising: callers that need existence
    guarantees have already checked, and a dataset should not fail to record
    its provenance because one snapshot was pruned.
    """
    versions: set[str] = set()
    unique = list(dict.fromkeys(snapshot_ids))
    for start in range(0, len(unique), SNAPSHOT_INSERT_CHUNK):
        batch = unique[start : start + SNAPSHOT_INSERT_CHUNK]
        placeholders = ", ".join(["?"] * len(batch))
        rows = connection.execute(
            f"SELECT json_extract(payload, '$.extractor_versions') FROM snapshots "
            f"WHERE snapshot_id IN ({placeholders})",
            list(batch),
        ).fetchall()
        for row in rows:
            versions.update(json.loads(row[0]))
    return tuple(sorted(versions))


def latest_quality_scoreboard(connection: duckdb.DuckDBPyConnection) -> QualityScoreboard | None:
    row = connection.execute(
        "SELECT payload FROM quality_scoreboards ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return QualityScoreboard.model_validate(json.loads(row[0])) if row else None


def all_health(connection: duckdb.DuckDBPyConnection) -> list[ProviderHealth]:
    rows = connection.execute("SELECT payload FROM provider_health ORDER BY provider_id").fetchall()
    return [ProviderHealth.model_validate(json.loads(row[0])) for row in rows]


def all_watermarks(connection: duckdb.DuckDBPyConnection) -> list[Watermark]:
    rows = connection.execute("SELECT payload FROM watermarks ORDER BY provider_id, query_id").fetchall()
    return [Watermark.model_validate(json.loads(row[0])) for row in rows]


def get_watermark(connection: duckdb.DuckDBPyConnection, provider_id: str, query_id: str) -> Watermark | None:
    row = connection.execute(
        "SELECT payload FROM watermarks WHERE provider_id=? AND query_id=?", [provider_id, query_id]
    ).fetchone()
    return Watermark.model_validate(json.loads(row[0])) if row else None


def sighting(connection: duckdb.DuckDBPyConnection, document_id: str) -> DocumentSighting | None:
    row = connection.execute(
        "SELECT document_id, first_seen_at, latest_seen_at, sighting_count, providers "
        "FROM document_sightings WHERE document_id = ?",
        [document_id],
    ).fetchone()
    if row is None:
        return None
    return DocumentSighting(
        document_id=row[0],
        first_seen_at=row[1],
        latest_seen_at=row[2],
        sighting_count=int(row[3]),
        providers=tuple(str(row[4]).split(",")) if row[4] else (),
    )


def quarantine_count(connection: duckdb.DuckDBPyConnection, since: datetime | None = None) -> int:
    """How many records are set aside, optionally only recent ones.

    The watchdog asks about a window: a spike matters, a slowly accumulating
    historical total does not and would latch the alert on forever.
    """
    if since is None:
        row = connection.execute("SELECT COUNT(*) FROM quarantine").fetchone()
    else:
        row = connection.execute(
            "SELECT COUNT(*) FROM quarantine WHERE retrieval_timestamp >= ?", [since]
        ).fetchone()
    return cast(int, row[0]) if row else 0


def query_quarantine(
    connection: duckdb.DuckDBPyConnection,
    *,
    provider: str | None = None,
    record_type: str | None = None,
    failure_category: str | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[QuarantineRecord]:
    clauses: list[str] = []
    values: list[object] = []
    for field, value in (("provider", provider), ("record_type", record_type)):
        if value is not None:
            clauses.append(f"json_extract_string(payload, '$.{field}') = ?")
            values.append(value)
    if failure_category is not None:
        clauses.append("json_extract_string(payload, '$.failure_reason') ILIKE ?")
        values.append(f"%{failure_category}%")
    if time_from is not None:
        clauses.append("retrieval_timestamp >= ?")
        values.append(time_from)
    if time_to is not None:
        clauses.append("retrieval_timestamp <= ?")
        values.append(time_to)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = connection.execute(
        f"SELECT payload FROM quarantine{where} ORDER BY retrieval_timestamp DESC LIMIT ? OFFSET ?",
        [*values, limit, offset],
    ).fetchall()
    return [QuarantineRecord.model_validate(json.loads(row[0])) for row in rows]


def corrections(connection: duckdb.DuckDBPyConnection, event_ids: Sequence[str] | None = None) -> list[EventCorrection]:
    """Every correction on record, newest last. Never filtered by status."""
    if event_ids is None:
        rows = connection.execute(
            "SELECT payload FROM event_corrections ORDER BY invalidated_at, correction_id"
        ).fetchall()
    elif not event_ids:
        return []
    else:
        placeholders = ",".join("?" * len(event_ids))
        rows = connection.execute(
            f"SELECT payload FROM event_corrections WHERE event_id IN ({placeholders}) "
            "ORDER BY invalidated_at, correction_id",
            list(event_ids),
        ).fetchall()
    return [EventCorrection.model_validate(json.loads(row[0])) for row in rows]


def correction_count(connection: duckdb.DuckDBPyConnection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM event_corrections").fetchone()
    return cast(int, row[0]) if row else 0


def ineligible_event_ids(
    connection: duckdb.DuckDBPyConnection,
    *,
    as_of: datetime | None = None,
    mode: EligibilityMode = EligibilityMode.CORRECTED,
) -> frozenset[str]:
    return ineligible_ids(corrections(connection), as_of=as_of, mode=mode)


def eligible_signals_as_of(
    connection: duckdb.DuckDBPyConnection,
    forecast_origin: datetime,
    *,
    mode: EligibilityMode = EligibilityMode.CORRECTED,
) -> list[EventSignal]:
    """The research view of the corpus at an origin.

    `signals_as_of` stays raw on purpose -- audit, provenance and the
    integrity checks all need to see an invalidated observation exactly
    where it has always been. Everything that *counts* events reads this
    instead, so an artefact of a defect is present in the record and absent
    from the result.
    """
    events = signals_as_of(connection, forecast_origin)
    excluded = ineligible_event_ids(connection, as_of=forecast_origin, mode=mode)
    return [event for event in events if event.event_id not in excluded]


def provider_observability(connection: duckdb.DuckDBPyConnection) -> list[dict[str, object]]:
    rows = connection.execute("""
      SELECT provider_id, COUNT(*) AS attempts, AVG(CASE WHEN success THEN 1 ELSE 0 END) AS success_rate,
        AVG(CASE WHEN success THEN 0 ELSE 1 END) AS failure_rate, AVG(latency_ms) AS mean_latency_ms,
        MEDIAN(latency_ms) AS median_latency_ms,
        AVG(CASE WHEN success THEN documents_received ELSE NULL END) AS documents_per_success,
        MAX(CASE WHEN success THEN observed_at ELSE NULL END) AS last_success,
        MAX(CASE WHEN NOT success THEN observed_at ELSE NULL END) AS last_failure,
        SUM(CASE WHEN rate_limited THEN 1 ELSE 0 END) AS rate_limit_events,
        EXTRACT(EPOCH FROM (current_timestamp - MAX(CASE WHEN success THEN observed_at ELSE NULL END))) AS staleness_seconds
      FROM provider_attempts GROUP BY provider_id ORDER BY provider_id
    """).fetchall()
    columns = [item[0] for item in connection.description]
    return [dict(zip(columns, row, strict=False)) for row in rows]


def extraction_observability(connection: duckdb.DuckDBPyConnection) -> dict[str, object]:
    method_rows = connection.execute("""
      SELECT json_extract_string(payload, '$.extraction_method') AS extraction_method,
        COUNT(*) AS extraction_count
      FROM signals GROUP BY extraction_method ORDER BY extraction_method
    """).fetchall()
    totals = connection.execute("""
      SELECT COUNT(*), AVG(CAST(json_extract_string(payload, '$.confidence') AS DOUBLE)),
        AVG(CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE)),
        MIN(CAST(json_extract_string(payload, '$.confidence') AS DOUBLE)),
        MEDIAN(CAST(json_extract_string(payload, '$.confidence') AS DOUBLE)),
        MAX(CAST(json_extract_string(payload, '$.confidence') AS DOUBLE)),
        MIN(CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE)),
        MEDIAN(CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE)),
        MAX(CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE)) FROM signals
    """).fetchone()
    rejection_rows = connection.execute("""
      SELECT json_extract_string(payload, '$.record_type'), COUNT(*) FROM quarantine GROUP BY 1 ORDER BY 1
    """).fetchall()
    rejection_categories = connection.execute("""
      SELECT
        SUM(CASE WHEN json_extract_string(payload, '$.failure_reason') ILIKE '%json%' THEN 1 ELSE 0 END),
        SUM(CASE WHEN json_extract_string(payload, '$.failure_reason') ILIKE '%source%' THEN 1 ELSE 0 END),
        SUM(CASE WHEN json_extract_string(payload, '$.failure_reason') ILIKE '%schema%'
                      OR json_extract_string(payload, '$.failure_reason') ILIKE '%valid%' THEN 1 ELSE 0 END)
      FROM quarantine
    """).fetchone()
    docs_row = connection.execute("SELECT COUNT(*) FROM documents").fetchone()
    if totals is None or docs_row is None:
        raise StorageError("observability query returned no aggregate row")
    if rejection_categories is None:
        raise StorageError("extraction rejection query returned no aggregate row")
    docs = cast(int, docs_row[0])
    return {
        "accepted_by_method": {row[0]: row[1] for row in method_rows},
        "accepted_extractions": totals[0],
        "rejected_extractions": quarantine_count(connection),
        "average_events_per_document": totals[0] / docs if docs else 0.0,
        "mean_confidence": totals[1] or 0.0,
        "mean_relevance": totals[2] or 0.0,
        "confidence_distribution": {"min": totals[3] or 0.0, "median": totals[4] or 0.0, "max": totals[5] or 0.0},
        "relevance_distribution": {"min": totals[6] or 0.0, "median": totals[7] or 0.0, "max": totals[8] or 0.0},
        "rejections_by_type": {row[0]: row[1] for row in rejection_rows},
        "invalid_json": rejection_categories[0] or 0,
        "fabricated_or_unknown_source": rejection_categories[1] or 0,
        "schema_failures": rejection_categories[2] or 0,
    }


def dashboard_summary(connection: duckdb.DuckDBPyConnection) -> dict[str, object]:
    event_types = connection.execute("""
      SELECT json_extract_string(payload, '$.event_type'), COUNT(*) FROM signals GROUP BY 1 ORDER BY 1
    """).fetchall()
    entities = connection.execute("""
      SELECT json_extract_string(payload, '$.entity'), COUNT(*) FROM signals
      WHERE json_extract_string(payload, '$.entity') IS NOT NULL GROUP BY 1 ORDER BY 1
    """).fetchall()
    quarantine = connection.execute("""
      SELECT json_extract_string(payload, '$.record_type'), COUNT(*) FROM quarantine GROUP BY 1 ORDER BY 1
    """).fetchall()
    return {
        "event_type_distribution": {row[0]: row[1] for row in event_types},
        "tracked_entities": {row[0]: row[1] for row in entities},
        "latest_runs": [run.model_dump(mode="json") for run in query_runs(connection, limit=10)],
        "latest_snapshots": [snapshot.model_dump(mode="json") for snapshot in list_snapshots(connection, limit=10)],
        "quarantine_by_type": {row[0]: row[1] for row in quarantine},
    }


def metrics(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    def scalar(sql: str) -> int:
        row = connection.execute(sql).fetchone()
        if row is None:
            raise StorageError("metrics query returned no row")
        return int(cast(Any, row[0]))

    return {
        "runs_total": scalar("SELECT COUNT(*) FROM runs"),
        "runs_failed": scalar("SELECT COUNT(*) FROM runs WHERE status='FAILED'"),
        "provider_failures": scalar("SELECT COUNT(*) FROM provider_attempts WHERE NOT success"),
        "documents_ingested": scalar("SELECT COUNT(*) FROM documents"),
        "events_extracted": scalar("SELECT COUNT(*) FROM signals"),
        "events_rejected": scalar(
            "SELECT COUNT(*) FROM quarantine WHERE json_extract_string(payload, '$.record_type')='BAD_EXTRACTOR_OUTPUT'"
        ),
        "quarantine_total": scalar("SELECT COUNT(*) FROM quarantine"),
        "cache_hits": scalar(
            "SELECT COALESCE(SUM(CAST(json_extract_string(payload, '$.quality_summary.cache_hits') AS INTEGER)), 0) FROM runs"
        ),
        "cache_misses": scalar(
            "SELECT COALESCE(SUM(CAST(json_extract_string(payload, '$.quality_summary.cache_misses') AS INTEGER)), 0) FROM runs"
        ),
        "snapshot_builds": scalar("SELECT COUNT(*) FROM snapshots"),
        "replay_rows_generated": scalar("SELECT COALESCE(SUM(row_count), 0) FROM replay_datasets"),
    }

__all__ = [
    "all_health",
    "all_watermarks",
    "correction_count",
    "corrections",
    "dashboard_summary",
    "documents_as_of",
    "eligible_signals_as_of",
    "event_sources",
    "extraction_observability",
    "extractor_versions_for",
    "get_document",
    "get_event",
    "get_run",
    "get_snapshot",
    "get_watermark",
    "ineligible_event_ids",
    "latest_quality_scoreboard",
    "latest_run_as_of",
    "list_snapshots",
    "metrics",
    "provider_observability",
    "quarantine_count",
    "query_documents",
    "query_events",
    "query_quarantine",
    "query_runs",
    "runs_up_to",
    "sighting",
    "signals_as_of",
]
