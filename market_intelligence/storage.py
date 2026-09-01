from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

import duckdb

from .errors import FeatureContractError, ReplayIntegrityError, SnapshotMismatchError, StorageError
from .features import FEATURE_CONTRACT_VERSION
from .models import Document, EventSignal
from .operations import (
    DocumentSighting,
    IntelligenceSnapshot,
    ProviderHealth,
    QualityScoreboard,
    QuarantineRecord,
    ReplayDatasetManifest,
    RunManifest,
    Watermark,
    WatermarkStatus,
)

if TYPE_CHECKING:
    from .retrieval import ProviderAttempt

SCHEMA_VERSION = 3


#: Rows per bulk snapshot INSERT. Bounded so a 10,000-origin replay does not
#: build one statement with 30,000 bound parameters.
SNAPSHOT_INSERT_CHUNK = 500


class IntelligenceStore:
    """Local DuckDB store with idempotent primary-key upserts and JSON provenance."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.connection = duckdb.connect(self.path)
        self._migrate()

    def _migrate(self) -> None:
        """Idempotent, forward-only, non-destructive storage migration."""
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS schema_metadata (
              component VARCHAR PRIMARY KEY, version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
              document_id VARCHAR PRIMARY KEY, available_at TIMESTAMPTZ NOT NULL, payload JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS document_sightings (
              document_id VARCHAR PRIMARY KEY,
              first_seen_at TIMESTAMPTZ NOT NULL,
              latest_seen_at TIMESTAMPTZ NOT NULL,
              sighting_count BIGINT NOT NULL,
              providers VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS signals (
              event_id VARCHAR PRIMARY KEY, available_time TIMESTAMPTZ NOT NULL, payload JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS watermarks (
              provider_id VARCHAR, query_id VARCHAR, payload JSON NOT NULL, PRIMARY KEY(provider_id, query_id)
            );
            CREATE TABLE IF NOT EXISTS quarantine (
              record_id VARCHAR PRIMARY KEY, retrieval_timestamp TIMESTAMPTZ NOT NULL, payload JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
              snapshot_id VARCHAR PRIMARY KEY, forecast_origin TIMESTAMPTZ NOT NULL, payload JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS provider_health (
              provider_id VARCHAR PRIMARY KEY, payload JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
              run_id VARCHAR PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL,
              finished_at TIMESTAMPTZ NOT NULL, status VARCHAR NOT NULL,
              provider_ids VARCHAR[] NOT NULL, payload JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS quality_scoreboards (
              run_id VARCHAR PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS provider_attempts (
              run_id VARCHAR NOT NULL, provider_id VARCHAR NOT NULL, success BOOLEAN NOT NULL,
              latency_ms DOUBLE NOT NULL, documents_received INTEGER NOT NULL,
              rate_limited BOOLEAN NOT NULL, observed_at TIMESTAMPTZ NOT NULL
            );
            CREATE TABLE IF NOT EXISTS replay_datasets (
              dataset_id VARCHAR PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT current_timestamp,
              row_count INTEGER NOT NULL, payload JSON NOT NULL
            );
        """)
        row = self.connection.execute(
            "SELECT version FROM schema_metadata WHERE component='market_intelligence'"
        ).fetchone()
        current = row[0] if row else 0
        if current > SCHEMA_VERSION:
            raise StorageError(f"database schema {current} is newer than supported version {SCHEMA_VERSION}")
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_metadata VALUES ('market_intelligence', ?)", [SCHEMA_VERSION]
        )

    def schema_version(self) -> int:
        row = self.connection.execute(
            "SELECT version FROM schema_metadata WHERE component='market_intelligence'"
        ).fetchone()
        if row is None:
            raise StorageError("schema metadata is missing")
        return cast(int, row[0])

    def ready(self) -> bool:
        row = self.connection.execute("SELECT 1").fetchone()
        return self.schema_version() == SCHEMA_VERSION and row is not None and row[0] == 1

    def put_documents(self, documents: list[Document]) -> None:
        """Persist evidence. **The first write wins.**

        `INSERT OR REPLACE` here would be a corpus-destroying bug in forward
        collection. A document's id is derived from its URL and content, so
        every cycle that re-reads a feed rediscovers the same items with a fresh
        `available_at` -- and replacing would rewrite each document's
        availability to the latest collection time, forever. A replay at any
        past origin would then see nothing, because nothing would ever have been
        available before now.

        Rediscovery is real information, so it is not thrown away either: it is
        recorded as a sighting (B4.1.12), leaving the original availability
        untouched.
        """
        if not documents:
            return
        self.connection.executemany(
            "INSERT OR IGNORE INTO documents VALUES (?, ?, ?)",
            [(d.document_id, d.available_at, d.model_dump_json()) for d in documents],
        )
        self.record_sightings(documents)

    def record_sightings(self, documents: list[Document]) -> None:
        """Track first and latest sighting without touching stored availability."""
        if not documents:
            return
        for document in documents:
            row = self.connection.execute(
                "SELECT first_seen_at, sighting_count, providers FROM document_sightings WHERE document_id = ?",
                [document.document_id],
            ).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO document_sightings VALUES (?, ?, ?, ?, ?)",
                    [
                        document.document_id,
                        document.retrieved_at,
                        document.retrieved_at,
                        1,
                        document.provider,
                    ],
                )
                continue
            first_seen, count, providers = row
            merged = sorted({*str(providers).split(","), document.provider})
            self.connection.execute(
                """
                UPDATE document_sightings
                   SET latest_seen_at = greatest(latest_seen_at, ?),
                       first_seen_at = least(first_seen_at, ?),
                       sighting_count = ?,
                       providers = ?
                 WHERE document_id = ?
                """,
                [
                    document.retrieved_at,
                    document.retrieved_at,
                    int(count) + 1,
                    ",".join(merged),
                    document.document_id,
                ],
            )

    def sighting(self, document_id: str) -> DocumentSighting | None:
        row = self.connection.execute(
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

    def put_signals(self, signals: list[EventSignal]) -> None:
        if signals:
            known_docs = {row[0] for row in self.connection.execute("SELECT document_id FROM documents").fetchall()}
            orphaned = [signal.event_id for signal in signals if not set(signal.source_ids) <= known_docs]
            if orphaned:
                raise ReplayIntegrityError(f"events reference unknown documents: {', '.join(orphaned)}")
            self.connection.executemany(
                "INSERT OR REPLACE INTO signals VALUES (?, ?, ?)",
                [(s.event_id, s.available_time, s.model_dump_json()) for s in signals],
            )

    def persist_cycle(self, documents: list[Document], signals: list[EventSignal], watermarks: list[Watermark]) -> None:
        """Atomically store evidence and advance only successful watermarks."""
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.put_documents(documents)
            self.put_signals(signals)
            rows = [
                (w.provider_id, w.query_id, w.model_dump_json())
                for w in watermarks
                if w.status == WatermarkStatus.SUCCESS
            ]
            if rows:
                for watermark in (item for item in watermarks if item.status == WatermarkStatus.SUCCESS):
                    current = self.get_watermark(watermark.provider_id, watermark.query_id)
                    if current and (
                        watermark.last_retrieval_time < current.last_retrieval_time
                        or (
                            current.last_successful_available_time
                            and watermark.last_successful_available_time
                            and watermark.last_successful_available_time < current.last_successful_available_time
                        )
                    ):
                        raise StorageError("watermark advancement must be monotonic")
                self.connection.executemany("INSERT OR REPLACE INTO watermarks VALUES (?, ?, ?)", rows)
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def get_watermark(self, provider_id: str, query_id: str) -> Watermark | None:
        row = self.connection.execute(
            "SELECT payload FROM watermarks WHERE provider_id=? AND query_id=?", [provider_id, query_id]
        ).fetchone()
        return Watermark.model_validate(json.loads(row[0])) if row else None

    def all_watermarks(self) -> list[Watermark]:
        rows = self.connection.execute("SELECT payload FROM watermarks ORDER BY provider_id, query_id").fetchall()
        return [Watermark.model_validate(json.loads(row[0])) for row in rows]

    def put_quarantine(self, records: list[QuarantineRecord]) -> None:
        if records:
            self.connection.executemany(
                "INSERT OR REPLACE INTO quarantine VALUES (?, ?, ?)",
                [(r.record_id, r.retrieval_timestamp, r.model_dump_json()) for r in records],
            )

    def quarantine_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM quarantine").fetchone()
        return cast(int, row[0]) if row else 0

    def put_snapshot(self, snapshot: IntelligenceSnapshot) -> None:
        self.put_snapshots([snapshot])

    def put_snapshots(self, snapshots: Sequence[IntelligenceSnapshot]) -> None:
        """Persist snapshots. Membership is validated exactly as before.

        Two changes from the original, both measured (research/b31/PROFILE.md):

        * Membership is checked against the *union* of everything the batch
          references rather than per snapshot. `all subsets valid` and
          `union valid` are the same statement, and the per-snapshot form was
          O(origins x history) -- a hidden quadratic in a replay of many origins
          that all reference the same documents.
        * Rows go in through a chunked ``VALUES`` insert rather than
          ``executemany``. For a 200-snapshot batch carrying 22.9 MB of JSON
          that is 1,578 ms against 96 ms, a 16x difference, with identical rows
          and identical INSERT OR IGNORE semantics.
        """
        if not snapshots:
            return
        for snapshot in snapshots:
            if snapshot.feature_contract_version != FEATURE_CONTRACT_VERSION:
                raise FeatureContractError(f"unknown feature contract: {snapshot.feature_contract_version}")

        referenced_docs: set[str] = set()
        referenced_events: set[str] = set()
        for snapshot in snapshots:
            referenced_docs.update(snapshot.document_ids)
            referenced_events.update(snapshot.event_ids)

        known_docs = {row[0] for row in self.connection.execute("SELECT document_id FROM documents").fetchall()}
        known_events = {row[0] for row in self.connection.execute("SELECT event_id FROM signals").fetchall()}
        if not referenced_docs <= known_docs or not referenced_events <= known_events:
            raise ReplayIntegrityError("snapshot references unknown membership")

        rows = [
            (snapshot.snapshot_id, snapshot.forecast_origin, snapshot.model_dump_json()) for snapshot in snapshots
        ]
        for start in range(0, len(rows), SNAPSHOT_INSERT_CHUNK):
            chunk = rows[start : start + SNAPSHOT_INSERT_CHUNK]
            placeholders = ", ".join(["(?, ?, ?)"] * len(chunk))
            parameters: list[object] = []
            for row in chunk:
                parameters.extend(row)
            self.connection.execute(
                f"INSERT OR IGNORE INTO snapshots SELECT * FROM (VALUES {placeholders})", parameters
            )

    def get_snapshot(self, snapshot_id: str) -> IntelligenceSnapshot | None:
        row = self.connection.execute("SELECT payload FROM snapshots WHERE snapshot_id=?", [snapshot_id]).fetchone()
        return IntelligenceSnapshot.model_validate(json.loads(row[0])) if row else None

    def extractor_versions_for(self, snapshot_ids: Sequence[str]) -> tuple[str, ...]:
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
            rows = self.connection.execute(
                f"SELECT json_extract(payload, '$.extractor_versions') FROM snapshots "
                f"WHERE snapshot_id IN ({placeholders})",
                list(batch),
            ).fetchall()
            for row in rows:
                versions.update(json.loads(row[0]))
        return tuple(sorted(versions))

    def verify_snapshot(
        self, snapshot_id: str, configuration_fingerprint: str, feature_contract_version: str = FEATURE_CONTRACT_VERSION
    ) -> IntelligenceSnapshot:
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise SnapshotMismatchError("snapshot not found")
        rebuilt = IntelligenceSnapshot.create(
            snapshot.forecast_origin,
            list(snapshot.document_ids),
            list(snapshot.event_ids),
            snapshot.features,
            snapshot.provider_versions,
            list(snapshot.extractor_versions),
            snapshot.configuration_fingerprint,
            list(snapshot.source_hashes),
            snapshot.feature_contract_version,
        )
        if rebuilt.snapshot_id != snapshot.snapshot_id:
            raise ReplayIntegrityError("snapshot membership fingerprint is corrupt")
        if snapshot.configuration_fingerprint != configuration_fingerprint:
            raise SnapshotMismatchError("configuration fingerprint mismatch")
        if snapshot.feature_contract_version != feature_contract_version:
            raise FeatureContractError("feature contract mismatch")
        return snapshot

    def put_run(self, manifest: RunManifest) -> None:
        existing = self.get_run(manifest.run_id)
        if existing is not None and existing != manifest:
            raise ReplayIntegrityError("historical run manifests are immutable")
        self.connection.execute(
            "INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            [
                manifest.run_id,
                manifest.started_at,
                manifest.finished_at,
                manifest.status.value,
                list(manifest.provider_ids),
                manifest.model_dump_json(),
            ],
        )

    def put_quality_scoreboard(self, scoreboard: QualityScoreboard) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO quality_scoreboards VALUES (?, ?, ?)",
            [scoreboard.run_id, scoreboard.created_at, scoreboard.model_dump_json()],
        )

    def put_provider_attempts(self, run_id: str, attempts: Sequence[ProviderAttempt], observed_at: datetime) -> None:
        rows = [
            (
                run_id,
                a.provider_id,
                a.success,
                a.latency_ms,
                a.documents_received,
                a.rate_limited,
                observed_at,
            )
            for a in attempts
        ]
        if rows:
            self.connection.executemany("INSERT INTO provider_attempts VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    def put_dataset_manifest(self, manifest: ReplayDatasetManifest) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO replay_datasets(dataset_id, row_count, payload) VALUES (?, ?, ?)",
            [manifest.dataset_id, manifest.row_count, manifest.model_dump_json()],
        )

    def put_health(self, health: list[ProviderHealth]) -> None:
        if health:
            self.connection.executemany(
                "INSERT OR REPLACE INTO provider_health VALUES (?, ?)",
                [(item.provider_id, item.model_dump_json()) for item in health],
            )

    def all_health(self) -> list[ProviderHealth]:
        rows = self.connection.execute("SELECT payload FROM provider_health ORDER BY provider_id").fetchall()
        return [ProviderHealth.model_validate(json.loads(row[0])) for row in rows]

    def query_documents(
        self,
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
        rows = self.connection.execute(
            f"SELECT payload FROM documents{where} ORDER BY available_at, document_id LIMIT ? OFFSET ?",
            [*values, limit, offset],
        ).fetchall()
        return [Document.model_validate(json.loads(row[0])) for row in rows]

    def get_document(self, document_id: str) -> Document | None:
        row = self.connection.execute("SELECT payload FROM documents WHERE document_id=?", [document_id]).fetchone()
        return Document.model_validate(json.loads(row[0])) if row else None

    def query_events(
        self,
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
        rows = self.connection.execute(
            f"SELECT payload FROM signals{where} ORDER BY available_time, event_id LIMIT ? OFFSET ?",
            [*values, limit, offset],
        ).fetchall()
        return [EventSignal.model_validate(json.loads(row[0])) for row in rows]

    def get_event(self, event_id: str) -> EventSignal | None:
        row = self.connection.execute("SELECT payload FROM signals WHERE event_id=?", [event_id]).fetchone()
        return EventSignal.model_validate(json.loads(row[0])) if row else None

    def event_sources(self, event_id: str) -> list[Document]:
        event = self.get_event(event_id)
        if event is None:
            return []
        return [document for source_id in event.source_ids if (document := self.get_document(source_id)) is not None]

    def query_runs(
        self,
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
        rows = self.connection.execute(
            f"SELECT payload FROM runs{where} ORDER BY started_at DESC, run_id LIMIT ? OFFSET ?",
            [*values, limit, offset],
        ).fetchall()
        return [RunManifest.model_validate(json.loads(row[0])) for row in rows]

    def get_run(self, run_id: str) -> RunManifest | None:
        row = self.connection.execute("SELECT payload FROM runs WHERE run_id=?", [run_id]).fetchone()
        return RunManifest.model_validate(json.loads(row[0])) if row else None

    def latest_run_as_of(self, origin: datetime) -> RunManifest | None:
        row = self.connection.execute(
            "SELECT payload FROM runs WHERE finished_at <= ? ORDER BY finished_at DESC LIMIT 1", [origin]
        ).fetchone()
        return RunManifest.model_validate(json.loads(row[0])) if row else None

    def runs_up_to(self, origin: datetime) -> list[tuple[datetime, RunManifest]]:
        """All runs finished at or before ``origin``, ascending.

        Batched counterpart to :meth:`latest_run_as_of`. A DuckDB statement costs
        ~4 ms of parse and plan time even against an empty table, so asking once
        per origin costs ~4 s per 1,000 origins for information that one ordered
        read plus a binary search already contains.
        """
        rows = self.connection.execute(
            "SELECT finished_at, payload FROM runs WHERE finished_at <= ? ORDER BY finished_at", [origin]
        ).fetchall()
        return [(row[0], RunManifest.model_validate(json.loads(row[1]))) for row in rows]

    def list_snapshots(self, limit: int = 100, offset: int = 0) -> list[IntelligenceSnapshot]:
        rows = self.connection.execute(
            "SELECT payload FROM snapshots ORDER BY forecast_origin DESC LIMIT ? OFFSET ?", [limit, offset]
        ).fetchall()
        return [IntelligenceSnapshot.model_validate(json.loads(row[0])) for row in rows]

    def latest_quality_scoreboard(self) -> QualityScoreboard | None:
        row = self.connection.execute(
            "SELECT payload FROM quality_scoreboards ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return QualityScoreboard.model_validate(json.loads(row[0])) if row else None

    def query_quarantine(
        self,
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
        rows = self.connection.execute(
            f"SELECT payload FROM quarantine{where} ORDER BY retrieval_timestamp DESC LIMIT ? OFFSET ?",
            [*values, limit, offset],
        ).fetchall()
        return [QuarantineRecord.model_validate(json.loads(row[0])) for row in rows]

    def provider_observability(self) -> list[dict[str, object]]:
        rows = self.connection.execute("""
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
        columns = [item[0] for item in self.connection.description]
        return [dict(zip(columns, row, strict=False)) for row in rows]

    def extraction_observability(self) -> dict[str, object]:
        method_rows = self.connection.execute("""
          SELECT json_extract_string(payload, '$.extraction_method') AS extraction_method,
            COUNT(*) AS extraction_count
          FROM signals GROUP BY extraction_method ORDER BY extraction_method
        """).fetchall()
        totals = self.connection.execute("""
          SELECT COUNT(*), AVG(CAST(json_extract_string(payload, '$.confidence') AS DOUBLE)),
            AVG(CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE)),
            MIN(CAST(json_extract_string(payload, '$.confidence') AS DOUBLE)),
            MEDIAN(CAST(json_extract_string(payload, '$.confidence') AS DOUBLE)),
            MAX(CAST(json_extract_string(payload, '$.confidence') AS DOUBLE)),
            MIN(CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE)),
            MEDIAN(CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE)),
            MAX(CAST(json_extract_string(payload, '$.btc_relevance') AS DOUBLE)) FROM signals
        """).fetchone()
        rejection_rows = self.connection.execute("""
          SELECT json_extract_string(payload, '$.record_type'), COUNT(*) FROM quarantine GROUP BY 1 ORDER BY 1
        """).fetchall()
        rejection_categories = self.connection.execute("""
          SELECT
            SUM(CASE WHEN json_extract_string(payload, '$.failure_reason') ILIKE '%json%' THEN 1 ELSE 0 END),
            SUM(CASE WHEN json_extract_string(payload, '$.failure_reason') ILIKE '%source%' THEN 1 ELSE 0 END),
            SUM(CASE WHEN json_extract_string(payload, '$.failure_reason') ILIKE '%schema%'
                          OR json_extract_string(payload, '$.failure_reason') ILIKE '%valid%' THEN 1 ELSE 0 END)
          FROM quarantine
        """).fetchone()
        docs_row = self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()
        if totals is None or docs_row is None:
            raise StorageError("observability query returned no aggregate row")
        if rejection_categories is None:
            raise StorageError("extraction rejection query returned no aggregate row")
        docs = cast(int, docs_row[0])
        return {
            "accepted_by_method": {row[0]: row[1] for row in method_rows},
            "accepted_extractions": totals[0],
            "rejected_extractions": self.quarantine_count(),
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

    def dashboard_summary(self) -> dict[str, object]:
        event_types = self.connection.execute("""
          SELECT json_extract_string(payload, '$.event_type'), COUNT(*) FROM signals GROUP BY 1 ORDER BY 1
        """).fetchall()
        entities = self.connection.execute("""
          SELECT json_extract_string(payload, '$.entity'), COUNT(*) FROM signals
          WHERE json_extract_string(payload, '$.entity') IS NOT NULL GROUP BY 1 ORDER BY 1
        """).fetchall()
        quarantine = self.connection.execute("""
          SELECT json_extract_string(payload, '$.record_type'), COUNT(*) FROM quarantine GROUP BY 1 ORDER BY 1
        """).fetchall()
        return {
            "event_type_distribution": {row[0]: row[1] for row in event_types},
            "tracked_entities": {row[0]: row[1] for row in entities},
            "latest_runs": [run.model_dump(mode="json") for run in self.query_runs(limit=10)],
            "latest_snapshots": [snapshot.model_dump(mode="json") for snapshot in self.list_snapshots(limit=10)],
            "quarantine_by_type": {row[0]: row[1] for row in quarantine},
        }

    def metrics(self) -> dict[str, int]:
        def scalar(sql: str) -> int:
            row = self.connection.execute(sql).fetchone()
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

    def documents_as_of(self, forecast_origin: datetime) -> list[Document]:
        rows = self.connection.execute(
            "SELECT payload FROM documents WHERE available_at <= ? ORDER BY available_at", [forecast_origin]
        ).fetchall()
        return [Document.model_validate(json.loads(row[0])) for row in rows]

    def signals_as_of(self, forecast_origin: datetime) -> list[EventSignal]:
        rows = self.connection.execute(
            "SELECT payload FROM signals WHERE available_time <= ? ORDER BY available_time", [forecast_origin]
        ).fetchall()
        return [EventSignal.model_validate(json.loads(row[0])) for row in rows]

    def export_parquet(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for table in ("documents", "signals"):
            output = str((target / f"{table}.parquet").resolve()).replace("'", "''")
            self.connection.execute(f"COPY {table} TO '{output}' (FORMAT PARQUET)")

    def close(self) -> None:
        self.connection.close()
