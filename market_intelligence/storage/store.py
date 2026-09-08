"""The store: a connection, its lifecycle, and every write to the corpus.

What a write means here is narrower than usual, and the narrowness is the point.
Nothing in this file updates an observation in place. Documents and signals are
inserted first-write-wins, a rediscovered document keeps the availability it was
first seen with, and an observation later found to be an artefact is invalidated
by *appending* a correction row rather than by editing or deleting it. The
corpus is an availability record -- what was knowable at each past instant --
and a record that can be edited afterwards is not evidence of anything.

Two neighbours carry the rest:

* `schema.py` -- the shape of the corpus and the only DDL in the repository.
* `queries.py` -- every read, as functions over a connection, where the
  guarantee is determinism rather than identity.

`IntelligenceStore` remains the public surface and its API is unchanged;
`from market_intelligence.storage import IntelligenceStore` still resolves. The
read methods below delegate rather than reimplement, so there is exactly one
definition of each query.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import duckdb

from ..corrections import EligibilityMode, EventCorrection
from ..errors import (
    FeatureContractError,
    ReplayIntegrityError,
    SnapshotMismatchError,
    StorageError,
)
from ..features import FEATURE_CONTRACT_VERSION
from ..models import Document, EventSignal
from ..operations import (
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
from . import queries
from .schema import SNAPSHOT_INSERT_CHUNK, is_ready, migrate, schema_version

if TYPE_CHECKING:
    from ..retrieval import ProviderAttempt


class IntelligenceStore:
    """Local DuckDB store with idempotent primary-key upserts and JSON provenance."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.connection = duckdb.connect(self.path)
        self._migrate()

    def _migrate(self) -> None:
        migrate(self.connection)

    def schema_version(self) -> int:
        return schema_version(self.connection)

    def ready(self) -> bool:
        return is_ready(self.connection)

    # ------------------------------------------------------------------ writes

    def canonical_availability(self, documents: list[Document]) -> list[Document]:
        """Restamp rediscovered documents with the availability they already had.

        `put_documents` keeps the first write, so the store is the authority on
        when a document became available. Everything derived from a document
        downstream -- an event's identity, its availability, which cluster it
        lands in -- has to read that same authority, or the two disagree.

        They did. Extraction ran on the freshly retrieved objects, whose
        `available_at` is *this* cycle's retrieval time, so re-reading an
        unchanged feed produced a brand-new event for a document that had not
        changed and whose stored availability had not moved. One press release
        sitting in a feed for a week became one event per cycle -- at a
        three-hour cadence, fifty-six of them -- each with a later availability
        than the last.

        That is worse than noise. B4's readiness gate counts events, so the gate
        would have opened on duplicates of a handful of announcements and
        reported a corpus ready for study when it held almost nothing.
        """
        if not documents:
            return documents
        known = dict(
            self.connection.execute(
                "SELECT document_id, available_at FROM documents WHERE document_id IN "
                f"({','.join('?' * len(documents))})",
                [d.document_id for d in documents],
            ).fetchall()
        )
        return [
            document.model_copy(update={"available_at": known[document.document_id]})
            if document.document_id in known
            else document
            for document in documents
        ]

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

    def put_quarantine(self, records: list[QuarantineRecord]) -> None:
        if records:
            self.connection.executemany(
                "INSERT OR REPLACE INTO quarantine VALUES (?, ?, ?)",
                [(r.record_id, r.retrieval_timestamp, r.model_dump_json()) for r in records],
            )

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

    # ------------------------------------------------------------- corrections

    def put_corrections(self, corrections: Sequence[EventCorrection]) -> int:
        """Record corrections. Append-only, and idempotent by construction.

        `INSERT OR IGNORE` on a content-addressed id: recording the same
        correction twice writes one row, and no correction can ever overwrite
        another. Returns how many were genuinely new, so a caller re-running a
        correction script can tell "already applied" from "just applied".
        """
        if not corrections:
            return 0
        before = self.correction_count()
        self.connection.executemany(
            "INSERT OR IGNORE INTO event_corrections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    correction.correction_id,
                    correction.event_id,
                    correction.status.value,
                    correction.reason,
                    correction.invalidated_at,
                    correction.invalidated_by_version,
                    correction.source_bug,
                    correction.notes,
                    correction.model_dump_json(),
                )
                for correction in corrections
            ],
        )
        return self.correction_count() - before

    # ------------------------------------------------------------------- reads
    #
    # Defined in `queries.py` and delegated to here. The methods stay on the
    # store because that is the API every caller and every test already uses;
    # the implementations moved because none of them needs a store, only a
    # connection, and a read that can be exercised without constructing one is
    # a read that gets exercised.

    def documents_as_of(self, forecast_origin: datetime) -> list[Document]:
        return queries.documents_as_of(self.connection, forecast_origin)

    def signals_as_of(self, forecast_origin: datetime) -> list[EventSignal]:
        return queries.signals_as_of(self.connection, forecast_origin)

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
        return queries.query_documents(
            self.connection,
            provider=provider,
            publisher=publisher,
            query=query,
            published_from=published_from,
            published_to=published_to,
            available_from=available_from,
            available_to=available_to,
            forecast_origin=forecast_origin,
            limit=limit,
            offset=offset,
        )

    def get_document(self, document_id: str) -> Document | None:
        return queries.get_document(self.connection, document_id)

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
        return queries.query_events(
            self.connection,
            entity=entity,
            event_type=event_type,
            direction=direction,
            minimum_relevance=minimum_relevance,
            minimum_confidence=minimum_confidence,
            available_from=available_from,
            available_to=available_to,
            forecast_origin=forecast_origin,
            limit=limit,
            offset=offset,
        )

    def get_event(self, event_id: str) -> EventSignal | None:
        return queries.get_event(self.connection, event_id)

    def event_sources(self, event_id: str) -> list[Document]:
        return queries.event_sources(self.connection, event_id)

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
        return queries.query_runs(
            self.connection,
            status=status,
            provider=provider,
            started_from=started_from,
            started_to=started_to,
            limit=limit,
            offset=offset,
        )

    def get_run(self, run_id: str) -> RunManifest | None:
        return queries.get_run(self.connection, run_id)

    def latest_run_as_of(self, origin: datetime) -> RunManifest | None:
        return queries.latest_run_as_of(self.connection, origin)

    def runs_up_to(self, origin: datetime) -> list[tuple[datetime, RunManifest]]:
        return queries.runs_up_to(self.connection, origin)

    def list_snapshots(self, limit: int = 100, offset: int = 0) -> list[IntelligenceSnapshot]:
        return queries.list_snapshots(self.connection, limit, offset)

    def get_snapshot(self, snapshot_id: str) -> IntelligenceSnapshot | None:
        return queries.get_snapshot(self.connection, snapshot_id)

    def extractor_versions_for(self, snapshot_ids: Sequence[str]) -> tuple[str, ...]:
        return queries.extractor_versions_for(self.connection, snapshot_ids)

    def latest_quality_scoreboard(self) -> QualityScoreboard | None:
        return queries.latest_quality_scoreboard(self.connection)

    def all_health(self) -> list[ProviderHealth]:
        return queries.all_health(self.connection)

    def all_watermarks(self) -> list[Watermark]:
        return queries.all_watermarks(self.connection)

    def get_watermark(self, provider_id: str, query_id: str) -> Watermark | None:
        return queries.get_watermark(self.connection, provider_id, query_id)

    def sighting(self, document_id: str) -> DocumentSighting | None:
        return queries.sighting(self.connection, document_id)

    def quarantine_count(self, since: datetime | None = None) -> int:
        return queries.quarantine_count(self.connection, since)

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
        return queries.query_quarantine(
            self.connection,
            provider=provider,
            record_type=record_type,
            failure_category=failure_category,
            time_from=time_from,
            time_to=time_to,
            limit=limit,
            offset=offset,
        )

    def corrections(self, event_ids: Sequence[str] | None = None) -> list[EventCorrection]:
        return queries.corrections(self.connection, event_ids)

    def correction_count(self) -> int:
        return queries.correction_count(self.connection)

    def ineligible_event_ids(
        self,
        *,
        as_of: datetime | None = None,
        mode: EligibilityMode = EligibilityMode.CORRECTED,
    ) -> frozenset[str]:
        return queries.ineligible_event_ids(self.connection, as_of=as_of, mode=mode)

    def eligible_signals_as_of(
        self,
        forecast_origin: datetime,
        *,
        mode: EligibilityMode = EligibilityMode.CORRECTED,
    ) -> list[EventSignal]:
        return queries.eligible_signals_as_of(self.connection, forecast_origin, mode=mode)

    def provider_observability(self) -> list[dict[str, object]]:
        return queries.provider_observability(self.connection)

    def extraction_observability(self) -> dict[str, object]:
        return queries.extraction_observability(self.connection)

    def dashboard_summary(self) -> dict[str, object]:
        return queries.dashboard_summary(self.connection)

    def metrics(self) -> dict[str, int]:
        return queries.metrics(self.connection)

    def export_parquet(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for table in ("documents", "signals"):
            output = str((target / f"{table}.parquet").resolve()).replace("'", "''")
            self.connection.execute(f"COPY {table} TO '{output}' (FORMAT PARQUET)")

    def close(self) -> None:
        self.connection.close()
