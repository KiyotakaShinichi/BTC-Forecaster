from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb

from .models import Document, EventSignal
from .operations import IntelligenceSnapshot, ProviderHealth, QuarantineRecord, Watermark, WatermarkStatus


class IntelligenceStore:
    """Local DuckDB store with idempotent primary-key upserts and JSON provenance."""

    def __init__(self, path: str | Path):
        self.connection = duckdb.connect(str(path))
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
              document_id VARCHAR PRIMARY KEY, available_at TIMESTAMPTZ NOT NULL, payload JSON NOT NULL
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
        """)

    def put_documents(self, documents: list[Document]) -> None:
        if documents:
            self.connection.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", [
                (d.document_id, d.available_at, d.model_dump_json()) for d in documents
            ])

    def put_signals(self, signals: list[EventSignal]) -> None:
        if signals:
            self.connection.executemany("INSERT OR REPLACE INTO signals VALUES (?, ?, ?)", [
                (s.event_id, s.available_time, s.model_dump_json()) for s in signals
            ])

    def persist_cycle(self, documents: list[Document], signals: list[EventSignal], watermarks: list[Watermark]) -> None:
        """Atomically store evidence and advance only successful watermarks."""
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.put_documents(documents)
            self.put_signals(signals)
            rows = [(w.provider_id, w.query_id, w.model_dump_json()) for w in watermarks if w.status == WatermarkStatus.SUCCESS]
            if rows:
                self.connection.executemany("INSERT OR REPLACE INTO watermarks VALUES (?, ?, ?)", rows)
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def get_watermark(self, provider_id: str, query_id: str) -> Watermark | None:
        row = self.connection.execute("SELECT payload FROM watermarks WHERE provider_id=? AND query_id=?", [provider_id, query_id]).fetchone()
        return Watermark.model_validate(json.loads(row[0])) if row else None

    def all_watermarks(self) -> list[Watermark]:
        rows = self.connection.execute("SELECT payload FROM watermarks ORDER BY provider_id, query_id").fetchall()
        return [Watermark.model_validate(json.loads(row[0])) for row in rows]

    def put_quarantine(self, records: list[QuarantineRecord]) -> None:
        if records:
            self.connection.executemany("INSERT OR REPLACE INTO quarantine VALUES (?, ?, ?)",
                [(r.record_id, r.retrieval_timestamp, r.model_dump_json()) for r in records])

    def quarantine_count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]

    def put_snapshot(self, snapshot: IntelligenceSnapshot) -> None:
        self.connection.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?)",
            [snapshot.snapshot_id, snapshot.forecast_origin, snapshot.model_dump_json()])

    def put_health(self, health: list[ProviderHealth]) -> None:
        if health:
            self.connection.executemany("INSERT OR REPLACE INTO provider_health VALUES (?, ?)",
                [(item.provider_id, item.model_dump_json()) for item in health])

    def all_health(self) -> list[ProviderHealth]:
        rows = self.connection.execute("SELECT payload FROM provider_health ORDER BY provider_id").fetchall()
        return [ProviderHealth.model_validate(json.loads(row[0])) for row in rows]

    def documents_as_of(self, forecast_origin: datetime) -> list[Document]:
        rows = self.connection.execute("SELECT payload FROM documents WHERE available_at <= ? ORDER BY available_at", [forecast_origin]).fetchall()
        return [Document.model_validate(json.loads(row[0])) for row in rows]

    def signals_as_of(self, forecast_origin: datetime) -> list[EventSignal]:
        rows = self.connection.execute("SELECT payload FROM signals WHERE available_time <= ? ORDER BY available_time", [forecast_origin]).fetchall()
        return [EventSignal.model_validate(json.loads(row[0])) for row in rows]

    def export_parquet(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for table in ("documents", "signals"):
            output = str((target / f"{table}.parquet").resolve()).replace("'", "''")
            self.connection.execute(f"COPY {table} TO '{output}' (FORMAT PARQUET)")

    def close(self) -> None:
        self.connection.close()
