from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb

from .models import Document, EventSignal


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
        """)

    def put_documents(self, documents: list[Document]) -> None:
        self.connection.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", [
            (d.document_id, d.available_at, d.model_dump_json()) for d in documents
        ])

    def put_signals(self, signals: list[EventSignal]) -> None:
        self.connection.executemany("INSERT OR REPLACE INTO signals VALUES (?, ?, ?)", [
            (s.event_id, s.available_time, s.model_dump_json()) for s in signals
        ])

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
