"""The shape of the corpus, and the one place it changes.

Separated from the store because it answers a different question. Everything
else in this package reads or writes rows; this decides what a row *is*, runs
once when a database is opened, and is the only code in the repository allowed
to issue DDL.

The migration is idempotent, forward-only and non-destructive. There is no
`DROP`, no `ALTER ... DROP COLUMN` and no rewrite of an existing row anywhere in
it, and that is a property of the corpus rather than a preference: the store
holds an availability record -- what was knowable at each past instant -- and a
record you can edit afterwards is not evidence of anything. A database written
by a newer version is refused rather than silently downgraded, because opening
it with an older schema would read columns that mean something else.
"""

from __future__ import annotations

from typing import cast

import duckdb

from ..errors import StorageError

#: Bumped when the DDL below changes in a way an older build cannot read. A
#: database carrying a higher number is refused, not migrated backwards.
SCHEMA_VERSION = 3

#: The component name under which the version is recorded. One row, one
#: component: this database belongs to one system.
COMPONENT = "market_intelligence"

#: Rows per bulk statement. A property of the engine rather than of any one
#: query, which is why it lives beside the schema and is shared by the writer
#: that inserts snapshots and the reader that looks their versions back up: a
#: 10,000-origin replay would otherwise build a single statement carrying
#: 30,000 bound parameters.
SNAPSHOT_INSERT_CHUNK = 500

#: Every table, created if absent. Ordering is irrelevant -- each statement is
#: independent and idempotent -- so this reads as a description of the corpus
#: rather than as a sequence of steps.
SCHEMA_DDL = """
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
-- Append-only. Nothing here updates or deletes an observation; a
-- correction is a new row stating what is now known about one, and
-- the observation itself stays exactly as it was written.
CREATE TABLE IF NOT EXISTS event_corrections (
  correction_id VARCHAR PRIMARY KEY,
  event_id VARCHAR NOT NULL,
  status VARCHAR NOT NULL,
  reason VARCHAR NOT NULL,
  invalidated_at TIMESTAMPTZ NOT NULL,
  invalidated_by_version VARCHAR NOT NULL,
  source_bug VARCHAR NOT NULL,
  notes VARCHAR NOT NULL,
  payload JSON NOT NULL
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
"""


def migrate(connection: duckdb.DuckDBPyConnection) -> None:
    """Bring a connection up to `SCHEMA_VERSION`. Safe to call on every open."""
    connection.execute(SCHEMA_DDL)
    row = connection.execute(
        f"SELECT version FROM schema_metadata WHERE component='{COMPONENT}'"
    ).fetchone()
    current = row[0] if row else 0
    if current > SCHEMA_VERSION:
        raise StorageError(
            f"database schema {current} is newer than supported version {SCHEMA_VERSION}"
        )
    connection.execute(
        "INSERT OR REPLACE INTO schema_metadata VALUES (?, ?)", [COMPONENT, SCHEMA_VERSION]
    )


def schema_version(connection: duckdb.DuckDBPyConnection) -> int:
    row = connection.execute(
        f"SELECT version FROM schema_metadata WHERE component='{COMPONENT}'"
    ).fetchone()
    if row is None:
        raise StorageError("schema metadata is missing")
    return cast(int, row[0])


def is_ready(connection: duckdb.DuckDBPyConnection) -> bool:
    """True when the schema is current *and* the connection answers a query.

    Both halves matter. A connection that reports the right version but cannot
    execute is not a usable store, and a readiness probe that only checks
    metadata would call it healthy.
    """
    row = connection.execute("SELECT 1").fetchone()
    return schema_version(connection) == SCHEMA_VERSION and row is not None and row[0] == 1


__all__ = [
    "COMPONENT",
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "SNAPSHOT_INSERT_CHUNK",
    "is_ready",
    "migrate",
    "schema_version",
]
