"""Immutable dataset catalog (B3.1.12).

A catalog entry is the answer to "what is this file and can I trust it". It
records the origin schedule, the fingerprints of everything that fed the build,
the output hash, and the commit that produced it.

Two rules make it worth having:

**Entries are immutable.** Registering an existing ``dataset_id`` with different
contents is an error, not an update. A dataset id is derived from its inputs and
its output hash, so two different builds cannot legitimately share one; if they
appear to, something upstream is not as deterministic as it claims.

**Re-registering identical contents is fine.** Rebuilding the same dataset from
the same inputs and getting the same id is a successful reproduction, and
failing that would make the catalog hostile to the thing it exists to encourage.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from .errors import ReplayIntegrityError


class DatasetCatalogEntry(BaseModel):
    """One immutable record of a generated historical dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    origin_start: datetime
    origin_end: datetime
    origin_frequency: str
    row_count: int
    feature_contract_version: str
    configuration_fingerprint: str
    source_fingerprint: str
    membership_hash: str
    extractor_versions: tuple[str, ...]
    output_format: str
    output_file_hash: str
    created_at: datetime
    git_sha: str
    chunk_count: int = 1
    mode: str = "OPTIMIZED"

    def identity(self) -> dict[str, Any]:
        """The fields that must not differ between two entries with one id."""
        return {
            "dataset_id": self.dataset_id,
            "origin_start": self.origin_start.isoformat(),
            "origin_end": self.origin_end.isoformat(),
            "origin_frequency": self.origin_frequency,
            "row_count": self.row_count,
            "feature_contract_version": self.feature_contract_version,
            "configuration_fingerprint": self.configuration_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "membership_hash": self.membership_hash,
            "output_file_hash": self.output_file_hash,
        }


class DatasetCatalog:
    """Append-only registry of generated datasets, stored in DuckDB."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_catalog (
              dataset_id VARCHAR PRIMARY KEY,
              origin_start TIMESTAMPTZ NOT NULL,
              origin_end TIMESTAMPTZ NOT NULL,
              row_count BIGINT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL,
              payload JSON NOT NULL
            )
            """
        )

    def register(self, entry: DatasetCatalogEntry) -> DatasetCatalogEntry:
        """Record a dataset. Refuses to overwrite a different one.

        Re-registering byte-identical contents is a no-op, so a reproduced build
        is not punished.
        """
        existing = self.get(entry.dataset_id)
        if existing is not None:
            if existing.identity() != entry.identity():
                raise ReplayIntegrityError(
                    f"dataset {entry.dataset_id} is already registered with different contents; "
                    "catalog entries are immutable"
                )
            return existing

        self.connection.execute(
            "INSERT INTO dataset_catalog VALUES (?, ?, ?, ?, ?, ?)",
            [
                entry.dataset_id,
                entry.origin_start,
                entry.origin_end,
                entry.row_count,
                entry.created_at,
                entry.model_dump_json(),
            ],
        )
        return entry

    def get(self, dataset_id: str) -> DatasetCatalogEntry | None:
        row = self.connection.execute(
            "SELECT payload FROM dataset_catalog WHERE dataset_id = ?", [dataset_id]
        ).fetchone()
        return DatasetCatalogEntry.model_validate(json.loads(row[0])) if row else None

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[DatasetCatalogEntry]:
        rows = self.connection.execute(
            "SELECT payload FROM dataset_catalog ORDER BY created_at DESC, dataset_id LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
        return [DatasetCatalogEntry.model_validate(json.loads(row[0])) for row in rows]

    def latest_covering(
        self, configuration_fingerprint: str, feature_contract_version: str
    ) -> DatasetCatalogEntry | None:
        """Most recent dataset built under a given config and contract.

        The starting point for incremental extension: an append is only allowed
        onto a dataset whose contract and configuration match the request.
        """
        for entry in self.list_datasets(limit=200):
            if (
                entry.configuration_fingerprint == configuration_fingerprint
                and entry.feature_contract_version == feature_contract_version
            ):
                return entry
        return None


__all__ = ["DatasetCatalog", "DatasetCatalogEntry"]
