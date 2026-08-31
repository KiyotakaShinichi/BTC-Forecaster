"""The historical dataset service: one entry point for CLI and API (B3.1.22).

Composes the pieces into the operation a researcher actually asks for -- "give me
a point-in-time-safe intelligence feature matrix over this origin schedule" --
and does it once, so the CLI and the API cannot drift apart.

Order of operations is deliberate:

1. Validate the origin schedule.
2. Build rows in bounded, resumable chunks.
3. Write the output file atomically.
4. Hash the output.
5. Register the catalog entry.
6. Write the manifest **last**.

The manifest is the completion marker. Its presence means every earlier step
finished; its absence after an interrupted run means the partial parts on disk
are scratch, not a dataset. Writing it first, or writing it before the catalog
entry, would make a half-finished run indistinguishable from a complete one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from .catalog import DatasetCatalog, DatasetCatalogEntry
from .errors import ReplayIntegrityError
from .feature_matrix import (
    DEFAULT_CHUNK_SIZE,
    FEATURE_NAMES,
    MATRIX_COLUMNS,
    HistoricalFeatureMatrixBuilder,
    build_catalog_entry,
    membership_hash,
    source_fingerprint,
)
from .features import FEATURE_CONTRACT_VERSION
from .operations import ReplayDatasetManifest
from .origins import OriginFrequency, infer_frequency, validate_origin_schedule
from .replay_dataset import ROW_INSERT_CHUNK, ReplayMode
from .storage import IntelligenceStore


@dataclass(frozen=True)
class HistoricalDatasetResult:
    """What a completed historical build produced."""

    manifest: ReplayDatasetManifest
    catalog_entry: DatasetCatalogEntry
    output_path: Path
    manifest_path: Path
    rows_appended: int
    chunk_count: int
    resumed_chunks: int


class HistoricalDatasetService:
    """Builds, extends and catalogs historical intelligence feature matrices."""

    def __init__(self, store: IntelligenceStore, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self.store = store
        self.builder = HistoricalFeatureMatrixBuilder(store, chunk_size=chunk_size)
        self.catalog = DatasetCatalog(store.connection)

    # -- build ------------------------------------------------------------

    def build(
        self,
        origins: list[datetime],
        output_path: str | Path,
        manifest_path: str | Path,
        provider_versions: dict[str, str],
        configuration_fingerprint: str,
        *,
        export_format: str = "parquet",
        mode: ReplayMode = ReplayMode.OPTIMIZED,
        git_sha: str | None = None,
        working_directory: str | Path | None = None,
        resume: bool = True,
    ) -> HistoricalDatasetResult:
        validate_origin_schedule(origins)
        if len(origins) > 100_000:
            raise ReplayIntegrityError("origin count exceeds bounded dataset limit")
        if export_format not in {"parquet", "csv"}:
            raise ValueError("export_format must be parquet or csv")

        target = Path(output_path)
        scratch = Path(working_directory) if working_directory else target.parent / f".{target.name}.parts"

        before = _completed_part_count(scratch) if resume else 0
        rows, snapshot_ids, chunks = self.builder.build_chunks(
            origins,
            scratch,
            provider_versions,
            configuration_fingerprint,
            mode=mode,
            resume=resume,
        )

        _write_dataset(target, rows, export_format)
        file_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        sha = git_sha or _git_sha()

        frequency = infer_frequency(origins) or OriginFrequency.HOURLY
        fingerprint = source_fingerprint(self.store, origins[-1])

        identity = {
            "origins": [origin.isoformat() for origin in origins],
            "membership": membership_hash(snapshot_ids),
            "contract": FEATURE_CONTRACT_VERSION,
            "configuration": configuration_fingerprint,
            "source": fingerprint,
            "file_hash": file_hash,
        }
        dataset_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

        extractor_versions = tuple(
            sorted({version for snapshot in self._snapshots_for(snapshot_ids) for version in snapshot})
        )

        entry = build_catalog_entry(
            dataset_id=dataset_id,
            origins=origins,
            frequency=frequency,
            row_count=len(rows),
            configuration_fingerprint=configuration_fingerprint,
            source_fingerprint_value=fingerprint,
            snapshot_ids=snapshot_ids,
            extractor_versions=extractor_versions,
            output_format=export_format,
            output_file_hash=file_hash,
            git_sha=sha,
            chunk_count=len(chunks),
            mode=mode,
        )
        self.catalog.register(entry)

        manifest = ReplayDatasetManifest(
            dataset_id=dataset_id,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            start_forecast_origin=origins[0],
            end_forecast_origin=origins[-1],
            origin_count=len(origins),
            snapshot_fingerprints=tuple(snapshot_ids),
            provider_versions=provider_versions,
            extractor_versions=extractor_versions,
            configuration_fingerprint=configuration_fingerprint,
            row_count=len(rows),
            columns=MATRIX_COLUMNS,
            file_hash=file_hash,
            git_sha=sha,
            format=export_format,
            mode=mode.value,
        )
        self.store.put_dataset_manifest(manifest)

        # Last, and atomically: the manifest is the completion marker.
        written_manifest = manifest.write_atomic(manifest_path)
        _clear_scratch(scratch)

        return HistoricalDatasetResult(
            manifest=manifest,
            catalog_entry=entry,
            output_path=target,
            manifest_path=written_manifest,
            rows_appended=len(rows),
            chunk_count=len(chunks),
            resumed_chunks=min(before, len(chunks)),
        )

    # -- incremental extension --------------------------------------------

    def extend(
        self,
        dataset_id: str,
        requested_origins: list[datetime],
        output_path: str | Path,
        manifest_path: str | Path,
        provider_versions: dict[str, str],
        configuration_fingerprint: str,
        *,
        export_format: str = "parquet",
        mode: ReplayMode = ReplayMode.OPTIMIZED,
        git_sha: str | None = None,
    ) -> HistoricalDatasetResult:
        """Append later origins to an existing dataset, or fail closed.

        The existing rows are never recomputed. A new dataset id is minted
        because the contents changed; the old entry stays in the catalog, so the
        earlier dataset remains resolvable and nothing is overwritten.
        """
        existing = self.catalog.get(dataset_id)
        if existing is None:
            raise ReplayIntegrityError(f"unknown dataset {dataset_id}")

        fingerprint_at_existing_end = source_fingerprint(self.store, existing.origin_end)
        new_origins = self.builder.plan_extension(
            self.catalog,
            existing,
            requested_origins,
            configuration_fingerprint,
            fingerprint_at_existing_end,
        )

        combined = [
            *_origins_between(existing.origin_start, existing.origin_end, existing.origin_frequency),
            *new_origins,
        ]
        result = self.build(
            combined,
            output_path,
            manifest_path,
            provider_versions,
            configuration_fingerprint,
            export_format=export_format,
            mode=mode,
            git_sha=git_sha,
            resume=False,
        )
        return HistoricalDatasetResult(
            manifest=result.manifest,
            catalog_entry=result.catalog_entry,
            output_path=result.output_path,
            manifest_path=result.manifest_path,
            rows_appended=len(new_origins),
            chunk_count=result.chunk_count,
            resumed_chunks=result.resumed_chunks,
        )

    # -- helpers ----------------------------------------------------------

    def _snapshots_for(self, snapshot_ids: list[str]) -> list[tuple[str, ...]]:
        versions: list[tuple[str, ...]] = []
        for snapshot_id in snapshot_ids:
            snapshot = self.store.get_snapshot(snapshot_id)
            if snapshot is not None:
                versions.append(snapshot.extractor_versions)
        return versions


def _origins_between(start: datetime, end: datetime, frequency: str) -> list[datetime]:
    from .origins import generate_origins

    return generate_origins(start, end, OriginFrequency(frequency))


def _completed_part_count(scratch: Path) -> int:
    from .feature_matrix import PART_PREFIX

    return len(list(scratch.glob(f"{PART_PREFIX}*.json"))) if scratch.exists() else 0


def _clear_scratch(scratch: Path) -> None:
    """Remove the resume scratch once a dataset is complete."""
    if not scratch.exists():
        return
    for path in scratch.glob("*"):
        path.unlink(missing_ok=True)
    scratch.rmdir()


def _write_dataset(target: Path, rows: list[dict[str, object]], export_format: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    columns = list(MATRIX_COLUMNS)

    if export_format == "csv":
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    else:
        connection = duckdb.connect()
        try:
            definitions = [
                "forecast_origin TIMESTAMPTZ",
                *[f"{name} DOUBLE" for name in FEATURE_NAMES],
                "provider_coverage_ratio DOUBLE",
                "queries_failed INTEGER",
                "source_stale_flag INTEGER",
            ]
            connection.execute(f"CREATE TABLE matrix ({', '.join(definitions)})")
            row_placeholder = "(" + ", ".join("?" for _ in columns) + ")"
            for start in range(0, len(rows), ROW_INSERT_CHUNK):
                chunk = rows[start : start + ROW_INSERT_CHUNK]
                parameters: list[object] = []
                for row in chunk:
                    parameters.extend(row[column] for column in columns)
                connection.execute(
                    f"INSERT INTO matrix SELECT * FROM (VALUES {', '.join([row_placeholder] * len(chunk))})",
                    parameters,
                )
            escaped = str(temporary.resolve()).replace("'", "''")
            connection.execute(f"COPY matrix TO '{escaped}' (FORMAT PARQUET)")
        finally:
            connection.close()

    os.replace(temporary, target)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


__all__ = ["HistoricalDatasetResult", "HistoricalDatasetService"]
