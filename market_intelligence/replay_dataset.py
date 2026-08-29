from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from bisect import bisect_right
from datetime import datetime
from enum import Enum
from pathlib import Path

import duckdb

from .errors import ReplayIntegrityError
from .features import FEATURE_CONTRACT_VERSION, FEATURE_DEFINITIONS
from .operations import ReplayDatasetManifest
from .services import SnapshotService
from .storage import IntelligenceStore

#: Rows per bulk INSERT into the parquet staging table.
ROW_INSERT_CHUNK = 500


class ReplayMode(str, Enum):
    """Which snapshot path built a dataset. Recorded in the manifest."""

    REFERENCE = "REFERENCE"
    OPTIMIZED = "OPTIMIZED"


class ReplayDatasetBuilder:
    def __init__(self, store: IntelligenceStore):
        self.store, self.snapshots = store, SnapshotService(store)

    def build(
        self,
        origins: list[datetime],
        output_path: str | Path,
        manifest_path: str | Path,
        provider_versions: dict[str, str],
        configuration_fingerprint: str,
        export_format: str = "parquet",
        git_sha: str | None = None,
        mode: ReplayMode = ReplayMode.OPTIMIZED,
    ) -> ReplayDatasetManifest:
        if not origins or origins != sorted(origins) or len(set(origins)) != len(origins):
            raise ReplayIntegrityError("forecast origins must be non-empty, unique, and ordered")
        if len(origins) > 100_000:
            raise ReplayIntegrityError("origin count exceeds bounded dataset limit")
        rows = []
        build = (
            self.snapshots.build_many
            if mode is ReplayMode.REFERENCE
            else self.snapshots.build_many_bulk
        )
        snapshots = build(origins, provider_versions, configuration_fingerprint)
        feature_names = [definition.name for definition in FEATURE_DEFINITIONS]

        # One ordered read instead of one query per origin. Runs are ascending,
        # so the latest run at an origin is a binary search.
        runs = self.store.runs_up_to(origins[-1])
        run_times = [finished_at for finished_at, _ in runs]

        for origin, snapshot in zip(origins, snapshots, strict=True):
            position = bisect_right(run_times, origin)
            latest_run = runs[position - 1][1] if position else None
            quality = latest_run.quality_summary if latest_run else {}
            row = {
                "forecast_origin": origin.isoformat(),
                **{name: snapshot.features[name] for name in feature_names},
                "provider_coverage_ratio": float(quality.get("provider_coverage_ratio", 0.0)),
                "queries_failed": int(quality.get("queries_failed", 0)),
                "source_stale_flag": int(quality.get("source_stale_flag", 1 if latest_run is None else 0)),
            }
            rows.append(row)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        columns = list(rows[0])
        if export_format == "csv":
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        elif export_format == "parquet":
            connection = duckdb.connect()
            try:
                definitions = [
                    "forecast_origin TIMESTAMPTZ",
                    *[f"{name} DOUBLE" for name in feature_names],
                    "provider_coverage_ratio DOUBLE",
                    "queries_failed INTEGER",
                    "source_stale_flag INTEGER",
                ]
                connection.execute(f"CREATE TABLE replay ({', '.join(definitions)})")
                # A VALUES insert rather than executemany, for the same reason
                # put_snapshots uses one: DuckDB's executemany costs milliseconds
                # per row regardless of row size.
                row_placeholder = "(" + ", ".join("?" for _ in columns) + ")"
                for start in range(0, len(rows), ROW_INSERT_CHUNK):
                    chunk = rows[start : start + ROW_INSERT_CHUNK]
                    parameters: list[object] = []
                    for row in chunk:
                        parameters.extend(row[column] for column in columns)
                    connection.execute(
                        f"INSERT INTO replay SELECT * FROM (VALUES {', '.join([row_placeholder] * len(chunk))})",
                        parameters,
                    )
                escaped = str(temporary.resolve()).replace("'", "''")
                connection.execute(f"COPY replay TO '{escaped}' (FORMAT PARQUET)")
            finally:
                connection.close()
        else:
            raise ValueError("export_format must be parquet or csv")
        os.replace(temporary, target)
        file_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        sha = git_sha or self._git_sha()
        identity = {
            "origins": [item.isoformat() for item in origins],
            "snapshots": [s.snapshot_id for s in snapshots],
            "contract": FEATURE_CONTRACT_VERSION,
            "configuration": configuration_fingerprint,
            "file_hash": file_hash,
        }
        dataset_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        extractor_versions = sorted({version for snapshot in snapshots for version in snapshot.extractor_versions})
        manifest = ReplayDatasetManifest(
            dataset_id=dataset_id,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            start_forecast_origin=origins[0],
            end_forecast_origin=origins[-1],
            origin_count=len(origins),
            snapshot_fingerprints=tuple(s.snapshot_id for s in snapshots),
            provider_versions=provider_versions,
            extractor_versions=tuple(extractor_versions),
            configuration_fingerprint=configuration_fingerprint,
            row_count=len(rows),
            columns=tuple(columns),
            file_hash=file_hash,
            git_sha=sha,
            format=export_format,
            mode=mode.value,
        )
        self.store.put_dataset_manifest(manifest)
        manifest.write_atomic(manifest_path)  # written last
        return manifest

    @staticmethod
    def _git_sha() -> str:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"
