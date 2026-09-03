"""Historical feature matrix: chunked, resumable, extendable (B3.1.8, B3.1.10, B3.1.11).

One point-in-time-safe row per forecast origin. No BTC target, no returns, no
fusion -- that is B4's problem, and mixing it in here would make the intelligence
dataset depend on the price series it is supposed to be evaluated against.

Three operational properties, each of which exists because long historical
replays fail partway:

**Chunked.** Origins are processed in bounded batches so peak memory tracks the
chunk, not the run. A 10,000-origin replay holds one chunk of evidence at a time.

**Resumable.** Each completed chunk writes a part file and a progress record. An
interrupted run restarts from the first incomplete chunk instead of from zero,
and the partial output is never mistaken for a finished dataset because the
manifest -- written last, atomically -- is the completion marker.

**Extendable.** Appending later origins to an existing dataset is allowed only
when the feature contract, configuration, source history and cadence all match.
Anything else fails closed rather than silently rebuilding history, because a
dataset whose early rows came from a different contract than its late rows is
worse than no dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .catalog import DatasetCatalog, DatasetCatalogEntry
from .errors import ReplayIntegrityError
from .features import FEATURE_CONTRACT_VERSION, FEATURE_DEFINITIONS
from .origins import OriginFrequency, infer_frequency, validate_origin_schedule
from .replay_dataset import ReplayMode
from .services import SnapshotService
from .storage import IntelligenceStore

#: Origins per chunk. Bounds peak memory and gives resumption a granularity.
DEFAULT_CHUNK_SIZE = 500

PROGRESS_FILENAME = "_progress.json"
PART_PREFIX = "part-"

FEATURE_NAMES: tuple[str, ...] = tuple(definition.name for definition in FEATURE_DEFINITIONS)

#: Operational columns required by the B3 contract alongside the seven features.
OPERATIONAL_COLUMNS: tuple[str, ...] = (
    "provider_coverage_ratio",
    "queries_failed",
    "source_stale_flag",
)

MATRIX_COLUMNS: tuple[str, ...] = ("forecast_origin", *FEATURE_NAMES, *OPERATIONAL_COLUMNS)


@dataclass(frozen=True)
class ChunkResult:
    """One completed chunk of the matrix."""

    index: int
    origin_start: datetime
    origin_end: datetime
    row_count: int
    snapshot_ids: tuple[str, ...]
    part_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "origin_start": self.origin_start.isoformat(),
            "origin_end": self.origin_end.isoformat(),
            "row_count": self.row_count,
            "part": self.part_path.name,
        }


@dataclass
class MatrixBuildState:
    """Progress across chunks, persisted so a run can resume."""

    origin_count: int
    chunk_size: int
    configuration_fingerprint: str
    feature_contract_version: str
    completed: list[dict[str, object]] = field(default_factory=list)

    @property
    def completed_rows(self) -> int:
        total = 0
        for chunk in self.completed:
            value = chunk["row_count"]
            total += value if isinstance(value, int) else 0
        return total

    def to_dict(self) -> dict[str, object]:
        return {
            "origin_count": self.origin_count,
            "chunk_size": self.chunk_size,
            "configuration_fingerprint": self.configuration_fingerprint,
            "feature_contract_version": self.feature_contract_version,
            "completed": self.completed,
            "complete": False,
        }


def source_fingerprint(store: IntelligenceStore, horizon: datetime) -> str:
    """Fingerprint of the evidence visible up to ``horizon``.

    Derived from document and event identity rather than content, because the
    content hashes are already inside the snapshots. Its job is to detect a
    source history that changed underneath an incremental extension.

    Eligibility is part of that history. A correction recorded after a matrix
    was built changes which events belong in it, so the fingerprint has to move
    -- otherwise an incremental extension quietly keeps rows research is no
    longer allowed to count.
    """
    documents = store.documents_as_of(horizon)
    events = store.eligible_signals_as_of(horizon)
    digest = hashlib.sha256()
    digest.update(f"documents:{len(documents)}\n".encode())
    for document in documents:
        digest.update(f"{document.document_id}|{document.available_at.isoformat()}\n".encode())
    digest.update(f"events:{len(events)}\n".encode())
    for event in events:
        digest.update(f"{event.event_id}|{event.available_time.isoformat()}\n".encode())
    return digest.hexdigest()


class HistoricalFeatureMatrixBuilder:
    """Builds one point-in-time-safe row per origin, in bounded chunks."""

    def __init__(self, store: IntelligenceStore, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.store = store
        self.snapshots = SnapshotService(store)
        self.chunk_size = chunk_size

    # -- row construction -------------------------------------------------

    def _rows_for(
        self,
        origins: list[datetime],
        provider_versions: dict[str, str],
        configuration_fingerprint: str,
        mode: ReplayMode,
    ) -> tuple[list[dict[str, object]], list[str]]:
        build = (
            self.snapshots.build_many if mode is ReplayMode.REFERENCE else self.snapshots.build_many_bulk
        )
        snapshots = build(origins, provider_versions, configuration_fingerprint)

        runs = self.store.runs_up_to(origins[-1])
        run_times = [finished_at for finished_at, _ in runs]

        from bisect import bisect_right

        rows: list[dict[str, object]] = []
        for origin, snapshot in zip(origins, snapshots, strict=True):
            position = bisect_right(run_times, origin)
            latest_run = runs[position - 1][1] if position else None
            quality = latest_run.quality_summary if latest_run else {}
            rows.append(
                {
                    "forecast_origin": origin.isoformat(),
                    **{name: snapshot.features[name] for name in FEATURE_NAMES},
                    "provider_coverage_ratio": float(quality.get("provider_coverage_ratio", 0.0)),
                    "queries_failed": int(quality.get("queries_failed", 0)),
                    "source_stale_flag": int(
                        quality.get("source_stale_flag", 1 if latest_run is None else 0)
                    ),
                }
            )
        return rows, [snapshot.snapshot_id for snapshot in snapshots]

    # -- chunked build ----------------------------------------------------

    def build_chunks(
        self,
        origins: list[datetime],
        working_directory: str | Path,
        provider_versions: dict[str, str],
        configuration_fingerprint: str,
        *,
        mode: ReplayMode = ReplayMode.OPTIMIZED,
        resume: bool = True,
    ) -> tuple[list[dict[str, object]], list[str], list[ChunkResult]]:
        """Process origins in bounded chunks, resuming completed ones.

        Returns the assembled rows, the snapshot ids, and the chunk records.
        """
        validate_origin_schedule(origins)
        directory = Path(working_directory)
        directory.mkdir(parents=True, exist_ok=True)

        state = self._load_state(directory) if resume else None
        if state is not None and (
            state.origin_count != len(origins)
            or state.chunk_size != self.chunk_size
            or state.configuration_fingerprint != configuration_fingerprint
            or state.feature_contract_version != FEATURE_CONTRACT_VERSION
        ):
            # The in-progress run was for a different request. Resuming into it
            # would interleave two datasets, so start fresh instead.
            state = None
            self._clear_parts(directory)

        if state is None:
            state = MatrixBuildState(
                origin_count=len(origins),
                chunk_size=self.chunk_size,
                configuration_fingerprint=configuration_fingerprint,
                feature_contract_version=FEATURE_CONTRACT_VERSION,
            )
            self._clear_parts(directory)

        batches = [origins[i : i + self.chunk_size] for i in range(0, len(origins), self.chunk_size)]
        chunks: list[ChunkResult] = []
        rows: list[dict[str, object]] = []
        snapshot_ids: list[str] = []

        for index, batch in enumerate(batches):
            part_path = directory / f"{PART_PREFIX}{index:05d}.json"
            if index < len(state.completed) and part_path.exists():
                payload = json.loads(part_path.read_text(encoding="utf-8"))
                rows.extend(payload["rows"])
                snapshot_ids.extend(payload["snapshot_ids"])
                chunks.append(
                    ChunkResult(
                        index=index,
                        origin_start=batch[0],
                        origin_end=batch[-1],
                        row_count=len(payload["rows"]),
                        snapshot_ids=tuple(payload["snapshot_ids"]),
                        part_path=part_path,
                    )
                )
                continue

            chunk_rows, chunk_snapshots = self._rows_for(
                batch, provider_versions, configuration_fingerprint, mode
            )
            _write_atomic_json(part_path, {"rows": chunk_rows, "snapshot_ids": chunk_snapshots})

            rows.extend(chunk_rows)
            snapshot_ids.extend(chunk_snapshots)
            chunk = ChunkResult(
                index=index,
                origin_start=batch[0],
                origin_end=batch[-1],
                row_count=len(chunk_rows),
                snapshot_ids=tuple(chunk_snapshots),
                part_path=part_path,
            )
            chunks.append(chunk)

            state.completed = [c.to_dict() for c in chunks]
            _write_atomic_json(directory / PROGRESS_FILENAME, state.to_dict())

        return rows, snapshot_ids, chunks

    # -- resume bookkeeping ------------------------------------------------

    @staticmethod
    def _load_state(directory: Path) -> MatrixBuildState | None:
        """Read progress from a previous run, or None if there is none.

        A malformed progress file is treated as absent rather than fatal: the
        worst case is redoing work, and refusing to start because a scratch file
        is corrupt would be the wrong trade for a resumable job.
        """
        path = directory / PROGRESS_FILENAME
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return MatrixBuildState(
                origin_count=int(payload["origin_count"]),
                chunk_size=int(payload["chunk_size"]),
                configuration_fingerprint=str(payload["configuration_fingerprint"]),
                feature_contract_version=str(payload["feature_contract_version"]),
                completed=list(payload.get("completed", [])),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _clear_parts(directory: Path) -> None:
        """Discard a previous run's partial output.

        Called when the in-progress work belongs to a different request.
        Resuming into it would interleave two datasets in one output.
        """
        for part in sorted(directory.glob(f"{PART_PREFIX}*.json")):
            part.unlink(missing_ok=True)
        (directory / PROGRESS_FILENAME).unlink(missing_ok=True)

    # -- incremental extension --------------------------------------------

    def plan_extension(
        self,
        catalog: DatasetCatalog,
        existing: DatasetCatalogEntry,
        requested_origins: list[datetime],
        configuration_fingerprint: str,
        current_source_fingerprint: str,
    ) -> list[datetime]:
        """Origins that may legitimately be appended to ``existing``.

        Fails closed on any mismatch. Silently rebuilding history under a changed
        contract is the failure mode this exists to prevent -- it produces a file
        whose rows do not all mean the same thing, and nothing downstream can
        detect that.
        """
        validate_origin_schedule(requested_origins)

        if existing.feature_contract_version != FEATURE_CONTRACT_VERSION:
            raise ReplayIntegrityError(
                f"feature contract changed ({existing.feature_contract_version} -> "
                f"{FEATURE_CONTRACT_VERSION}); rebuild rather than extend"
            )
        if existing.configuration_fingerprint != configuration_fingerprint:
            raise ReplayIntegrityError("configuration fingerprint changed; rebuild rather than extend")
        if existing.source_fingerprint != current_source_fingerprint:
            raise ReplayIntegrityError(
                "source history changed for the covered window; rebuild rather than extend"
            )

        cadence = infer_frequency(requested_origins)
        if cadence is None or cadence.value != existing.origin_frequency:
            raise ReplayIntegrityError(
                f"origin cadence changed (dataset is {existing.origin_frequency}); rebuild rather than extend"
            )

        new_origins = [origin for origin in requested_origins if origin > existing.origin_end]
        if not new_origins:
            raise ReplayIntegrityError(
                f"nothing to extend: every requested origin is at or before {existing.origin_end.isoformat()}"
            )
        expected_first = existing.origin_end + cadence.step
        if new_origins[0] != expected_first:
            raise ReplayIntegrityError(
                f"extension must continue the cadence; expected first new origin "
                f"{expected_first.isoformat()}, got {new_origins[0].isoformat()}"
            )
        return new_origins


def membership_hash(snapshot_ids: list[str]) -> str:
    """Aggregate membership fingerprint over a dataset's snapshots."""
    digest = hashlib.sha256()
    for snapshot_id in snapshot_ids:
        digest.update(snapshot_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_catalog_entry(
    *,
    dataset_id: str,
    origins: list[datetime],
    frequency: OriginFrequency,
    row_count: int,
    configuration_fingerprint: str,
    source_fingerprint_value: str,
    snapshot_ids: list[str],
    extractor_versions: tuple[str, ...],
    output_format: str,
    output_file_hash: str,
    git_sha: str,
    chunk_count: int,
    mode: ReplayMode,
) -> DatasetCatalogEntry:
    return DatasetCatalogEntry(
        dataset_id=dataset_id,
        origin_start=origins[0],
        origin_end=origins[-1],
        origin_frequency=frequency.value,
        row_count=row_count,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        configuration_fingerprint=configuration_fingerprint,
        source_fingerprint=source_fingerprint_value,
        membership_hash=membership_hash(snapshot_ids),
        extractor_versions=extractor_versions,
        output_format=output_format,
        output_file_hash=output_file_hash,
        created_at=datetime.now(tz=timezone.utc),
        git_sha=git_sha,
        chunk_count=chunk_count,
        mode=mode.value,
    )


def _write_atomic_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return path


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "FEATURE_NAMES",
    "MATRIX_COLUMNS",
    "OPERATIONAL_COLUMNS",
    "PART_PREFIX",
    "PROGRESS_FILENAME",
    "ChunkResult",
    "HistoricalFeatureMatrixBuilder",
    "MatrixBuildState",
    "build_catalog_entry",
    "membership_hash",
    "source_fingerprint",
]
