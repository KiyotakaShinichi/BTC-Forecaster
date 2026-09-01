"""B4.48 / B4.49 / B4.50 — run artifacts, resource telemetry, and the manifest.

Same discipline as the B3.1 dataset manifest, for the same reason: **the manifest
is written last**, so its presence means every artifact before it completed. A
run directory without a manifest holds scratch, not results.

Two rules beyond that:

**A completed run is never overwritten (B4.49).** Each run gets its own directory
named by its run id. Re-running with the same inputs produces the same id and
refuses to clobber; re-running with different inputs produces a different id and
sits alongside. Negative results are the main output of this track, and a
negative result that can be quietly replaced is worth very little.

**Telemetry is recorded, not optimised against (B4.48).** Timings and counts go
in the manifest so a later run can be compared. B3.1 already established that
replay is not the bottleneck; nothing here should be tuned until a measurement
says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from pydantic import BaseModel, ConfigDict

from ..features import FEATURE_CONTRACT_VERSION
from .contracts import B4DataError, EvidenceTier

MANIFEST_FILENAME = "run_manifest.json"
RUN_CONTRACT_VERSION = "b4-run-v1"


class ResourceTelemetry(BaseModel):
    """B4.48. What the run cost, recorded for comparison rather than tuning."""

    model_config = ConfigDict(frozen=True)

    stage_seconds: dict[str, float]
    peak_python_mb: float | None
    origin_count: int
    event_count: int
    market_bar_count: int


class RunManifest(BaseModel):
    """B4.50. Everything needed to identify, audit and reproduce one run."""

    model_config = ConfigDict(frozen=True)

    run_contract_version: str = RUN_CONTRACT_VERSION
    run_id: str
    created_at: datetime
    source_git_sha: str
    #: The B3.1 intelligence dataset this run's features came from, when one
    #: was built. None when no intelligence corpus existed to build from.
    intelligence_dataset_id: str | None
    intelligence_row_count: int
    target_dataset_fingerprint: str
    feature_contract_version: str
    target_contract_version: str
    preregistration_hash: str
    study_spec_hashes: dict[str, str]
    market_series_fingerprints: dict[str, str]
    origin_count: int
    event_counts: dict[str, int]
    #: How many candidate signals fell in each evidence tier. The headline
    #: number of this track: a run where everything is RETROSPECTIVE_ONLY or
    #: absent has not validated anything.
    evidence_tier_counts: dict[str, int]
    result_hashes: dict[str, str]
    software_versions: dict[str, str]
    telemetry: ResourceTelemetry
    notes: tuple[str, ...] = ()

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()


class RunRecorder:
    """Accumulates a run's artifacts, then writes the manifest last."""

    def __init__(self, directory: str | Path, *, run_id: str) -> None:
        self.directory = Path(directory)
        self.run_id = run_id
        self.result_hashes: dict[str, str] = {}
        self.stage_seconds: dict[str, float] = {}
        self._peak_bytes = 0
        if (self.directory / MANIFEST_FILENAME).exists():
            raise B4DataError(
                f"{self.directory} already holds a completed run; B4 runs are never overwritten"
            )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one stage and fold its peak allocation into the run's."""
        tracing = not tracemalloc.is_tracing()
        if tracing:
            tracemalloc.start()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            self._peak_bytes = max(self._peak_bytes, peak)
            if tracing:
                tracemalloc.stop()

    def write_json(self, name: str, payload: Any) -> Path:
        """Write one artifact atomically and record its content hash."""
        target = self.directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
        self.result_hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        return target

    @property
    def peak_python_mb(self) -> float | None:
        return self._peak_bytes / 1_048_576 if self._peak_bytes else None

    def finish(
        self,
        *,
        source_git_sha: str | None = None,
        intelligence_dataset_id: str | None,
        intelligence_row_count: int,
        target_dataset_fingerprint: str,
        target_contract_version: str,
        preregistration_hash: str,
        study_spec_hashes: dict[str, str],
        market_series_fingerprints: dict[str, str],
        origin_count: int,
        event_counts: dict[str, int],
        evidence_tier_counts: dict[str, int],
        market_bar_count: int,
        notes: Sequence[str] = (),
    ) -> RunManifest:
        """Write the manifest. Nothing else may be written to the run after this."""
        manifest = RunManifest(
            run_id=self.run_id,
            created_at=datetime.now(timezone.utc),
            source_git_sha=source_git_sha or git_sha(),
            intelligence_dataset_id=intelligence_dataset_id,
            intelligence_row_count=intelligence_row_count,
            target_dataset_fingerprint=target_dataset_fingerprint,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            target_contract_version=target_contract_version,
            preregistration_hash=preregistration_hash,
            study_spec_hashes=dict(study_spec_hashes),
            market_series_fingerprints=dict(market_series_fingerprints),
            origin_count=origin_count,
            event_counts=dict(event_counts),
            evidence_tier_counts=dict(evidence_tier_counts),
            result_hashes=dict(self.result_hashes),
            software_versions=software_versions(),
            telemetry=ResourceTelemetry(
                stage_seconds=dict(self.stage_seconds),
                peak_python_mb=self.peak_python_mb,
                origin_count=origin_count,
                event_count=sum(event_counts.values()),
                market_bar_count=market_bar_count,
            ),
            notes=tuple(notes),
        )
        target = self.directory / MANIFEST_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = manifest.model_dump(mode="json")
        payload["manifest_hash"] = manifest.content_hash()
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        return manifest


def read_manifest(path: str | Path) -> RunManifest:
    """Load a run manifest and verify it has not been edited since."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    declared = payload.pop("manifest_hash", None)
    manifest = RunManifest.model_validate(payload)
    if declared is not None and declared != manifest.content_hash():
        raise B4DataError(f"{path}: the manifest hash does not match its contents")
    return manifest


def verify_run(directory: str | Path) -> RunManifest:
    """Confirm a run directory is complete and its artifacts match the manifest.

    Re-hashes the files rather than trusting the manifest's own record, which
    is the only version of this check worth having.
    """
    root = Path(directory)
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise B4DataError(
            f"{root} has no {MANIFEST_FILENAME}: the run did not complete, so its files are scratch"
        )
    manifest = read_manifest(manifest_path)
    for name, expected in manifest.result_hashes.items():
        artifact = root / name
        if not artifact.exists():
            raise B4DataError(f"{root}: manifest references {name}, which is missing")
        actual = hashlib.sha256(artifact.read_text(encoding="utf-8").encode()).hexdigest()
        if actual != expected:
            raise B4DataError(f"{root}: {name} does not match the hash recorded in the manifest")
    return manifest


def make_run_id(*components: str) -> str:
    """Deterministic id from the inputs, so the same run reproduces its name."""
    digest = hashlib.sha256("|".join(components).encode()).hexdigest()
    return f"b4-{digest[:16]}"


def evidence_tier_counts(tiers: Sequence[EvidenceTier | None]) -> dict[str, int]:
    counts = {tier.value: 0 for tier in EvidenceTier}
    counts["UNKNOWN"] = 0
    for tier in tiers:
        counts[tier.value if tier else "UNKNOWN"] += 1
    return counts


def software_versions() -> dict[str, str]:
    import platform  # noqa: PLC0415 -- only needed when a manifest is written

    versions = {"python": platform.python_version()}
    for module_name in ("pydantic", "duckdb", "yfinance"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "not installed"
    return versions


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


__all__ = [
    "MANIFEST_FILENAME",
    "RUN_CONTRACT_VERSION",
    "ResourceTelemetry",
    "RunManifest",
    "RunRecorder",
    "evidence_tier_counts",
    "git_sha",
    "make_run_id",
    "read_manifest",
    "software_versions",
    "verify_run",
]
