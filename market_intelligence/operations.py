from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WatermarkStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class Watermark(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider_id: str
    query_id: str
    last_successful_available_time: datetime | None = None
    last_retrieval_time: datetime
    last_document_id: str | None = None
    status: WatermarkStatus


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"


class ProviderHealth(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider_id: str
    state: HealthState
    last_success: datetime | None = None
    last_failure: datetime | None = None
    failure_count: int = 0
    latency_ms: float = 0.0
    documents_returned: int = 0
    stale_duration_seconds: float | None = None
    rate_limited: bool = False


class MissingnessReport(BaseModel):
    provider_available: bool
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    documents_seen: int = Field(ge=0)
    queries_successful: int = Field(ge=0)
    queries_failed: int = Field(ge=0)


class QuarantineRecord(BaseModel):
    record_id: str
    failure_reason: str
    provider: str
    retrieval_timestamp: datetime
    raw_response_hash: str
    record_type: str

    @staticmethod
    def from_raw(reason: str, provider: str, retrieved_at: datetime, raw: str, record_type: str) -> "QuarantineRecord":
        digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
        material = f"{provider}|{retrieved_at.isoformat()}|{digest}|{reason}"
        return QuarantineRecord(
            record_id=hashlib.sha256(material.encode()).hexdigest(),
            failure_reason=reason,
            provider=provider,
            retrieval_timestamp=retrieved_at,
            raw_response_hash=digest,
            record_type=record_type,
        )


class QualityIssue(BaseModel):
    code: str
    record_id: str | None = None
    detail: str


class QualityReport(BaseModel):
    valid: bool
    documents_checked: int
    events_checked: int
    issues: list[QualityIssue]


class IntelligenceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    snapshot_id: str
    forecast_origin: datetime
    document_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    features: dict[str, float]
    provider_versions: dict[str, str]
    extractor_versions: tuple[str, ...]
    configuration_fingerprint: str
    source_hashes: tuple[str, ...]
    feature_contract_version: str

    @staticmethod
    def create(
        forecast_origin: datetime,
        document_ids: list[str],
        event_ids: list[str],
        features: dict[str, float],
        provider_versions: dict[str, str],
        extractor_versions: list[str],
        configuration_fingerprint: str,
        source_hashes: list[str],
        feature_contract_version: str,
    ) -> "IntelligenceSnapshot":
        membership = {
            "forecast_origin": forecast_origin.isoformat(),
            "document_ids": sorted(document_ids),
            "event_ids": sorted(event_ids),
            "features": features,
            "provider_versions": provider_versions,
            "extractor_versions": sorted(set(extractor_versions)),
            "configuration_fingerprint": configuration_fingerprint,
            "source_hashes": sorted(source_hashes),
            "feature_contract_version": feature_contract_version,
        }
        snapshot_id = hashlib.sha256(json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return IntelligenceSnapshot(
            snapshot_id=snapshot_id,
            forecast_origin=forecast_origin,
            document_ids=tuple(membership["document_ids"]),
            event_ids=tuple(membership["event_ids"]),
            features=features,
            provider_versions=provider_versions,
            extractor_versions=tuple(membership["extractor_versions"]),
            configuration_fingerprint=configuration_fingerprint,
            source_hashes=tuple(membership["source_hashes"]),
            feature_contract_version=feature_contract_version,
        )


class RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime
    configuration_fingerprint: str
    providers_attempted: int
    queries_attempted: int
    documents_accepted: int
    documents_rejected: int
    events_accepted: int
    events_rejected: int
    quality_summary: dict[str, Any]
    watermark_changes: int
    software_source_sha: str
    status: RunStatus = RunStatus.SUCCESS
    provider_ids: tuple[str, ...] = ()

    def write_atomic(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return target


class QualityScoreboard(BaseModel):
    run_id: str
    created_at: datetime
    documents_accepted: int = 0
    duplicates_removed: int = 0
    events_accepted: int = 0
    events_rejected: int = 0
    orphan_events: int = 0
    future_availability_violations: int = 0
    invalid_schema_records: int = 0
    quarantined_records: int = 0
    provider_coverage_ratio: float = Field(ge=0.0, le=1.0)
    stale_providers: int = 0


class BackfillManifest(BaseModel):
    from_time: datetime
    to_time: datetime
    window_hours: int = Field(ge=1, le=24 * 31)
    completed_window_ends: tuple[datetime, ...] = ()
    max_windows: int = Field(ge=1, le=10_000)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class ReplayDatasetManifest(BaseModel):
    dataset_id: str
    feature_contract_version: str
    start_forecast_origin: datetime
    end_forecast_origin: datetime
    origin_count: int
    snapshot_fingerprints: tuple[str, ...]
    provider_versions: dict[str, str]
    extractor_versions: tuple[str, ...]
    configuration_fingerprint: str
    row_count: int
    columns: tuple[str, ...]
    file_hash: str
    git_sha: str
    format: str

    def write_atomic(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return target
