"""Immutable evidence contracts for A3 forward shadow validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ForecastMode(str, Enum):
    FORWARD_SHADOW = "FORWARD_SHADOW"
    RETROSPECTIVE_REPLAY = "RETROSPECTIVE_REPLAY"
    UNCOMMITTED_DRY_RUN = "UNCOMMITTED_DRY_RUN"


class Timeliness(str, Enum):
    ON_TIME = "ON_TIME"
    LATE_FORECAST = "LATE_FORECAST"
    MISSED = "MISSED"


class SupportState(str, Enum):
    IN_SUPPORT = "IN_SUPPORT"
    OUT_OF_SUPPORT = "OUT_OF_SUPPORT"
    UNKNOWN = "UNKNOWN"


class Readiness(str, Enum):
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    INSUFFICIENT_FORWARD_EVIDENCE = "INSUFFICIENT_FORWARD_EVIDENCE"
    FAILED_FORWARD_VALIDATION = "FAILED_FORWARD_VALIDATION"
    READY_FOR_PROMOTION_REVIEW = "READY_FOR_PROMOTION_REVIEW"


def require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


@dataclass(frozen=True)
class DataProvenance:
    provider: str
    retrieved_at: datetime
    latest_bar_timestamp: datetime
    data_cutoff: datetime
    row_count: int
    data_hash: str

    def __post_init__(self) -> None:
        for name in ("retrieved_at", "latest_bar_timestamp", "data_cutoff"):
            require_utc(getattr(self, name), name)
        if self.latest_bar_timestamp > self.data_cutoff:
            raise ValueError("latest bar is beyond data cutoff")
        if self.data_cutoff > self.retrieved_at:
            raise ValueError("data cutoff cannot be in the future at retrieval")


@dataclass(frozen=True)
class ForwardForecastRecord:
    forecast_id: str
    forecast_mode: ForecastMode
    forecast_origin: datetime
    created_at: datetime
    instrument: str
    horizon_bars: int
    target_timestamp: datetime
    model_id: str
    model_version: str
    model_registry_version: str
    training_start: datetime
    training_end: datetime
    data_cutoff: datetime
    latest_observation_time: datetime
    feature_contract_version: str
    feature_snapshot_hash: str
    training_data_hash: str
    model_config_hash: str
    source_code_sha: str
    origin_price: float
    point_forecast: float
    expected_return: float
    return_quantiles: dict[str, float] | None
    raw_direction_probability: float | None
    calibration_version: str | None
    promotion_status_at_issue: str
    data_freshness_seconds: float
    model_status: str
    support_state: SupportState
    drift_diagnostics: dict[str, Any]
    training_seconds: float
    forecast_seconds: float
    timeliness: Timeliness
    forecast_payload_hash: str
    previous_forecast_hash: str | None
    research_only: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        for name in (
            "forecast_origin",
            "created_at",
            "target_timestamp",
            "training_start",
            "training_end",
            "data_cutoff",
            "latest_observation_time",
        ):
            require_utc(getattr(self, name), name)
        if self.latest_observation_time > self.data_cutoff:
            raise ValueError("forecast consumed data beyond cutoff")
        if self.training_end != self.latest_observation_time:
            raise ValueError("training end must equal latest observation")
        if self.target_timestamp <= self.forecast_origin:
            raise ValueError("target must follow forecast origin")
        if (
            self.forecast_mode is ForecastMode.FORWARD_SHADOW
            and self.created_at < self.forecast_origin
        ):
            raise ValueError("official forecast cannot be backdated")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ForwardForecastRecord:
        data = dict(payload)
        for name in (
            "forecast_origin",
            "created_at",
            "target_timestamp",
            "training_start",
            "training_end",
            "data_cutoff",
            "latest_observation_time",
        ):
            data[name] = datetime.fromisoformat(data[name])
        data["forecast_mode"] = ForecastMode(data["forecast_mode"])
        data["support_state"] = SupportState(data["support_state"])
        data["timeliness"] = Timeliness(data["timeliness"])
        data.pop("research_only", None)
        return cls(**data)


@dataclass(frozen=True)
class OutcomeRecord:
    outcome_id: str
    forecast_id: str
    target_horizon_bars: int
    matured_at: datetime
    outcome_data_retrieved_at: datetime
    actual_return: float
    actual_direction: int
    actual_price: float
    target_bar_timestamps: tuple[datetime, ...]
    target_data_hash: str
    target_provider: str
    target_definition_version: str
    scoring_version: str
    point_error: float
    squared_error: float
    direction_correct: bool
    previous_outcome_hash: str | None
    outcome_payload_hash: str

    def __post_init__(self) -> None:
        require_utc(self.matured_at, "matured_at")
        require_utc(self.outcome_data_retrieved_at, "outcome_data_retrieved_at")
        if self.outcome_data_retrieved_at < self.matured_at:
            raise ValueError("outcome retrieved before maturity")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OutcomeRecord:
        data = dict(payload)
        data["matured_at"] = datetime.fromisoformat(data["matured_at"])
        data["outcome_data_retrieved_at"] = datetime.fromisoformat(
            data["outcome_data_retrieved_at"]
        )
        data["target_bar_timestamps"] = tuple(
            datetime.fromisoformat(item) for item in data["target_bar_timestamps"]
        )
        return cls(**data)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    scheduled_origin: datetime
    actual_start: datetime
    actual_end: datetime
    source_sha: str
    registry_hash: str
    market_data_fingerprint: str
    models_attempted: tuple[str, ...]
    models_succeeded: tuple[str, ...]
    models_failed: dict[str, str]
    forecast_ids: tuple[str, ...]
    lateness_state: Timeliness
    environment_versions: dict[str, str]
    errors: tuple[str, ...]
    manifest_hash: str

    def __post_init__(self) -> None:
        for name in ("scheduled_origin", "actual_start", "actual_end"):
            require_utc(getattr(self, name), name)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunManifest:
        data = dict(payload)
        for name in ("scheduled_origin", "actual_start", "actual_end"):
            data[name] = datetime.fromisoformat(data[name])
        for name in ("models_attempted", "models_succeeded", "forecast_ids", "errors"):
            data[name] = tuple(data[name])
        data["lateness_state"] = Timeliness(data["lateness_state"])
        return cls(**data)
