"""Issuance and scoring services for pre-committed A3 evidence."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..data.snapshot import MarketSnapshot, frame_digest
from ..models import registry as model_registry
from ..models.base import TrainingWindow
from .contracts import (
    ForecastMode,
    ForwardForecastRecord,
    OutcomeRecord,
    RunManifest,
    SupportState,
    Timeliness,
)
from .ledger import EvidenceLedger, IntegrityError
from .registry import ShadowModelSpec, ShadowRegistry, canonical_json

SCORING_VERSION = "a3-scoring-v1"


def utc(value: datetime | pd.Timestamp) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return cast(datetime, stamp.tz_convert("UTC").to_pydatetime())


def source_sha(root: Path = Path.cwd()) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
    ).strip()


def environment_versions() -> dict[str, str]:
    packages = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "statsmodels"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return packages


def completed_frame(snapshot: MarketSnapshot, *, now: datetime) -> pd.DataFrame:
    """Exclude a UTC daily bar until the following midnight has passed."""
    now_utc = utc(now)
    if now_utc.tzinfo is None:
        raise ValueError("now must be timezone aware")
    frame = snapshot.frame.copy()
    availability = pd.DatetimeIndex(frame.index) + pd.Timedelta(days=1)
    permitted = frame.loc[availability <= pd.Timestamp(now_utc)]
    if permitted.empty:
        raise ValueError("no fully completed bars available")
    return permitted


def timeliness(registry: ShadowRegistry, *, origin: datetime, created_at: datetime) -> Timeliness:
    expected = origin + timedelta(
        days=1, seconds=int(registry.schedule["scheduled_issue_offset_seconds"])
    )
    latest = expected + timedelta(seconds=int(registry.schedule["max_lateness_seconds"]))
    return Timeliness.ON_TIME if created_at <= latest else Timeliness.LATE_FORECAST


def _support(frame: pd.DataFrame) -> tuple[SupportState, dict[str, Any]]:
    returns = frame["close"].pct_change().dropna()
    if len(returns) < 60:
        return SupportState.UNKNOWN, {"reason": "fewer_than_60_returns"}
    recent = float(returns.tail(30).std())
    historical = returns.iloc[:-30].rolling(30).std().dropna()
    if historical.empty:
        return SupportState.UNKNOWN, {"reason": "no_historical_volatility_support"}
    low, high = float(historical.quantile(0.01)), float(historical.quantile(0.99))
    state = SupportState.IN_SUPPORT if low <= recent <= high else SupportState.OUT_OF_SUPPORT
    return state, {
        "recent_volatility": recent,
        "training_p01": low,
        "training_p99": high,
        "missing_values": int(frame.isna().sum().sum()),
        "data_gaps": int(
            (pd.DatetimeIndex(frame.index).to_series().diff() > pd.Timedelta(days=1)).sum()
        ),
    }


def _forecast_payload(
    spec: ShadowModelSpec,
    registry: ShadowRegistry,
    snapshot: MarketSnapshot,
    frame: pd.DataFrame,
    *,
    created_at: datetime,
    code_sha: str,
    mode: ForecastMode,
    previous_hash: str | None,
) -> dict[str, Any]:
    missing = [column for column in spec.data_requirements if column not in frame]
    if missing:
        raise ValueError(f"missing model inputs: {missing}")
    model = model_registry.build(spec.model_id, **dict(spec.config))
    training_start_clock = time.perf_counter()
    model.fit(TrainingWindow(frame))
    training_seconds = time.perf_counter() - training_start_clock
    forecast_start_clock = time.perf_counter()
    result = model.predict(registry.horizon_bars)
    forecast_seconds = time.perf_counter() - forecast_start_clock
    origin = utc(frame.index[-1])
    target = utc(result.index[-1])
    origin_price = float(frame["close"].iloc[-1])
    point = float(result.point.iloc[-1])
    quantiles = None
    if result.lower is not None and result.upper is not None and result.interval_level is not None:
        tail = (1 - result.interval_level) / 2
        quantiles = {
            f"{tail:.6f}": float(result.lower.iloc[-1] / origin_price - 1),
            f"{1 - tail:.6f}": float(result.upper.iloc[-1] / origin_price - 1),
        }
    support, drift = _support(frame)
    body: dict[str, Any] = {
        "forecast_mode": mode.value,
        "forecast_origin": origin.isoformat(),
        "created_at": created_at.isoformat(),
        "instrument": registry.instrument,
        "horizon_bars": registry.horizon_bars,
        "target_timestamp": target.isoformat(),
        "model_id": spec.model_id,
        "model_version": spec.model_version,
        "model_registry_version": registry.registry_version,
        "training_start": utc(frame.index[0]).isoformat(),
        "training_end": origin.isoformat(),
        "data_cutoff": origin.isoformat(),
        "latest_observation_time": origin.isoformat(),
        "feature_contract_version": registry.feature_contract_version,
        "feature_snapshot_hash": frame_digest(frame),
        "training_data_hash": frame_digest(frame),
        "model_config_hash": sha256(canonical_json(dict(spec.config))).hexdigest(),
        "source_code_sha": code_sha,
        "origin_price": origin_price,
        "point_forecast": point,
        "expected_return": point / origin_price - 1,
        "return_quantiles": quantiles,
        "raw_direction_probability": None,
        "calibration_version": None,
        "promotion_status_at_issue": "UNPROMOTED",
        "data_freshness_seconds": (created_at - (origin + timedelta(days=1))).total_seconds(),
        "model_status": "OBSERVATION_ONLY",
        "support_state": support.value,
        "drift_diagnostics": drift,
        "training_seconds": training_seconds,
        "forecast_seconds": forecast_seconds,
        "timeliness": timeliness(registry, origin=origin, created_at=created_at).value,
        "previous_forecast_hash": previous_hash,
        "research_only": True,
    }
    digest = sha256(canonical_json(body)).hexdigest()
    body["forecast_payload_hash"] = digest
    body["forecast_id"] = f"a3-{digest[:24]}"
    ForwardForecastRecord.from_dict(body)
    return body


def _issue_run_unlocked(
    snapshot: MarketSnapshot,
    ledger: EvidenceLedger,
    *,
    registry: ShadowRegistry | None = None,
    now: datetime | None = None,
    code_sha: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    registry = registry or ShadowRegistry.load()
    created = utc(now or datetime.now(UTC))
    start = created
    frame = completed_frame(snapshot, now=created)
    if snapshot.ticker != registry.instrument:
        raise ValueError("snapshot instrument does not match shadow registry")
    code = code_sha or source_sha()
    mode = ForecastMode.UNCOMMITTED_DRY_RUN if dry_run else ForecastMode.FORWARD_SHADOW
    previous = None
    current = ledger.forecast_payloads()
    if current:
        previous = str(current[-1]["forecast_payload_hash"])
    attempted, succeeded, failures, payloads = [], [], {}, []
    for spec in registry.models:
        attempted.append(spec.model_id)
        try:
            payload = _forecast_payload(
                spec,
                registry,
                snapshot,
                frame,
                created_at=created,
                code_sha=code,
                mode=mode,
                previous_hash=previous,
            )
            payloads.append(payload)
            succeeded.append(spec.model_id)
            previous = str(payload["forecast_payload_hash"])
        except Exception as exc:
            failures[spec.model_id] = f"{type(exc).__name__}: {exc}"
    if dry_run:
        return {"mode": mode.value, "forecasts": payloads, "committed": False, "failures": failures}
    origin = utc(frame.index[-1])
    # The public entry point holds the run lock across calculation and persistence.
    if True:
        snapshot_dir = ledger.root / "snapshots" / frame_digest(frame)
        if snapshot_dir.exists():
            preserved = MarketSnapshot.load(snapshot_dir)
            if frame_digest(preserved.frame) != frame_digest(frame):
                raise IntegrityError("preserved snapshot conflict")
        else:
            preserved_snapshot = MarketSnapshot.build(
                frame,
                ticker=snapshot.ticker,
                provider=snapshot.manifest.provider,
                provider_version=snapshot.manifest.provider_version,
                retrieved_at=created,
                note="A3 immutable issuance snapshot",
                normalise=False,
            )
            preserved_snapshot.save(snapshot_dir)
        forecast_ids = []
        for payload in payloads:
            forecast_id, _ = ledger.append_forecast(payload)
            forecast_ids.append(forecast_id)
        end = datetime.now(UTC)
        manifest_body = {
            "run_id": f"a3-run-{sha256(canonical_json([origin.isoformat(), code, forecast_ids])).hexdigest()[:20]}",
            "scheduled_origin": origin.isoformat(),
            "actual_start": start.isoformat(),
            "actual_end": end.isoformat(),
            "source_sha": code,
            "registry_hash": registry.registry_hash,
            "market_data_fingerprint": frame_digest(frame),
            "models_attempted": attempted,
            "models_succeeded": succeeded,
            "models_failed": failures,
            "forecast_ids": forecast_ids,
            "lateness_state": timeliness(registry, origin=origin, created_at=created).value,
            "environment_versions": environment_versions(),
            "errors": list(failures.values()),
        }
        manifest_body["manifest_hash"] = sha256(canonical_json(manifest_body)).hexdigest()
        RunManifest.from_dict(manifest_body)
        ledger.append(
            ledger.manifests_path, "manifest", manifest_body
        )  # manifest is deliberately last
    return {"mode": mode.value, "forecasts": payloads, "manifest": manifest_body, "committed": True}


def issue_run(
    snapshot: MarketSnapshot,
    ledger: EvidenceLedger,
    *,
    registry: ShadowRegistry | None = None,
    now: datetime | None = None,
    code_sha: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return _issue_run_unlocked(
            snapshot, ledger, registry=registry, now=now, code_sha=code_sha, dry_run=True
        )
    with ledger.lock():
        return _issue_run_unlocked(
            snapshot, ledger, registry=registry, now=now, code_sha=code_sha, dry_run=False
        )


def record_missed(
    ledger: EvidenceLedger, *, scheduled_origin: datetime, model_ids: tuple[str, ...], reason: str
) -> str:
    """Record absence as evidence; never reconstruct a forward forecast."""
    body = {
        "scheduled_origin": utc(scheduled_origin).isoformat(),
        "model_ids": list(model_ids),
        "state": Timeliness.MISSED.value,
        "reason": reason,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    return ledger.append(ledger.schedule_path, "missed", body)


def score_pending(
    outcome_snapshot: MarketSnapshot,
    ledger: EvidenceLedger,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    retrieved = utc(now or datetime.now(UTC))
    outcomes = {item["forecast_id"] for item in ledger.outcome_payloads()}
    scored = pending = unavailable = 0
    frame = completed_frame(outcome_snapshot, now=retrieved)
    for forecast in ledger.forecast_payloads():
        if (
            forecast["forecast_id"] in outcomes
            or forecast["forecast_mode"] != ForecastMode.FORWARD_SHADOW.value
        ):
            continue
        target = datetime.fromisoformat(forecast["target_timestamp"])
        matured = target + timedelta(days=1)
        if retrieved < matured:
            pending += 1
            continue
        origin = datetime.fromisoformat(forecast["forecast_origin"])
        target_rows = frame.loc[
            (frame.index > pd.Timestamp(origin)) & (frame.index <= pd.Timestamp(target))
        ]
        if (
            len(target_rows) != int(forecast["horizon_bars"])
            or pd.Timestamp(target) not in target_rows.index
        ):
            unavailable += 1
            continue
        actual_price = float(target_rows["close"].iloc[-1])
        actual_return = actual_price / float(forecast["origin_price"]) - 1
        point_error = float(forecast["point_forecast"]) - actual_price
        body: dict[str, Any] = {
            "forecast_id": forecast["forecast_id"],
            "target_horizon_bars": forecast["horizon_bars"],
            "matured_at": matured.isoformat(),
            "outcome_data_retrieved_at": retrieved.isoformat(),
            "actual_return": actual_return,
            "actual_direction": int(np.sign(actual_return)),
            "actual_price": actual_price,
            "target_bar_timestamps": [utc(stamp).isoformat() for stamp in target_rows.index],
            "target_data_hash": frame_digest(target_rows),
            "target_provider": outcome_snapshot.manifest.provider,
            "target_definition_version": ShadowRegistry.load().target_definition_version,
            "scoring_version": SCORING_VERSION,
            "point_error": point_error,
            "squared_error": point_error**2,
            "direction_correct": bool(
                np.sign(float(forecast["expected_return"])) == np.sign(actual_return)
            ),
            "previous_outcome_hash": ledger.outcome_payloads()[-1]["outcome_payload_hash"]
            if ledger.outcome_payloads()
            else None,
        }
        digest = sha256(canonical_json(body)).hexdigest()
        body["outcome_payload_hash"] = digest
        body["outcome_id"] = f"a3-outcome-{digest[:24]}"
        OutcomeRecord.from_dict(body)
        ledger.append_outcome(body)
        scored += 1
    return {"scored": scored, "pending": pending, "unavailable": unavailable}
