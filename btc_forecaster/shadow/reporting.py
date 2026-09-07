"""Status, audit, integrity, and deterministic export for A3 evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from ..data.snapshot import MarketSnapshot, frame_digest
from .evaluation import forward_report
from .ledger import EvidenceLedger, IntegrityError
from .registry import ShadowRegistry, canonical_json


def status(ledger: EvidenceLedger, registry: ShadowRegistry | None = None) -> dict[str, Any]:
    registry = registry or ShadowRegistry.load()
    forecasts = ledger.forecast_payloads()
    outcomes = ledger.outcome_payloads()
    origins = sorted({item["forecast_origin"] for item in forecasts})
    return {
        "registry_version": registry.registry_version,
        "registry_hash": registry.registry_hash,
        "latest_scheduled_origin": origins[-1] if origins else None,
        "latest_completed_origin": max(
            (
                item["forecast_origin"]
                for item in forecasts
                if any(outcome["forecast_id"] == item["forecast_id"] for outcome in outcomes)
            ),
            default=None,
        ),
        "pending_outcomes": len(forecasts) - len(outcomes),
        "matured_outcomes": len(outcomes),
        "missed_forecasts": sum(
            record["payload"].get("state") == "MISSED"
            for record in ledger.records(ledger.schedule_path)
        ),
        "late_forecasts": sum(item["timeliness"] == "LATE_FORECAST" for item in forecasts),
        **forward_report(ledger),
        "trading_recommendation": None,
    }


def audit_forecast(ledger: EvidenceLedger, forecast_id: str) -> dict[str, Any]:
    matches = [item for item in ledger.forecast_payloads() if item["forecast_id"] == forecast_id]
    if len(matches) != 1:
        raise KeyError(f"forecast {forecast_id!r} not found uniquely")
    outcomes = [item for item in ledger.outcome_payloads() if item["forecast_id"] == forecast_id]
    return {
        "forecast": matches[0],
        "outcome": outcomes[0] if outcomes else None,
        "integrity_status": "VERIFIED",
    }


def verify(ledger: EvidenceLedger, registry: ShadowRegistry | None = None) -> dict[str, Any]:
    registry = registry or ShadowRegistry.load()
    counts = ledger.verify()
    forecasts = ledger.forecast_payloads()
    previous_forecast = None
    seen_keys: set[tuple[str, str, str]] = set()
    for forecast in forecasts:
        key = (forecast["model_id"], forecast["forecast_origin"], forecast["forecast_mode"])
        if key in seen_keys:
            raise IntegrityError("duplicate forecast key")
        seen_keys.add(key)
        expected = forecast["forecast_payload_hash"]
        body = {
            key: value
            for key, value in forecast.items()
            if key not in {"forecast_id", "forecast_payload_hash"}
        }
        if (
            sha256(canonical_json(body)).hexdigest() != expected
            or forecast["forecast_id"] != f"a3-{expected[:24]}"
        ):
            raise IntegrityError("forecast payload hash mismatch")
        if forecast["previous_forecast_hash"] != previous_forecast:
            raise IntegrityError("forecast payload chain mismatch")
        previous_forecast = expected
        snapshot_path = ledger.root / "snapshots" / forecast["training_data_hash"]
        if not snapshot_path.exists():
            raise IntegrityError("forecast snapshot missing")
        preserved = MarketSnapshot.load(snapshot_path)
        if frame_digest(preserved.frame) != forecast["training_data_hash"]:
            raise IntegrityError("forecast data fingerprint mismatch")
    previous_outcome = None
    for outcome in ledger.outcome_payloads():
        expected = outcome["outcome_payload_hash"]
        body = {
            key: value
            for key, value in outcome.items()
            if key not in {"outcome_id", "outcome_payload_hash"}
        }
        if (
            sha256(canonical_json(body)).hexdigest() != expected
            or outcome["outcome_id"] != f"a3-outcome-{expected[:24]}"
        ):
            raise IntegrityError("outcome payload hash mismatch")
        if outcome["previous_outcome_hash"] != previous_outcome:
            raise IntegrityError("outcome payload chain mismatch")
        previous_outcome = expected
    manifests = [record["payload"] for record in ledger.records(ledger.manifests_path)]
    for manifest in manifests:
        if manifest["registry_hash"] != registry.registry_hash:
            raise IntegrityError("manifest registry hash mismatch")
        expected_manifest = manifest["manifest_hash"]
        manifest_body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if sha256(canonical_json(manifest_body)).hexdigest() != expected_manifest:
            raise IntegrityError("manifest payload hash mismatch")
        forecast_ids = {item["forecast_id"] for item in ledger.forecast_payloads()}
        if not set(manifest["forecast_ids"]).issubset(forecast_ids):
            raise IntegrityError("manifest references missing forecast")
    for outcome in ledger.outcome_payloads():
        forecast = audit_forecast(ledger, outcome["forecast_id"])["forecast"]
        if (
            outcome["outcome_data_retrieved_at"] < outcome["matured_at"]
            or outcome["target_horizon_bars"] != forecast["horizon_bars"]
        ):
            raise IntegrityError("outcome timing or horizon invalid")
    return {**counts, "registry_hash": registry.registry_hash, "integrity": "VERIFIED"}


def export(
    ledger: EvidenceLedger, destination: Path, registry: ShadowRegistry | None = None
) -> Path:
    registry = registry or ShadowRegistry.load()
    payload = {
        "registry": {"version": registry.registry_version, "hash": registry.registry_hash},
        "forecasts": ledger.forecast_payloads(),
        "outcomes": ledger.outcome_payloads(),
        "manifests": [record["payload"] for record in ledger.records(ledger.manifests_path)],
        "integrity": verify(ledger, registry),
        "metrics": forward_report(ledger),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json(payload) + b"\n")
    return destination
