"""Standard-library append-only, hash-chained A3 evidence ledger."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

from .registry import canonical_json


class IntegrityError(RuntimeError):
    pass


class LockHeldError(RuntimeError):
    pass


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


class EvidenceLedger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.forecasts_path = root / "forecasts.jsonl"
        self.outcomes_path = root / "outcomes.jsonl"
        self.manifests_path = root / "manifests.jsonl"
        self.schedule_path = root / "schedule.jsonl"
        self.lock_path = root / ".shadow-run.lock"

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise LockHeldError("official shadow run lock is held") from exc
        try:
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)

    def records(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        previous = None
        for record in records:
            stored = record.get("record_hash")
            body = {key: value for key, value in record.items() if key != "record_hash"}
            if (
                body.get("previous_record_hash") != previous
                or sha256(canonical_json(body)).hexdigest() != stored
            ):
                raise IntegrityError(f"hash-chain failure in {path.name}")
            previous = stored
        return records

    def append(self, path: Path, kind: str, payload: Any) -> str:
        records = self.records(path)
        body = {
            "kind": kind,
            "payload": jsonable(payload),
            "previous_record_hash": records[-1]["record_hash"] if records else None,
        }
        record_hash = sha256(canonical_json(body)).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({**body, "record_hash": record_hash}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record_hash

    def forecast_payloads(self) -> list[dict[str, Any]]:
        return [record["payload"] for record in self.records(self.forecasts_path)]

    def outcome_payloads(self) -> list[dict[str, Any]]:
        return [record["payload"] for record in self.records(self.outcomes_path)]

    def append_forecast(self, payload: dict[str, Any]) -> tuple[str, bool]:
        existing = [
            item
            for item in self.forecast_payloads()
            if item["model_id"] == payload["model_id"]
            and item["forecast_origin"] == payload["forecast_origin"]
            and item["forecast_mode"] == payload["forecast_mode"]
        ]
        if existing:
            excluded = {
                "forecast_id",
                "forecast_payload_hash",
                "created_at",
                "data_freshness_seconds",
                "training_seconds",
                "forecast_seconds",
                "timeliness",
                "previous_forecast_hash",
            }
            original_content = {
                key: value for key, value in existing[0].items() if key not in excluded
            }
            rerun_content = {key: value for key, value in payload.items() if key not in excluded}
            if canonical_json(original_content) == canonical_json(rerun_content):
                return str(existing[0]["forecast_id"]), False
            raise IntegrityError("conflicting forecast for the same model/origin/mode")
        self.append(self.forecasts_path, "forecast", payload)
        return str(payload["forecast_id"]), True

    def append_outcome(self, payload: dict[str, Any]) -> tuple[str, bool]:
        existing = [
            item
            for item in self.outcome_payloads()
            if item["forecast_id"] == payload["forecast_id"]
        ]
        if existing:
            if existing[0]["outcome_payload_hash"] == payload["outcome_payload_hash"]:
                return str(existing[0]["outcome_id"]), False
            raise IntegrityError("conflicting outcome for forecast")
        self.append(self.outcomes_path, "outcome", payload)
        return str(payload["outcome_id"]), True

    def verify(self) -> dict[str, int]:
        forecasts = self.records(self.forecasts_path)
        outcomes = self.records(self.outcomes_path)
        manifests = self.records(self.manifests_path)
        ids = {record["payload"]["forecast_id"] for record in forecasts}
        for outcome in outcomes:
            if outcome["payload"]["forecast_id"] not in ids:
                raise IntegrityError("outcome references unknown forecast")
        return {"forecasts": len(forecasts), "outcomes": len(outcomes), "manifests": len(manifests)}
