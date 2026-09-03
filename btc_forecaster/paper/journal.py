"""Append-only, hash-chained JSONL paper-trade journal."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, cast


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Enum):
        return value.value
    return value


class AppendOnlyJournal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event_type: str, payload: Any) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        previous = self._last_hash()
        body = {"event_type": event_type, "payload": _jsonable(payload), "previous_hash": previous}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
        digest = sha256(encoded.encode()).hexdigest()
        record = {**body, "record_hash": digest}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return digest

    def read_verified(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        previous = None
        for record in records:
            digest = record.pop("record_hash")
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if (
                record["previous_hash"] != previous
                or sha256(encoded.encode()).hexdigest() != digest
            ):
                raise ValueError("journal integrity failure")
            record["record_hash"] = digest
            previous = digest
        return records

    def _last_hash(self) -> str | None:
        records = self.read_verified()
        return records[-1]["record_hash"] if records else None
