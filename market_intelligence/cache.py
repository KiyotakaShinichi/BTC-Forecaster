from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class DeterministicCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(query: str, start: datetime, end: datetime, provider: str, extractor_version: str) -> str:
        material = json.dumps({"query": query, "start": start.isoformat(), "end": end.isoformat(),
            "provider": provider, "extractor_version": extractor_version}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, key: str, value: Any) -> Path:
        path, temporary = self.root / f"{key}.json", self.root / f".{key}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
        return path
