from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Callable

from .operations import BackfillManifest


class BackfillRunner:
    def __init__(self, run_window: Callable[[object, object], None]):
        self.run_window = run_window

    def run(self, manifest: BackfillManifest, progress_path: str | Path) -> BackfillManifest:
        target = Path(progress_path)
        if target.exists():
            saved = BackfillManifest.model_validate_json(target.read_text(encoding="utf-8"))
            if (saved.from_time, saved.to_time, saved.window_hours) != (manifest.from_time, manifest.to_time, manifest.window_hours):
                raise ValueError("existing backfill manifest does not match requested bounds")
            manifest = saved.model_copy(update={"max_windows": manifest.max_windows})
        completed = set(manifest.completed_window_ends)
        cursor, windows = manifest.from_time, 0
        while cursor < manifest.to_time and windows < manifest.max_windows:
            end = min(cursor + timedelta(hours=manifest.window_hours), manifest.to_time)
            if end not in completed:
                self.run_window(cursor, end)
                completed.add(end)
                updated = manifest.model_copy(update={"completed_window_ends": tuple(sorted(completed))})
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                temporary.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
                os.replace(temporary, target)
                manifest = updated
            cursor, windows = end, windows + 1
        return manifest
