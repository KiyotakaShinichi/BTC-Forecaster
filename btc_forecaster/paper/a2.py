"""Load-bearing bridge to the frozen A2 promotion evidence."""

from __future__ import annotations

import csv
from pathlib import Path

DEFAULT_PROMOTION_PATH = Path("research/runs/2026-08-29-a2-benchmark/promotion.csv")


def promoted_models(path: Path = DEFAULT_PROMOTION_PATH) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return tuple(row["model"] for row in rows if row["decision"] == "PROMOTE")


def current_live_permission(path: Path = DEFAULT_PROMOTION_PATH) -> dict[str, object]:
    promoted = promoted_models(path)
    return {
        "promoted_models": promoted,
        "live_candidates": 0,
        "allowed_live_leverage": 0.0,
        "live_trading_eligible": False,
    }
