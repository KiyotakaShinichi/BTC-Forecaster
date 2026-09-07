"""Run output and the manifest that makes it reproducible.

Everything written here is regenerable and therefore gitignored (ARTIFACTS.md).
What makes an output directory worth promoting to ``research/runs/<date>/`` is
``manifest.json``: it records the data snapshot hash, the full configuration,
every model's hyperparameters, the fold boundaries and the environment, so the
run can be reconstructed rather than merely admired.

Filenames are stable across runs so the dashboard and API can find them without
discovery logic.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..timebase import UTC

FORECAST_CSV = "forecast_results.csv"
SUMMARY_JSON = "forecast_summary.json"
BACKTEST_CSV = "backtest_results.csv"
BACKTEST_SUMMARY_CSV = "backtest_summary.csv"
HISTORY_CSV = "historical_prices.csv"
MANIFEST_JSON = "manifest.json"
DIAGNOSTICS_JSON = "diagnostics.json"

#: Legacy filename the pre-Track-A dashboard reads. Written as a copy of
#: FORECAST_CSV so an existing deployment keeps working across the change.
LEGACY_FORECAST_CSV = "nextgen_hybrid_forecast_results_montecarlo.csv"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


def environment_fingerprint() -> dict:
    """Versions that can change a numerical result between runs."""
    versions: dict[str, str | None] = {}
    for module in ("numpy", "pandas", "scipy", "statsmodels", "prophet", "xgboost", "arch"):
        try:
            versions[module] = getattr(__import__(module), "__version__", "unknown")
        except ImportError:
            versions[module] = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _git_commit(),
    }


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


@dataclass
class ArtifactWriter:
    """Writes a run's outputs into one directory."""

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def path(self, filename: str) -> Path:
        return self.output_dir / filename

    def write_json(self, filename: str, payload: dict) -> Path:
        target = self.path(filename)
        target.write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
        )
        return target

    def write_frame(self, filename: str, frame: pd.DataFrame, *, index: bool = True) -> Path:
        target = self.path(filename)
        frame.to_csv(target, index=index)
        return target

    # -- the specific artifacts ------------------------------------------

    def write_forecast(self, forecast: pd.DataFrame) -> list[Path]:
        primary = self.write_frame(FORECAST_CSV, forecast)
        legacy = self.write_frame(LEGACY_FORECAST_CSV, forecast)
        return [primary, legacy]

    def write_history(self, close: pd.Series, *, bars: int = 365) -> Path:
        tail = close.iloc[-bars:].to_frame(name="close")
        tail.index.name = "date"
        return self.write_frame(HISTORY_CSV, tail)

    def write_backtest(self, per_fold: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
        written = [self.write_frame(BACKTEST_CSV, per_fold, index=False)]
        if not summary.empty:
            written.append(self.write_frame(BACKTEST_SUMMARY_CSV, summary))
        return written

    def write_manifest(self, manifest: dict) -> Path:
        payload = {"generated_at": pd.Timestamp.now(tz=UTC).isoformat(), **manifest}
        return self.write_json(MANIFEST_JSON, payload)

    def write_summary(self, summary: dict) -> Path:
        return self.write_json(SUMMARY_JSON, summary)

    def write_diagnostics(self, diagnostics: dict) -> Path:
        return self.write_json(DIAGNOSTICS_JSON, diagnostics)

    def listing(self) -> list[dict]:
        return [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in sorted(self.output_dir.glob("*"))
            if path.is_file()
        ]


__all__ = [
    "BACKTEST_CSV",
    "BACKTEST_SUMMARY_CSV",
    "DIAGNOSTICS_JSON",
    "FORECAST_CSV",
    "HISTORY_CSV",
    "LEGACY_FORECAST_CSV",
    "MANIFEST_JSON",
    "SUMMARY_JSON",
    "ArtifactWriter",
    "environment_fingerprint",
]
