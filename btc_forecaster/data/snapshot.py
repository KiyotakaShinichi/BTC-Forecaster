"""Deterministic market-data snapshots with provenance.

A forecast is only reproducible if the data behind it is. ``yfinance`` returns a
different series every day, and silently revises history; a result produced
against "BTC-USD, whatever Yahoo said that afternoon" cannot be re-derived.

A snapshot pins the data and records how it was obtained:

    data.csv        the frame itself, in a canonical serialisation
    manifest.json   ticker, provider, retrieval time, window, rows, schema,
                    timezone and the SHA-256 of data.csv

Loading verifies the hash, so a corrupted or hand-edited snapshot fails loudly
rather than quietly changing a result. The snapshot directory is gitignored
(see ARTIFACTS.md) -- the manifest is small enough to travel with a promoted
research run and is what makes that run reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..timebase import UTC, to_utc_timestamp
from .contracts import SCHEMA_VERSION, normalise_market_frame, validate_market_frame

#: Fixed precision for the canonical serialisation. 12 significant digits round
#: trips float64 for price and volume magnitudes while keeping the hash stable
#: across platforms, which ``repr``-based output does not guarantee.
_FLOAT_FORMAT = "%.12g"

MANIFEST_FILENAME = "manifest.json"
DATA_FILENAME = "data.csv"


class SnapshotIntegrityError(RuntimeError):
    """Raised when a snapshot's contents do not match its manifest."""


def canonical_bytes(frame: pd.DataFrame) -> bytes:
    """Serialise a market frame deterministically.

    The same frame must produce the same bytes on every platform and pandas
    version, because these bytes are what gets hashed.
    """
    ordered = frame.sort_index()
    ordered = ordered[sorted(ordered.columns)]
    csv = ordered.to_csv(
        float_format=_FLOAT_FORMAT,
        date_format="%Y-%m-%dT%H:%M:%S%z",
        lineterminator="\n",
    )
    return csv.encode("utf-8")


def frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_bytes(frame)).hexdigest()


@dataclass(frozen=True)
class SnapshotManifest:
    """Everything needed to say what a snapshot is and where it came from."""

    ticker: str
    provider: str
    retrieved_at: str
    start: str
    end: str
    rows: int
    columns: tuple[str, ...]
    timezone: str
    frequency: str
    sha256: str
    schema_version: str = SCHEMA_VERSION
    provider_version: str | None = None
    missing_bars: int = 0
    note: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> SnapshotManifest:
        data = dict(payload)
        data["columns"] = tuple(data.get("columns", ()))
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class MarketSnapshot:
    """A validated market frame plus its provenance."""

    frame: pd.DataFrame
    manifest: SnapshotManifest

    @classmethod
    def build(
        cls,
        frame: pd.DataFrame,
        *,
        ticker: str,
        provider: str,
        provider_version: str | None = None,
        retrieved_at: datetime | pd.Timestamp | None = None,
        note: str | None = None,
        normalise: bool = True,
    ) -> MarketSnapshot:
        prepared = normalise_market_frame(frame) if normalise else frame
        report = validate_market_frame(prepared)

        retrieval = (
            to_utc_timestamp(retrieved_at)
            if retrieved_at is not None
            else pd.Timestamp.now(tz=UTC)
        )

        manifest = SnapshotManifest(
            ticker=ticker,
            provider=provider,
            provider_version=provider_version,
            retrieved_at=retrieval.isoformat(),
            start=report.start.isoformat(),
            end=report.end.isoformat(),
            rows=report.rows,
            columns=tuple(prepared.columns),
            timezone="UTC",
            frequency="D",
            sha256=frame_digest(prepared),
            missing_bars=len(report.missing_bars),
            note=note,
        )
        return cls(frame=prepared, manifest=manifest)

    # -- persistence ------------------------------------------------------

    def save(self, directory: Path | str) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / DATA_FILENAME).write_bytes(canonical_bytes(self.frame))
        (path / MANIFEST_FILENAME).write_text(self.manifest.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path | str, *, verify: bool = True) -> MarketSnapshot:
        path = Path(directory)
        manifest_path = path / MANIFEST_FILENAME
        data_path = path / DATA_FILENAME
        if not manifest_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"no snapshot at {path}")

        manifest = SnapshotManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        frame = read_canonical_csv(data_path)

        if verify:
            digest = frame_digest(frame)
            if digest != manifest.sha256:
                raise SnapshotIntegrityError(
                    f"snapshot at {path} does not match its manifest\n"
                    f"  manifest sha256: {manifest.sha256}\n"
                    f"  actual   sha256: {digest}"
                )
        validate_market_frame(frame)
        return cls(frame=frame, manifest=manifest)

    # -- convenience ------------------------------------------------------

    @property
    def ticker(self) -> str:
        return self.manifest.ticker

    def __len__(self) -> int:
        return len(self.frame)

    def describe(self) -> str:
        m = self.manifest
        return (
            f"{m.ticker} via {m.provider} | {m.rows} bars "
            f"{m.start[:10]}..{m.end[:10]} | sha256={m.sha256[:12]} "
            f"| retrieved {m.retrieved_at[:19]}Z"
        )


def read_canonical_csv(path: Path | str) -> pd.DataFrame:
    """Read a snapshot's ``data.csv`` back into a contract-satisfying frame."""
    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(UTC)
    else:
        frame.index = frame.index.tz_convert(UTC)
    frame.index = frame.index.normalize()
    frame.index.name = "date"
    return frame.sort_index()


__all__ = [
    "DATA_FILENAME",
    "MANIFEST_FILENAME",
    "MarketSnapshot",
    "SnapshotIntegrityError",
    "SnapshotManifest",
    "canonical_bytes",
    "frame_digest",
    "read_canonical_csv",
]
