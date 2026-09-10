"""Immutable inputs, canonical outputs, and the digest that ties them together.

Two problems, one module.

**The input is not immutable.** yfinance revises history silently: A6 and A2
pulled nearly the same span and got different bytes. So this module does not
pretend a snapshot is permanent. It records exactly which bytes were used --
the canonical-frame digest that :mod:`btc_forecaster.data.snapshot` already
verifies on load, and the raw file hash -- and it refuses to call a rerun a
reproduction when the input hash differs from the record. A different input is
a different experiment, however similar the dates.

**The output must be byte-stable.** Two runs of the same configuration on the
same input must produce the same canonical files, byte for byte. That rules
out: wall-clock timestamps and run ids inside canonical files, unordered
dictionaries, platform-dependent float formatting, and gzip headers carrying a
modification time. Timings are real and are recorded -- in a separate,
non-canonical file that no digest covers.

Floats are written at twelve significant digits, the convention
:mod:`btc_forecaster.data.snapshot` already uses. That makes the text of a
number deterministic given its value. It does not make floating-point
arithmetic identical across CPUs and BLAS builds, and nothing here claims so:
byte identity is asserted for the same environment, and the manifest records
the environment.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.snapshot import MarketSnapshot, frame_digest

#: The one float format for every canonical artifact.
CANONICAL_FLOAT = "%.12g"

#: What the input may and may not be used for, written into every manifest.
DATA_LICENCE = (
    "Market data retrieved from Yahoo Finance through yfinance. Yahoo's terms do "
    "not grant redistribution, so the snapshot, the row-level predictions and "
    "the realised returns are not committed. The hashes are, and they are what "
    "makes a rerun checkable."
)

#: Reproducibility states. There is no state in which a changed input counts.
INPUT_MATCHES_RECORD = "INPUT_MATCHES_RECORD"
INPUT_CHANGED = "INPUT_CHANGED_NOT_REPRODUCIBLE"
NO_RECORDED_INPUT = "NO_RECORDED_INPUT"


class InputChangedError(RuntimeError):
    """The input hash differs from the one a result was recorded against."""


# -- canonical serialisation ----------------------------------------------


def canonical_float(value: float) -> float | None:
    """Twelve significant digits; non-finite values become ``None``.

    JSON has no NaN or infinity, and ``allow_nan`` would write a non-standard
    token that another parser may reject or read differently.
    """
    if not math.isfinite(value):
        return None
    return float(CANONICAL_FLOAT % value)


def _canonicalise(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _canonicalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalise(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return canonical_float(float(value))
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"cannot serialise {type(value).__name__} canonically")


def canonical_json(value: object) -> str:
    """Sorted keys, no whitespace, fixed float text, no NaN tokens."""
    return json.dumps(_canonicalise(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_csv(frame: pd.DataFrame) -> bytes:
    """A frame as bytes that depend only on its contents and column order.

    Row order is the caller's contract: sort before calling. Nothing here sorts
    silently, because a sort key chosen in the wrong place is how two
    "identical" files end up differing.
    """
    text = frame.to_csv(
        index=False,
        float_format=CANONICAL_FLOAT,
        lineterminator="\n",
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )
    return text.encode("utf-8")


def deterministic_gzip(data: bytes) -> bytes:
    """gzip without a timestamp in its header, so equal input gives equal output."""
    return gzip.compress(data, compresslevel=9, mtime=0)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def result_digest(parts: dict[str, bytes]) -> str:
    """One hash over named canonical contents, independent of insertion order.

    Each part contributes ``name`` and the hash of its **uncompressed** bytes, so
    a compressor upgrade cannot change a result digest.
    """
    lines = [f"{name}\0{sha256_hex(parts[name])}" for name in sorted(parts)]
    return sha256_hex("\n".join(lines).encode("utf-8"))


# -- the input --------------------------------------------------------------


@dataclass(frozen=True)
class InputManifest:
    """Which bytes a result was computed from, and what is known about them."""

    ticker: str
    provider: str
    provider_version: str | None
    retrieved_at: str
    start: str
    end: str
    rows: int
    columns: tuple[str, ...]
    timezone: str
    frequency: str
    missing_bars: int
    #: Digest of the canonical frame -- what ``MarketSnapshot.load`` verifies.
    frame_sha256: str
    #: Digest of the file as it sits on disk, when there is a file.
    file_sha256: str | None
    preprocessing_version: str
    feature_specs: tuple[str, ...]
    target: dict

    def as_dict(self) -> dict:
        return {
            "source": {
                "ticker": self.ticker,
                "provider": self.provider,
                "provider_version": self.provider_version,
                "retrieved_at": self.retrieved_at,
            },
            "range": {"start": self.start, "end": self.end},
            "rows": self.rows,
            "columns": list(self.columns),
            "timezone": self.timezone,
            "frequency": self.frequency,
            "missing_bars": self.missing_bars,
            "frame_sha256": self.frame_sha256,
            "file_sha256": self.file_sha256,
            "preprocessing_version": self.preprocessing_version,
            "feature_specs": list(self.feature_specs),
            "target": self.target,
            "licence": DATA_LICENCE,
            "provider_revises_history": True,
            "immutability": (
                "The provider is not immutable; this record is. A rerun counts as a "
                "reproduction only if frame_sha256 matches."
            ),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_dict())

    def digest(self) -> str:
        return sha256_hex(self.canonical_json().encode("utf-8"))


def input_manifest(
    snapshot: MarketSnapshot,
    *,
    file_bytes: bytes | None,
    preprocessing_version: str,
    feature_specs: tuple[str, ...],
    target: dict,
) -> InputManifest:
    """Describe a loaded snapshot, re-deriving the frame digest rather than trusting it."""
    manifest = snapshot.manifest
    digest = frame_digest(snapshot.frame)
    if digest != manifest.sha256:
        raise InputChangedError(
            f"snapshot frame hashes to {digest}, its own manifest says {manifest.sha256}"
        )
    return InputManifest(
        ticker=manifest.ticker,
        provider=manifest.provider,
        provider_version=manifest.provider_version,
        retrieved_at=manifest.retrieved_at,
        start=manifest.start,
        end=manifest.end,
        rows=manifest.rows,
        columns=tuple(manifest.columns),
        timezone=manifest.timezone,
        frequency=manifest.frequency,
        missing_bars=manifest.missing_bars,
        frame_sha256=digest,
        file_sha256=sha256_hex(file_bytes) if file_bytes is not None else None,
        preprocessing_version=preprocessing_version,
        feature_specs=feature_specs,
        target=target,
    )


def input_status(recorded_sha256: str | None, actual_sha256: str) -> str:
    """Whether a rerun on this input may be called a reproduction."""
    if recorded_sha256 is None:
        return NO_RECORDED_INPUT
    return INPUT_MATCHES_RECORD if recorded_sha256 == actual_sha256 else INPUT_CHANGED


def require_recorded_input(recorded_sha256: str, actual_sha256: str) -> None:
    """Refuse to proceed as a reproduction on a different input."""
    if input_status(recorded_sha256, actual_sha256) != INPUT_MATCHES_RECORD:
        raise InputChangedError(
            "the input is not the one this result was recorded against\n"
            f"  recorded frame sha256: {recorded_sha256}\n"
            f"  actual   frame sha256: {actual_sha256}\n"
            "A different input is a different experiment; its results are not a "
            "reproduction of the recorded ones."
        )


def load_snapshot(path: Path | str) -> tuple[MarketSnapshot, bytes]:
    """Load and hash-verify a snapshot directory; return it with its raw file bytes."""
    directory = Path(path)
    snapshot = MarketSnapshot.load(directory, verify=True)
    return snapshot, (directory / "data.csv").read_bytes()


__all__ = [
    "CANONICAL_FLOAT",
    "DATA_LICENCE",
    "INPUT_CHANGED",
    "INPUT_MATCHES_RECORD",
    "NO_RECORDED_INPUT",
    "InputChangedError",
    "InputManifest",
    "canonical_csv",
    "canonical_float",
    "canonical_json",
    "deterministic_gzip",
    "input_manifest",
    "input_status",
    "load_snapshot",
    "require_recorded_input",
    "result_digest",
    "sha256_hex",
]
