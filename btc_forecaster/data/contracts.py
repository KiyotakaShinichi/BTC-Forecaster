"""The market data contract: what a valid price frame is, enforced on ingest.

Every provider must hand back a frame that satisfies :func:`validate_market_frame`.
Downstream code -- features, models, backtests -- is then entitled to assume a
UTC-indexed, gap-checked, strictly-ordered daily frame, and does not re-validate.

The pre-Track-A pipeline discovered its own columns with
``[c for c in df.columns if "close" in c.lower()][0]``, which silently selects
"Adj Close" or "Close" depending on dict ordering. Column resolution is now
explicit and preference-ordered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..timebase import BAR_DURATION, to_utc_index

#: Columns every downstream component may assume exist.
REQUIRED_COLUMNS: tuple[str, ...] = ("close", "volume")

#: Columns carried through when a provider supplies them.
OPTIONAL_COLUMNS: tuple[str, ...] = ("open", "high", "low")

CANONICAL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

#: Provider column name -> canonical name, in preference order. "close" is
#: preferred over "adj close": the pipeline models log price of the traded
#: series, and BTC-USD has no dividends or splits to adjust for, so an adjusted
#: column would only introduce provider-specific revisions.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "close": ("close", "adj close", "adj_close", "adjclose", "price"),
    "volume": ("volume", "vol"),
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
}

SCHEMA_VERSION = "1"


class SchemaViolation(ValueError):
    """Raised when a frame does not satisfy the market data contract."""


@dataclass(frozen=True)
class FrameReport:
    """Non-fatal observations about an otherwise valid frame."""

    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_bars: tuple[pd.Timestamp, ...] = field(default=())
    duplicate_bars: tuple[pd.Timestamp, ...] = field(default=())

    @property
    def is_gapless(self) -> bool:
        return not self.missing_bars


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex when given a ticker list; flatten it."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(part) for part in col if str(part) != "").strip()
            for col in df.columns
        ]
    return df


def _resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Find the provider column backing ``canonical``, honouring alias order.

    Aliases are tried in preference order across *all* columns before moving to
    the next alias, so "Close" always wins over "Adj Close" regardless of the
    order the provider happened to emit them in.
    """
    lowered = {str(c).lower().strip(): c for c in df.columns}

    def canonicalise(name: str) -> str:
        return name.replace("_", " ").strip()

    for alias in _COLUMN_ALIASES[canonical]:
        target = canonicalise(alias)
        for name, original in lowered.items():
            if canonicalise(name) == target:
                return original

    # MultiIndex-flattened names look like "Close_BTC-USD". Match on the segment
    # before the first underscore so we never confuse "Adj Close_BTC-USD" for
    # "Close_BTC-USD".
    for alias in _COLUMN_ALIASES[canonical]:
        target = canonicalise(alias)
        for name, original in lowered.items():
            if canonicalise(name.split("_", 1)[0]) == target:
                return original
    return None


def normalise_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map a raw provider frame onto the canonical schema.

    Renames columns, coerces the index to UTC daily bar labels, sorts, and drops
    everything outside the canonical schema. Does not validate -- call
    :func:`validate_market_frame` on the result.
    """
    df = _flatten_columns(df)

    resolved: dict[str, str] = {}
    for canonical in CANONICAL_COLUMNS:
        source = _resolve_column(df, canonical)
        if source is not None:
            resolved[canonical] = source

    missing = [c for c in REQUIRED_COLUMNS if c not in resolved]
    if missing:
        raise SchemaViolation(
            f"provider frame is missing required column(s) {missing}; "
            f"available columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {canonical: pd.to_numeric(df[source], errors="coerce") for canonical, source in resolved.items()}
    )
    out.index = to_utc_index(df.index)
    out.index.name = "date"
    out = out[[c for c in CANONICAL_COLUMNS if c in out.columns]]
    return out.sort_index()


def validate_market_frame(df: pd.DataFrame, *, allow_gaps: bool = True) -> FrameReport:
    """Assert the market data contract holds. Raises :class:`SchemaViolation`.

    ``allow_gaps`` defaults to True: crypto trades every day, but providers do
    drop bars, and a missing bar is a data-quality fact to report rather than a
    reason to refuse the whole series. Gaps are surfaced in the returned report
    so callers can decide.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaViolation(f"expected a DataFrame, got {type(df).__name__}")
    if df.empty:
        raise SchemaViolation("market frame is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaViolation(f"missing required column(s): {missing}")

    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise SchemaViolation(f"index must be a DatetimeIndex, got {type(index).__name__}")
    if index.tz is None:
        raise SchemaViolation("index must be timezone-aware; the contract is UTC")
    if str(index.tz) not in {"UTC", "utc"}:
        raise SchemaViolation(f"index must be UTC, got {index.tz}")
    if not (index == index.normalize()).all():
        raise SchemaViolation("daily bar labels must be normalised to UTC midnight")
    if not index.is_monotonic_increasing:
        raise SchemaViolation("index must be sorted ascending")

    duplicates = tuple(index[index.duplicated()].unique())
    if duplicates:
        raise SchemaViolation(f"duplicate bar labels: {[str(d.date()) for d in duplicates[:5]]}")

    for column in REQUIRED_COLUMNS:
        series = df[column]
        if not np.issubdtype(series.dtype, np.number):
            raise SchemaViolation(f"column '{column}' must be numeric, got {series.dtype}")

    close = df["close"]
    if close.isna().any():
        n = int(close.isna().sum())
        raise SchemaViolation(f"'close' contains {n} NaN value(s); price is not optional")
    if (close <= 0).any():
        raise SchemaViolation("'close' contains non-positive values; log price is undefined")

    expected = pd.date_range(index.min(), index.max(), freq="D", tz=index.tz)
    missing_bars = tuple(expected.difference(index))
    if missing_bars and not allow_gaps:
        raise SchemaViolation(
            f"{len(missing_bars)} missing daily bar(s), first: {missing_bars[0].date()}"
        )

    return FrameReport(
        rows=len(df),
        start=index.min(),
        end=index.max(),
        missing_bars=missing_bars,
        duplicate_bars=duplicates,
    )


def reindex_to_regular_daily(df: pd.DataFrame, *, method: str = "ffill") -> pd.DataFrame:
    """Fill provider gaps onto a gapless daily grid.

    Forward-filling a price is point-in-time safe -- it repeats the last *known*
    close and never looks forward. Volume is filled with 0.0 instead, because
    carrying a stale volume forward would fabricate activity that did not occur.
    Use deliberately: the filled bars are synthetic and inflate any metric that
    counts observations.
    """
    if method != "ffill":
        raise ValueError("only forward fill is point-in-time safe for prices")

    grid = pd.date_range(df.index.min(), df.index.max(), freq="D", tz=df.index.tz, name="date")
    out = df.reindex(grid)
    price_columns = [c for c in out.columns if c != "volume"]
    out[price_columns] = out[price_columns].ffill()
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0.0)
    return out


def slice_to_window(
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Inclusive window slice on bar labels, in UTC."""
    from ..timebase import to_utc_timestamp

    out = df
    if start is not None:
        out = out[out.index >= to_utc_timestamp(start).normalize()]
    if end is not None:
        out = out[out.index <= to_utc_timestamp(end).normalize()]
    return out


def bars_span(df: pd.DataFrame) -> int:
    """Calendar days covered, inclusive of both endpoints."""
    return int((df.index.max() - df.index.min()) / BAR_DURATION) + 1


__all__ = [
    "CANONICAL_COLUMNS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "FrameReport",
    "SchemaViolation",
    "bars_span",
    "normalise_market_frame",
    "reindex_to_regular_daily",
    "slice_to_window",
    "validate_market_frame",
]
