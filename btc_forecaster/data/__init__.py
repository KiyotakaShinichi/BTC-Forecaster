"""Market data: the schema contract, providers, and reproducible snapshots."""

from .contracts import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    FrameReport,
    SchemaViolation,
    normalise_market_frame,
    reindex_to_regular_daily,
    slice_to_window,
    validate_market_frame,
)
from .providers import (
    InMemoryProvider,
    MarketDataProvider,
    SnapshotProvider,
    YFinanceProvider,
    load_or_fetch,
)
from .snapshot import (
    MarketSnapshot,
    SnapshotIntegrityError,
    SnapshotManifest,
    frame_digest,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "FrameReport",
    "InMemoryProvider",
    "MarketDataProvider",
    "MarketSnapshot",
    "SchemaViolation",
    "SnapshotIntegrityError",
    "SnapshotManifest",
    "SnapshotProvider",
    "YFinanceProvider",
    "frame_digest",
    "load_or_fetch",
    "normalise_market_frame",
    "reindex_to_regular_daily",
    "slice_to_window",
    "validate_market_frame",
]
