"""Market data providers.

Every provider satisfies :class:`MarketDataProvider` and returns a
contract-satisfying frame (see :mod:`btc_forecaster.data.contracts`).

Network access is confined to :class:`YFinanceProvider`, and ``yfinance`` is
imported lazily *inside* the fetch call so the rest of the platform -- and the
whole unit test suite -- can be imported without it installed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from ..timebase import UTC, to_utc_timestamp
from .contracts import normalise_market_frame, slice_to_window, validate_market_frame
from .snapshot import MarketSnapshot


@runtime_checkable
class MarketDataProvider(Protocol):
    """Anything that can produce a validated daily market frame.

    Track B's exogenous signal sources are expected to implement a sibling
    protocol rather than this one: they are not OHLCV and carry a non-zero
    publication lag. See ``docs/seams.md``.
    """

    name: str

    def fetch(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp | None = None,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a frame satisfying the market data contract."""
        ...


class YFinanceProvider:
    """Daily bars from Yahoo Finance. The only component that touches the network."""

    name = "yfinance"

    def __init__(self, *, auto_adjust: bool = False) -> None:
        # auto_adjust=False keeps the raw traded close. BTC-USD has no splits or
        # dividends, so adjustment would only introduce provider revisions that
        # break snapshot reproducibility.
        self.auto_adjust = auto_adjust

    def _module(self):
        try:
            import yfinance  # noqa: PLC0415 - deliberately lazy
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "yfinance is required for live data. Install the optional extra:\n"
                '    pip install "btc-forecaster[data]"'
            ) from exc
        return yfinance

    @property
    def version(self) -> str | None:
        try:
            return getattr(self._module(), "__version__", None)
        except ImportError:  # pragma: no cover
            return None

    def fetch(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp | None = None,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        yf = self._module()
        raw = yf.download(
            ticker,
            start=None if start is None else to_utc_timestamp(start).strftime("%Y-%m-%d"),
            end=None if end is None else to_utc_timestamp(end).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=self.auto_adjust,
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"{self.name} returned no rows for {ticker!r}")
        frame = normalise_market_frame(raw)
        validate_market_frame(frame)
        return frame

    def snapshot(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp | None = None,
        end: str | datetime | pd.Timestamp | None = None,
        note: str | None = None,
    ) -> MarketSnapshot:
        frame = self.fetch(ticker, start=start, end=end)
        return MarketSnapshot.build(
            frame,
            ticker=ticker,
            provider=self.name,
            provider_version=self.version,
            retrieved_at=pd.Timestamp.now(tz=UTC),
            note=note,
            normalise=False,
        )


class SnapshotProvider:
    """Serves a previously saved snapshot. Offline, deterministic, hash-verified."""

    name = "snapshot"

    def __init__(self, directory: Path | str, *, verify: bool = True) -> None:
        self.directory = Path(directory)
        self._snapshot = MarketSnapshot.load(self.directory, verify=verify)

    @property
    def snapshot(self) -> MarketSnapshot:
        return self._snapshot

    def fetch(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp | None = None,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if ticker != self._snapshot.ticker:
            raise ValueError(
                f"snapshot at {self.directory} holds {self._snapshot.ticker!r}, not {ticker!r}"
            )
        return slice_to_window(self._snapshot.frame, start, end).copy()


class InMemoryProvider:
    """Serves a frame handed to it. The test suite's data source."""

    name = "in-memory"

    def __init__(self, frame: pd.DataFrame, *, ticker: str = "TEST-USD") -> None:
        self.frame = normalise_market_frame(frame) if "close" not in frame.columns else frame.sort_index()
        validate_market_frame(self.frame)
        self.ticker = ticker

    def fetch(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp | None = None,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return slice_to_window(self.frame, start, end).copy()


def load_or_fetch(
    ticker: str,
    *,
    cache_dir: Path | str,
    start: str | datetime | pd.Timestamp | None = None,
    end: str | datetime | pd.Timestamp | None = None,
    provider: MarketDataProvider | None = None,
    refresh: bool = False,
) -> MarketSnapshot:
    """Return a snapshot from the local cache, fetching only if needed.

    Keeping this explicit -- rather than fetching on every run -- is what makes
    a research session reproducible: repeated runs read the same bytes, and the
    manifest records exactly which vintage they were.
    """
    cache = Path(cache_dir)
    if not refresh:
        try:
            return MarketSnapshot.load(cache)
        except FileNotFoundError:
            pass

    live = provider if provider is not None else YFinanceProvider()
    if isinstance(live, YFinanceProvider):
        snap = live.snapshot(ticker, start=start, end=end)
    else:
        snap = MarketSnapshot.build(
            live.fetch(ticker, start=start, end=end),
            ticker=ticker,
            provider=getattr(live, "name", type(live).__name__),
            normalise=False,
        )
    snap.save(cache)
    return snap


__all__ = [
    "InMemoryProvider",
    "MarketDataProvider",
    "SnapshotProvider",
    "YFinanceProvider",
    "load_or_fetch",
]
