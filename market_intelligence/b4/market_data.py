"""B4.20 — cross-asset market data behind a provider seam.

Two implementations: a fixture provider that unit tests use (no network, ever)
and a yfinance adapter for research runs. The seam exists so a study is bound to
the *contract* rather than to one vendor — swapping providers should change
provenance metadata and nothing else.

Fetched series are cached to disk as JSON keyed by the request, because a
research result that cannot be reproduced without re-hitting a third-party API
is not reproducible. The cache stores the series exactly as validated, so a
replay reads the same numbers even if the vendor silently revises history —
and if it does revise, the fingerprint changes and the run manifest says so.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    B4DataError,
    EvidenceTier,
    MarketBar,
    MarketSeries,
    SourceDomain,
    TimestampConvention,
    require_utc,
)

#: Intervals this module knows how to convert into a bar period.
INTERVAL_SECONDS: dict[str, int] = {
    "1h": 3_600,
    "1d": 86_400,
}


class MarketDataProvider(ABC):
    """Seam for a historical OHLC provider."""

    name: str

    @abstractmethod
    def fetch(
        self,
        series_id: str,
        ticker: str,
        domain: SourceDomain,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> MarketSeries: ...


class FixtureMarketDataProvider(MarketDataProvider):
    """Deterministic in-memory provider. The only one unit tests may use."""

    name = "fixture"

    def __init__(self, series: dict[str, MarketSeries]) -> None:
        self._series = dict(series)

    def fetch(
        self,
        series_id: str,
        ticker: str,
        domain: SourceDomain,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> MarketSeries:
        if series_id not in self._series:
            raise B4DataError(f"no fixture series {series_id!r}")
        series = self._series[series_id]
        lower = require_utc(start, "start") if start else None
        upper = require_utc(end, "end") if end else None
        bars = tuple(
            bar
            for bar in series.bars
            if (lower is None or bar.period_start >= lower) and (upper is None or bar.period_start < upper)
        )
        return series.model_copy(update={"bars": bars})


class YFinanceProvider(MarketDataProvider):
    """Research-run adapter. Imports yfinance lazily so importing B4 never does.

    `auto_adjust=False` and the raw `Close` are used deliberately: an adjusted
    series is restated whenever a corporate action occurs, which would make a
    historical study's inputs change under it. None of the tickers B4 uses pay
    dividends or split, so the adjustment is a no-op that only adds a restatement
    risk.
    """

    name = "yfinance"

    def fetch(
        self,
        series_id: str,
        ticker: str,
        domain: SourceDomain,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> MarketSeries:
        if interval not in INTERVAL_SECONDS:
            raise B4DataError(f"unsupported interval {interval!r}")
        import yfinance  # noqa: PLC0415 — deliberately lazy; unit tests never import it

        handle = yfinance.Ticker(ticker)
        if start is not None or end is not None:
            frame = handle.history(
                start=require_utc(start, "start") if start else None,
                end=require_utc(end, "end") if end else None,
                interval=interval,
                auto_adjust=False,
            )
        else:
            frame = handle.history(
                period="730d" if interval == "1h" else "max", interval=interval, auto_adjust=False
            )

        if frame is None or frame.empty:
            raise B4DataError(f"{ticker}: provider returned no rows")

        source_timezone = str(frame.index.tz) if frame.index.tz is not None else "naive"
        period_seconds = INTERVAL_SECONDS[interval]
        bars: list[MarketBar] = []
        rejected: list[str] = []
        for stamp, row in frame.iterrows():
            values = [row["Open"], row["High"], row["Low"], row["Close"]]
            if any(value is None or (isinstance(value, float) and math.isnan(value)) for value in values):
                continue  # a holiday placeholder, not an observation
            moment = stamp.to_pydatetime()
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            volume = row.get("Volume", 0.0)
            try:
                bars.append(
                    MarketBar(
                        period_start=moment.astimezone(timezone.utc),
                        period_seconds=period_seconds,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(volume) if volume is not None and not math.isnan(float(volume)) else 0.0,
                    )
                )
            except (B4DataError, ValueError) as error:
                # Yahoo's older futures history carries rows whose high/low are
                # unpopulated placeholders, so the close falls outside its own
                # range. Quarantine them: the alternative is either to abort a
                # whole series over rows decades before BTC existed, or to relax
                # the bar contract until it stops catching real corruption.
                rejected.append(f"{moment.date().isoformat()}: {error}")

        if not bars:
            raise B4DataError(f"{ticker}: every row was missing or incoherent after validation")

        return MarketSeries(
            series_id=series_id,
            ticker=ticker,
            domain=domain,
            provider=self.name,
            timestamp_convention=TimestampConvention.PERIOD_OPEN,
            evidence_tier=EvidenceTier.PIT_VALIDATED,
            source_timezone=source_timezone,
            bars=tuple(bars),
            rejected_bar_count=len(rejected),
            rejection_summary=tuple(rejected[:20]),
        )


class CachingMarketDataProvider(MarketDataProvider):
    """Disk cache in front of any provider, so a research run is replayable."""

    name = "cached"

    def __init__(self, inner: MarketDataProvider, cache_directory: str | Path) -> None:
        self.inner = inner
        self.cache_directory = Path(cache_directory)

    def _path(self, series_id: str, interval: str) -> Path:
        safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in series_id)
        return self.cache_directory / f"{safe}.{interval}.json"

    def fetch(
        self,
        series_id: str,
        ticker: str,
        domain: SourceDomain,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> MarketSeries:
        path = self._path(series_id, interval)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return MarketSeries.model_validate(payload)
        series = self.inner.fetch(series_id, ticker, domain, interval, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(series.model_dump_json(indent=1), encoding="utf-8")
        temporary.replace(path)
        return series


def last_available_index(series: MarketSeries, origin: datetime) -> int:
    """Index of the last bar whose close was knowable at `origin`, or -1.

    The whole point-in-time contract for market data reduces to this function,
    so it is deliberately the only place the comparison is written.
    """
    origin = require_utc(origin, "origin")
    low, high = 0, len(series.bars)
    while low < high:
        middle = (low + high) // 2
        if series.bars[middle].available_at <= origin:
            low = middle + 1
        else:
            high = middle
    return low - 1


def close_as_of(series: MarketSeries, origin: datetime) -> float | None:
    """Last close available at `origin`. None when the series has not started."""
    index = last_available_index(series, origin)
    return series.bars[index].close if index >= 0 else None


def first_close_after(series: MarketSeries, origin: datetime) -> tuple[datetime, float] | None:
    """The earliest close that becomes available strictly after `origin`.

    Used for targets, where "strictly after" is the requirement rather than
    "at or before". Returns the availability time alongside the value so a
    caller can assert the ordering rather than trust it.
    """
    origin = require_utc(origin, "origin")
    for bar in series.bars:
        if bar.available_at > origin:
            return bar.available_at, bar.close
    return None


def align_periods(series: MarketSeries) -> timedelta:
    """The bar period of a series, as a timedelta."""
    if not series.bars:
        raise B4DataError(f"{series.series_id}: empty series has no period")
    return timedelta(seconds=series.period_seconds)


def series_to_records(series: MarketSeries) -> list[dict[str, Any]]:
    """Flat records for manifests and result files."""
    return [
        {
            "period_start": bar.period_start.isoformat(),
            "available_at": bar.available_at.isoformat(),
            "close": bar.close,
        }
        for bar in series.bars
    ]


__all__ = [
    "INTERVAL_SECONDS",
    "CachingMarketDataProvider",
    "FixtureMarketDataProvider",
    "MarketDataProvider",
    "YFinanceProvider",
    "align_periods",
    "close_as_of",
    "first_close_after",
    "last_available_index",
    "series_to_records",
]
