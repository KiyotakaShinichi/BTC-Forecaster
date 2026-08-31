"""B4.0 — produce the historical data availability inventory from evidence.

Fetches the market series B4 can actually obtain, counts what the intelligence
store actually holds, and records every category that has nothing as an explicit
DATA_UNAVAILABLE row. Writes research/market_intelligence/b4/data_inventory.json.

Run:
    python scripts/b4_data_audit.py --cache research/market_intelligence/b4/cache
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from market_intelligence.b4.audit import (
    SourceInventoryEntry,
    inventory_intelligence_store,
    inventory_market_series,
    unavailable_entry,
    write_inventory,
)
from market_intelligence.b4.contracts import SourceDomain
from market_intelligence.b4.market_data import (
    CachingMarketDataProvider,
    MarketDataProvider,
    YFinanceProvider,
)
from market_intelligence.storage import IntelligenceStore

YAHOO_LICENSE = (
    "Yahoo Finance API, personal/research use under Yahoo's terms; redistribution of raw "
    "quotes is not granted, so only derived statistics are published"
)

MARKET_REQUESTS: tuple[tuple[str, str, SourceDomain, str, str], ...] = (
    ("btc_usd_hourly", "BTC-USD", SourceDomain.BTC_MARKET, "1h", "BTC hourly OHLCV"),
    ("btc_usd_daily", "BTC-USD", SourceDomain.BTC_MARKET, "1d", "BTC daily OHLCV"),
    ("dxy_daily", "DX-Y.NYB", SourceDomain.MACRO_MARKET, "1d", "US dollar index daily"),
    ("gold_daily", "GC=F", SourceDomain.CROSS_ASSET, "1d", "COMEX gold front future daily"),
    ("oil_daily", "CL=F", SourceDomain.CROSS_ASSET, "1d", "WTI crude front future daily"),
    ("sp500_daily", "^GSPC", SourceDomain.CROSS_ASSET, "1d", "S&P 500 index daily"),
    ("nasdaq_daily", "^IXIC", SourceDomain.CROSS_ASSET, "1d", "Nasdaq Composite daily"),
    ("vix_daily", "^VIX", SourceDomain.CROSS_ASSET, "1d", "CBOE volatility index daily"),
    ("ust10y_daily", "^TNX", SourceDomain.MACRO_MARKET, "1d", "US 10-year yield daily"),
    ("eth_usd_daily", "ETH-USD", SourceDomain.CRYPTO_MARKET_STRUCTURE, "1d", "ETH daily OHLCV"),
    ("bnb_usd_daily", "BNB-USD", SourceDomain.CRYPTO_MARKET_STRUCTURE, "1d", "BNB daily OHLCV"),
    ("xrp_usd_daily", "XRP-USD", SourceDomain.CRYPTO_MARKET_STRUCTURE, "1d", "XRP daily OHLCV"),
    ("ltc_usd_daily", "LTC-USD", SourceDomain.CRYPTO_MARKET_STRUCTURE, "1d", "LTC daily OHLCV"),
)

# Categories B4 was asked about for which nothing exists. Each records *why*,
# because "we did not find any" and "it cannot be obtained" are different
# findings and only one of them is a blocker.
KNOWN_ABSENT: tuple[tuple[str, SourceDomain, str, str], ...] = (
    (
        "entity-statements:Donald Trump / Elon Musk / Michael Saylor / Jerome Powell",
        SourceDomain.ENTITY_STATEMENT,
        "dated public statements by watchlist entities",
        "SocialStatementProvider is an abstract seam with no lawful contracted implementation "
        "configured, and no statements have ever been persisted. Historical statement archives "
        "cannot be reconstructed with defensible availability timestamps.",
    ),
    (
        "regulatory-events:SEC / CFTC",
        SourceDomain.REGULATORY_EVENT,
        "dated regulatory actions and filings",
        "The RSS provider is configured with an empty feed list and disabled. No regulatory "
        "corpus has been collected.",
    ),
    (
        "etf-events:issuer flows and approvals",
        SourceDomain.ETF_EVENT,
        "ETF creation/redemption flows and approval events",
        "No flow provider exists in the codebase and none is configured. Daily ETF flow data is "
        "a commercial product; no contract is assumed.",
    ),
    (
        "onchain:whale transfers",
        SourceDomain.ONCHAIN_WHALE,
        "large BTC transfers with exchange-inflow/outflow context",
        "WhaleDataProvider is an abstract seam with no implementation configured, and no "
        "observations have been persisted. Transfer context (INFLOW/OUTFLOW/CUSTODY/INTERNAL) "
        "cannot be inferred without a labelled address dataset.",
    ),
    (
        "news-web:general crypto news corpus",
        SourceDomain.NEWS_WEB,
        "retrieved news/web documents about BTC",
        "Both configured providers are disabled placeholders (empty feed_urls, unset search "
        "endpoint). Retrieving articles today about past events would produce RETROSPECTIVE_ONLY "
        "evidence, which B4.35 forbids using as point-in-time forecasting evidence.",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("research/market_intelligence/b4/cache"))
    parser.add_argument("--database", type=Path, default=None, help="optional intelligence store to count")
    parser.add_argument(
        "--output", type=Path, default=Path("research/market_intelligence/b4/data_inventory.json")
    )
    parser.add_argument("--offline", action="store_true", help="use only cached series")
    args = parser.parse_args()

    provider: MarketDataProvider = CachingMarketDataProvider(YFinanceProvider(), args.cache)
    entries: list[SourceInventoryEntry] = []

    for series_id, ticker, domain, interval, description in MARKET_REQUESTS:
        try:
            series = provider.fetch(series_id, ticker, domain, interval)
        except Exception as error:  # noqa: BLE001 -- an audit records failures, it does not abort
            entries.append(
                unavailable_entry(
                    f"yfinance:{ticker}", domain, description, f"fetch failed: {type(error).__name__}: {error}"
                )
            )
            print(f"  {series_id:<18} FAILED {type(error).__name__}", flush=True)
            continue
        entry = inventory_market_series(series, data_type=description, license_status=YAHOO_LICENSE)
        entries.append(entry)
        print(
            f"  {series_id:<18} {len(series):>6} bars  "
            f"{entry.start_date.date() if entry.start_date else '?'} -> "
            f"{entry.end_date.date() if entry.end_date else '?'}",
            flush=True,
        )

    if args.database is not None:
        store = IntelligenceStore(args.database)
        try:
            entries.extend(inventory_intelligence_store(store))
        finally:
            store.close()

    for source, domain, data_type, reason in KNOWN_ABSENT:
        entries.append(unavailable_entry(source, domain, data_type, reason))

    write_inventory(entries, args.output)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": len(entries),
        "suitable": sum(1 for entry in entries if entry.suitable_for_historical_study),
        "unavailable": sum(1 for entry in entries if not entry.suitable_for_historical_study),
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
