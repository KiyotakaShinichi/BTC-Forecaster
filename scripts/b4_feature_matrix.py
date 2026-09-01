"""B4.39 / B4.40 — build the historical intelligence feature matrix and join it.

Uses the B3.1 `HistoricalDatasetService` unchanged: chunked, resumable,
manifest-last, catalogued. No BTC target is computed inside the intelligence
dataset; the join to outcomes happens afterwards, on `forecast_origin` alone.

Run against the live store, this currently produces a **structurally empty**
matrix -- one row per origin, every feature at its missing-data default, and a
provider coverage ratio of zero -- because no intelligence corpus exists. That is
the artifact worth having: it proves the path from origins to a joined research
table works end to end, and it shows exactly what is missing rather than
implying the step was skipped.

It is a separate script from `b4_run_study.py` on purpose. The study's expensive
stage is the cross-asset bootstrap; rebuilding a matrix that is empty by
construction has no reason to sit behind twenty minutes of resampling.

Run:
    python scripts/b4_feature_matrix.py --database data/intelligence.duckdb
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from market_intelligence.b4.contracts import SourceDomain
from market_intelligence.b4.market_data import (
    CachingMarketDataProvider,
    MarketDataProvider,
    YFinanceProvider,
    close_as_of,
)
from market_intelligence.b4.targets import (
    DEFAULT_HORIZONS_DAILY,
    assert_targets_are_future_only,
    build_targets,
    join_targets_by_origin,
)
from market_intelligence.historical import HistoricalDatasetService
from market_intelligence.origins import OriginFrequency, generate_origins
from market_intelligence.storage import IntelligenceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("research/market_intelligence/b4/cache"))
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("research/market_intelligence/b4/feature_matrix")
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-01")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    database = args.database or (args.output / "intelligence.duckdb")

    provider: MarketDataProvider = CachingMarketDataProvider(YFinanceProvider(), args.cache)
    btc = provider.fetch("btc_usd_daily", "BTC-USD", SourceDomain.BTC_MARKET, "1d")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    origins = generate_origins(start, end, OriginFrequency.DAILY)
    origins = [origin for origin in origins if close_as_of(btc, origin) is not None]
    print(f"origins: {len(origins)} daily, {origins[0].date()} -> {origins[-1].date()}")

    store = IntelligenceStore(database)
    try:
        documents = _count(store, "documents")
        signals = _count(store, "signals")
        print(f"store: {documents} documents, {signals} signals")

        service = HistoricalDatasetService(store, chunk_size=100)
        result = service.build(
            origins,
            args.output / "intelligence-features.parquet",
            args.output / "intelligence-features.manifest.json",
            {"b4": "research-run"},
            "B4_RESEARCH_V1",
            export_format="csv",
        )
    finally:
        store.close()

    print(
        f"matrix: {result.manifest.row_count} rows x {len(result.manifest.columns)} columns, "
        f"{result.chunk_count} chunks, mode {result.manifest.mode}"
    )
    print(f"dataset_id {result.manifest.dataset_id[:16]}  file_hash {result.manifest.file_hash[:16]}")

    # --- the join, and what it shows -------------------------------------
    feature_rows = _read_matrix(result.output_path)
    targets = build_targets(btc, origins, DEFAULT_HORIZONS_DAILY)
    assert_targets_are_future_only(btc, targets, DEFAULT_HORIZONS_DAILY)
    joined = join_targets_by_origin(feature_rows, targets)

    coverage = [
        value
        for row in feature_rows
        if isinstance(value := row.get("provider_coverage_ratio"), float)
    ]
    non_zero_features = sum(
        1
        for row in feature_rows
        for key, value in row.items()
        if key.startswith(("sentiment_", "event_count", "high_relevance", "regulatory", "whale", "macro"))
        and isinstance(value, float)
        and value != 0.0
    )

    summary = {
        "origins": len(origins),
        "documents_in_store": documents,
        "signals_in_store": signals,
        "matrix_rows": result.manifest.row_count,
        "matrix_columns": len(result.manifest.columns),
        "dataset_id": result.manifest.dataset_id,
        "file_hash": result.manifest.file_hash,
        "joined_rows": len(joined),
        "rows_with_any_non_zero_intelligence_feature": non_zero_features,
        "mean_provider_coverage_ratio": (sum(coverage) / len(coverage) if coverage else None),
        "verdict": (
            "STRUCTURALLY EMPTY — the join works and produces one row per origin, but every "
            "intelligence feature is at its missing-data default because no corpus exists"
            if non_zero_features == 0
            else "the matrix carries non-zero intelligence features"
        ),
    }
    path = args.output / "feature_matrix_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _count(store: IntelligenceStore, table: str) -> int:
    row = store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _read_matrix(path: Path) -> list[dict[str, object]]:
    """Read the CSV matrix back, restoring the origin as a datetime for the join."""
    import csv  # noqa: PLC0415

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            parsed: dict[str, object] = {}
            for key, value in record.items():
                if key == "forecast_origin":
                    parsed[key] = datetime.fromisoformat(value)
                elif value in ("", None):
                    parsed[key] = None
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
