"""Profile the market-intelligence replay path before optimising it (B3.1.0).

Answers four questions with measurement rather than intuition:

1. Where does wall time actually go? (cProfile, cumulative and total time)
2. How many SQL statements are issued, and how does that scale with origins?
3. How many rows are scanned?
4. Is the cost O(origins x history) or something else?

Run:
    python scripts/profile_replay.py --records 1000 --origins 100
    python scripts/profile_replay.py --records 1000 --origins 200 --json report.json

Origin counts are swept so the growth curve is measured, not assumed.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from market_intelligence.models import (
    Direction,
    Document,
    EventSignal,
    EventType,
    ExtractionMethod,
    SignalCategory,
)
from market_intelligence.replay_dataset import ReplayDatasetBuilder
from market_intelligence.storage import IntelligenceStore

BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)


class CountingConnection:
    """Transparent proxy over a DuckDB connection that counts statements.

    Wrapping rather than patching duckdb keeps the store's own code path
    untouched, so the counts describe the real implementation.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.execute_count = 0
        self.executemany_count = 0
        self.statements: dict[str, int] = {}

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self.execute_count += 1
        key = " ".join(sql.split())[:90]
        self.statements[key] = self.statements.get(key, 0) + 1
        return self._inner.execute(sql, *args, **kwargs)

    def executemany(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self.executemany_count += 1
        key = "MANY: " + " ".join(sql.split())[:84]
        self.statements[key] = self.statements.get(key, 0) + 1
        return self._inner.executemany(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_fixture(record_count: int) -> tuple[list[Document], list[EventSignal]]:
    """Same data shape as market_intelligence.benchmark, so numbers compare."""
    documents: list[Document] = []
    events: list[EventSignal] = []
    for index in range(record_count):
        available = BASE - timedelta(minutes=index % 120)
        document_id = f"benchmark-document-{index}"
        documents.append(
            Document(
                document_id=document_id,
                url=f"https://example.invalid/benchmark/{index}",
                publisher="Synthetic Benchmark",
                title=f"Synthetic benchmark item {index}",
                published_at=available,
                retrieved_at=available,
                available_at=available,
                text_hash=Document.content_hash(document_id),
                query="synthetic benchmark",
                provider="benchmark",
            )
        )
        events.append(
            EventSignal(
                event_id=f"benchmark-event-{index}",
                event_time=available,
                available_time=available,
                source_ids=(document_id,),
                category=SignalCategory.WEB_EVENT,
                event_type=EventType.REGULATION,
                direction=Direction.UNKNOWN,
                sentiment=0.0,
                btc_relevance=0.5,
                novelty=0.5,
                confidence=0.5,
                expected_horizon_hours=24,
                summary="Synthetic benchmark event",
                extractor_version="benchmark-v1",
                extraction_method=ExtractionMethod.FIXTURE,
            )
        )
    return documents, events


def measure(record_count: int, origin_count: int, *, profile: bool) -> dict[str, Any]:
    documents, events = build_fixture(record_count)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        store = IntelligenceStore(directory / "profile.duckdb")
        store.put_documents(documents)
        store.put_signals(events)

        counter = CountingConnection(store.connection)
        store.connection = counter  # type: ignore[assignment]

        origins = [BASE + timedelta(minutes=index) for index in range(origin_count)]
        builder = ReplayDatasetBuilder(store)

        def run() -> None:
            builder.build(
                origins,
                directory / "profile.parquet",
                directory / "profile.manifest.json",
                {"benchmark": "v1"},
                "benchmark",
                git_sha="profile",
            )

        tracemalloc.start()
        started = time.perf_counter()
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
            run()
            profiler.disable()
        else:
            profiler = None
            run()
        wall_ms = (time.perf_counter() - started) * 1000
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result: dict[str, Any] = {
            "records": record_count,
            "origins": origin_count,
            "wall_ms": round(wall_ms, 1),
            "ms_per_origin": round(wall_ms / origin_count, 3),
            "origins_per_second": round(origin_count / (wall_ms / 1000), 2),
            "peak_memory_mb": round(peak_bytes / 1_048_576, 1),
            "sql_execute_count": counter.execute_count,
            "sql_executemany_count": counter.executemany_count,
            "sql_by_statement": dict(
                sorted(counter.statements.items(), key=lambda kv: -kv[1])[:10]
            ),
        }

        if profiler is not None:
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats("tottime").print_stats(22)
            result["profile_tottime"] = stream.getvalue()

        store.connection = counter._inner  # type: ignore[assignment]
        store.close()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument(
        "--origins",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200],
        help="sweep, so the growth curve is measured rather than assumed",
    )
    parser.add_argument("--profile-at", type=int, default=100, help="origin count to cProfile")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for origin_count in args.origins:
        outcome = measure(args.records, origin_count, profile=origin_count == args.profile_at)
        results.append(outcome)
        print(
            f"origins={origin_count:>6}  wall={outcome['wall_ms']:>10.1f} ms  "
            f"{outcome['ms_per_origin']:>8.3f} ms/origin  "
            f"{outcome['origins_per_second']:>8.2f} origins/s  "
            f"sql={outcome['sql_execute_count']:>6}  "
            f"peak={outcome['peak_memory_mb']:>6.1f} MB"
        )

    print("\n--- scaling ---")
    for earlier, later in zip(results, results[1:], strict=False):
        origin_ratio = later["origins"] / earlier["origins"]
        time_ratio = later["wall_ms"] / earlier["wall_ms"]
        print(
            f"{earlier['origins']:>5} -> {later['origins']:<5} "
            f"origins x{origin_ratio:.1f}  time x{time_ratio:.2f}  "
            f"(linear would be x{origin_ratio:.1f})"
        )

    profiled = next((r for r in results if "profile_tottime" in r), None)
    if profiled:
        print(f"\n--- cProfile at {profiled['origins']} origins (tottime) ---")
        print(profiled["profile_tottime"])
        print("--- SQL statements ---")
        for statement, count in profiled["sql_by_statement"].items():
            print(f"  {count:>6}x  {statement}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
