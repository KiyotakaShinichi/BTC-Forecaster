"""Reference vs optimised replay benchmark (B3.1.17, B3.1.19).

Each measurement gets a **fresh store**. Running both modes against one store
would let the first run warm the page cache and pre-populate the snapshots table,
so the second mode's `INSERT OR IGNORE` would find its rows already present --
which flatters whichever mode runs second.

Repeats are reported as median plus range, because a single run on a laptop is
not a measurement. Cold start is included, not hidden: the store is created and
loaded inside the timed setup for every repetition.

Run:
    python scripts/bench_replay.py --records 1000 --origins 100 500 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import tracemalloc
from datetime import timedelta
from pathlib import Path
from typing import Any

from market_intelligence.services import SnapshotService
from market_intelligence.storage import IntelligenceStore

from profile_replay import BASE, build_fixture

PROVIDERS = {"benchmark": "v1"}
CONFIG = "benchmark"


def run_once(mode: str, record_count: int, origin_count: int, *, measure_memory: bool) -> tuple[float, float]:
    """One timed replay in a fresh store. Returns (wall ms, peak MB)."""
    documents, events = build_fixture(record_count)
    origins = [BASE + timedelta(minutes=index) for index in range(origin_count)]

    with tempfile.TemporaryDirectory() as temporary:
        store = IntelligenceStore(Path(temporary) / "bench.duckdb")
        store.put_documents(documents)
        store.put_signals(events)
        service = SnapshotService(store)

        if measure_memory:
            tracemalloc.start()
        started = time.perf_counter()
        if mode == "reference":
            service.build_many(origins, PROVIDERS, CONFIG)
        else:
            service.build_many_bulk(origins, PROVIDERS, CONFIG)
        wall_ms = (time.perf_counter() - started) * 1000
        peak_mb = 0.0
        if measure_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1_048_576
        store.close()
    return wall_ms, peak_mb


def equivalence_holds(record_count: int, origin_count: int) -> bool:
    """Confirm the two modes still agree at this size before quoting a speedup."""
    documents, events = build_fixture(record_count)
    origins = [BASE + timedelta(minutes=index) for index in range(origin_count)]
    with tempfile.TemporaryDirectory() as temporary:
        store = IntelligenceStore(Path(temporary) / "equiv.duckdb")
        store.put_documents(documents)
        store.put_signals(events)
        service = SnapshotService(store)
        reference = [s.snapshot_id for s in service.build_many(origins, PROVIDERS, CONFIG)]
        optimized = [s.snapshot_id for s in service.build_many_bulk(origins, PROVIDERS, CONFIG)]
        store.close()
    return reference == optimized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--origins", type=int, nargs="+", default=[100, 250, 500])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-reference-above", type=int, default=2000, help="reference is slow; cap it")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    print(f"records={args.records}  repeats={args.repeats}  (fresh store per run, cold start included)\n")
    header = f"{'origins':>8} {'reference ms':>26} {'optimized ms':>26} {'speedup':>9} {'origins/s':>11} {'peak MB':>9}"
    print(header)
    print("-" * len(header))

    for origin_count in args.origins:
        optimized_runs = [
            run_once("optimized", args.records, origin_count, measure_memory=index == 0)
            for index in range(args.repeats)
        ]
        optimized_ms = [run[0] for run in optimized_runs]
        peak_mb = optimized_runs[0][1]

        if origin_count <= args.skip_reference_above:
            reference_ms = [
                run_once("reference", args.records, origin_count, measure_memory=False)[0]
                for _ in range(args.repeats)
            ]
        else:
            reference_ms = []

        optimized_median = statistics.median(optimized_ms)
        reference_median = statistics.median(reference_ms) if reference_ms else float("nan")
        speedup = reference_median / optimized_median if reference_ms else float("nan")

        def span(values: list[float]) -> str:
            if not values:
                return f"{'skipped':>26}"
            return f"{statistics.median(values):>12.1f} [{min(values):.0f}-{max(values):.0f}]".rjust(26)

        print(
            f"{origin_count:>8} {span(reference_ms)} {span(optimized_ms)} "
            f"{speedup:>8.1f}x {origin_count / (optimized_median / 1000):>10.1f} {peak_mb:>9.1f}"
        )
        results.append(
            {
                "records": args.records,
                "origins": origin_count,
                "reference_ms_median": reference_median,
                "reference_ms_runs": reference_ms,
                "optimized_ms_median": optimized_median,
                "optimized_ms_runs": optimized_ms,
                "speedup": speedup,
                "origins_per_second": origin_count / (optimized_median / 1000),
                "optimized_peak_mb": peak_mb,
            }
        )

    smallest = min(args.origins)
    print(f"\nequivalence at {smallest} origins: {'IDENTICAL' if equivalence_holds(args.records, smallest) else 'DIVERGED'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
