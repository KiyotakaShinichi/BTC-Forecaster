"""Historical feature matrix memory benchmark (B3.1.19).

The chunked builder makes one claim worth testing: peak memory should track the
*chunk*, not the length of the run. That claim is only partly true, and this
script is how the difference was found rather than assumed.

Two sweeps:

**Chunk sweep.** Fixed origin count, varying chunk size. Isolates the replay
working set -- the snapshots, documents and events alive while one chunk is in
flight.

**Origin sweep.** Fixed chunk size, growing origin count. Exposes what chunking
does *not* bound: ``build_chunks`` accumulates every row so the service can hash
and write one file, so the assembled rows are O(origins) no matter how small the
chunks are.

Peak is ``tracemalloc``, which counts Python allocations only -- DuckDB's native
arena is outside it. That understates absolute RSS and is the right instrument
anyway, because the question here is whether *our* structures grow, not how much
the database caches.

Run:
    python scripts/bench_matrix.py --records 1000 --origins 250 500 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
import tracemalloc
from datetime import timedelta
from pathlib import Path
from typing import Any

from profile_replay import BASE, build_fixture

from market_intelligence.historical import HistoricalDatasetService
from market_intelligence.storage import IntelligenceStore

PROVIDERS = {"benchmark": "v1"}
CONFIG = "benchmark"


def run_once(record_count: int, origin_count: int, chunk_size: int) -> dict[str, Any]:
    """One chunked matrix build in a fresh store. Store load is not timed."""
    documents, events = build_fixture(record_count)
    origins = [BASE + timedelta(minutes=index) for index in range(origin_count)]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        store = IntelligenceStore(root / "bench.duckdb")
        store.put_documents(documents)
        store.put_signals(events)
        service = HistoricalDatasetService(store, chunk_size=chunk_size)

        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = service.build(
            origins,
            root / "matrix.parquet",
            root / "matrix.manifest.json",
            PROVIDERS,
            CONFIG,
            git_sha="benchmark",
        )
        wall_ms = (time.perf_counter() - started) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        output_kb = (root / "matrix.parquet").stat().st_size / 1024
        store.close()

    return {
        "records": record_count,
        "origins": origin_count,
        "chunk_size": chunk_size,
        "chunks": result.chunk_count,
        "rows": result.manifest.row_count,
        "wall_ms": wall_ms,
        "peak_mb": peak / 1_048_576,
        "output_kb": output_kb,
    }


def _header(varying: str) -> None:
    header = (
        f"{varying:>11} {'chunks':>7} {'rows':>7} {'wall ms':>10} "
        f"{'origins/s':>10} {'peak MB':>9} {'peak KB/origin':>15}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)


def _row(row: dict[str, Any], varying: str) -> None:
    """Printed as each measurement lands: a long sweep should be watchable."""
    per_origin = row["peak_mb"] * 1024 / row["origins"]
    print(
        f"{row[varying]:>11} {row['chunks']:>7} {row['rows']:>7} {row['wall_ms']:>10.0f} "
        f"{row['origins'] / (row['wall_ms'] / 1000):>10.1f} {row['peak_mb']:>9.2f} {per_origin:>15.2f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--origins", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("--fixed-origins", type=int, default=1000)
    parser.add_argument("--fixed-chunk", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    print(f"records={args.records}  (fresh store per run; store load excluded from the timer)\n")

    print(f"CHUNK SWEEP -- {args.fixed_origins} origins, chunk size varies", flush=True)
    _header("chunk_size")
    chunk_rows = []
    for size in args.chunk_sizes:
        measured = _median_of(args.repeats, args.records, args.fixed_origins, size)
        _row(measured, "chunk_size")
        chunk_rows.append(measured)

    print()
    print(f"ORIGIN SWEEP -- chunk size {args.fixed_chunk}, origin count varies", flush=True)
    _header("origins")
    origin_rows = []
    for count in args.origins:
        measured = _median_of(args.repeats, args.records, count, args.fixed_chunk)
        _row(measured, "origins")
        origin_rows.append(measured)

    first, last = origin_rows[0], origin_rows[-1]
    origin_growth = last["origins"] / first["origins"]
    memory_growth = last["peak_mb"] / first["peak_mb"] if first["peak_mb"] else float("nan")
    print(
        f"\n{origin_growth:.0f}x the origins costs {memory_growth:.2f}x the peak Python memory "
        f"({first['peak_mb']:.2f} MB -> {last['peak_mb']:.2f} MB)."
    )
    print(
        "Chunking bounds the replay working set, not the assembled rows: build_chunks holds "
        "every row so the service can hash and write one file."
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"chunk_sweep": chunk_rows, "origin_sweep": origin_rows}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


def _median_of(repeats: int, records: int, origins: int, chunk_size: int) -> dict[str, Any]:
    runs = [run_once(records, origins, chunk_size) for _ in range(repeats)]
    merged = dict(runs[0])
    merged["wall_ms"] = statistics.median(run["wall_ms"] for run in runs)
    merged["peak_mb"] = statistics.median(run["peak_mb"] for run in runs)
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
