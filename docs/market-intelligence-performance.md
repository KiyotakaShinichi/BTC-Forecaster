# Market-intelligence local performance

Measured on 2026-08-28 using Python 3.14.3 and DuckDB 1.5.5 on the local Windows development environment. These are representative observations, not CI thresholds or cross-machine guarantees.

| Operation | Size | Wall time |
|---|---:|---:|
| SQL-filtered document query plus validation | 1,000 documents | 221.219 ms |
| SQL-filtered event query plus validation | 1,000 events | 209.815 ms |
| Immutable point-in-time snapshot | 1,000 documents + 1,000 events | 634.649 ms |
| Parquet replay dataset with persisted membership snapshots | 1,000 origins | 156,761.826 ms |

The initial replay implementation repeatedly queried and validated the entire evidence set for every origin and exceeded three minutes. B3 changed bulk replay to read bounded evidence once and batch snapshot integrity validation/insertion. Snapshot membership and point-in-time checks were not weakened.

Reproduce with:

```powershell
python -m market_intelligence.benchmark
```

## B3.1 replay optimisation

Measured on 2026-08-31, same machine, median of three runs with a fresh store
and cold start inside the timer. The B3 row above is left as recorded; it was
measured with a different protocol and is not directly comparable.

| Operation | Size | Wall time |
|---|---:|---:|
| Reference replay at the B3.1 branch base `a4fb0ea` | 1,000 origins | 51,949 ms |
| Reference replay, after the chunked-`VALUES` storage fix | 1,000 origins | 17,282 ms |
| Bulk replay (`build_many_bulk`) | 1,000 origins | 7,205 ms |
| Bulk replay | 10,000 origins | 111,899 ms |
| Chunked historical feature matrix | 10,000 origins | 263,686 ms |

End-to-end 7.2x at 1,000 origins, decomposing into 3.01x from storage batching
and 2.40x from the replay engine. Peak Python memory for the chunked matrix
builder is 34.9 MB at 10,000 origins — 1.22x the 1,000-origin figure — because
chunk size, not run length, sets the ceiling. `build_many_bulk` is O(origins) in
memory (1.34 GB at 10,000 origins) and is not the supported way to build a long
history.

Full method, decomposition and remaining bottlenecks: `research/b31/PERFORMANCE.md`.

Reproduce with:

```powershell
python scripts/bench_replay.py --records 1000 --origins 100 1000 --repeats 3
python scripts/bench_matrix.py --records 1000 --origins 250 1000 --repeats 3
```
