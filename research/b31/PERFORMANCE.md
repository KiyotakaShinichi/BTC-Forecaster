# B3.1.17–B3.1.21 — Replay optimisation and historical materialisation, measured

Companion to `PROFILE.md`, which recorded the bottleneck **before** any change.
This records what the change actually bought, where it did not help, and one
defect the measurement found that the tests had missed.

Raw numbers: `base_reference.json` (branch base), `replay_bench.json`,
`matrix_memory.json`, `matrix_memory_10k.json`.

## Method

Every figure below is the **median of three** runs of `scripts/bench_replay.py`
or `scripts/bench_matrix.py`, on one machine, same synthetic fixture (1,000
documents / 1,000 events), **fresh store per run with cold start inside the
timer**. Ranges are printed, not hidden.

The baseline was re-measured on the same machine on the same day, in a detached
worktree at the branch base `a4fb0ea`, rather than quoted from the published B3
figure. Comparing today's optimised path against a number measured on other
hardware months earlier would attribute the hardware to the optimisation.

### The first run is always the slowest

| series | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| base, 1,000 origins | 97,327 | 51,669 | 51,949 |
| optimised, 1,000 origins | 15,866 | 6,971 | 7,205 |
| optimised, 10,000 origins | 198,116 | 111,899 | 111,638 |

Every series shows it, in both directions, so it is a property of the machine
(page cache, allocator warm-up), not of either implementation. A single-run
benchmark of the baseline would have overstated it by ~1.9x, which would have
manufactured most of a speedup out of nothing. Medians are used throughout, and
this is why.

## B3.1.17 — Performance

1,000 origins, 1,000-record history:

| stage | median ms | range | origins/s |
| --- | ---: | --- | ---: |
| base `a4fb0ea` — reference engine, `executemany` storage | 51,949 | 51,669–97,327 | 19.2 |
| branch — reference engine, chunked-`VALUES` storage | 17,282 | 16,857–19,389 | 57.9 |
| branch — bulk engine, chunked-`VALUES` storage | **7,205** | 6,971–15,866 | **138.8** |

100 origins: 2,855 ms to 1,113 ms (2.6x). 10,000 origins, optimised only:
111,899 ms, 89.4 origins/s; the reference was skipped there because a single run
would have taken about nine minutes.

## B3.1.18 — Where the speedup actually comes from

End-to-end at 1,000 origins is **7.2x** (51,949 to 7,205 ms). It decomposes:

| contribution | factor |
| --- | ---: |
| storage: `executemany` replaced by chunked `INSERT ... SELECT * FROM (VALUES ...)` | 3.01x |
| replay engine: bulk windows replacing the per-origin re-scan | 2.40x |
| product | 7.2x |

Both numbers matter and reporting only one would mislead in opposite directions.
Quoting 7.2x as "the replay optimisation" would credit the engine with the
storage fix. Quoting 2.40x would hide two thirds of the improvement a caller
actually experiences.

The storage share is that large because DuckDB's `executemany` costs roughly
4 ms per statement regardless of row size — 1,000 origins meant 1,000
statements. The batched form issues two.

Against the ~156 s figure published for B3 the improvement reads as ~21.8x, but
that comparison spans machines and is not claimed here.

## B3.1.19 — Memory

**The bulk replay primitive is O(origins), and always was.** Peak Python
allocation, 1,000-record history:

| origins | base `build_many` | optimised `build_many_bulk` |
| ---: | ---: | ---: |
| 100 | 20.1 MB | 14.4 MB |
| 1,000 | 140.6 MB | 134.9 MB |
| 10,000 | not run | 1,339.9 MB |

The optimised engine is slightly *below* the baseline at equal origin counts, so
CPU was not traded for memory. But 1.34 GB at 10,000 origins is real: both
implementations hold every constructed snapshot until the batch ends. Calling
`build_many_bulk` with a 10,000-origin list is not the supported way to build a
long history, and this is the measurement that says so.

**The chunked builder is what bounds it.** Same fixture, through
`HistoricalFeatureMatrixBuilder`:

| origins | chunks | wall ms | origins/s | peak MB | peak KB/origin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 2 | 8,597 | 29.1 | 27.90 | 114.28 |
| 500 | 3 | 15,328 | 32.6 | 28.06 | 57.47 |
| 1,000 | 5 | 30,890 | 32.4 | 28.53 | 29.21 |
| 10,000 | 50 | 263,686 | 37.9 | **34.87** | 3.57 |

**10x the origins costs 1.22x the peak.** The residual growth is the assembled
row list, which `build_chunks` holds so the service can hash and write one file:
measured at about 0.7 KB per origin, so 10,000 origins contributes roughly 7 MB.
Bounded in practice, but it is growth, and it is stated rather than rounded away.

Chunk size, not run length, sets the ceiling — 1,000 origins throughout:

| chunk size | chunks | peak MB |
| ---: | ---: | ---: |
| 50 | 20 | 8.59 |
| 200 | 5 | 28.53 |
| 1,000 | 1 | 135.04 |

Peak tracks chunk size almost exactly (about 0.135 MB per origin in flight),
which is the claim chunking is supposed to support, measured rather than
asserted.

The matrix path runs at ~32 origins/s against the bulk primitive's ~139. The
difference is the materialisation work the primitive does not do: a JSON part
file per chunk for resumption, output hashing, catalog registration and the
parquet write. That is the price of resumability, and it is bounded.

## B3.1.20 — A query-complexity defect the benchmark found

The regression test asserted that SQL statements do not grow with origin count.
It passed. It was also testing the wrong scope: it wrapped
`HistoricalFeatureMatrixBuilder.build_chunks`, while
`HistoricalDatasetService.build` collected extractor versions with **one
`get_snapshot` per origin** immediately afterwards — one round trip and one full
snapshot payload parse each, which is precisely the per-origin pattern bulk
replay exists to remove.

Nothing in the suite failed, because nothing in the suite looked there. The
memory benchmark is what exposed it: the matrix path was three times slower per
origin than the primitive it wraps, and that gap had to come from somewhere.

Replaced with a batched `IntelligenceStore.extractor_versions_for` that extracts
just the versions array server-side, 500 ids per statement. Measured with the
statement counter:

| implementation | 6 origins | 24 origins | growth |
| --- | ---: | ---: | ---: |
| one `get_snapshot` per origin | 18 | 36 | +18 |
| batched | 13 | 13 | **0** |

The regression test now wraps the whole service, not just the builder. Dataset
ids are unchanged across the fix, so this was throughput only, not semantics.

## B3.1.21 — Storage index and query audit: still no

`PROFILE.md` found the database was never the bottleneck, and nothing since has
changed that. Statement counts are now flat in origin count — 13 statements for
both 6 and 24 origins through the full service — and the reads are bounded
single scans over `available_at` / `available_time` ordered data.

DuckDB is a columnar analytical engine with zone maps over its row groups; a
secondary index on a table this size would target the ~2% of wall time that is
SQL. Materialised buckets would additionally introduce a second definition of
each feature window, which is the failure mode this track exists to avoid.

**No indexes, no materialised aggregates and no pre-computed buckets were
added.** That decision is evidence-driven and reversible if a larger history
changes the profile.

## Remaining bottlenecks

1. **`FeatureAggregator` arithmetic is still Python-level.** Bulk windowing
   removed the redundant re-scanning; the per-origin reductions remain. A
   columnar rewrite is the next material win, and it would break bit-exactness
   with the reference unless done very carefully — which is why it was not
   attempted here.
2. **Snapshot membership serialisation.** Each snapshot payload still embeds the
   full document-id and source-hash membership; at large histories this
   dominates the persisted bytes.
3. **`build_many_bulk` memory is O(origins).** Callers building long histories
   should use `HistoricalDatasetService`, which chunks. The primitive could take
   a callback instead of returning a list; it does not today.
