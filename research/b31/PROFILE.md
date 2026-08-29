# B3.1.0 — Replay path profile, before any optimisation

Measured on the branch base `a4fb0ea`, before edits, with
`scripts/profile_replay.py`. Raw numbers in `profile_baseline.json`.

The point of this document is to record what the bottleneck **actually is**, so
the optimisation targets measured cost rather than the first plausible story.
The first plausible story was wrong.

## Method

- Same synthetic data shape as `market_intelligence.benchmark` (N documents, N
  events, all within 120 minutes of a base instant), so numbers are comparable
  to the existing published benchmark.
- `CountingConnection` transparently proxies the DuckDB connection and counts
  statements. The store's own code path is untouched, so counts describe the
  real implementation.
- `cProfile` sorted by `tottime`, `tracemalloc` for peak allocation.
- Origin counts swept so the growth curve is **measured**, not assumed.

## Result 1 — cost is linear in origins, and also grows with history

Origins swept at 1,000 documents / 1,000 events:

| origins | wall (ms) | ms/origin | origins/s | SQL statements | peak MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 2,612 | 104.5 | 9.6 | 30 | 10.1 |
| 50 | 4,155 | 83.1 | 12.0 | 55 | 13.4 |
| 100 | 11,174 | 111.7 | 9.0 | 105 | 20.1 |
| 200 | 22,175 | 110.9 | 9.0 | 205 | 33.5 |

`100 -> 200` origins doubles the time (×1.98). Per-origin cost is flat at
~111 ms.

History swept at 50 origins:

| documents / events | wall (ms) | ms/origin |
| ---: | ---: | ---: |
| 250 | 1,874 | 37.5 |
| 500 | 2,798 | 56.0 |
| 1,000 | 4,757 | 95.1 |

History ×4 gives time ×2.5, and per-origin cost tracks history size.

**Conclusion: the path is `O(origins × history)`.** At ~111 ms/origin with a
1,000-record history, 1,000 origins extrapolates to ~111 s, consistent with the
~156,762 ms figure published for B3.

## Result 2 — the bottleneck is NOT the database

This was the hypothesis worth killing first. `SnapshotService.build_many`
already performs **one** bounded read for documents and **one** for events; it
does not re-query per origin.

SQL statements for 100 origins — 105 total:

```
100x  SELECT payload FROM runs WHERE finished_at <= ? ORDER BY finished_at DESC LIMIT 1
  1x  SELECT payload FROM documents WHERE available_at <= ? ORDER BY available_at
  1x  SELECT payload FROM signals  WHERE available_time <= ? ORDER BY available_time
  1x  SELECT document_id FROM documents
  1x  SELECT event_id FROM signals
  1x  MANY: INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?)
  1x  INSERT OR IGNORE INTO replay_datasets(...)
```

Only one statement scales with origins: `latest_run_as_of`, called once per
origin by `ReplayDatasetBuilder.build`. It is a `LIMIT 1` against a tiny table
and accounts for roughly 2% of wall time — worth fixing on principle, but it is
not the bottleneck.

Rows scanned: 1,000 documents + 1,000 events, once each. The bounded read is
already correct.

## Result 3 — the bottleneck is Python-level re-scanning of history

`cProfile` at 100 origins, 11.17 s total, sorted by `tottime`:

| tottime (s) | share | function |
| ---: | ---: | --- |
| 5.784 | **52%** | `aggregation.py:11 FeatureAggregator.aggregate` |
| 1.300 | 12% | `json/encoder.py:207 iterencode` |
| 0.735 | 7% | `services.py:54 SnapshotService.build_many` |
| 0.725 | 6% | `executemany` (snapshot payload persistence) |
| 0.239 | 2% | `aggregation.py:16 <genexpr>` |
| 0.228 | 2% | `aggregation.py:38 _weighted_mean` |
| 0.226 | 2% | `builtins.sum` |
| 0.197 | 2% | pydantic `validate_python` |

`aggregate` alone is 52% of `tottime` and 61% of cumulative time. The genexpr
call counts tell the story: **100,100 calls** = 100 origins × ~1,001 events. The
entire eligible event list is re-scanned, in Python, for every origin.

### Why `aggregate` costs what it does

```python
eligible  = [s for s in signals if s.available_time <= forecast_origin]   # pass 1
recent24  = [s for s in eligible if s.available_time > origin - 24h]      # pass 2
recent72  = [s for s in eligible if s.available_time > origin - 72h]      # pass 3
weight    = sum(... for s in recent24)                                    # pass 4
weighted  = sum(... for s in recent24)                                    # pass 5
regulatory = [s for s in recent72 if ...]                                 # pass 6
inflows    = [s for s in recent72 if ...]                                 # pass 7
macro      = [s for s in recent72 if ...]                                 # pass 8
```

Eight full passes per origin, each doing Pydantic attribute access
(`s.available_time`, `s.btc_relevance`, …) on every event. Pydantic v2
attribute reads are not free, and there is no vectorisation.

### The second cost: membership serialisation

`IntelligenceSnapshot.create` builds a membership dict containing **all**
document ids and **all** source hashes, `json.dumps` it, and SHA-256s the
result — once per origin. At 1,000 documents that is ~2,000 hex strings
serialised per origin. This is the 12% in `iterencode`, plus most of the
`executemany` cost, because each persisted snapshot payload embeds the same
2,000 hashes.

### The third cost: per-origin membership filtering

`build_many` runs, per origin:

```python
eligible_documents = [d for d in documents if d.available_at <= origin]        # O(D)
document_ids = {d.document_id for d in eligible_documents}                     # O(D)
eligible_events = [e for e in events
                   if e.available_time <= origin
                   and set(e.source_ids) <= document_ids]                      # O(E), allocates a set per event per origin
```

## What this implies for the optimisation

1. **Do not add database machinery.** The reads are already bounded and
   correct. Indexes, materialised buckets and pre-aggregation would target 2%
   of the cost. B3.1.5 and B3.1.21 are answered by this measurement: not
   justified on this evidence.
2. **Eliminate the per-origin Python scan.** Events arrive sorted by
   `available_time`; window membership is therefore a contiguous range locatable
   by binary search, and the arithmetic is a set of reductions over a columnar
   slice rather than eight Python passes over Pydantic objects.
3. **An event's eligibility is a function of the data, not of the origin.** The
   condition `set(source_ids) <= available_document_ids` is equivalent to
   `max(available_time, max(source document available_at)) <= origin`. Computing
   that effective availability once per event turns a per-origin set-subset test
   into a per-event precomputation.
4. **Cache the membership serialisation across origins that share a document
   set.** The document-id and source-hash portions only change when a new
   document becomes available.
5. **Batch `latest_run_as_of`.** One ordered read plus a binary search replaces
   N queries.

Every one of these preserves feature semantics, window boundaries, missingness
behaviour and snapshot identity exactly — which is the constraint that matters,
and is what the equivalence fixture exists to prove.
