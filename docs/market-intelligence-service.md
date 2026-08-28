# BTC Market Intelligence Service

This subsystem is an independently operated, point-in-time evidence service for research. It is **not trading advice, not a trading bot, not a causal inference system, and not a guarantee of price movement**.

## Architecture and safety

Imports are side-effect free. Network retrieval, credentials, DuckDB connections, collection, replay, and watermark advancement occur only through explicit constructors or function calls. The standalone FastAPI factory never modifies the forecasting API or model code.

The flow is:

```text
configured providers → planned queries → bounded retrieval → provenance deduplication
→ validated extraction → durable evidence/watermarks → run/quality registries
→ immutable snapshots → point-in-time replay feature datasets
```

Every endpoint that accepts `forecast_origin` filters documents by `available_at <= forecast_origin` and events by `available_time <= forecast_origin`. Event detail also verifies every supporting document was eligible. Publication time is never treated as proof the system knew an item earlier than its availability time.

## Installation and checks

Runtime dependencies are isolated from forecasting dependencies:

```powershell
python -m pip install -r requirements-market-intelligence.txt
python -m pip install -r requirements-market-intelligence-dev.txt
python -m ruff check --config market_intelligence/ruff.toml market_intelligence btc-intel.py btc-intel-api.py tests/test_intelligence_*.py
python -m mypy --config-file market_intelligence/mypy.ini market_intelligence btc-intel.py btc-intel-api.py
python -m pytest -q --cov=market_intelligence --cov-branch --cov-fail-under=75 tests
python -m compileall -q market_intelligence tests btc-intel.py btc-intel-api.py
```

The initial B3 measurement was 79%; the final statement/branch result is 83.13%. CI ratchets at a conservative 75% and must not be lowered merely to pass.

## API

Start the local research service explicitly:

```powershell
python btc-intel-api.py --db data/intelligence.duckdb --host 127.0.0.1 --port 8081
```

Read endpoints cover health/readiness, runs, documents, events, event traceability, timeline, provider and extraction observability, quality, snapshots, versioned features, quarantine, metrics, and the `/dashboard` research view. `POST /replay` builds a point-in-time snapshot. Pagination is capped at 500, time ranges at 366 days, SQL predicates are fixed and parameterized, and no API accepts a filesystem path or SQL fragment. Typed operational failures are returned without stack traces. `X-Request-ID` supports request-to-storage/replay correlation.

Readiness checks only local DuckDB access and schema compatibility. It does not require internet, a live provider, or a paid LLM.

## CLI and offline demo

CLI and API call the same `IntelligenceReadService`, `SnapshotService`, storage, quality, aggregation, and replay implementations.

```powershell
python btc-intel.py demo --output-dir demo-output
python btc-intel.py --db data/intelligence.duckdb replay --origin 2026-08-28T08:00:00Z
python btc-intel.py --db data/intelligence.duckdb dataset --origins origins.txt --output replay.parquet --manifest replay.manifest.json --config-fingerprint CONFIG_SHA
```

Demo records are curated synthetic fixtures under a dedicated demo database. They are never presented as current intelligence and require no network, keys, or LLM.

## Storage and integrity

DuckDB schema version `3` is installed through an idempotent, forward-only migration. Existing B/B2 document, event, watermark, quarantine, snapshot, and health tables remain readable. Storage rejects orphan events, unknown snapshot members, corrupted snapshot fingerprints, unknown feature-contract versions, non-monotonic watermarks, and attempts to rewrite historical run facts.

Runs use deterministic statuses: `SUCCESS` for complete valid collection, `PARTIAL_SUCCESS` when some providers fail but evidence remains, `FAILED` when none succeed, and `DEGRADED` when collection succeeds but quality validation fails.

## Replay dataset contract

The canonical export is Parquet; CSV is available for inspection. Each row contains `forecast_origin`, the seven feature-contract-v1 intelligence features, and separate `provider_coverage_ratio`, `queries_failed`, and `source_stale_flag` fields. A zero event or sentiment value therefore remains distinguishable from missing provider coverage.

Dataset manifests are written last and contain the dataset ID, feature contract, origin bounds/count, snapshot fingerprints, provider/extractor versions, configuration fingerprint, columns, file hash, Git SHA, and export format. BTC prices, returns, targets, labels, final weights, and fusion logic are deliberately absent.

## Quality, health, dashboard, and limitations

Quality is a scoreboard of individual counts, not an arbitrary scalar score. Provider observability reports uptime, latency, documents per successful query, failures, rate limits, and staleness. Extraction observability reports method counts, rejection categories, events per document, and confidence/relevance distributions. These are operational measurements, never bullish/bearish features.

The dashboard is labeled `MARKET INTELLIGENCE / RESEARCH` and includes provider health, run/snapshot metadata, document/event counts, event types, tracked entities, quality, quarantine, and feature history without claiming price causality.

The gold set remains nine curated regression cases, versioned by `gold_manifest.json`; its report explicitly states that the sample is too small to establish production accuracy. Live deployment still requires contracted providers, verified official feed URLs, environment-provisioned credentials, vendor mappings, and empirical extraction calibration.
