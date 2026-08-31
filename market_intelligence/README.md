# External Market Intelligence (Track B)

This package collects and normalizes external information into structured, source-backed signals. It does not trade, select forecast weights, or produce a final forecast.

## Boundaries

- `SearchProvider` supports deterministic fixtures, configured RSS feeds, and generic JSON search APIs. Operators must supply credentials through environment variables and configure sources they are allowed to access.
- `EventExtractor` supports deterministic fixtures and a provider-independent structured-LLM adapter. The LLM response is parsed as JSON and fully validated; prose never becomes a feature.
- `Document.available_at` and `EventSignal.available_time` are the only timestamps used for point-in-time eligibility. Live adapters conservatively use retrieval time unless a provider supplies a provable first-seen time, and extraction rejects signals dated before their cited sources. Callers must enforce `available_at <= forecast_origin`; the fixture provider, DuckDB store, and aggregator do so.
- Documents retain URL, publisher, title, publication/retrieval/availability timestamps, author, query, provider, and SHA-256 content provenance. Events retain one or more document IDs.
- Entity weights use `EntityConfig` and are tunable inputs to empirical evaluation—not statements about fame, politics, or causal importance.

## Future fusion contract

The quantitative pipeline should consume only the numeric dictionary returned by:

```python
features = FeatureAggregator().aggregate(signals, forecast_origin)
```

Current keys are `sentiment_mean_24h`, `sentiment_weighted_24h`, `event_count_24h`, `high_relevance_event_count_24h`, `regulatory_signal_72h`, `whale_exchange_inflow_signal_72h`, and `macro_news_signal_72h`. The caller should join these by `forecast_origin` and learn their value/weight inside point-in-time walk-forward evaluation.

Live provider work remaining: select contracted search/news/market/on-chain vendors, map their response formats, define operator-approved RSS feeds, load credentials from deployment environment, and empirically calibrate entity importance and extraction quality.

## B2 operations

`configuration.py` validates provider and watchlist configuration. `QueryPlanner` creates stable query IDs from enabled watch entities without interpreting fame or configured importance as market direction. The example configuration deliberately contains no asserted live feed URLs; operators must add verified, permitted sources.

`MultiProviderRetriever` provides bounded retries, exponential backoff/jitter hooks, timeout isolation, rate-limit reporting, and partial-success results. Cross-provider deduplication uses canonical URLs, exact content hashes, or the corroborated combination of normalized title, publisher, and close publication time. All retrieval provenance is retained.

`run_intelligence_cycle` is the single operational entry point:

```text
plan → retrieve → normalize/deduplicate → cache → extract → validate
     → durable store/watermarks → aggregate → quality/health → manifest
```

Extraction failures quarantine metadata while preserving successfully retrieved documents. Successful evidence storage and watermark advancement share a DuckDB transaction. Failed provider/query attempts never advance watermarks. Missingness and provider health are returned separately from numeric features, so an outage is not interpreted as zero sentiment.

`ReplayService.replay(forecast_origin, ...)` includes only documents with `available_at <= forecast_origin` and events with `available_time <= forecast_origin` whose sources are also eligible. It persists a membership fingerprint rather than copying raw content. Feature definitions and missing-value semantics are versioned in `features.py`.

## CLI and automation

Install the layered requirements, then invoke the scheduler-friendly script:

```powershell
python -m pip install -r requirements-market-intelligence.txt
python btc-intel.py --db data/intelligence.duckdb collect --config market_intelligence/config.example.json --origin 2026-08-28T08:00:00Z --manifest data/manifests/run.json
python btc-intel.py --db data/intelligence.duckdb replay --origin 2026-08-28T08:00:00Z
python btc-intel.py --db data/intelligence.duckdb aggregate --origin 2026-08-28T08:00:00Z
python btc-intel.py --db data/intelligence.duckdb health
python btc-intel.py --db data/intelligence.duckdb quality --origin 2026-08-28T08:00:00Z
```

Commands contain no scheduler dependency and can be called by cron, GitHub Actions, or Windows Task Scheduler. `backfill` records deterministic bounded windows and an atomic resumable progress manifest; an operational deployment should bind its window callback to the same collection service with an approved provider configuration.

The RSS adapter is for operator-verified public feeds. The generic JSON adapter is a seam for legitimate contracted search APIs; no vendor contract is assumed. `SocialStatementProvider` intentionally requires a lawful API implementation and contains no X/Twitter scraping. `WhaleDataProvider` carries explicit transfer context and defaults unknown transfers to `UNKNOWN`; it never invents wallet labels.

The manually defined gold labels in `gold_fixtures.json` exercise regulation, monetary policy, ETF, exchange incident, whale, irrelevant, ambiguous social, duplicate, and multi-source cases. `evaluation.py` compares extraction quality only; its outputs are not forecast weights, and the small fixture set is not evidence of production accuracy.

## B3.1 historical feature materialisation

`replay-dataset` builds a point-in-time-safe intelligence feature matrix over a
generated origin schedule, in bounded resumable chunks, and registers it in an
immutable dataset catalog:

```powershell
python btc-intel.py --db data/intelligence.duckdb replay-dataset --start 2026-01-01T00:00:00Z --end 2026-03-01T00:00:00Z --frequency HOURLY --output data/features.parquet --manifest data/features.manifest.json --config-fingerprint PROD_V1
python btc-intel.py --db data/intelligence.duckdb replay-dataset --extend <dataset_id> --start 2026-03-01T00:00:00Z --end 2026-04-01T00:00:00Z --output data/features-v2.parquet --manifest data/features-v2.manifest.json --config-fingerprint PROD_V1
python btc-intel.py --db data/intelligence.duckdb catalog
```

The same `HistoricalDatasetService` backs `POST /replay/dataset`, `GET /datasets`
and `GET /datasets/{dataset_id}`, so the CLI and API cannot drift; a test asserts
it. An interrupted build resumes from the first incomplete chunk, and the
manifest is written last, so partial output is never mistaken for a dataset.
Extension fails closed on a changed feature contract, configuration, cadence,
schedule continuity, or source history.

`--mode REFERENCE` runs the pre-optimisation engine, which is retained as the
correctness oracle. Both modes produce the same dataset id: mode is provenance,
not semantics. Measurements and the decisions they drove are in
`research/b31/PROFILE.md` and `research/b31/PERFORMANCE.md`.

## B3 service and replay datasets

The import-safe FastAPI factory, SQL-filtered read repository, schema-v3 migration, replay dataset builder, offline demo, dashboard, observability contracts, static checks, and service operations are documented in `docs/market-intelligence-service.md`. Representative local timings are recorded in `docs/market-intelligence-performance.md`.
