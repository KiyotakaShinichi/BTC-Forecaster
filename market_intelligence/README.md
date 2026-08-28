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
