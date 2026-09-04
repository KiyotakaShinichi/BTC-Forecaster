# A3 forward shadow validation

A3 collects genuinely forward, pre-committed quantitative forecasts. It never
places trades, never reconstructs missed forecasts, and never promotes a model.
All official evidence is `research_only`; C0 continues to return `NO_TRADE`,
with live eligibility false and live leverage `0x`.

## A2 model and horizon audit

A2 used expanding daily UTC training windows, 30 scored daily bars, a 30-bar
embargo, 36 folds, and price-path outputs with 95% intervals. Model fitting,
feature selection, and any nested selection occurred inside each training
window. Runtime multiples below are relative to random walk and come from the
frozen A2 benchmark.

| Model ID | Implementation / frozen configuration | Features and output | A2 result | Cost | A3 |
|---|---|---|---|---:|---|
| `random_walk` | `RandomWalk`, 95% interval | log price/returns; flat point path + interval | comparison baseline, not promoted | 1.0x | baseline |
| `random_walk_drift` | `RandomWalkWithDrift`, 95% | endpoint log drift; path + interval | rejected, MAE skill 0.0040 | 1.08x | excluded |
| `historical_mean_return` | trailing 365-bar arithmetic return | close returns; path + interval | rejected, skill -0.0884 | 1.28x | excluded |
| `arima` | `ArimaModel(1,1,1)`, 95% | log price; path + analytic interval | rejected, skill -0.0021 | 119.76x | observation candidate |
| `ets` | additive damped trend, no seasonality | log price; path + interval | rejected, skill -0.0657 | 36.68x | excluded |
| `prophet` | weekly/yearly seasonality | log price; path + interval | rejected, skill -1.2912 | 801.22x | excluded |
| `prophet_xgb_hybrid` | frozen A2 XGB parameters, 200 rounds, GARCH/500 simulations | causal lag/rolling/EMA features; path + interval | rejected, skill -1.4057 | 1898.52x | excluded |
| `xgboost_causal_retuned` | nested 12-candidate selection, up to 300 rounds | causal return features; next-return path + interval | rejected, skill -0.0801 | 1779.09x | excluded |

All A2 candidates remain reproducible from the committed model implementations,
manifest, constraints, and snapshot fingerprint. Optional models require their
declared extras. A3 deliberately freezes only the always-available no-change
baseline and ARIMA, the closest named non-baseline on A2 MAE skill. It does not
reinterpret that negative result as evidence of edge.

## Frozen registry and time contract

The packaged `registry-v1.json` freezes `random_walk`, `arima(1,1,1)`, their
versions/configuration, expanding training rule, feature contracts, BTC-USD,
and a 30-daily-bar horizon. Its canonical SHA-256 is
`2b9ae47f1fe587f0d7846af9ee3de5c76b79800489878f0739811e779e5e4ea4`.
Changing it requires a new registry version; the old file and evidence remain.

All persisted timestamps are aware UTC. A daily bar is labeled by its period
start and becomes usable only after the following UTC midnight. The forecast
origin is the latest completed bar label. Scheduled issuance is 00:30 UTC;
more than six hours late is `LATE_FORECAST`. Absence is recorded as `MISSED`,
never backfilled. Historical reconstruction must use `RETROSPECTIVE_REPLAY` and
cannot enter the official `FORWARD_SHADOW` evidence tier.

The exact hash-verified input snapshot is preserved under its content hash.
Forecasts record provider, retrieval/cutoff times, latest observation, row
count, data/feature/config/source hashes, support diagnostics, and runtime.
Rows whose availability is beyond issuance are structurally excluded.

## Evidence lifecycle

`ForwardForecastRecord`, outcome, and run-manifest contracts are separate.
JSONL ledgers are append-only and hash chained. Forecast IDs are content
addressed. An identical model/origin rerun is a no-op that preserves original
creation time; a different result fails integrity. The run lock spans both
calculation and persistence, and the manifest is written last.

Scoring requires all 30 target bars and waits until the final target bar is
available. Partial or missing horizons remain pending/unavailable. Outcomes are
new immutable records and never modify forecast rows. Candidate/baseline metrics
use paired origins only.

Promotion readiness is descriptive only. Fewer than 100 scored forecasts is
`INSUFFICIENT_FORWARD_EVIDENCE`; uncertainty intervals require at least 50
paired observations and use deterministic block bootstrap. Passing metrics can
only produce `READY_FOR_PROMOTION_REVIEW`, never `PROMOTED`.

## Commands and scheduler exits

```text
btc-forecast shadow-run --snapshot PATH [--dry-run]
btc-forecast shadow-run --live
btc-forecast shadow-score --snapshot PATH
btc-forecast shadow-status
btc-forecast shadow-audit FORECAST_ID
btc-forecast shadow-verify
btc-forecast shadow-export --output PATH
```

`--live` is explicit; CI is entirely synthetic/offline. Exit `0` means success,
`3` means the official-run lock is held, and `4` means nothing was produced.
Other failures use the CLI's existing nonzero failure behavior. Dry-run output
is labeled `UNCOMMITTED_DRY_RUN` and cannot later become official evidence.
