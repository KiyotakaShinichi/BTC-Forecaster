# BTC Forecaster

A forecasting **research platform** for BTC-USD, built around point-in-time
correctness and honest evaluation.

The question it exists to answer is not "what will BTC do" but **"does this model
beat a random walk, on identical folds, out of sample?"** — and to be able to
report "no" when the answer is no.

It has two layers, and they are honest about different things:

| Layer | What it is | Where it stands |
|---|---|---|
| **Quantitative core** (`btc_forecaster/`) | Causal features, model adapters, walk-forward backtesting, honest evaluation | Working. **Nothing beats the random walk.** |
| **Market intelligence** (`market_intelligence/`) | Point-in-time collection of official announcements, and the machinery to replay a corpus as of any past instant | Working, collecting nothing yet: no host. **Historical validation returned HOLD.** |

Neither has produced a tradeable edge, and the platform is built so that saying
so is easy.

> Research and educational project. Not financial advice. No live trading, no
> execution and no leverage: there is no broker integration in this repository.

---

## What changed, and why it matters

This started as a script that reported **68.97% directional accuracy, p=0.031**
on a 90-day holdout. That number was not real. Three independent defects
produced it, all verifiable against `research/legacy/bayesianCutoff.py`:

**1. The model was shown the answer.** Features and target sat on the same bar:

```python
df["roll_mean_ret_7"] = df["return"].rolling(7).mean()   # includes return[D]
df["ema_7"]           = df["close"].ewm(span=7).mean()   # includes close[D]
train_df["residual"]  = train_df["log_close"] - prophet_train_log
dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df["residual"])
```

The rolling window at row `D` includes `return[D]`, which is a deterministic
function of `close[D]`. `ema_7[D]` contains `close[D]` outright. The target was
the residual of `log_close[D]`. The model was asked to predict a closing price
it had been handed.

**2. The cutoff was chosen on the test set.** 48 candidate start dates were
scored on the last 90 days of the series, and the winner's holdout was
*that same window*. With 48 candidates, a 0.69 maximum is close to what
selection noise alone produces.

**3. The uncertainty bands did not widen.** The Monte Carlo drew one independent
shock per horizon step instead of accumulating along the path. GARCH conditional
variance mean-reverts within weeks, so the 95% band was roughly constant width a
year out, when it should grow like `sqrt(h)`. Nothing measured coverage, so
nothing noticed.

Every one of these now has a test that fails if it returns.

**What the corrected numbers look like.** Same model, same data, leaks removed
([full run](research/runs/2026-08-28-track-a-baseline/README.md)):

| model | MAE | skill vs RW | dir. acc | coverage (nominal 95%) |
| --- | ---: | ---: | ---: | ---: |
| random_walk_drift | 3,480 | +0.096 | 0.889 | 0.956 |
| random_walk *(baseline)* | 3,849 | 0.000 | 0.444 | 0.944 |
| prophet | 9,405 | −1.444 | 0.222 | 0.367 |
| **prophet_xgb_hybrid** | **10,159** | **−1.640** | 0.244 | 0.233 |

The hybrid is the worst model tested: 2.6x the random walk's error, directional
accuracy below a coin, and a stated 95% interval that contained 23% of outcomes.
Nothing about the model changed — only that it can no longer see the price it is
predicting. The apparent skill was the leaks.

(3 folds, so this ranks nothing among the top five — `mae_std` swamps those
gaps. It does establish that the leak-free numbers are bad.) The frozen
before-state is in
[`research/runs/2026-04-02/`](research/runs/2026-04-02/README.md).

Both numbers are now **tamper-evident**: `btc_forecaster/evidence.py` records
them as data and `tests/test_evidence.py` reads the archived JSON back, failing
if either drifts. Retuning the corrected reference until it looks better is
forbidden; building a named challenger and scoring both through identical folds
is the sanctioned route. See [`docs/benchmark.md`](docs/benchmark.md).

---

## Architecture

```
btc_forecaster/
    timebase.py      the time contract: UTC, event_time, available_time, origins
    config/          frozen run configuration
    data/            schema contract, providers, hash-verified snapshots
    features/        causal features, train-only selection, supervised alignment
    models/          ForecastModel contract, baselines, ARIMA/SARIMAX/ETS,
                     Prophet, XGBoost hybrid, GARCH volatility
    evaluation/      point and interval metrics, skill scores
    backtesting/     the walk-forward engine every model is scored through
    diagnostics/     stationarity, autocorrelation, ARCH, normality + corrections
    artifacts/       run output and its manifest
    pipeline.py      orchestration
    cli.py           btc-forecast
```

```
market_intelligence/
    collection/      feeds, syndication parsing, matching, clustering, readiness
    ops/             scheduled runs, storage paths, integrity, backup, watchdog
    storage/         schema.py (the only DDL), store.py (every write),
                     queries.py (every read, in a total order)
    commands/        one module per command family, behind a registry
    corrections.py   the append-only ledger and the eligibility contract
    reports.py       corpus status, providers, operations -- shared with the API
    logs.py          structured JSON logging, and what it refuses to print
    b4/              the historical event-study engine
    cli.py           btc-intel: the parser, and nothing else
```

Nothing in either core imports Prophet, XGBoost, arch, matplotlib, yfinance or
the network at module scope, so both are importable — and the whole unit suite
runs — without them and offline.

### The time contract

A daily bar labelled `D` covers `[D, D+1)`. Its close is only determined at
`D+1`, so:

```
event_time     = D + 1 day        the close is determined
available_time = event_time + publication_lag
```

A forecast issued at `forecast_origin` may use an observation **iff**
`available_time <= forecast_origin`. That single inequality is the whole leakage
rule, and `assert_available()` is its only enforcement point.

Features are **causal** (row `D` uses bars `<= D`) but that is not enough to be
*usable*: row `D` still contains bar `D`'s close. `to_supervised(step=1)` shifts
features so the row predicting bar `T` comes from bar `T-1`. `step=0` is
rejected outright — it is precisely the original defect.

---

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on POSIX

# Reproducible: the exact transitive closure, every artifact verified by hash.
pip install --require-hashes -r requirements.lock
pip install --no-deps -e .
```

Or resolve fresh against the declared bounds — faster to iterate on, not
byte-reproducible:

```bash
pip install -e ".[all]" -c constraints.txt
```

Python 3.11 or newer. `requirements.lock` is generated by `scripts/lock.sh`;
see [`DEPENDENCIES.md`](DEPENDENCIES.md) for which file is the authority.

Nothing above needs configuring — the CLI is flag-driven and the test suite is
offline. For a service (the API, or the collector on a host), copy the template:

```bash
cp .env.example .env               # then edit; .env is gitignored
set -a; . ./.env; set +a           # nothing loads it automatically, by design
```

[`.env.example`](.env.example) documents every variable the code reads, sorted
into REQUIRED / OPTIONAL / DEPLOYMENT-ONLY / RESEARCH-ONLY. A test checks it
against the source, so a variable that exists and is undocumented fails the
build rather than surfacing on a host.

```bash
btc-forecast models               # what can be built here
btc-forecast diagnose             # stationarity / autocorrelation / ARCH
btc-forecast backtest             # walk-forward comparison, no forward forecast
btc-forecast benchmark            # the full study: nested tuning, promotion verdicts
btc-forecast run                  # backtest, then forecast forward
```

Useful flags:

```bash
btc-forecast backtest --models random_walk,arima,ets --folds 8 --wf-horizon 30
btc-forecast run --mode rolling --window-bars 730 --embargo-bars 5
btc-forecast run --horizon 90 --primary-model arima --no-plots
```

Every flag has an environment-variable equivalent (`TICKER`, `HORIZON_DAYS`,
`OUTPUT_DIR`, `WF_FOLDS`, …) — the original names are unchanged.

---

## Reading a run

`btc-forecast run` prints a model comparison, then a verdict:

```
data      : 3527 bars 2017-01-01..2026-08-28
            provider=yfinance sha256=056b866b16fe
backtest  : 3 expanding folds, horizon=30, embargo=0, min_train=900

model comparison (walk-forward means, lower MAE is better):
                               mae        rmse    mase  directional_accuracy  interval_coverage
random_walk_drift       3,480.1898  4,455.4098  7.7887                0.8889             0.9556
ets                     3,801.7574  5,018.6515  8.4151                0.4444             0.9556
arima                   3,840.2669  5,033.7429  8.7406                0.5556             0.9333
random_walk             3,848.5724  5,042.3343  8.7847                0.4444             0.9444
historical_mean_return  4,286.7264  5,611.8315  9.2442                0.4444             0.5556
prophet                 9,404.9897 10,036.7518 18.0914                0.2222             0.3667
prophet_xgb_hybrid     10,158.5059 10,791.1421 18.9160                0.2444             0.2333

verdict:
  prophet_xgb_hybrid did NOT beat random_walk on walk-forward MAE across
  identical folds. The forward forecast should be read as a scenario, not a
  prediction, and the added complexity is not currently earning anything.
```

(Real output, not an illustration — the run is archived in
[`research/runs/2026-08-28-track-a-baseline/`](research/runs/2026-08-28-track-a-baseline/README.md).)

That verdict is the product. A platform that can only report success is not
measuring anything.

**How to read the columns.** `mase < 1` beats a naive forecast on the training
scale. `interval_coverage` should sit near the nominal level — both 0.60 and
1.00 are miscalibrated, which is why coverage is reported but never used to rank.
`directional_accuracy` is now measured against the last observed close (the
tradeable quantity); `path_directional_accuracy` is the original metric, kept so
historical numbers stay interpretable. Check `mae_std` before believing any of it.

---

## The A6 model zoo

Forty runnable models across eight families -- baselines, autoregressive and
state-space, GARCH, linear ML, trees, kernel/local, quantile and conformal, and
seven compact deep architectures on a numpy autodiff engine with
finite-difference-checked gradients.

```bash
python -m btc_forecaster.research.model_zoo --list
python -m btc_forecaster.research.model_zoo --train-rows 1000 --output research/runs/a6-model-zoo
```

Nothing beat the random walk. Twenty-eight of thirty-nine models are
significantly different from the naive forecast after Benjamini-Hochberg and all
twenty-eight are worse; none of the forty is positive in every temporal block;
mean pairwise error correlation is 0.945. The models that lose hardest are the
ones with capacity.

**1,000-row A6 results are resource-constrained exploratory evidence and do not
supersede A2's historical promotion study.** A6 fits every model once on a
deterministic 1,000-row budget and scores it on one frozen holdout; A2 ran 36
walk-forward folds. Nothing in A6 is promoted, nothing can be, and the paper
engine stays fail-closed. See [`docs/model-zoo.md`](docs/model-zoo.md) and the
run evidence in [`research/runs/a6-model-zoo/`](research/runs/a6-model-zoo/).

## The A7 walk-forward robustness layer

A6's question asked again: from 1,417 daily origins rather than one block, at
1, 3, 7 and 30 bars, with 250 to 2,000 rows of rolling history and with all of
it, across early, middle and late periods. Eleven A6 models, their
configurations frozen, every horizon scored against the random walk at the same
origins. Preregistered before the run; the input hash-pinned; the output
byte-reproducible.

```bash
python -m btc_forecaster.research.walk_forward config    # prints the configuration digest
python -m btc_forecaster.research.snapshot verify data/snapshots/BTC-USD --expect 39b93e34520a496692ccc2156b8fe5c88c88b31bd9e9903a6fad74f282004bd3
python -m btc_forecaster.research.walk_forward run --data data/snapshots/BTC-USD \
    --expect-input 39b93e34520a496692ccc2156b8fe5c88c88b31bd9e9903a6fad74f282004bd3 --output <scratch dir> --workers 4
python -m btc_forecaster.research.walk_forward verify <scratch dir>
```

A reproduction is a run whose result digest equals the committed one,
`b3fba7227e82e13a…`, and it needs the pinned snapshot, which is not committed.
A fresh clone checks the committed evidence instead:

```bash
python -m btc_forecaster.research.walk_forward verify --committed research/runs/a7-walk-forward
```

That checks every committed canonical file against its manifest and — with the
recorded hash of the predictions it does not have — against the result digest.
Plain `verify` names `predictions.csv.gz` as absent: the predictions are
regenerated from the snapshot, not redistributed.

**Nothing beats the random walk here either.** 0 of 200 configurations are
significantly better than the naive forecast after Benjamini-Hochberg;
103 are significantly worse. The largest positive skill is +0.37%, not
significant, and the best configuration at every horizon loses in the most
recent period. Decision: `ROBUSTLY_UNINTERESTING`. No model is promoted, no A8
is proposed, and the paper engine stays fail-closed.

A2, A6 and A7 are different experiments -- a 31-step price path over 36
folds, next-bar forecasts over one 705-bar holdout after a single 1,000-row
fit, and a walk-forward over four horizons -- and their numbers are not
comparable. They agree on the question they share. See
[`docs/walk-forward.md`](docs/walk-forward.md) and the evidence in
[`research/runs/a7-walk-forward/`](research/runs/a7-walk-forward/).

## Quantitative research is frozen

**`QUANT_RESEARCH_FROZEN`.** A2, A6 and A7 asked one question three ways and got
one answer: no tested model beats the naive forecast. The project does not
currently possess validated evidence of a tradable predictive edge. That is a
claim about these models under these designs, not about whether BTC can ever
be forecast. No model is promoted until it clears `NO_MODEL_PROMOTION_UNTIL`,
live trading stays disabled, and `tests/test_quant_freeze.py` enforces both.

[`docs/quant-research-status.md`](docs/quant-research-status.md) is the record:
lineage, digests, policy and provenance.
[`docs/quant-research-handoff.md`](docs/quant-research-handoff.md) is what was
learned, where the evidence stops, and what would justify reopening.

---

## Models

| Name | Family | Notes |
| --- | --- | --- |
| `random_walk` | baseline | `P[T+h] = P[T]`. The hypothesis to disprove. |
| `random_walk_drift` | baseline | Classical endpoint drift. |
| `historical_mean_return` | baseline | Trailing arithmetic mean simple return. |
| `arima` / `arima_auto` | statistical | `arima_auto` selects order by AIC **inside each fold**. |
| `sarimax` | statistical | Seasonal + exogenous dynamic regression seam. |
| `ets` | statistical | Damped additive trend on log price. |
| `prophet` | structural | Preserved from the original pipeline. |
| `prophet_xgb_hybrid` | hybrid | The original headline model, with the leaks fixed. |

Adding one is `registry.register(...)`; it is then scored through the same folds
against the same baseline with the same metrics.

---

## API and dashboard

```bash
pip install -e ".[all]"
python api_server.py               # http://localhost:8010  (docs at /docs)
```

`GET /health` · `GET /status` · `POST /run` · `GET /latest` · `GET /artifacts`

`POST /run` accepts the original fields plus `models`, `primary_model`,
`baseline_model`, `walk_forward_folds`, `walk_forward_horizon`,
`min_train_bars` and `embargo_bars`. Set `API_TOKEN` to require an `X-API-Key`
header; `ALLOWED_OUTPUT_ROOT` confines any `output_dir` override;
`S3_ARTIFACT_BUCKET` uploads artifacts after a successful run.

A fresh clone has an empty `out/`, so `/latest` returns 404 until a run happens.
That is intentional — see [`ARTIFACTS.md`](ARTIFACTS.md).

---

## Docker

```bash
docker build -t btc-forecaster .                    # batch run
docker run --rm -v "$PWD/out:/app/out" btc-forecaster

docker build -f Dockerfile.api -t btc-forecast-api .
docker run --rm -p 8010:8010 -v "$PWD/out:/app/out" btc-forecast-api
# or: docker compose up --build
```

---

## Tests

```bash
pytest                                    # everything, no network, no live data
pytest tests/test_leakage.py -v           # the quantitative causality guard
pytest tests/test_intelligence_*.py -q    # the collector
bash scripts/coverage.sh                  # quantitative core, gated at 78%
```

The two suites run under separate CI jobs because they have different
dependencies and different lint contracts; between them they cover the whole
repository. `pytest` with no arguments runs both, which needs `.[all]`.

The suite is offline by construction: every series comes from seeded generators
in `btc_forecaster/testing.py`. `tests/test_leakage.py` is the centrepiece — it
verifies causality by **perturbing the future and asserting the past does not
move**, which catches centred windows, negative shifts and whole-sample
normalisation that reading a column name will not. It also proves the guard
itself fires, because a check that never fails is worthless.

---

## Reproducibility

Every run writes `manifest.json` with the data snapshot's SHA-256, the full
configuration, model hyperparameters, fold boundaries, package versions and the
git commit. Data is pinned as a hash-verified snapshot rather than re-fetched,
so repeated runs read the same bytes — `yfinance` returns a different series
every day and silently revises history.

A promoted research run without its manifest is an anecdote.

---

## Layout

- [`ARTIFACTS.md`](ARTIFACTS.md) — what is committed, what is generated, and why
- [`docs/benchmark.md`](docs/benchmark.md) — what the benchmark measures, how to read it, what it cannot tell you
- [`docs/seams.md`](docs/seams.md) — where external signals attach (Track B contract)
- [`docs/model-zoo.md`](docs/model-zoo.md) — Track A6: forty models on a 1,000-row budget, and why it does not supersede A2
- [`docs/walk-forward.md`](docs/walk-forward.md) — Track A7: A6's question from 1,417 origins at four horizons, preregistered, and why nothing survived
- [`docs/quant-research-status.md`](docs/quant-research-status.md) — the quantitative freeze: A2 → A6 → A7 lineage, the promotion policy, the live-trading gate, the provenance audit
- [`docs/quant-research-handoff.md`](docs/quant-research-handoff.md) — what the quant research established, where its evidence stops, and what would justify reopening it
- [`docs/b5-event-study.md`](docs/b5-event-study.md) — Track B5: is there a point-in-time intelligence corpus to study? Preregistered Gate 1, and why the answer is not yet
- [`DEPLOYMENT.md`](DEPLOYMENT.md), [`AWS_BACKEND_API.md`](AWS_BACKEND_API.md) — deployment
- [`DEPENDENCIES.md`](DEPENDENCIES.md) — the one dependency contract, and why the other files exist
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to build it, what CI enforces, and the four rules that are about the science rather than the code
- [`CHANGELOG.md`](CHANGELOG.md) — the milestones, including the negative results
- [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md) — running the collector on a host
- [`deploy/COLLECTION_FREEZE.md`](deploy/COLLECTION_FREEZE.md) — the frozen collection contracts
- [`corrections/README.md`](corrections/README.md) — why invalidated evidence is kept
- `research/market_intelligence/b4/` — the historical event study and its HOLD verdict
- `research/legacy/` — superseded implementations, kept deliberately
- `research/runs/` — frozen, dated run evidence

---

## Market intelligence

The second layer asks whether *what regulators and central banks actually
announce* helps forecast BTC. Answering that honestly turns out to be mostly a
data-provenance problem, not a modelling one.

### Availability is retrieval, never publication

A document becomes usable at the moment it was **retrieved**, not the moment its
publisher stamped it. A three-week-old press release first seen today became
usable today; treating its publication date as availability fabricates three
weeks of hindsight. Rediscovering a document later never moves that first
availability forward.

This is the same inequality as the quantitative core's time contract, applied to
a different kind of evidence, and it has the same consequence: it cannot be
recovered after the fact. A corpus collected late is not the corpus that existed.

### Historical validation returned HOLD

`research/market_intelligence/b4/` holds a full event-study run — preregistered
hypotheses, matched controls, block bootstrap, Benjamini-Hochberg correction —
against historical intelligence. The finding was that **nothing survives, and
most of it was never testable**: no point-in-time-valid historical corpus exists,
because availability cannot be reconstructed for documents collected after the
fact. The verdict was HOLD rather than a negative result, which is a different
and more honest claim: the question was not answerable with that evidence.

### Forward collection

So the corpus is built forward instead, one bounded cycle at a time, from six
public government feeds (SEC press and administrative proceedings, two Federal
Reserve feeds, CFTC, BLS). Two further feeds are retired but kept resolvable, so
a manifest that names them still reads.

The collector is a **frozen deployment candidate** and is **not deployed** — no
persistent host is available, so nothing is currently collecting. See
[`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md) to run it and
[`deploy/COLLECTION_FREEZE.md`](deploy/COLLECTION_FREEZE.md) for the contracts
that are frozen and what changing one requires.

`btc-intel corpus-status` reports readiness against B4's thresholds. It says
`NOT_READY`, and will for months: the remaining bottleneck is **elapsed calendar
time**, which is not an engineering problem and cannot be worked around.

### Corrections, not deletions

An observation that turns out to be an artefact of a defect is **invalidated,
never deleted**. A correction is a new row in an append-only ledger; the
observation stays exactly where it was written. Two views follow — a raw one for
audit and integrity, and an eligible one that research counts — so a bad record
is *preserved physically and excluded scientifically*. Deleting it would leave a
corpus that cannot explain its own history, and a corpus whose contents can
change without leaving evidence is not evidence.

`tests/test_intelligence_silent_empty.py` is the counterpart to
`tests/test_leakage.py`: fifteen defects that each made a cycle report success
while writing nothing, which is the failure mode this layer is most prone to
because its output is *plausible* — a green run with zero documents looks exactly
like a quiet news day.

### Point-in-time event study (B5): stopped at Gate 1

B5 asked whether independently timestamped external information is associated
with BTC behaviour beyond what price alone shows. Its first gate, preregistered
with B4's own thresholds, asked whether there is a point-in-time corpus to study —
and there is not: six documents and seven events from two days of collection,
none of which clears B4's extraction-confidence floor. Decision
**`INTELLIGENCE_CORPUS_INSUFFICIENT`**; no event study was run.

Two things block it besides elapsed time. The collector's rule-based extractor
assigns every event confidence 0.35 by construction, below the 0.5 floor, so it
can never produce a countable event. And `corpus-status` never supplies
collection coverage, so its readiness report cannot say `READY_FOR_VALIDATION`
however long collection runs. Both are recorded, neither is changed by B5. See
[`docs/b5-event-study.md`](docs/b5-event-study.md) and the committed result in
[`research/market_intelligence/b5/gate1/`](research/market_intelligence/b5/gate1/).

---

## Known limitations

- **Directional accuracy is not profitability.** No transaction costs, slippage
  or position sizing are modelled. Nothing here says a strategy makes money.
- **The market may simply not be forecastable at this frequency.** A benchmark
  that keeps returning "no skill" is not necessarily broken — that is the result
  this platform was built to be able to report.
- Multi-step hybrid forecasts are **recursive**: only the origin bar's features
  are real, later steps are built from the model's own simulated prices, and
  volume cannot be simulated at all. Long-horizon output is a scenario.
- The legacy hybrid's XGBoost hyperparameters remain the frozen output of an
  Optuna search run against a different cutoff, a different feature set and a
  leaky evaluation. They are deliberately **not** retuned, so the preserved
  reference stays comparable to itself; `xgboost_causal_retuned` is the
  challenger that does tune, nested inside each fold.
- Monte Carlo shocks are Gaussian. Crypto returns are not; Jarque-Bera rejects
  normality decisively, so tail risk is understated even with correct
  accumulation.
- Interval coverage is measured but not yet calibrated against.
- The binomial direction test remains uncorrected for multiple comparisons; its
  caveat travels with the result rather than being buried in a comment.
- **The intelligence corpus is nearly empty and cannot be hurried.** It holds a
  handful of documents from a few manual cycles. B4 readiness needs 30 events
  across 3 publishers spanning 180 days; that is months of elapsed time, and
  collecting faster does not help because availability is retrieval time.
- **Nothing is collecting.** The collector is verified and frozen but has no
  host, so the corpus is not growing. `ops-watch` reports `COLLECTION_STALE`,
  which is the absence of a deployment rather than a fault.
- The administrative-proceedings feed titles its entries with respondent names,
  so topical filtering matches almost nothing from it. That is a property of the
  source, recorded as `subject_bearing=False`, and deliberately not worked
  around by loosening the filter.
- The two layers are not connected. There is no fusion model, and the seams in
  [`docs/seams.md`](docs/seams.md) describe where one could attach, not one that
  does.

## License

Add a license before open-sourcing.
