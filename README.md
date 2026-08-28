# BTC Forecaster

A forecasting **research platform** for BTC-USD, built around point-in-time
correctness and honest evaluation.

The question it exists to answer is not "what will BTC do" but **"does this model
beat a random walk, on identical folds, out of sample?"** — and to be able to
report "no" when the answer is no.

> Research and educational project. Not financial advice.

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

Nothing in the core imports Prophet, XGBoost, arch, matplotlib or yfinance at
module scope, so it is importable — and the whole unit suite runs — without them
and without a network connection.

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
pip install -e ".[all]" -c constraints.txt
```

```bash
btc-forecast models               # what can be built here
btc-forecast diagnose             # stationarity / autocorrelation / ARCH
btc-forecast backtest             # walk-forward comparison, no forward forecast
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
pytest                      # full suite, no network, no live data
pytest tests/test_leakage.py -v
```

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
- [`docs/seams.md`](docs/seams.md) — where external signals attach (Track B contract)
- [`DEPLOYMENT.md`](DEPLOYMENT.md), [`AWS_BACKEND_API.md`](AWS_BACKEND_API.md) — deployment
- `research/legacy/` — superseded implementations, kept deliberately
- `research/runs/` — frozen, dated run evidence

---

## Known limitations

- Multi-step hybrid forecasts are **recursive**: only the origin bar's features
  are real, later steps are built from the model's own simulated prices, and
  volume cannot be simulated at all. Long-horizon output is a scenario.
- The XGBoost hyperparameters are the frozen output of an Optuna search run
  against a different cutoff, a different feature set and a leaky evaluation.
  Re-tuning them under the walk-forward engine is open work.
- Monte Carlo shocks are Gaussian. Crypto returns are not; Jarque-Bera rejects
  normality decisively, so tail risk is understated even with correct
  accumulation.
- Interval coverage is measured but not yet calibrated against.
- The binomial direction test remains uncorrected for multiple comparisons; its
  caveat travels with the result rather than being buried in a comment.

## License

Add a license before open-sourcing.
