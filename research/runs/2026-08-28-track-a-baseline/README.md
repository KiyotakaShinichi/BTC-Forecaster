# Run artifacts — first honest baseline (2026-08-28)

The first run of the Track A platform on live data, and the "after" to pair with
`research/runs/2026-04-02/`. **This is a negative result, and it is the point.**

Config: BTC-USD, 3527 bars 2017-01-01..2026-08-28, snapshot
`sha256=056b866b16fe…`, 3 expanding folds, 30-bar scored horizon,
`min_train_bars=900`, 90-day forward forecast, 300 Monte Carlo paths.
Full provenance in `manifest.json`.

## Result

| model | MAE | MAE std | skill vs RW | dir. acc | coverage (nominal 95%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_walk_drift | 3,480 | 1,355 | **+0.096** | 0.889 | 0.956 |
| ets | 3,802 | 1,529 | +0.012 | 0.444 | 0.956 |
| arima | 3,840 | 1,403 | +0.002 | 0.556 | 0.933 |
| random_walk *(baseline)* | 3,849 | 1,390 | 0.000 | 0.444 | 0.944 |
| historical_mean_return | 4,287 | 1,856 | −0.114 | 0.444 | 0.556 |
| prophet | 9,405 | 6,586 | −1.444 | 0.222 | 0.367 |
| **prophet_xgb_hybrid** | **10,159** | 8,003 | **−1.640** | 0.244 | 0.233 |

## What this shows

**The hybrid is the worst model tested.** It has 2.6× the random walk's error,
directional accuracy of 0.244 — materially *worse* than a coin — and its stated
95% interval contained 23% of outcomes.

Compare the same model's pre-Track-A holdout: 68.97% directional accuracy at
p=0.031. Nothing about the model changed. What changed is that it can no longer
see `close[D]` while predicting `close[D]`, the cutoff is no longer chosen on the
window it is scored on, and the interval now has to widen with horizon. The
apparent skill was the leaks.

Prophet alone is nearly as bad, which locates most of the damage in the
trend/seasonality extrapolation rather than in the XGBoost correction: fitting
yearly seasonality to a non-stationary crypto price and extrapolating it 30 days
produces large, confidently wrong forecasts.

The best performer is `random_walk_drift`, a three-line model. Its +0.096 MAE
skill is **not** significant at 3 folds — `mae_std` (1,355) is comparable to the
gap it wins by (368). Treat it as "indistinguishable from the random walk", not
as a discovery.

## Caveats, stated because they matter

- **3 folds is not enough to rank anything.** Fold dispersion dominates every
  difference except the hybrid's and Prophet's, which are large enough to survive
  it. This run establishes that the leak-free numbers are bad; it does not
  establish an ordering among the top five.
- **`random_walk_drift`'s 0.889 directional accuracy is not skill.** All 90
  scored bars are compared against three origin closes, and BTC trended over
  those windows, so the outcomes are heavily serially dependent. The pooled
  binomial test's own caveat field says exactly this.
- The 90-day forward forecast (\$72,605, 95% PI \$36,540–\$157,148) comes from
  the worst model in the table, because `primary_model` defaults to preserving
  the original headline. It is a scenario. `forecast_summary.json` says so in
  its `interpretation` field.

## Reproduce

```bash
btc-forecast run --folds 3 --wf-horizon 30 --horizon 90 --min-train-bars 900
```

with `MONTE_CARLO_RUNS=300`. The data snapshot hash in `manifest.json` pins the
exact vintage; a later run against refreshed data will differ.
