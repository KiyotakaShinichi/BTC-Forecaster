# A2 benchmark — the first statistically defensible comparison (2026-08-29)

**Run id** `20260829T052637Z-3a2c4c3b` · **data** BTC-USD, 3,527 bars
2017-01-01→2026-08-28, snapshot `sha256=056b866b16fee19a…` · **36 expanding
folds**, 30-bar scored horizon, 30-bar embargo, `min_train=900` · **0 fold
failures**. Full provenance in `manifest.json`.

## Headline: nothing was promoted, and two models are significantly worse than doing nothing

| model | MAE | skill vs RW | skill 95% CI | dir. acc (step 31) | vs null | coverage | worst/median | cost× |
| --- | ---: | ---: | :---: | ---: | :---: | ---: | ---: | ---: |
| random_walk_drift | 7,421 | +0.0040 | [−0.277, +0.113] | 0.611 | indist. | 0.981 | 6.4 | 1.0 |
| **random_walk** *(baseline)* | 7,451 | 0.000 | — | 0.333 | **below** | 0.977 | 5.1 | 1.0 |
| arima | 7,466 | −0.0021 | [−0.009, +0.001] | 0.500 | indist. | 0.972 | 5.1 | 121 |
| ets | 7,940 | −0.0657 | [−0.221, +0.003] | 0.361 | **below** | 0.995 | 4.5 | 34 |
| xgboost_causal_retuned | 8,048 | −0.0801 | [−0.204, +0.081] | 0.444 | indist. | 0.962 | 4.8 | 1,858 |
| historical_mean_return | 8,110 | −0.0884 | [−0.550, +0.095] | 0.611 | indist. | 0.919 | 9.0 | 1.3 |
| prophet | 17,072 | −1.2912 | **[−3.68, −0.67]** | 0.639 | indist. | 0.602 | 7.7 | 756 |
| **prophet_xgb_hybrid** | 17,924 | **−1.4057** | **[−4.89, −0.92]** | 0.667 | indist. | 0.818 | 8.6 | 1,910 |

All seven challengers **REJECTED** under the pre-declared policy. None cleared
the 0.02 MAE-skill bar against the naive baseline.

**The one result with statistical support:** `prophet` and `prophet_xgb_hybrid`
have bootstrap skill intervals that **exclude zero on the negative side**. They
are not merely unhelpful — they are significantly worse than predicting no
change. Every other model is statistically indistinguishable from the random
walk.

## The directional finding, which changes how every such number should be read

Over 31-day windows in this sample **BTC rose 61.1% of the time**. So the null
for a directional claim is not 0.5, it is `max(base_rate, 1−base_rate)` = 0.611
— the accuracy of the best *constant* predictor, which requires no model.

| model | predicts "up" | hit rate |
| --- | ---: | ---: |
| random_walk_drift | **100%** | 0.6111 |
| prophet_xgb_hybrid | 66.7% | 0.6667 |
| historical_mean_return | 72.2% | 0.6111 |
| *always-up constant* | 100% | **0.6111** |

`random_walk_drift` scores **exactly the base rate** because it *is* a constant
"up" predictor. Against 0.5 its interval [0.500, 0.722] reads as an edge; it
contains no information. The hybrid's 0.667 is a whisker above what predicting
"up" every day buys for free.

**No model's interval lies above the null.** Two (`random_walk`, `ets`) lie
*below* it — a flat forecast is classified "not up", so it predicts down every
day in a rising sample. That is a metric convention, not a directional failure.

## Where the damage comes from (A2.17)

All 25 largest failures belong to `prophet` and `prophet_xgb_hybrid`, all in
`low_vol/bull`, and the worst cluster is a single origin:

```
origin 2021-11-15  ->  target 2022-01-13   step 59
   predicted move  +146%
   actual move      −33%
   absolute error  $113,762
```

The top of the bull market. Prophet extrapolated the trend into a regime turn.
That single origin is most of why the hybrid's MAE is 2.4× the baseline's.

## Regime analysis, and why no regime model was added (A2.14 / A2.15)

Point-in-time labels at each forecast origin, trailing data only. MAE by regime:

| regime | origins | prophet_xgb_hybrid | random_walk |
| --- | ---: | ---: | ---: |
| high_vol/bear | 5 | 3,494 | 3,347 |
| high_vol/bull | **2** | 12,193 | 12,889 |
| low_vol/bear | 10 | 26,706 | 7,615 |
| low_vol/bull | 16 | 18,622 | 8,201 |
| low_vol/sideways | 3 | 12,800 | 6,120 |

**Decision: no regime model was added.** The descriptive analysis does not
justify one. Cells contain 2–16 origins; two origins cannot estimate anything.
A Markov-switching model would mean fitting additional regime and transition
parameters from data that already cannot separate eight models on the aggregate.
A2.15 permits one "only if justified" — it is not, and adding it would be
complexity for its own sake.

## Residual diagnostics, corrected for multiple testing (A2.18 / A2.11)

Raw p-values point at the Prophet-based models: Ljung-Box **p ≈ 0.005** for both
(residual serial correlation, structure left behind), against **p > 0.83** for
every other model. The hybrid also shows a biased mean residual (t = −2.03,
systematically over-predicting).

**But nothing survives correction.** Across the 24 residual tests,
Benjamini-Hochberg lifts the smallest p-value from 0.005 to **0.062**:

```
prophet:ljung_box             p=0.0047 -> BH 0.0621   not significant
prophet_xgb_hybrid:ljung_box  p=0.0052 -> BH 0.0621   not significant
```

Treat the mis-specification signal as **suggestive, not established**. With 36
residuals these tests have little power, and expected false positives at α=0.05
across 24 tests is 1.2.

## Stability disqualifies everything anyway (A2.16)

`worst_to_median` runs **4.5–9.0 for every model**, against a policy threshold of
3.0. Per-fold MAE spans $288 to $85,084. Even a model with positive skill would
have come out INCONCLUSIVE on stability alone.

## Nested tuning bought essentially nothing (A2.3 / A2.4)

`xgboost_causal_retuned` selected, in the final fold, `max_depth=4,
learning_rate=0.03, subsample=0.6, colsample_bytree=1.0, min_child_weight=5,
reg_lambda=20.0` from 12 candidates over 3 inner splits (3,131–3,311 training
rows each, 90-bar validation, 5-bar inner embargo).

The **spread between best and worst candidate was 0.00014** in validation MAE on
log returns. The search did not meaningfully discriminate — which is the honest
outcome on a target with this signal-to-noise ratio, and is recorded rather than
hidden.

## What the challenger did achieve

`xgboost_causal_retuned` is **17× closer to the baseline** than the legacy hybrid
(−0.080 vs −1.406) and its skill interval contains zero while the hybrid's does
not. Dropping Prophet and targeting stationary log returns removed nearly all of
the gap. It still does not beat the random walk — but it fails like a reasonable
model rather than a broken one, and its residuals are clean where the hybrid's
are not.

## Reproduce

```bash
btc-forecast benchmark --folds 36 --wf-horizon 30 --min-train-bars 900 --embargo-bars 30
```

with `MONTE_CARLO_RUNS=500`. The snapshot hash in `manifest.json` pins the data
vintage; a later run against refreshed data will differ.

## Caveats

- 36 origins is a weak basis for ranking. Most skill intervals are wide enough
  to contain both meaningful improvement and meaningful harm.
- Direction is measured at step 31 (embargo 30 + 1), not step 1. It is the
  shortest distance this fold design scores, and it is a *month* ahead.
- Price-level MAE is dominated by the price level; BTC ranged ~$3.5k–$100k over
  the sample, which is most of the fold dispersion.
- No transaction costs, slippage or position sizing. **Directional accuracy is
  not profitability.**
