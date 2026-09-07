# The A2 benchmark

What it measures, how to read it, and what it cannot tell you.

---

## The result this project must not bury

The original pipeline reported **68.97% directional accuracy, p = 0.031**. That
number was produced by leakage, not skill. Re-scored with the leaks removed, the
same model gets **24.4% directional accuracy and −1.64 MAE skill against a random
walk** — worse than a coin, and 2.6× the naive forecast's error.

Both results are frozen and tamper-evident in `btc_forecaster/evidence.py`:

| Label | Archive | Status |
| --- | --- | --- |
| `LEGACY_INVALID_RESULT` | `research/runs/2026-04-02/` | **INVALID** — do not cite |
| `LEAKAGE_CORRECTED_REFERENCE` | `research/runs/2026-08-28-track-a-baseline/` | **VALID**, negative, preserved deliberately |

`tests/test_evidence.py` reads the archived JSON back and fails if either number
drifts. That mechanism exists because the natural failure mode of a forecasting
project is to keep tuning until the number looks acceptable and then report the
last one — each retune individually defensible, the drift invisible.

Retuning the reference is forbidden. Building a **named challenger** and scoring
both through identical folds is the sanctioned route, which is what
`xgboost_causal_retuned` is.

---

## What the benchmark measures

```bash
btc-forecast benchmark --folds 36 --wf-horizon 30 --min-train-bars 900 --embargo-bars 30
```

**Outer folds.** Expanding window, 36 origins spaced ~73 bars apart across
2019–2026, each forecasting 30 bars after a 30-bar embargo. The fold count is
chosen for regime coverage and step-1 sample size, *not* to reach a significance
threshold. The embargo equals the horizon: features run on windows up to 60 bars,
so training targets near the boundary and test-period features would otherwise
overlap.

**Nested model selection.** Any tuning happens inside each outer fold's training
history, on forward, embargoed inner splits. Because `fit()` receives a
`TrainingWindow` and nothing else, a model *cannot* see the outer test window —
that is structural, not a discipline.

**Target.** Price level is primary, because the preserved reference was scored on
it and switching metrics would break comparability. Return-space views are
reported alongside. See `btc_forecaster/evaluation/targets.py` for the audit.

---

## How to read the table

| Column | Read it as |
| --- | --- |
| `mae`, `rmse` | Price-level error. Dominated by the price level; compare within a run, not across periods. |
| `mae_skill_vs_random_walk` | The number that matters. Positive means better than naive. |
| `mase` | Below 1 beats a naive forecast on the training scale. |
| `dir_acc_step1` + `dir_ci_*` | One-step directional accuracy with a block-bootstrap interval. **Read the interval, not the point.** |
| `dir_beats_coin` | Whether that interval excludes 0.5. |
| `interval_coverage` | Should sit near nominal. Both 0.60 and 1.00 are miscalibrated. |
| `mae_worst_to_median` | Worst fold relative to typical. Above ~3 means a failure mode a mean will hide. |
| `cost_multiple_vs_cheapest` | Median-fold compute relative to the cheapest model. |

**Directional accuracy is reported at step 1 across origins, not pooled.** Scoring
30 bars against a single origin gives 30 *correlated* observations: on a trending
series every bar in a fold agrees. That artefact is what produced
`random_walk_drift`'s 0.889 in the Track A baseline run, and it is why the
benchmark reports the step-1 sample and an interval around it.

---

## Inference

The legacy binomial p-value assumed independent Bernoulli trials. Walk-forward
direction outcomes are not independent: horizons overlap, bars within a fold move
together, and volatility clusters. Under positive dependence a naive test
overstates significance.

* **Directional uncertainty** — stationary bootstrap (Politis–Romano) over the
  per-origin step-1 hit series, block length `n^(1/3)`.
* **Pairwise comparison** — Diebold–Mariano with Newey–West HAC variance at
  `h−1` lags and the Harvey–Leybourne–Newbold small-sample correction.
* **Nested pairs are refused, not reported.** DM is not asymptotically normal for
  nested models, and `random_walk` is nested in `arima`, `sarimax`, `ets`,
  `random_walk_drift` and `historical_mean_return` — most of the comparisons this
  project wants to make. Those rows are flagged and excluded from the
  multiple-testing correction; correcting a statistic that was never valid would
  launder it.
* **Multiple testing** — Benjamini–Hochberg across the K−1 challenger-vs-baseline
  comparisons, with raw and adjusted p-values both shown.

What none of this fixes: a block bootstrap handles short-range dependence, not
structural breaks. A study spanning several crypto regimes is not
covariance-stationary in the strict sense DM assumes. **Prefer the intervals to
the p-values.**

---

## Promotion policy

Declared before the aggregates were inspected. A challenger is **PROMOTED** only
if it clears all five:

1. MAE skill ≥ 0.02 against the *naive baseline* (≥ 0.10 above a 20× compute
   multiple).
2. Bootstrap interval on that skill excludes zero.
3. Worst fold ≤ 3× the median fold.
4. Interval coverage within 0.15 of nominal.
5. At least 10 folds.

Failing (1) is **REJECT**. Failing a later criterion is **INCONCLUSIVE** — "we
showed it does not work" and "we could not show it works" are different results
and are kept distinct.

**A2 promotes nothing into production.** There is no production model to replace;
these are research verdicts.

---

## Reproducing a run

Every run writes `manifest.json` **last**, so its presence means the run
completed. It carries the run id, dataset SHA-256, fold boundaries, per-fold
feature and hyperparameter selections, model descriptions, the promotion policy,
the target and inference audits, environment fingerprint, and a re-verification
of the preserved evidence.

`write_benchmark` refuses to write into a directory that already holds a
manifest. Completed research runs are immutable.

---

## What this cannot tell you

- **Directional accuracy is not profitability.** No transaction costs, no
  slippage, no position sizing, no borrow. Nothing here says a strategy makes
  money.
- **Skill on 36 folds is a weak claim.** Fold dispersion on daily BTC is large
  relative to any effect present. Check `mae_worst_to_median` and the interval
  before believing a ranking.
- **Multi-step forecasts from the tree models are recursive.** Only step 1 uses
  real features; later steps compound the model's own output, and volume cannot
  be simulated at all. Long-horizon output is a scenario, not a prediction.
- **Monte Carlo shocks are Gaussian.** Jarque–Bera rejects normality on BTC
  returns decisively, so tail risk is understated even with correct accumulation.
- **The market may simply not be forecastable at this frequency.** A benchmark
  that keeps returning "no skill" is not necessarily broken. That is the result
  the platform was built to be able to report.

> Research and educational project. Not financial advice, and no guarantee of
> trading profitability.
