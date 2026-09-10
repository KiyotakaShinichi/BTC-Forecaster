# Track A7 — walk-forward robustness

> **A7 re-asks A6's question from many origins, at four horizons, with five
> amounts of history, across three periods. It is built to make a spurious edge
> harder to manufacture, not easier to find. It can reach three decisions, and
> none of them promotes a model.**

This document was committed **before the BTC benchmark ran**. Sections 1–11 and
the engine validation are the preregistration: the design, the data, the
targets, the models, the tests and every gate threshold. Sections 12–17 are
filled in afterwards from the run's canonical files, and the commit history
shows which came first.

## 1. Research question

A6 fitted forty models once, on 1,000 rows, and scored them on one holdout
block at a one-bar horizon. Nothing beat the random walk: 28 of 39 were
significantly worse than the naive forecast after Benjamini-Hochberg, and none
was better.

A6 named its own limits: one partition, one horizon, one target, n = 1,000. A7
asks whether its conclusion survives those limits being relaxed:

- from **many forecast origins** rather than one block;
- at **1, 3, 7 and 30 bars** rather than one;
- with **250, 500, 1,000 and 2,000 rows** of rolling history, and with all of it;
- across **early, middle and late** periods separately.

If nothing survives, the negative result is the conclusion, and it is preserved
as the answer — not treated as a reason to try another model family.

## 2. Benchmark design

Every (model, horizon, window) is a **job**. A job walks twelve refit folds in
order: at each fold's first origin the model is estimated on that fold's
training window, and its parameters are then frozen while the forecast origin
advances one bar at a time to the end of the fold. State and features see each
new bar; estimation does not.

Every job is scored at the **same daily origins**. A comparison across
different origins is partly a comparison of the origins.

Jobs run in spawned worker processes with linear algebra pinned to one thread,
so the worker count changes the run time and nothing else — a test asserts one
worker and two produce byte-identical results.

Scale: 11 models × 4 horizons × 5 windows = 220 jobs, 12 refits each, plus 22
leakage attacks (every model, at the shortest and longest horizon).

## 3. Data provenance

The input is the snapshot A6 used, so A6 and A7 differ in design and never in
data:

| | |
|---|---|
| Source | Yahoo Finance via yfinance 1.2.0, `BTC-USD` |
| Retrieved | 2026-09-07T02:53:49Z |
| Range | 2017-01-01 → 2026-09-07, 3,536 daily bars, one missing |
| Frame sha256 | `39b93e34520a496692ccc2156b8fe5c88c88b31bd9e9903a6fad74f282004bd3` |

**The provider is not immutable; the record is.** yfinance revises history
silently — A6 and A2 pulled nearly the same span and got different bytes. So the
run records exactly which bytes it used, and
`python -m btc_forecaster.research.snapshot verify <dir> --expect <sha>` exits 3
on any other input. A rerun on a different input is a different experiment, and
the benchmark refuses to call it a reproduction.

The snapshot is **not committed**: Yahoo's terms do not grant redistribution.
Nor are the row-level predictions, which contain realised returns. The hashes
are.

## 4. Target and horizon definitions

One target for every horizon, model and baseline:

    y_h(t) = ln( close[t + h] / close[t] )

the natural-log return over **h bars** from the forecast origin `t`, the last
bar whose close is observable. Dimensionless.

- At `h = 1` this is exactly A6's target.
- Horizons count bars, not calendar days: "three bars later" always exists.
- MAE grows with `h` by construction. It is never compared across horizons;
  skill against the naive forecast, within a horizon, is.
- It is **not A2's target**. A2 scored a 31-step price path in USD.

## 5. Rolling-origin methodology

Origins are consecutive bars from the first at which the largest fixed window
can be filled with settled, fully realised rows — warm-up, then 2,000 rows, then
the 30 bars those rows need — to the last bar that still has a 30-bar target
inside the data. The same origins serve every horizon.

They are partitioned twice, contiguously and deterministically: into **twelve
refit folds**, and into **three blocks** — early, middle, late.

On the input above, computed by `schedule_for` before the benchmark ran:

| | |
|---|---|
| Origins | **1,417** daily origins, 2022-09-21 → 2026-08-07 |
| Refit folds | 12 of 118–119 origins, refitting 2022-09-21, 2023-01-18, … 2026-04-12 |
| Early block | 2022-09-21 → 2024-01-06, 473 origins |
| Middle block | 2024-01-07 → 2025-04-22, 472 origins |
| Late block | 2025-04-23 → 2026-08-07, 472 origins |
| `rolling-2000` at fold 0 | extent starts on 2017-01-01, the first bar of the data — the schedule is exactly tight, not loose |
| `expanding` | 2,000–2,029 rows at fold 0, growing to 3,299–3,328 at fold 11 |
| Configuration digest | `ba369d2f65153de73fa7b94fbd9d37fb7d7bb9477b0a90b6cfc501624c39ec5c` — the run is this design only if its `config.json` hashes to it |

Training rows must have their target **realised by the refit origin**. For
`h > 1` the last `h − 1` feature rows before each origin are excluded, because
their targets end after it.

## 6. Training-window definitions

| Window | Rows at each refit |
|---|---|
| `rolling-250` / `-500` / `-1000` / `-2000` | the most recent realised rows |
| `expanding` | every realised row since the start of the data |

Every (fold, window) declares an **extent** — the first raw bar anything in it
may read — and features are rebuilt from inside it: a 60-bar warm-up, then the
window. This is stricter than A6 on purpose: A6's EMA feature carries a fading
memory of every bar since 2017, so a "250-row" model would still read older bars,
faintly. Here it cannot, and an adversary checks.

Deep models stop early on the last 20% of their own window, with the `h − 1`
rows whose targets overlap the held-out block purged.

## 7. Baselines

| Baseline | Forecast of `y_h(t)` |
|---|---|
| `naive_last_value` | 0 — the random walk. **Every comparison is against it.** |
| `random_walk_drift` | `h` × the window's mean one-bar return |
| constant direction | the direction the window's realised `h`-bar targets favoured |

The constant-direction baseline replaces a 50/50 coin. This series rises on
roughly 53% of days, so "always up" beats a coin while knowing nothing.

## 8. Models evaluated

The A6 registry decides what exists and whether it can run here; A7 keeps no
second list. The curated eleven:

| Model | Family | How it reaches `h` bars |
|---|---|---|
| `naive_last_value`, `random_walk_drift` | baseline | iterated |
| `ar_p`, `arima`, `theta`, `local_level`, `local_linear_trend` | statistical | iterated |
| `xgboost` | tree ensemble | direct |
| `ridge` | linear | direct |
| `mlp`, `lstm` | deep | direct |

**Iterated:** models that declare the `MULTI_STEP` capability estimate on
one-bar returns and run their own recursion `h` bars forward with frozen
parameters. That capability was added to A6's series adapters for A7 — the only
change to A6 code. It equals the A6 forecast at `h = 1` to machine precision,
it equals statsmodels' own `forecast()` for every state-space model, and **A6's
committed result digest `b266a50d…` reproduces unchanged** with it in place.

**Direct:** every other model is trained on the `h`-bar target itself.

No model is retuned. Every configuration is A6's.

## 9. Metrics

- **Point:** MAE, RMSE, MASE and skill against naive (`1 − MAE / MAE_naive`, on
  the same origins), and bias.
- **Direction:** accuracy, balanced accuracy, MCC, and the constant-direction
  null. A forecast of exactly zero is a wrong call, as in A6.
- **Stability:** skill in each of the twelve folds and three blocks.
- **Not computed:** pinball loss, Brier score, calibration error and QLIKE. No
  curated model declares quantiles, a direction probability or a variance, and
  nothing is invented to fill the cells.

## 10. Statistical testing

- **Diebold-Mariano** on absolute loss against naive, over paired origins —
  A2's implementation, with the Harvey-Leybourne-Newbold correction.
- **HAC lags** = `max(h − 1, ⌊4 (n/100)^(2/9)⌋)`. Walk-forward origins are one
  bar apart, so an `h`-bar forecast overlaps its neighbours; and an absolute-loss
  differential on daily returns inherits volatility clustering even at `h = 1`.
  More lags make the test harder to pass. A2's function gained an optional
  `hac_lags` for this, defaulting to its original `h − 1`.
- **Benjamini-Hochberg** at 0.05 over the **primary family**: every testable
  (model, horizon, window). About 200 tests — roughly ten raw rejections are
  expected from noise alone, and that number is reported beside the results.
  Per-horizon families are reported too and do not gate.
- **Block-bootstrap intervals** on skill.
- Fewer than 100 paired origins, or a nested pair: reported, not tested.

Three distinct claims, never merged: **statistically different** (BH, two-sided),
**statistically better** (different, and in the model's favour), **practically
useful** (better, by at least one percent skill).

## 11. Stability rules

A configuration is a **`ROBUST_RESEARCH_CANDIDATE`** only if it clears all seven
gates:

| Gate | Threshold |
|---|---|
| `aggregate_skill` | MAE skill against naive > 0 |
| `bh_better` | significantly better after Benjamini-Hochberg, α = 0.05 |
| `late_block` | skill in the late block > 0 |
| `fold_majority` | positive skill in ≥ 75% of refit folds (9 of 12) |
| `practical` | skill ≥ 0.01 |
| `leakage` | passed every walk-forward adversary; *not checked counts as failed* |
| `resources` | every fold ran, within its A6 resource class |

The benchmark reaches one decision:

- **`ROBUST_RESEARCH_CANDIDATE`** — something cleared every gate. It merits a
  separate, preregistered study. Not a trade.
- **`FRAGILE_SIGNAL`** — nothing cleared every gate, but something looked like a
  signal: positive skill with a one-sided raw p < 0.05, or BH significance that
  failed another gate. Each is listed with the gates it failed.
- **`ROBUSTLY_UNINTERESTING`** — nothing looked like a signal.

There is no fourth state. The thresholds are part of the configuration digest,
so a threshold moved after seeing a result makes a different run.

### Deviation from this preregistration: the `resources` gate

*Recorded after the first canonical BTC run.*

The table above says every fold must run "within its A6 resource class". The
first canonical run showed that this made the canonical result depend on
machine load. Fifteen LSTM folds — `h = 1` on the expanding and 2,000-row
windows, `h = 3` on the expanding window — took 121–240 s against a 120 s
budget while four worker processes shared four cores; the same kind of fold
takes about 48 s alone. A status that moves with load cannot sit inside a
digest that claims byte-identical reproduction: a rerun with fewer workers
would have produced a different digest for reasons unrelated to the result.

So `resources` now means **every fold completed**, which is deterministic.
Budget overruns are reported in full in `run_info.json`, outside the digest,
together with any deep-model fold that hit A6's own 300 s training cap — the
one case in which the forecasts themselves would depend on timing.

This **relaxes** a gate after a result existed, which is why it is recorded
here rather than folded into the table. It cannot have changed the decision,
and that was checked from the first run's files before the change was made:
no configuration had a raw signal or a Benjamini-Hochberg-significant
improvement, and none failed the `resources` gate alone. The first run is kept
outside the repository as the record of the defect; the committed run is the
rerun under the corrected code.

## Engine validation

A benchmark reporting "nothing works" on BTC is indistinguishable from a broken
one. So the engine is run on synthetic worlds where the answer is known, and it
must:

- find **no candidate** in pure noise;
- **detect** an AR(1) signal through the iterated path, and a signal carried by
  last week's mean return through the direct path;
- refuse a signal that **stops half-way** — aggregate skill may survive, the
  late block and fold gates must not;
- let the drift baseline win in a strongly trending world;
- refuse a **leaky oracle** that reads the realised target: it scores perfect
  skill, clears every numeric gate, and is rejected by the leakage gate.

Five adversaries attack every model through the code path the benchmark uses:
poisoning a future row, a future target, a future feature cell, or the bar just
outside the window must change nothing; shifting the valid past must change the
forecast — otherwise the first four prove nothing.

### What the engine actually did on those worlds

Observed with the preregistered gates, unmodified (one horizon, two windows,
six refit folds, 900 bars). Recorded here beside the expectations, including the
one that was not met.

| World | Expected | Observed | |
|---|---|---|---|
| Noise | no candidate | `ROBUSTLY_UNINTERESTING`; no configuration cleared the gate | met |
| Trend | drift beats naive | drift, AR, and ridge clear every gate in both windows | met |
| AR(1) | found via the iterated path | `ar_p` clears every gate in both windows, +6.0–6.5% skill, positive in every fold | met |
| Stopped signal | refused by the late-block and fold gates | `ar_p` fails both, in both windows | met |
| Feature signal | found via the direct path | **found, not certified** — see below | **not met as a candidate** |
| Leaky oracle | refused by the leakage gate | perfect skill; fails `leakage` and nothing else | met |

**The deviation.** On the feature-signal world, ridge on the expanding window is
significantly better than naive after Benjamini-Hochberg, by 3.3% skill, and
positive in the late block — but positive in only four of six refit folds,
under the three-in-four gate. The benchmark calls it `FRAGILE_SIGNAL`, failing
`fold_majority` and nothing else.

That is detection without certification. The preregistration expected
certification, and at this sample size the fold gate withholds it. The gate
behaves as specified; an effect that is real but inconsistent across folds is
exactly what it exists to withhold candidacy from. The direct path's ability to
clear every gate is shown on the trend world, where ridge does.

The injected signal was **not** strengthened until it passed. That would have
tuned the validation to its result. The test asserts what was observed and
cites this paragraph. What it implies for BTC is stated in section 16: an effect
of a few percent skill that is not consistent across folds will be reported as
fragile, by design.

## 12. Results

From the committed run in [`research/runs/a7-walk-forward/`](../research/runs/a7-walk-forward/),
result digest `b3fba7227e82e13ab2c112cb44169857aeb567cf7cb2a6f375bd7792f0d13790`: 1,417 origins, 200
gated configurations — 10 models × 4 horizons × 5 windows, each against
the naive forecast.

**Decision: `ROBUSTLY_UNINTERESTING`.** No configuration cleared the gate, and
none looked like a signal: not one had positive skill with a one-sided raw
p-value below 0.05.

| Primary family: 200 comparisons at α = 0.05 | |
|---|---|
| raw rejections | 113 — against 10 expected from noise |
| significant after Benjamini-Hochberg | 103 |
| of which significantly **better** than naive | **0** |
| of which significantly **worse** | 103 |

Significantly worse by horizon: 26 at h = 1, 26 at h = 3, 26 at h = 7, 25 at h = 30; the rest
are indistinguishable from naive.

29 of 200 configurations have positive aggregate skill
(4 at h = 1, 11 at h = 3, 6 at h = 7, 8 at h = 30). The largest is
+0.37% — `arima`, h = 3, rolling-2000 —
37% of the one-percent practical floor, and not significant.

The best configuration at each horizon, and where its skill came from:

| h | best | skill | early | middle | late | folds positive |
|---|---|---|---|---|---|---|
| 1 | `local_level` rolling-2000 | +0.22% | +0.36% | +0.37% | -0.14% | 8 of 12 |
| 3 | `arima` rolling-2000 | +0.37% | +0.44% | +0.75% | -0.20% | 6 of 12 |
| 7 | `ar_p` rolling-2000 | +0.05% | +0.83% | +0.51% | -1.43% | 5 of 12 |
| 30 | `ar_p` rolling-2000 | +0.20% | +2.23% | +3.61% | -6.64% | 6 of 12 |

Gate failures across the 200 configurations:
`aggregate_skill` 171, `bh_better` 200, `fold_majority` 193, `late_block` 186, `leakage` 0, `practical` 200, `resources` 0.

**Leakage:** every model passed every adversary that applies to it —
5 adversaries, 108 applicable checks, at h = 1 and h = 30 on the
rolling-250 window in the last fold. `valid_past` does not apply to `naive_last_value`, which forecasts a constant by construction.

**Direction** (descriptive; the gate does not test it): the best balanced
accuracy at each horizon is 0.526, 0.535, 0.526, 0.556,
and no MCC exceeds +0.14.

**Resources** (non-canonical, `run_info.json`): 2,214 s of wall
clock on 4 workers of an Intel Core i5-10210U laptop (four cores), and
6,600 s of fitting and forecasting summed across folds, 80% of it LSTM.
No fold overran its A6 budget under parallel load, and no deep-model fold hit A6's 300 s
training cap, so no forecast depends on machine speed.

## 13. Negative findings

- **Nothing beats the random walk.** 0 of 200
  configurations are significantly better; 103 are
  significantly worse. At no horizon and in no window.
- **The small positive skills belong to the random walk's neighbours** — local
  level, AR, ARIMA and drift, whose forecasts sit close to zero; 21 of the
  29 are on the two longest windows. The largest is +0.37%.
- **Every horizon's best configuration loses in the late block**
  (2025-04-23 → 2026-08-07) after positive early and middle
  blocks. Whatever edge exists belongs to a period. Of the 29 positive
  configurations, 17 fail the late-block gate and 22 fail fold majority.
- **Capacity hurts at every horizon.** MLP, LSTM, XGBoost and ridge are worse than
  naive in all 80 of their configurations and, averaged across windows,
  worse at every longer horizon: MLP from -29.1% at h = 1
  to -110.2% at h = 30. Ridge and XGBoost do best with all
  the history there is — the expanding window is their best at every horizon —
  and still never reach zero. The networks show no consistent benefit from more
  history.
- **One pattern is consistent, and negligible.** Local level at h = 3 is
  positive in all five windows (+0.09% to +0.14%) — at
  most 14% of the practical floor, and not significant.
- **Theta loses everywhere**, from -0.6% to
  -44.7%, and its average loss grows with every
  longer horizon: its forecast function extrapolates a trend in returns, which
  compounds.
- **Drift needs a long history.** It beats naive only at h ≥ 3 and only on the
  two longest windows (at most +0.29%), and loses on both short
  windows at every horizon: BTC's drift is real on average and too noisy to
  estimate from a year or two of data.

## 14. Comparison with A2

| | A2 | A7 |
|---|---|---|
| What is forecast | a 30-step **price path** | one **log return** over `h` bars |
| Where it is scored | the price at step 31 (after a one-bar embargo) | `ln(C[t+h] / C[t])` at every origin |
| Units | USD | dimensionless |
| Origins | 36 widely spaced folds | 1,417 consecutive daily origins, 12 refits |
| Models | 8, A2's adapters | 11, A6's registry |
| Outcome | 0 of 8 promoted | *see section 17* |

**The numbers are not comparable, including at `h = 30`.** A7's 30-bar horizon
is the nearest thing to A2's 31-step path, and it is still a different
measurement. A2's MAE is the dollar error of a price level, dominated by
wherever the price happened to be. A7's is the error of a scale-free return, at
every daily origin rather than 36 spaced ones. A2's best model, drift at
+0.003957, and any A7 number are in different units and cannot be set side by
side.

What the two share is a question — *does anything beat the random walk?* — and
the answer can agree or disagree without the numbers being comparable. That
qualitative comparison is in section 17.

## 15. Comparison with A6

Same input bytes, and at h = 1 the same target. The closest A7 configuration to
A6's design is h = 1 on the 1,000-row rolling window — but scored at 1,417
origins with twelve refits instead of one block with one fit:

| model | A6 skill, one block | A7 skill, h = 1, `rolling-1000` | A7 q |
|---|---|---|---|
| `random_walk_drift` | -0.19% | -0.32% | 0.036 |
| `ar_p` | -0.13% | -0.41% | 0.18 |
| `arima` | -0.16% | -0.20% | 0.46 |
| `theta` | -3.18% | -1.40% | 0.00087 |
| `local_level` | +0.00% | +0.12% | 0.46 |
| `local_linear_trend` | -0.45% | -0.31% | 0.31 |
| `xgboost` | -22.65% | -7.82% | 1.6e-08 |
| `ridge` | -16.03% | -2.10% | 0.00063 |
| `mlp` | -65.32% | -32.14% | 7.8e-14 |
| `lstm` | -29.24% | -10.44% | 9.2e-12 |

**`CONSISTENT`.** The ordering is essentially A6's: the two networks are the
worst in both, then gradient boosting and ridge, with the statistical models
near zero. Refitting on recent data every 118 origins leaves the learned
models with 13% to 49% of their A6 losses —
and none of them crosses zero. The near-zero models move by fractions of a
percent, in both directions. A6's one-block, n = 1,000 conclusion survives many origins and
refits at h = 1, and A7 extends it to 3, 7 and 30 bars and to four more amounts
of history.

## 16. Limitations

These hold whatever the result.

- **One instrument, one bar frequency, one source.** Daily BTC-USD from yfinance,
  which revises history silently. The input hash pins the bytes used; it cannot
  make the provider stable.
- **Eleven models, A6's configurations, no tuning.** Deliberate. Tuning across
  220 configurations would manufacture edges, which is what A7 exists to
  prevent. A model that needs tuning to work shows here at its A6 setting.
- **Parameters are frozen within each fold** of about 118 origins. A model
  refitted every day could behave differently.
- **A6's eleven causal price and volume features, nothing exogenous.** Adding
  intelligence signals to quant models is out of scope by rule.
- **The fold gate trades power for robustness.** Engine validation showed it: a
  genuine effect of about 3% skill, significant after correction, was classified
  `FRAGILE_SIGNAL` because it was positive in four of six folds. On BTC a small
  but real effect that is not consistent across twelve folds is reported as
  fragile, not as a candidate. This is by design, and it is still a limitation.
- **Diebold-Mariano assumes a covariance-stationary loss differential.** At
  `h = 30` consecutive origins share 29 of 30 bars, so the effective sample is
  far smaller than 1,417. The HAC lags account for the overlap; they cannot
  create information.
- **The primary family is large.** About 220 comparisons under one
  Benjamini-Hochberg correction makes significance hard to reach. The
  per-horizon families are reported and do not gate.
- **Configurations are not independent.** Windows overlap in information,
  horizons share origins, models share features. The gate's counts describe the
  table, not independent experiments.
- **Absolute loss only.** No transaction costs, no position sizing, no economic
  evaluation. Direction is reported against the constant-direction null. No
  curated model declares a distribution, so nothing probabilistic is scored.
- **A6's deep models stop training at 300 s of wall time.** If a fold ever hit
  that cap, its forecasts would depend on machine speed. `run_info.json` lists
  every fold that did; the committed run's count is in section 12.
- **The resource budget is reported, not gated.** Whether a fold overran it
  depends on how many processes shared the machine, so overruns are recorded
  outside the result — see the deviation after section 11.
- **Byte identity is asserted for one environment.** The canonical text is fixed
  at twelve significant digits; different CPUs or BLAS builds may still differ
  in the last bits. The manifest records the environment.
- **The data and predictions are not committed.** A reproduction needs the same
  snapshot bytes. The hash says whether you have them.

## 17. Decision

**`ROBUSTLY_UNINTERESTING`.**

- **Against the random walk:** 0 of 200 configurations
  better, 103 worse. The best, +0.37%, is not significant,
  is 37% of the practical floor, and is negative in the late block.
- **Candidate signals:** none. **Fragile signals:** none — nothing reached even a
  raw one-sided p < 0.05 with positive skill.
- **What failed stability:** every positive configuration failed significance
  and the practical floor; 17 of 29 also failed the late block,
  22 fold majority.
- **A2, A6 and A7 agree** on the question they share — nothing beats the random
  walk — across three designs whose numbers are not comparable: 0 of 8 promoted,
  0 of 39 better, 0 of 200 better.
- **A8 is not proposed.** A7's stop condition applies: the random walk remains
  dominant, and no concrete question survived the framework. Another model
  family on the same eleven price and volume features is exactly what this
  result argues against. If there is a future quantitative track, the only
  defensible starting point is a new, preregistered source of information — and
  that is a decision for its own proposal, not a conclusion of this one.
- **Nothing is promoted. Live trading remains disabled**, and A7 has no state
  that could change that.
