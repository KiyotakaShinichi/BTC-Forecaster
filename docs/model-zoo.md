# Track A6 — the resource-constrained model zoo

> **1,000-row A6 results are resource-constrained exploratory evidence and do
> not supersede A2's historical promotion study.**

That sentence is the whole scientific status of this package, and everything
below is an elaboration of it. A6 registers 43 models, fits the 40 that can
honestly run on 1,000 deterministically chosen training rows, and scores them on
one frozen holdout block of 705 bars. A2 ran 8 models through 36 walk-forward
folds and promoted none. The two are not comparable, and the second remains the
authoritative historical evidence.

## Why build it anyway

Breadth is cheap and ignorance is not. Before A6 the repository could not answer
questions it should be able to answer:

- Does *anything* beat a random walk on this series when you try forty things
  rather than eight? **No.** Twenty-eight of thirty-nine models are
  significantly different from the naive forecast after Benjamini-Hochberg, and
  all twenty-eight are worse. Zero are better. The three with positive raw
  skill -- all state-space models -- are indistinguishable from naive
  (q ≥ 0.57).
- Do different model families make different mistakes, or the same ones? **The
  same ones.** Mean pairwise error correlation across the forty is 0.945.
- How much data does each architecture need before it stops being noise? **More
  than this.** Every model with real capacity — MLP, TCN, LSTM, N-BEATS, the
  boosting family — is among the worst performers, and none of the forty is
  positive in every temporal block.

The full evidence is in
[`research/runs/a6-model-zoo/`](../research/runs/a6-model-zoo/), which carries
its own generated README. These answers are worth having before anyone proposes
a preregistered study — not because they settle anything, but because they
narrow what is worth preregistering.

## The 1,000-row constraint

The training budget is the **contiguous tail of the training partition**. Not a
sample: there is no seed, no shuffle, and two builds select bit-identical rows.

Contiguity matters more than it looks. Randomly sampling 1,000 rows from twelve
years would hand a sequence model a shuffled series and let a tabular model
interpolate between neighbours it should have had to extrapolate past.

The data is cut three ways and the rules are structural:

| Partition | What happens there |
|---|---|
| `TRAIN` | parameters are estimated, and nowhere else |
| `DEV` | every choice: early stopping, the at-most-three-configuration selection |
| `HOLDOUT` | scored once, at the end, never read while choosing anything |

The budget is the last 1,000 rows of `TRAIN`. If `TRAIN` holds fewer, the budget
is all of it and the shortfall is recorded — never topped up from `DEV`.

## Taxonomy

Generated from the registry, which is the single source of truth. Run
`python -m btc_forecaster.research.model_zoo --list` for the live version.

| Family | Models | What distinguishes them |
|---|---|---|
| `BASELINE` | 3 | constants. The hypotheses everything else must disprove |
| `STATISTICAL` | 10 | AR, ARIMA, Theta, state-space level/trend/cycle, and three seasonal models excluded on measurement |
| `VOLATILITY` | 3 | GARCH, EGARCH, GJR — conditional variance, **not** direction |
| `LINEAR_ML` | 8 | what is penalised, and how error is measured |
| `TREE_ENSEMBLE` | 7 | one capacity ceiling, six ways of reaching it |
| `KERNEL_LOCAL` | 2 | RBF SVR and KNN — analogue forecasting |
| `PROBABILISTIC` | 3 | quantile regression, quantile boosting, split conformal |
| `DEEP` | 7 | MLP, CNN, TCN, GRU, LSTM, transformer, N-BEATS |

Every model declares its **capabilities**, and asking for an undeclared one
raises rather than guessing. A deterministic point forecaster has no probability
to report, and a benchmark cell left blank is more honest than one filled by a
Gaussian nobody chose.

## What A6 says about A2

Nothing, mostly — and the table below says which parts of "nothing" are which.
Every A6 claim is classified against A2's frozen evidence using four labels:

- **`NOT_COMPARABLE`** — the two studies do not measure the same quantity. No
  inference either way.
- **`EXPLORATORY_ONLY`** — an A6 observation with no A2 counterpart. It is a
  candidate for a future study, not a result.
- **`CONSISTENT`** — A6 saw the same qualitative thing under a different design.
  Corroboration, not confirmation.
- **`INTERESTING_FOR_FUTURE_VALIDATION`** — worth a preregistered test, and not
  evidence until it gets one.

| Claim | A2 | A6 | Verdict |
|---|---|---|---|
| Headline error figures | MAE ≈ 7,451 **USD**, MASE ≈ 18 | MAE ≈ 0.0164 **log return**, MASE ≈ 1.00 | `NOT_COMPARABLE` |
| Forecast target | price path, scored at step 31 | next-bar log return, one step | `NOT_COMPARABLE` |
| Data | snapshot `056b866b…`, 3,527 rows | snapshot `39b93e34…`, 3,536 rows | `NOT_COMPARABLE` |
| Design | 36 walk-forward folds | 1 fixed partition | `NOT_COMPARABLE` |
| `random_walk_drift` skill | **+0.003957** (best, still rejected) | **−0.001909** (q = 0.46) | `NOT_COMPARABLE` |
| Directional accuracy of drift | 0.6176 | 0.4993 | `NOT_COMPARABLE` |
| Nothing beat the random walk | 0 of 8 promoted | 0 of 40 significantly better | `CONSISTENT` |
| Gradient boosting loses to naive | `xgboost_causal_retuned` −0.080081 | `xgboost` −0.226512 (q < 0.0001) | `CONSISTENT` |
| High-capacity models lose hardest | Prophet hybrid −1.405653, worst of 8 | MLP DM +15.4, TCN +11.4, LSTM +9.9 — the worst of 40 | `CONSISTENT` |
| ARIMA sits just under naive | −0.002062 | −0.001580 (q = 0.58) | `CONSISTENT` |
| `uc_stochastic_cycle` best at +0.000716 | not in A2's model set | q = 0.5662 | `EXPLORATORY_ONLY` |
| Three seasonal models excluded on measured absence of weekly seasonality | not tested | ACF(7) = −0.018 vs ±0.062 band, day-of-week ANOVA p = 0.946, STL strength 0.065 — TRAIN only | `EXPLORATORY_ONLY` |
| Deep architectures on a numpy autodiff engine | no deep family | 7 models, gradient-checked | `EXPLORATORY_ONLY` |
| No model is positive in every temporal block | per-fold stability reported | 0 of 40 | `INTERESTING_FOR_FUTURE_VALIDATION` |
| Mean pairwise error correlation 0.945 across eight families | not measured | measured, no ensemble built | `INTERESTING_FOR_FUTURE_VALIDATION` |
| Conditional variance is predictable where the mean is not | not tested | GARCH family scored on QLIKE, separately from direction | `INTERESTING_FOR_FUTURE_VALIDATION` |

Two rows deserve their own sentence.

**The drift sign flip is not a contradiction.** A2's `random_walk_drift` beat a
zero-drift random walk over a 31-step price path across 36 folds; A6's loses to a
zero-return forecast on next-bar log returns over one block. A drift term helps
when errors compound over thirty-one steps and hurts when there is one step for
it to be wrong about. Reading either number as a correction of the other requires
ignoring what each measured.

**`xgboost` here is not A2's `xgboost_causal_retuned`.** That model selected its
hyperparameters by nested inner validation inside every one of 36 folds. This one
is frozen at the family capacity ceiling and fitted once. The names are one word
apart and the studies are not.

And the snapshots differ for a reason worth stating: yfinance silently revised
the history between the two pulls. Truncating the newer snapshot to A2's last bar
gives the same 3,527 rows and a *different* hash. A6 cannot reproduce A2's bytes,
which is one more reason these are two studies rather than one study run twice.

## Running the benchmark

```bash
# what is registered, and why anything is excluded
python -m btc_forecaster.research.model_zoo --list

# the canonical run
python -m btc_forecaster.research.model_zoo \
    --train-rows 1000 \
    --output research/runs/a6-model-zoo

# the sample-efficiency subset at 250 / 500 / 1000
python -m btc_forecaster.research.model_zoo --sample-efficiency
```

It needs a hash-verified market snapshot, which is **not committed** — Yahoo's
terms do not grant redistribution. Run `btc-forecast snapshot` first, or pass
`--data` to an existing one.

On the Windows laptop used for this track, fitting and scoring the forty models
took between about 75 and 155 seconds across runs of identical code -- a
two-fold spread from machine load alone, visible in every model including the
numpy networks that import nothing. The manifest records each run's figure as
`wall_clock_seconds`, and one-time library imports separately as
`import_seconds`, so neither is charged to a model. The full command, with the
sample-efficiency subset and forty serialized artifacts written and
hash-checked, takes a few minutes. It is deliberately **not** a CI job. CI runs the registry
check, the leakage adversaries, the synthetic worlds and a four-model smoke
benchmark; the complete run is an explicit research command.

Two runs of it agree: the same `results_table_sha256`, byte-identical
`predictions.csv`, the same training fingerprint on every model. If yours
disagrees while the snapshot hash matches, that is a defect worth reporting.

## Adding a model

1. Subclass `ZooModel` in the right `adapters/` module. Implement `_fit` and
   `_predict_point`; declare any richer capability and implement its hook.
2. Register it, with a `description` and any `notes` that belong on its card.
3. Add the adapter module to `registry.ADAPTER_MODULES` if it is new — a module
   absent from that tuple is a module whose models silently do not exist.
4. Freeze its configuration. At most three development-only candidates are
   permitted; record the alternatives and their DEV scores, not only the winner.
5. If it cannot honestly run here, register it as
   `UNSUITABLE_FOR_CONSTRAINED_LAB` **with the measurement that excludes it**.
   The registry refuses an unexplained exclusion.

Everything else — the results table, the capability matrix, the model cards, the
documentation counts above — is generated from the registry. There is no second
list to update.

## How the results are protected

- **Leakage.** `EvaluationContext.history_at` stops at the forecast origin, one
  bar before the target. The window builder asserts on the recorded
  `feature_bar` rather than on array positions. `tests/test_a6_leakage.py`
  poisons future rows, future targets and single future cells and asserts
  nothing moves — with companion assertions that poisoning the *past* does move
  it, because a leakage test that cannot detect a leak is decoration.
- **The budget.** `TrainingSet` refuses a `series` or `close` that is not
  aligned with its rows, and `tests/test_a6_budget.py` checks that every
  state-space model estimates on exactly its budget and that every non-deep
  model's forecasts move when the budget does. The guard exists because the
  bound was once not enforced -- see the correction below.
- **The NOISE_ONLY control.** A synthetic world with no structure. A model that
  beats the mean on it has found something that is not there, and the finding is
  a bug report. Every other claim depends on that control finding nothing.
- **Gradients.** The deep family runs on a numpy autodiff engine in this
  repository, checked against central finite differences. A wrong gradient does
  not raise — it produces a model that trains, converges and learns nothing.
- **Determinism.** One seed, contiguous rows, fingerprinted training data. Two
  runs produce bit-identical forecasts.
- **Multiplicity.** Thirty-nine comparisons at alpha 0.05 yield 1.95 significant
  results from nothing. Raw p-values, Benjamini-Hochberg q-values and that
  expected false-positive count are reported together, so a reader can see the
  number of free hits before reading the hits.
- **Nothing promotes.** There is no `promoted` field. Every result carries
  `EXPLORATORY`, `assert_nothing_promoted` refuses a manifest that says
  otherwise, and the paper-trading engine stays fail-closed.

## Correction

An earlier version of this run, committed as `76927fd` and `83b8925`, did not
enforce the training budget on five models. `arima`, `local_level`,
`holt_linear_trend`, `local_linear_trend` and `uc_stochastic_cycle` estimated on
all 2,302 returns up to the end of TRAIN instead of the 1,000-row budget. Not
leakage -- nothing after TRAIN was seen -- but 2.3 times the data the other
thirty-five models were given, while the recorded training fingerprint said
1,000. It showed up as ARIMA scoring identically at budgets of 250, 500 and
1,000, where any model that estimates something should move.

Fixed in `ef0f730`: the dataset now hands over only the budget's bars and
`TrainingSet` refuses anything else. The run in this directory is the corrected
one. Only those five models' numbers moved, and none crossed a significance
threshold:

| model | before | after | q after |
|---|---|---|---|
| `arima` | −0.002729 | −0.001580 | 0.5778 |
| `local_level` | +0.000170 | +0.000000 (rounds to zero) | 0.6554 |
| `holt_linear_trend` | +0.000464 | +0.000277 | 0.9214 |
| `local_linear_trend` | −0.003109 | −0.004547 | 0.3098 |
| `uc_stochastic_cycle` | −0.003494 | +0.000716 | 0.5662 |

The headline did not change: twenty-eight of thirty-nine significantly worse than
naive, none better, none positive in every temporal block, mean pairwise error
correlation 0.945. The commit messages of the two superseded evidence commits
quote the old figures, and history is not rewritten; this section is the record
of which figures changed and why.

## Limitations, stated once

- One partition. A single holdout is not a distribution, and the block analysis
  exists because a mean skill across it can describe nothing.
- n = 1,000. A regime in which almost nothing is decidable.
- One horizon, one target, one instrument.
- The deep models are compact by necessity. "A transformer did not work here"
  is a statement about a 2,433-parameter transformer on 977 sequences.
- No ensemble. The error-diversity analysis is preparation for a possible future
  study, and A6 builds none.
