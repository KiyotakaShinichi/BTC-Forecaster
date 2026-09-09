# Track A6 — the resource-constrained model zoo

> **1,000-row A6 results are resource-constrained exploratory evidence and do
> not supersede A2's historical promotion study.**

That sentence is the whole scientific status of this package, and everything
below is an elaboration of it. A6 fits 43 registered models on 1,000
deterministically chosen training rows and scores them on one frozen holdout
block. A2 ran 8 models through 36 walk-forward folds and promoted none. The two
are not comparable, and the second remains the authoritative historical
evidence.

## Why build it anyway

Breadth is cheap and ignorance is not. Before A6 the repository could not answer
questions it should be able to answer:

- Does *anything* beat a random walk on this series when you try forty things
  rather than eight?
- Do different model families make different mistakes, or the same ones?
- How much data does each architecture need before it stops being noise?

The answers are in [`research/runs/a6-model-zoo/`](../research/runs/a6-model-zoo/).
They are worth having before anyone proposes a preregistered study — not because
they settle anything, but because they narrow what is worth preregistering.

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

## What is *not* comparable to A2

| | A2 | A6 |
|---|---|---|
| Design | 36 walk-forward folds | 1 fixed partition |
| Training rows | full history per fold | 1,000 |
| Models | 8 | 43 registered, 40 runnable |
| Target | next-bar log return | next-bar log return |
| Data | snapshot `056b866b…` | snapshot `39b93e34…` |
| Purpose | promotion study | exploratory breadth |
| Outcome | 0 promoted | 0 promoted, **and none can be** |

The data differs for a reason worth stating: the two snapshots cover almost the
same span, and yfinance silently revised the history between them. Truncating
the newer pull to A2's last bar gives the same 3,527 rows and a *different*
hash. A6 therefore cannot reproduce A2's bytes, which is one more reason the two
studies are `NOT_COMPARABLE` rather than merely different.

`xgboost` here is **not** A2's `xgboost_causal_retuned`. That model selected
hyperparameters by nested inner validation inside every one of 36 folds; this
one is frozen at the family capacity ceiling and fitted once. The names are one
word apart and the studies are not.

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

The full zoo takes roughly three minutes and is deliberately **not** a CI job.
CI runs the registry check, the leakage adversaries, the synthetic worlds and a
four-model smoke benchmark; the complete run is an explicit research command.

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
- **The NOISE_ONLY control.** A synthetic world with no structure. A model that
  beats the mean on it has found something that is not there, and the finding is
  a bug report. Every other claim depends on that control finding nothing.
- **Gradients.** The deep family runs on a numpy autodiff engine in this
  repository, checked against central finite differences. A wrong gradient does
  not raise — it produces a model that trains, converges and learns nothing.
- **Determinism.** One seed, contiguous rows, fingerprinted training data. Two
  runs produce bit-identical forecasts.
- **Multiplicity.** Forty comparisons at alpha 0.05 yield two significant
  results from nothing. Raw p-values, Benjamini-Hochberg q-values and the
  expected false-positive count are reported together.
- **Nothing promotes.** There is no `promoted` field. Every result carries
  `EXPLORATORY`, `assert_nothing_promoted` refuses a manifest that says
  otherwise, and the paper-trading engine stays fail-closed.

## Limitations, stated once

- One partition. A single holdout is not a distribution, and the block analysis
  exists because a mean skill across it can describe nothing.
- n = 1,000. A regime in which almost nothing is decidable.
- One horizon, one target, one instrument.
- The deep models are compact by necessity. "A transformer did not work here"
  is a statement about a 2,433-parameter transformer on 977 sequences.
- No ensemble. The error-diversity analysis is preparation for a possible future
  study, and A6 builds none.
