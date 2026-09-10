# Quantitative research handoff

> **The project does not currently possess validated evidence of a tradable
> predictive edge.**

Status: **`QUANT_RESEARCH_FROZEN`**, since 2026-09-11 (Track A7.1). The record —
lineage, digests, promotion policy, provenance — is
[`quant-research-status.md`](quant-research-status.md). This document is the
narrative for whoever picks the work up next: what was tried, what it showed,
where the evidence stops, and what it would take to start again.

The defensible conclusion is narrower than the headline invites. **The models and
configurations tested here do not provide validated evidence of out-of-sample
predictive edge under the designs that tested them.** This document does not
claim that BTC is inherently unpredictable, that a future edge is impossible, or
that the model zoo proves no strategy can work. It claims what was tested, how,
and what came back.

---

## 1. What we tested

One question, asked three ways, on one data source — daily BTC-USD bars from
Yahoo Finance through yfinance: **does any forecasting model beat the naive
forecast out of sample?**

- **A2 — a promotion study.** Seven challengers against the naive forecast:
  random walk with drift, historical mean return, ARIMA, ETS, Prophet, a
  Prophet–XGBoost hybrid and a causally retuned XGBoost. 36 expanding walk-forward
  folds across 2019–2026, tuning nested inside each fold, USD price level scored
  at steps 31–60 after a 30-bar embargo, and a five-criterion promotion policy
  declared before any aggregate was read. [`benchmark.md`](benchmark.md)
- **A6 — a model zoo.** 43 models attempted and 40 run, across eight families:
  baselines, autoregressive and state-space, GARCH, linear ML, trees,
  kernel/local, quantile and conformal, and seven compact deep architectures
  (MLP, LSTM, GRU, TCN, 1-D CNN, N-BEATS, transformer). Next-bar log return, every
  model fitted once on a 1,000-row budget, scored on one 705-bar holdout.
  [`model-zoo.md`](model-zoo.md)
- **A7 — a robustness study of A6's answer.** Eleven of A6's models at their A6
  configurations, untuned: naive, drift, AR, ARIMA, Theta, local level, local
  linear trend, XGBoost, ridge, MLP and LSTM. Cumulative log returns over 1, 3, 7
  and 30 bars; rolling windows of 250 to 2,000 rows and an expanding one; 1,417
  daily origins with twelve refits; early, middle and late periods; five leakage
  adversaries; every gate preregistered; 200 comparisons under one
  Benjamini-Hochberg family. [`walk-forward.md`](walk-forward.md)

Before all three there was the project's original headline — 68.97% directional
accuracy, p = 0.031 — which turned out to be leakage. Corrected, that hybrid's
error is more than two and a half times the random walk's. Both numbers are
preserved, and `tests/test_evidence.py` fails if either changes.

A6 and A7 models saw eleven causal features built from BTC's own price and volume
history, and nothing else.

## 2. What we learned

- **The random walk was not beaten in any of the three designs.** A2 rejected all
  seven challengers. In A6, 28 of 39 models were significantly different from
  naive after correction, and all 28 were worse. In A7, 0 of 200 configurations
  were significantly better and 103 were significantly worse.
- **More capacity did worse, consistently.** A6's hardest losers were the models
  with capacity. In A7 the MLP, LSTM, XGBoost and ridge are worse than naive in
  all 80 of their configurations, and the MLP's loss grows with the horizon, from
  29% at one bar to 110% at thirty, averaged across windows.
- **The small positive skills belong to near-random-walks.** Local level, AR,
  ARIMA and drift — models whose forecasts sit close to zero — produce the only
  positive skills, the largest +0.37%. It is not significant, it is a third of
  the preregistered 1% practical floor, and at every horizon the best
  configuration loses in the most recent period.
- **A6's single block was not a fluke of that block.** At one bar, A7 re-asked
  A6's question from 1,417 origins with twelve refits and found the same ordering
  (`CONSISTENT`, [`walk-forward.md`](walk-forward.md) section 15).
- **The engine can see a signal when one exists.** On synthetic worlds A7 found
  an autoregressive signal and a trend, refused pure noise and a signal that
  stops halfway, and caught a leaky oracle with perfect skill. A planted
  feature-driven signal of about 3% skill was detected but classified fragile,
  because it held in only four of six folds. So the BTC result is not the silence
  of an engine that cannot hear — though it is the result of an engine with
  limited power for small, inconsistent effects.
- **Provenance failed more often than modelling did.** yfinance silently revised
  history between A2's and A6's pulls. Wall-clock timings leaked into a results
  digest twice, once in A6 and once in A7, and a library's first import was once
  charged to one model's time budget. And A6's results digest depended on the
  operating system's line ending, which nobody saw until CI recomputed it on
  Linux during A7.1. Each made a result depend on the machine, and each was
  found only because the digest was expected to reproduce.

## 3. What did not work

"Did not work" means *did not beat the naive forecast under the design that tested
it* — nothing more general.

| Approach | Where | What happened |
|---|---|---|
| Drift and mean-return models | A2, A6, A7 | Drift beats naive only at h ≥ 3 on the two longest windows, by at most +0.29%, and loses on both short windows at every horizon |
| Classical time series: ARIMA, ETS, Theta, state space | A2, A6, A7 | Near zero at best. Theta loses everywhere in A7, from −0.6% to −44.7%, worse with every longer horizon |
| Prophet and a Prophet–XGBoost hybrid | A2 | Rejected with MAE skill −1.29 and −1.41 against naive |
| Nested tuning | A2 | The causally retuned XGBoost was rejected, MAE skill −0.08 |
| Gradient boosting and trees | A2, A6, A7 | Worse than naive in every A7 configuration; XGBoost −7.8% at one bar on the 1,000-row window |
| Linear ML | A6, A7 | Worse everywhere; ridge does best with all the history there is and still never reaches zero |
| Compact deep models | A6 (seven), A7 (MLP, LSTM) | Among the worst in both. In A6 the MLP lost 65% and the LSTM 29% |
| Direction | A7, descriptive | Best balanced accuracy 0.526 to 0.556 by horizon, MCC at most 0.14. Not gated, and not an edge |

## 4. Why the random walk remains the reference

- **It is the forecast nothing beat**, in three designs with different targets,
  horizons and amounts of history.
- **It is the null a claim has to clear.** "Tomorrow's price is today's" is what a
  weakly efficient price series predicts. A model that beats another model while
  both lose to naive has shown nothing; skill is measured against naive or it is
  not measured.
- **It cannot overfit and it cannot leak.** It has no parameters, no features and
  no training window, which is exactly what makes it the right yardstick for
  models that have all three.

Keeping it as the reference is not a belief that BTC *is* a random walk. It is the
bar any future claim has to clear, and so far nothing has.

## 5. Why more generic model shopping is currently unjustified

- **The search has already been broad, and the answer uniform.** More than forty
  model specifications across eight families, from a parameter-free baseline to a
  transformer, gave the same result — and capacity made it worse. That is the
  signature of inputs with little learnable structure for this target, not of an
  under-explored model space.
- **Every model saw the same information.** A new family trained on the same
  eleven price and volume features asks the same question once more.
- **Every new model is another comparison.** A7's 200 comparisons at α = 0.05
  already imply about ten false positives from noise. Adding models until one
  passes is the forking-paths failure that preregistration exists to prevent, and
  this project's first headline was exactly that kind of artefact.
- **The stop condition was met.** A7's brief was to stop model research, rather
  than invent another model family, if the random walk stayed dominant or the
  signals proved fragile. It stayed dominant, and nothing even reached fragile.
- **The prize is small.** The best effect observed is a third of the practical
  floor, before transaction costs.

What would change this is new information or a genuinely different question, not
another model (section 7).

## 6. Known methodological limitations

These bound the negative result. None of them is evidence that an edge was
missed; each marks where the evidence stops.

- **One instrument, one frequency, one source.** Daily BTC-USD from yfinance,
  which revises history silently. Nothing here speaks to intraday data, other
  venues or other assets.
- **Only BTC's own history as input.** No exogenous information: no macro, no
  on-chain data, no order book, no news or intelligence signals.
- **Point forecasts scored by absolute error.** No transaction costs, sizing or
  economic evaluation, and no probabilistic scoring in A7, where no curated model
  declares a distribution. A model could lose on MAE and still carry useful
  information about volatility or tails; that is not what these studies gated.
- **Configurations frozen, deliberately.** A7 used A6's settings untuned, and
  froze parameters within each refit fold of about 118 origins.
- **A6 is one partition at n = 1,000.** Its deep models are compact by necessity:
  "a transformer did not work" is a statement about a 2,433-parameter transformer
  on 977 sequences.
- **A7's fold gate trades power for robustness.** In validation it classified a
  real effect of about 3% skill as fragile.
- **Overlap limits the effective sample.** At thirty bars, consecutive origins
  share 29 of 30 bars; the HAC correction fixes the variance and cannot add
  information. A7's configurations share origins, features and history, so its
  counts describe a table, not 200 independent experiments.
- **A2 scores price level,** which the price level dominates; its errors compare
  within a run, not across periods or studies.
- **Byte-identical reproduction is asserted for one environment.** Different CPUs
  or BLAS builds may differ in the last bits.

## 7. What would justify reopening quant research

Reopening needs one of these, written down before any result exists.

1. **A defect that invalidates A2, A6 or A7.** Leakage in a baseline, an error in
   the Diebold-Mariano or Benjamini-Hochberg implementation, a corrupted snapshot,
   a misaligned target. The response is to fix it and rerun the preregistered
   study on its pinned input — not to add models.
2. **A new source of information, with a stated mechanism.** Something the models
   have not seen, with timestamps that can be trusted for availability — for
   example point-in-time market-intelligence signals, once the forward corpus meets
   B4's readiness thresholds. The mechanism says why it should predict BTC and at
   what horizon, before anyone looks.
3. **A materially different question.** A different target (volatility,
   cost-aware returns), frequency or instrument, with its own baseline, practical
   floor and preregistration.

Whichever it is, the proposal names the comparison family and its correction, the
practical floor, the stability and leakage rules and the pinned snapshot before
the first run, and anything beyond research has to clear `NO_MODEL_PROMOTION_UNTIL`
in [`quant-research-status.md`](quant-research-status.md).

**Not reasons to reopen:** a new architecture or library release; a published
claim that BTC is predictable, not replicated out of sample on this data; a
positive result on one split, one window or in sample; relaxing A7's thresholds
after the fact; a model that ranks well among models that all lose to naive.

## 8. Relationship to market-intelligence research

- **A separate layer with a separate question.** `market_intelligence/` asks
  whether what regulators and central banks announce helps forecast BTC.
- **Its historical study returned `HOLD`, not a negative.** B4 found that no
  point-in-time-valid historical corpus exists, because availability cannot be
  reconstructed for documents collected after the fact. The question was not
  answerable with that evidence, which is a different claim from "it does not
  work".
- **Its corpus is being built forward, slowly.** The collector is a frozen
  deployment candidate, not deployed, and `btc-intel corpus-status` reports
  `NOT_READY`. The bottleneck is elapsed calendar time.
- **The quant freeze does not freeze it.** Its own contracts are frozen separately,
  in [`deploy/COLLECTION_FREEZE.md`](../deploy/COLLECTION_FREEZE.md).
- **They meet only through section 7.** A7 kept intelligence signals out of the
  quant models by rule. Bringing them in is a reopening under 7.2: its own
  preregistered question, run only once the corpus is ready, with availability
  meaning retrieval time.

The next research direction, if there is one, is a market-intelligence question
backed by enough point-in-time evidence to answer it — not another generic
model-shopping exercise.

## 9. Live trading: disabled

- **Live trading is disabled.** It is not implemented, and this repository
  connects to no broker.
- **The paper engine is fail-closed.** `LIVE_TRADING_ELIGIBLE = False` in
  `btc_forecaster/paper/decision.py`, and `current_live_permission()` returns no
  live candidates and zero leverage. The engine reads A2's `promotion.csv`, which
  promotes nothing.
- **A6 and A7 cannot change that.** Their manifests record no promoted model and
  live trading disabled, their runners refuse to write one that says otherwise,
  A7's decision states include nothing that promotes or trades, and no research
  module imports the paper engine.
- **The suite enforces it.** `tests/test_quant_freeze.py` pins each of those facts;
  `tests/test_paper_a2_regression.py` and `tests/test_shadow_a3.py` pin the paper
  side.

No research result enables live trading by itself — not this one, and not a
future positive one. A model that met every criterion of
`NO_MODEL_PROMOTION_UNTIL` would earn a confirmation study, not a trading system.
