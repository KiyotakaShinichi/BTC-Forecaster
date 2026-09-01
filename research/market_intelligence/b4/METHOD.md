# B4 — method, and the decisions that shaped it

Written before the results, so that the choices below can be read as the plan
they were rather than as a rationalisation of what came out. Numbers live in
`FINDINGS.md`, the run directories under `runs/`, and `data_inventory.json`.

## The finding that shaped everything else

The B4.0 audit was run first, and it decided what this track could honestly be.

**Obtainable, point-in-time defensible:** BTC (daily from 2014-09-17, hourly from
2024-09-01), DXY, gold, WTI, S&P 500, Nasdaq, VIX, the US 10-year, and four
large-cap crypto constituents for a market benchmark. Thirteen series.

**Not obtainable at all:** every intelligence domain. No news corpus, no entity
statements, no regulatory event log, no ETF flows, no whale observations. Both
configured providers are disabled placeholders; `SocialStatementProvider` and
`WhaleDataProvider` are abstract seams with no implementation; the store has
never held a single document.

That is not a gap that can be closed by fetching harder. Retrieving articles
today about a 2021 announcement produces evidence whose *historical
availability* cannot be established, and B4.35 forbids using it as
point-in-time forecasting evidence. A corpus assembled that way would be
`RETROSPECTIVE_ONLY`, and pooling it with market data would contaminate every
result in the track.

So B4 splits in two. The cross-asset questions get a real study. The event,
entity, whale and sentiment questions get a **complete, tested framework and an
explicit `INSUFFICIENT_DATA` result** — because a study that leaves no trace is
indistinguishable from one that was never asked for.

## Availability, not the label

A daily bar labelled 2026-08-30 covers a whole day. Its close is not knowable
until the day ends. Treating the label as the availability time leaks a day of
hindsight into an entire study, and it looks completely reasonable in code.

Every bar therefore carries both, and availability is derived conservatively:

    available_at(close) = period_start + period_length

Exact for 24/7 UTC series. For an exchange-local series it lands *later* than
the true close — the NYSE close on a Friday is knowable at 20:00 UTC while this
rule says 04:00 Saturday. Later is the safe direction: it can only weaken a
result, never falsely strengthen one. Modelling every venue's session and
holiday calendar would buy a few hours of freshness in exchange for a wide
surface of silent, venue-specific leakage bugs.

Targets live in a separate file joined to features by `forecast_origin` alone.
The base price is the last close available *at* the origin — exactly what a
feature may legitimately see — and every forward price is strictly after it. A
horizon that runs past the data is `None`, never a silently shortened window: a
truncated 72h return in a column labelled 72h is a differently-defined outcome
hiding in plain sight.

The leakage audit re-derives every outcome from the series rather than trusting
the stored row. That is the whole point — a bug in the target builder that also
wrote a matching `base_available_at` would satisfy any check that only read the
row back.

## Inference for dependent data

**Block bootstrap, not iid.** Returns are serially dependent and event windows
overlap heavily. An iid bootstrap would treat 500 overlapping 72-hour windows as
500 independent observations and produce intervals several times too narrow. A
test on a strongly autocorrelated AR(1) shows the one-observation bootstrap
giving an interval under two-thirds the width of the blocked one.

**Families declared before results.** Entities × event types × horizons ×
outcomes generates hundreds of tests, and at the 5% level roughly one in twenty
comes back "significant" from noise alone. Benjamini-Hochberg within
pre-declared families. A family of 200 pure-noise p-values yields the expected
handful of uncorrected passes and zero discoveries.

**Significance is not the finding.** Practical thresholds are 0.25 × the
development-period standard deviation of the outcome — in the outcome's own
units, arguable, and frozen in the preregistration. Sample adequacy is checked
*before* anything else, so an underpowered test that happens to look significant
is `INSUFFICIENT_SAMPLE`, not `SUPPORTED`.

**Effects in the outcome's units.** The cross-asset effect is the slope of BTC's
forward return on a *standardised* predictor — the expected forward-return change
per one-sd move. A correlation coefficient cannot be compared against a
threshold expressed in returns, so correlation is reported for context and never
as the headline.

## The three things event studies usually get wrong

**Forty articles are not forty events.** One announcement produces a wave of
coverage; counting each story separately inflates N by an order of magnitude and
shrinks every interval to match. Events are clustered by (type, entity) within a
declared window, chaining so a genuine multi-day news cycle stays one cluster.
Source breadth is kept as corroboration metadata, never as sample size.

**Overlapping windows are not independent.** The policy is declared per study,
and the effective non-overlapping count is reported next to the raw one. Ten
events an hour apart are ten observations at a 1h horizon and one at 24h.

**A move that already happened is not a response.** Pre-event returns are
computed for every study. A run-up at least half the size of the measured
response is flagged as consistent with reverse timing or leakage rather than
impact.

## Validating a framework with no data to validate it on

With no intelligence corpus in existence, the tests are the only evidence the
machinery works. They are built around two synthetic worlds with known answers:

* one where events really are followed by a 2% jump, which the engine recovers
  to within 0.4 percentage points and whose placebo distribution it clears;
* one where the same events sit on a series that knows nothing about them, where
  every horizon's interval must contain zero and the placebo must reproduce the
  "effect" routinely.

The second is the load-bearing one. A framework that only passes the first would
confirm every hypothesis B4 was asked to test.

## Preregistration and the split

The last 30% of the history is held out. The end of the history, not a random
sample: a random holdout from a time series leaks, because neighbouring
observations are nearly the same observation. Practical thresholds are derived
from the *development* period only — computing them on the data a result will be
read from is choosing the threshold after seeing the estimate.

The plan is hashed before the validation period is opened, and the file refuses
to be overwritten with a different plan. If the plan genuinely needs to change,
the revision goes in a new file so both stay on the record.

## What was deliberately not done

* **No Granger test was run.** It is gated on sample size and a stationarity
  screen. "Granger causality" carries more authority than the procedure earns,
  and running it on inputs that fail its assumptions launders a weak association
  into a causal-sounding claim.
* **No regime model was fitted.** A hidden-state model estimated on the full
  sample would use the future to label the past. Stratification is available
  only on labels computable at the origin, and is refused outright when cells
  are thin.
* **No propensity-score system.** Matching is a coarse volatility bucket and a
  calendar period. This is observational; an elaborate matching model would add
  the appearance of causal identification without the design that earns it.
* **No trading backtest, no Sharpe ratio, no equity curve, no position sizing.**
  B4.47. Those belong, if ever, after predictive value survives out-of-sample
  testing in a later track.
* **No weights.** The registry records what was tested and what came back.
  Assigning a weight would be signal fusion.
