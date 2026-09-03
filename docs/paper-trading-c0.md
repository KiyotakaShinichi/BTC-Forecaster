# C0 paper-trading research engine

C0 is a deterministic execution and risk research layer for a future validated
forecast distribution. It does not connect to exchanges and every decision is
marked `research_only`. The frozen A2 evidence promotes zero models, so the
current evidence-backed output is `NO_TRADE`, live eligibility is false, and
allowed live leverage is `0x`.

The core consumes `ForecastDistribution`, independent of model family, and
emits immutable `TradeDecision` records. A candidate requires every gate to
pass: model promotion, empirical calibration, fresh data, bounded uncertainty,
positive distribution-based EV after versioned costs, non-low confidence,
active portfolio risk, and green integrity. Missing evidence fails closed.

Confidence uses an empirically fitted, regularized Platt map with a minimum
sample requirement. Reliability bins, Brier score, and expected calibration
error are reported separately from the frozen LOW/MODERATE/HIGH participation
bands. Confidence never determines leverage.

Execution policies are predefined and versioned. OHLC simulations resolve a
same-bar stop/target collision conservatively as stop-first and label the result
`AMBIGUOUS_STOP_FIRST`. Every trade has a horizon-aligned time stop. The
append-only JSONL journal is hash chained so mutation is detected.

Research utilities include performance and no-trade metrics, predefined
execution ablations, chronology overlap checks, block bootstrap expectancy,
and a drawdown-based ruin proxy. Threshold selection must occur only before the
final temporal validation period.

Commands:

```text
btc-forecast opportunity
btc-forecast risk-status
btc-forecast journal PATH
```

`opportunity` deliberately reports only the current A2 permission state. It
does not fabricate a forecast or market price.
