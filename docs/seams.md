# Integration seams for external signals

Track A owns the quantitative core. Track B owns market-intelligence retrieval
(news, sentiment, on-chain, flow). This document is the contract between them.

**Track A implements none of that retrieval.** What it provides is the set of
places an external signal can attach without breaking point-in-time correctness,
and the machinery that will catch it if it does.

---

## The one rule

Everything reduces to the inequality in `btc_forecaster/timebase.py`:

```
available_time(observation) <= forecast_origin
```

where `available_time = event_time + publication_lag`, and `event_time` for a
daily bar is the **end** of that bar, not its label.

A price has zero publication lag: the close is known the instant the bar ends.
Almost nothing Track B produces will have zero lag, and several will have
*negative* effective lag if handled carelessly — a sentiment score computed from
an article corpus that was re-crawled last week carries next week's information
into last month's bar.

`assert_available()` is the single enforcement point. Use it.

---

## Seam 1 — `FeatureSpec.publication_lag` (the primary seam)

The cleanest attachment point. Subclass `FeatureSpec`, declare the lag, and the
existing machinery handles alignment, causality checking and the walk-forward
refit.

```python
from btc_forecaster.features.spec import FeatureSpec
import pandas as pd

class NewsSentiment(FeatureSpec):
    """Aggregate sentiment over articles published in a bar.

    publication_lag is 6 hours: the crawl and scoring pipeline runs on a
    schedule, and a score stamped at bar D is not queryable until D+1 06:00Z.
    """

    def __init__(self, source: pd.Series):
        super().__init__(name="news_sentiment", publication_lag=pd.Timedelta(hours=6))
        self._source = source

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        return self._source.reindex(frame.index).ffill().rename(self.name)
```

Requirements:

- `compute()` must be **causal**: row `D` uses only data from bars `<= D`.
  Forward-fill is safe; backward-fill and centred windows are not.
- Declare a truthful `publication_lag`. It is not decoration — a lagged signal
  from a past bar can still be unavailable at an origin, and a test in
  `tests/test_timebase.py` pins exactly that case.
- Verify with `assert_causal(frame, [your_spec])` before wiring anything up. It
  perturbs the future and asserts past values do not move, which catches leaks
  that reading the code will not.

Add the spec to a `FeatureSelector` result or pass it directly to
`build_feature_frame`. `to_supervised()` then applies the forecast-step shift,
so the feature row predicting bar `T` comes from bar `T - step`.

**This is the seam to prefer.** It composes with everything already built.

---

## Seam 2 — `TrainingWindow.exog`

For models that consume exogenous columns directly rather than through the
feature pipeline:

```python
window = TrainingWindow(frame=market_frame, exog=signal_frame)
```

`exog` must be indexed identically to the market frame, and each column must
already be point-in-time correct at its bar. The platform does not lag it for
you — that is the producer's job, declared through Seam 1.

---

## Seam 3 — `SarimaxModel(exog_columns=...)` — restricted

Dynamic regression with exogenous regressors. Fitted on `TrainingWindow.exog`,
forecast with `future_exog`.

**Hard constraint, and the one most likely to be got wrong:** `future_exog`
requires values over the entire forecast horizon, which means the regressor must
be **known in advance**.

| Legitimate | Not legitimate |
| --- | --- |
| Day of week, month, holiday flags | Sentiment index |
| Scheduled halving dates | News volume |
| Announced policy or listing dates | Whale flow |
| Contract expiry calendars | Anything requiring its own forecast |

Supplying realised future values of a signal that would have had to be forecast
is leakage of the most direct kind, and it produces spectacular backtest results
that evaporate live. If a Track B signal is not deterministic ahead of time, it
attaches via **Seam 1 as a lagged input**, not here.

---

## Seam 4 — `MarketDataProvider`

For alternative *price* sources (another exchange, an index, a different
vendor). Implement `fetch(ticker, start, end) -> pd.DataFrame` returning a frame
that satisfies `validate_market_frame`.

Non-OHLCV signal sources should **not** implement this protocol. They are not
price series, they carry publication lag, and they belong in Seams 1–2.

---

## Seam 5 — model registry

Register a model that consumes external signals without touching the core:

```python
from btc_forecaster.models import registry

registry.register(
    "sentiment_hybrid",
    lambda **kw: SentimentHybrid(**kw),
    family="intelligence",
    requires=("market_intelligence",),
    description="Price hybrid with lagged external sentiment features.",
)
```

It is then scored through the same walk-forward folds as the random walk,
against the same baseline, with the same metrics. That comparison is the point:
an intelligence-driven model that does not beat `random_walk` on identical folds
has not earned its data pipeline, however sophisticated the retrieval is.

---

## What Track B should expect to have to prove

Adding a signal source is easy. Showing it helps is not. The engine will ask for:

1. **Positive MAE skill against `random_walk`** on identical folds. Not
   directional accuracy above 0.5 — a model can beat a coin and still lose to
   the naive forecast.
2. **Stability across folds.** `mae_std` in the summary. One good fold out of six
   is noise.
3. **Interval calibration.** If the signal narrows the intervals, coverage must
   still land near nominal.
4. **A multiple-comparisons account.** Testing twenty signals and reporting the
   best one is a search, not a discovery. `expected_false_positives(20, 0.05)`
   is 1.0. Use `benjamini_hochberg`.

A negative result here is a real result and should be promoted to
`research/runs/` with its README saying so. See `ARTIFACTS.md`.

---

## Anti-patterns

Each of these has a test that catches it:

| Anti-pattern | Caught by |
| --- | --- |
| Signal aligned to the bar it predicts | `to_supervised(step=0)` raises |
| Backfilled or interpolated across gaps | `assert_causal` |
| Whole-sample normalisation of a signal | `assert_causal` |
| Signal selected once on all history | `FeatureSelection.fitted_end` audit |
| Revised/restated history treated as live | snapshot SHA-256 mismatch |
| Future sentiment as SARIMAX `future_exog` | not caught automatically — **this one is on you** |

The last row is the dangerous one. There is no way for the platform to tell a
legitimately-known-in-advance regressor from a leaked one by inspecting values.
It is a modelling judgement, and it has to be made honestly.
