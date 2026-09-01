# B4 — findings

Run `b4-3980bac0b72aeaa0`, manifest hash `11bc97ed1fc7a1b1`. Method and the
decisions behind it are in `METHOD.md`; the raw artifacts are in
`runs/b4-3980bac0b72aeaa0/`.

## The headline

**Nothing was carried forward.** Of 41 candidate signals, 21 were rejected and
20 could not be tested at all.

| decision | count | what it means |
| --- | ---: | --- |
| `CARRY_FORWARD` | **0** | nothing cleared the preregistered bar |
| `EXPLORATORY_ONLY` | **0** | nothing was worth carrying on exploratory grounds |
| `REJECT` | 21 | cross-asset features, tested and found wanting |
| `INSUFFICIENT_DATA` | 20 | intelligence studies, no data to test |

Those two failure modes are different and are not merged. A signal that could
not be tested has not been found wanting.

## 1. Every requested intelligence question is unanswerable

Twenty event studies were declared — regulation, monetary policy, ETF flows,
institutional adoption, exchange and security incidents, macro shocks,
geopolitics, liquidity events, generic entity statements, five named entities
(Trump, Musk, Powell, Saylor, SEC), and five whale transfer contexts studied
separately with `UNKNOWN` never pooled into inflow or outflow.

**All twenty ran. All twenty returned zero observations.**

The intelligence store holds 0 documents and 0 event signals. Both configured
providers are disabled placeholders — an empty RSS feed list and an unset search
endpoint. `SocialStatementProvider` and `WhaleDataProvider` are abstract seams
with no implementation. Nothing has ever been collected.

This is not closable by fetching harder. Retrieving articles today about a past
announcement yields evidence whose *historical availability* cannot be
established; B4.35 forbids using that as point-in-time forecasting evidence, and
pooling it with market data would contaminate every result in the track.

So for research questions 1–7 and 9–12, the answer is **INSUFFICIENT SAMPLE**,
not "no effect". The framework is built, tested against worlds with known
answers, and waiting for a corpus.

### The feature matrix says the same thing structurally

`feature_matrix/` holds an intelligence feature matrix built through the
unchanged B3.1 `HistoricalDatasetService` over 366 daily origins: 366 rows, 11
columns, 4 chunks, joined to BTC outcomes on `forecast_origin`. All 366 rows
join. **Zero rows carry a non-zero intelligence feature**, and the mean provider
coverage ratio is 0.0.

The path from origins to a joined research table works end to end. There is
simply nothing flowing through it.

## 2. Cross-asset lead/lag: no effect reaches even an uncorrected threshold

Seven assets (DXY, gold, WTI, S&P 500, Nasdaq, VIX, US 10-year) × three features
(1-day and 3-day lagged return, 7-day realized volatility) × three horizons (1d,
3d, 7d) = **63 declared cells, all 63 estimable**, over ~3,100 daily origins from
2018-01-01 to 2026-09-01.

| | |
| --- | ---: |
| cells reaching **uncorrected** p ≤ 0.05 | **0** |
| cells surviving Benjamini-Hochberg at q ≤ 0.10 | **0** |
| cells whose 95% interval excludes zero | **0** |
| largest \|correlation\| anywhere | 0.061 |

The six largest effects, all with intervals containing zero:

| feature @ horizon | n | slope per 1 sd | 95% interval | r | p | q |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| ust10y realized vol 7d @ 7d | 3159 | +0.00542 | [−0.00340, +0.01221] | +0.061 | 0.178 | 0.889 |
| oil realized vol 7d @ 7d | 3159 | +0.00405 | [−0.00478, +0.00952] | +0.045 | 0.322 | 0.889 |
| VIX realized vol 7d @ 7d | 3159 | +0.00396 | [−0.00694, +0.01293] | +0.044 | 0.488 | 0.889 |
| S&P realized vol 7d @ 7d | 3159 | +0.00362 | [−0.00703, +0.01052] | +0.040 | 0.442 | 0.889 |
| gold lagged return 3d @ 7d | 3099 | −0.00339 | [−0.00891, +0.00159] | −0.038 | 0.176 | 0.889 |
| DXY lagged return 3d @ 7d | 3100 | +0.00293 | [−0.00234, +0.00819] | +0.033 | 0.272 | 0.889 |

The preregistered practical bars, frozen from development-period dispersion
before the validation window was opened, were \|effect\| ≥ 0.0091 (1d), 0.0155
(3d) and 0.0245 (7d). **The largest effect measured is a fifth of the smallest
bar.**

So for research question 8 — do macro variables have useful lead/lag
relationships with BTC? — the answer at daily frequency over this window is
**NO EVIDENCE**: the associations are both statistically indistinguishable from
zero and, even taken at face value, an order of magnitude too small to matter.

## 3. The one result that did clear a statistical bar — and why it is still rejected

The Granger gate passed for all seven assets (3,165 paired daily returns, both
series through the stationarity screen). Of the seven bounded 5-lag tests, **one
survives Benjamini-Hochberg**:

    ust10y_daily -> btc_usd_daily     min p = 0.0024 at lag 5     q = 0.0170

and it is not a lone lucky lag: p = 0.0052, 0.0100 and 0.0024 at lags 3, 4 and 5.
The next best is oil at p = 0.032, q = 0.113 — not significant after correction.

**This is a real detection of lagged linear structure, and it changes nothing.**

The same predictor's effect sizes, from the lead/lag study on the same data:

| feature @ horizon | slope per 1 sd | 95% interval | r | q |
| --- | ---: | --- | ---: | ---: |
| ust10y lagged return 1d @ 1d | −0.00083 | [−0.00408, +0.00254] | −0.026 | 0.889 |
| ust10y lagged return 1d @ 3d | −0.00257 | [−0.00574, +0.00017] | −0.047 | 0.889 |
| ust10y lagged return 3d @ 1d | −0.00117 | [−0.00249, +0.00005] | −0.035 | 0.889 |
| ust10y lagged return 3d @ 3d | −0.00181 | [−0.00599, +0.00166] | −0.032 | 0.889 |

Between 0.03% and 0.26% of forward return per one-standard-deviation move in the
10-year yield, against a practical bar of 0.91%. **Four to a hundred times too
small**, with every interval containing zero.

This is exactly the case B4.17 exists for. At n = 3,165 a Granger F-test has
enormous power and will detect linear structure far below any magnitude that
matters. Reporting "US 10-year yields Granger-cause Bitcoin, q = 0.017" would be
true and grossly misleading. The registry rejects the underlying signal on
practical-significance grounds, and this paragraph is the reason the effect size
is reported next to every p-value in this track.

It is also, on its own terms, worth someone's attention later: it is the only
thing in 63 cells and 7 tests that survived correction, and if a future run with
intraday data reproduces it at a magnitude that matters, that would be a finding.

## 4. Negative findings, preserved (B4.46)

Stated plainly, because they are the output:

* **Trump** — INSUFFICIENT SAMPLE. No statement corpus exists.
* **Musk** — INSUFFICIENT SAMPLE. No statement corpus exists.
* **Powell / Federal Reserve** — INSUFFICIENT SAMPLE.
* **Saylor / institutional adoption** — INSUFFICIENT SAMPLE.
* **SEC / regulation** — INSUFFICIENT SAMPLE.
* **ETF flows** — INSUFFICIENT SAMPLE. No flow provider exists or is configured.
* **Whales (inflow, outflow, custody, internal, unknown)** — INSUFFICIENT SAMPLE
  in all five contexts, studied separately.
* **Sentiment, relevance, novelty, confidence** — INSUFFICIENT SAMPLE. The three
  predefined composites were named before results, as B4.8 requires, and none
  could be evaluated.
* **Gold** — NO EVIDENCE of a useful lead. \|r\| ≤ 0.038 across every declared
  lag and horizon.
* **Oil, DXY, S&P 500, Nasdaq, VIX** — NO EVIDENCE. Same picture.
* **US 10-year yield** — NO EVIDENCE on effect size, despite the corrected
  Granger detection above.

Not one entity importance prior in the watchlist configuration is supported by
any measurement. Every one of them is 1.0 by configuration, and B4.6 is explicit
that a configured prior is not an observed impact. After this run, all of them
remain exactly that: configuration.

## 5. Stability, concentration and decay

Meaningful only for the cross-asset candidates, since nothing else has data.

Every one of the 21 cross-asset candidates failed the stability gate — the
per-origin contributions do not hold a consistent sign across calendar years,
which is what one expects of a series that is noise. Seven of the 21 were also
flagged **fragile**: removing a single origin's contribution moves or flips the
estimate. Both are recorded per candidate in `signal_registry.json`.

Decay profiles were computed for all 20 event studies and are empty by
construction. No monotone decay was assumed anywhere; the profile reports the
peak horizon as measured.

## 6. Coverage bias, regimes, placebos

* **Coverage bias (B4.34)** — not assessable. It compares outcome dispersion in
  high- and low-coverage periods, and every period has zero coverage. The check
  correctly reports `INSUFFICIENT`.
* **Regime stratification (B4.27)** — INSUFFICIENT FOR REGIME ANALYSIS on the
  intelligence side. No hidden-state model was fitted; a regime model estimated
  on the full sample would use the future to label the past.
* **Placebo and matched controls (B4.18, B4.19)** — implemented and validated
  against synthetic worlds, but not run on real events, because there are none.
  In the synthetic world with a real 2% jump the placebo exceedance rate is
  ≤ 5%; in the null world it exceeds 5% routinely, as it must.

## 7. What would change the answer

1. **A collected intelligence corpus with genuine `available_at` timestamps.**
   Not a scrape of historical articles — those are `RETROSPECTIVE_ONLY` and
   cannot become point-in-time evidence. Forward collection from the existing
   B2/B3 pipeline, running from now, accumulating.
2. **Intraday data.** BTC hourly exists from 2024-09-01, and the 1h/6h horizons
   the intelligence features are defined over cannot be studied at daily
   frequency. A daily study cannot see a six-hour response.
3. **A contracted ETF-flow or on-chain provider.** Both are commercial products;
   no contract is assumed anywhere in this codebase.

Until at least the first of those exists, the honest position is that the
external-signal hypothesis has not been tested — not that it has failed.
