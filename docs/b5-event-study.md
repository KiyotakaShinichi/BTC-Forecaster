# B5 — point-in-time market-intelligence event study

**Status: Gate 1 preregistered, not yet decided.** The corpus-sufficiency thresholds
below are frozen in
[`research/market_intelligence/b5/preregistration.json`](../research/market_intelligence/b5/preregistration.json),
content hash `70777f0e45b0b8ddd6b6d84b4a9502bb4277d5a0d2b4631c59816610a50fcc9d`,
committed before the canonical Gate 1 audit exists. Sections 5 onward are filled
in once Gate 1 is decided, and only if the track gets that far.

B5 is research-only. The quantitative line is frozen
([`quant-research-status.md`](quant-research-status.md)); B5 builds no forecasting
model, trains nothing, and cannot promote anything. Live trading stays disabled.

---

## Before B5: what already exists

Read before a line of B5 was written, so that nothing is built twice.

- **The platform.** `market_intelligence/` is a point-in-time collector, a DuckDB
  store (schema version 3, append-only, migrated only forward), a replay layer
  and B4's research engine. Its collection semantics are frozen in
  [`deploy/COLLECTION_FREEZE.md`](../deploy/COLLECTION_FREEZE.md).
- **B4 asked this question historically and could not answer it.** Run
  `b4-3980bac0b72aeaa0` declared twenty intelligence event studies; all twenty
  returned zero observations, because the intelligence store held nothing. B4's
  finding was `HOLD` — the question was not answerable with that evidence — not a
  negative. It also established why a historical corpus cannot be assembled after
  the fact: an article retrieved today about a past announcement became *available
  to this system* today, and B4.35 forbids treating it as point-in-time evidence
  for an earlier instant. ([`research/market_intelligence/b4/FINDINGS.md`](../research/market_intelligence/b4/FINDINGS.md))
- **B4.1 built the forward path.** A collector over official feeds (SEC, Federal
  Reserve, CFTC, BLS), a readiness gate carrying B4's thresholds, deduplication,
  clustering, corrections and corpus snapshots. It is a frozen deployment
  candidate and is not deployed.
- **The point-in-time contract.** A document's `available_at` is its retrieval
  time, first write wins, and rediscovery never moves it. A document may carry the
  source's own `published_at`. An event carries `event_time` and
  `available_time`. Corrections are append-only: an invalidated event stays in the
  store and is excluded from anything that counts.
- **Deduplication and provenance.** Canonical URL is identity; clustering groups
  events of one type and entity within a declared window, and treats
  corroboration as metadata, never as sample size. Sightings, raw-evidence hashes
  and corpus snapshots with membership hashes record where everything came from.
- **Taxonomy and quality fields.** Five signal categories, eleven event types,
  six disclosure streams, source types; per event, sentiment, BTC relevance,
  novelty, extraction confidence and extraction method; per document, timestamp
  quality.
- **B4's engine already implements most of what an event study needs**:
  cluster-level counting, pre-event returns, placebos that preserve temporal
  structure, matched non-event controls, block bootstrap, Benjamini-Hochberg,
  practical-significance classification, decay, period stability,
  leave-one-period-out, concentration and future-only targets, each validated on
  synthetic worlds in `tests/test_intelligence_b4_*.py`. B5 does not rebuild any of
  it.
- **A gap in the readiness report, recorded and not fixed here.** `corpus-status`
  is the only caller of the readiness gate's `assess_family`, and it never passes
  a collection-coverage fraction, so the 80% coverage clause is always reported
  unmet and the report can never say `READY_FOR_VALIDATION`. The gate's own
  `coverage_fraction()` and the collection service's `successful_run_days()` have
  no callers. Readiness semantics are frozen
  ([`COLLECTION_FREEZE.md`](../deploy/COLLECTION_FREEZE.md) section 10), so B5 does
  not change them: its audit computes coverage itself, per family, with the gate's
  own `coverage_fraction`. Repairing the report is a B4.1 change for its own
  commit.

**What corpus exists.** Three intelligence stores are reachable from this
repository and this machine:

| Store | Documents | Events | Collection |
|---|---:|---:|---|
| B4's historical store (feature-matrix build) | 0 | 0 | none |
| A stray CLI database in an old integration worktree | 0 | 0 | none |
| The collector's state, at its default root on the development machine | 6 | 7 | 5 runs on 2026-09-02 and 2026-09-03, then none |

The third is the only corpus there is. The canonical Gate 1 audit is run on it.

## 1. Research question

> Does independently timestamped external information produce measurable and
> stable BTC market reactions after controlling for baseline market behaviour?

With secondary questions about category, event quality, novelty, decay,
stability, publisher concentration, incremental information over price behaviour,
and survival under multiple-testing correction.

B5 is an event study, not a machine-learning competition. Its first question is
narrower than any of those: **is there a point-in-time corpus sufficient to ask
them?** That is Gate 1, and if it fails the track stops.

## 2. Relationship to A2, A6 and A7

A2, A6 and A7 asked whether any model forecasting from BTC's own price and volume
history beats the naive forecast, and found none that does
([`quant-research-status.md`](quant-research-status.md)). B5 asks a different
question on different information: whether external events are *associated with*
BTC's behaviour beyond what ordinary movement at comparable times would produce.
Its numbers are not comparable to A2's, A6's or A7's, and nothing here reopens
quantitative model research.

## 3. Corpus sufficiency

### The thresholds

A corpus is sufficient only if at least one family meets every adequacy clause
**on its own**, and the corpus as a whole meets the integrity clauses.

| Clause | Threshold | Where it comes from |
|---|---|---|
| Independent events in the family (clusters) | ≥ 30 | B4 preregistration; B4.1 readiness |
| Effective events, non-overlapping at 168 h | ≥ 20 | B4 carry-forward policy; B4.1 readiness |
| Publishers in the family (union) | ≥ 3 | B4.1 readiness |
| Providers in the family | ≥ 1 | B4.1 readiness |
| Span, first to last event availability | ≥ 180 days | B4.1 readiness |
| Days in that span with a completed collection run | ≥ 80% | B4.1 readiness |
| Families meeting every clause above, each alone | ≥ 1 | B5 |
| BTC relevance and extraction confidence of a counted event | ≥ 0.5 each | B4 extraction-quality filters |
| Point-in-time-valid events whose event time is a source's publication time | ≥ 90% | B5 |
| Events with impossible timestamps | ≤ 5%, and always excluded | B5 |

The adequacy numbers are B4's, preregistered under content hash `389aa76f…` and
held by the B4.1 readiness gate. The B5 code holds that gate's own policy object,
and `tests/test_intelligence_b5_contracts.py` reads B4's committed preregistration
back and fails if any inherited number differs.

The B5 clauses only remove events. None can make a corpus sufficient that B4's
bar would call insufficient.

### Why these numbers, and not lower ones

A family at the threshold offers 20 effective events. With the standard deviation
B4 measured in its development period, a two-sided test at 5% with 80% power on
20 independent events can detect a mean effect of **2.27% at one day, 3.89% at
three days and 6.14% at seven** — 2.5 times B4's practical floors. The thresholds
are the least evidence at which a study can see even large effects; lowering them
would make it blind to anything smaller. Recorded in the preregistration as
`minimum_detectable_effects`.

### Families, never pooled

Each event type is its own family; whale transfers are judged per transfer
context (B4.1.32). Fifteen families in all, every one reported, including the
empty ones. A family short of the bar is reported as short. Merging sparse
families to reach a count is exactly what this gate refuses, so the pooled
corpus is shown for context and decides nothing.

### What was seen before this was written

Forensics required inspecting the corpus, so its size was known when these
thresholds were declared: six documents, seven events, three of them invalidated
by corrections, two clusters, two days of collection. The thresholds were not
chosen with that in view. They are B4's, frozen before B4.1's collector existed,
plus clauses that only tighten. No threshold B5 could have chosen would change the
outcome: no family holds more than one independent event against a bar of 30, and
no family spans more than a day against a bar of 180.

### Stop rule

If Gate 1 fails, B5 records `INTELLIGENCE_CORPUS_INSUFFICIENT` and stops. No
reaction is computed, no category is tested, no control is drawn.

## 4. Point-in-time definition

Two instants per event, kept separate:

- **`event_time`** — when the event happened, taken from a source's own
  publication timestamp where one exists.
- **`available_time`** — when this system first had it: the retrieval of its
  earliest source, first write wins. Rediscovery never moves it.

**Eligibility keys off `available_time`.** An audit, a study or a forecast at
instant *T* may count an event only if `available_time ≤ T`. A reaction window for
information value to this system opens at `available_time`; `event_time` is kept
for alignment and diagnostics, never as a substitute. A window shorter than the
gap between publication and retrieval cannot be studied from availability.

**An event is excluded, and counted as a point-in-time violation, if** its
`event_time` is after its `available_time`, its `available_time` is earlier than
its earliest source's availability, or a source claims publication after its own
retrieval. Each is an impossible record.

**An event has an identifiable event time** when `event_time` equals a source's
`published_at`. Otherwise it is recorded as aligned to retrieval, or to neither,
with its precision (intraday or date-only), and a study could not align its
reaction to publication.

**Retrospective evidence is never eligible.** Collecting an article today about an
event last year gives it `available_time` today. No backfilled publication time
substitutes for that.

Four adversaries enforce this in the audit's tests: an event moved beyond the audit
instant disappears; an event whose `event_time` moves while its availability does
not changes its alignment and nothing else; a future article cannot touch an
earlier audit; and a later re-extraction of an event cannot change what the audit
counted at an earlier instant.

## 5. Event taxonomy

*Pending: decided by Gate 1.*

## 6. Event windows

*Pending: decided by Gate 1.*

## 7. Controls

*Pending: decided by Gate 1.*

## 8. Statistical methodology

*Pending: decided by Gate 1.*

## 9. Multiple-testing methodology

*Pending: decided by Gate 1.*

## 10. Practical significance thresholds

*Pending: decided by Gate 1.*

## 11. Stability methodology

*Pending: decided by Gate 1.*

## 12. Publisher and entity confounding

*Pending: decided by Gate 1.*

## 13. Synthetic validation

*Pending: decided by Gate 1.*

## 14. Historical results

*Pending: decided by Gate 1.*

## 15. Negative findings

*Pending.*

## 16. Positive findings

*Pending.*

## 17. Fragile findings

*Pending.*

## 18. Limitations

*Pending.*

## 19. Decision

*Pending.*

## 20. Reproducibility

*Pending.*
