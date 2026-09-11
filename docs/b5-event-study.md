# B5 — point-in-time market-intelligence event study

**Decision: `INTELLIGENCE_CORPUS_INSUFFICIENT`. The track stopped at Gate 1.** No
point-in-time corpus exists that could support a defensible event study, so none
was run (section 19). The corpus-sufficiency thresholds were frozen in
[`research/market_intelligence/b5/preregistration.json`](../research/market_intelligence/b5/preregistration.json),
content hash `70777f0e45b0b8ddd6b6d84b4a9502bb4277d5a0d2b4631c59816610a50fcc9d`,
and committed before the canonical Gate 1 audit existed.

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

The existing taxonomy, unchanged: five signal categories, eleven event types, six
disclosure streams. Gate 1 judges fifteen families — one per event type, with
whale transfers split by transfer context — and reports every one, empty or not.

**Observed in the corpus**, after corrections: three `REGULATION` events (SEC)
and one `MONETARY_POLICY` event (CFTC). Every other type: none. The CFTC does not
set monetary policy, so that label looks like a misclassification by the
rule-based extractor. B5 records it as collected and does not relabel it:
relabelling is a correction for B4.1's append-only ledger, with its own evidence.

No category was analysed. There was nothing to analyse, and sparse categories were
not merged.

## 6. Event windows

**Not built; the study was not reached.** What the corpus would support, as
measured:

- Every point-in-time-valid event has an intraday event time taken from its
  source's publication timestamp, so sub-daily alignment to *publication* would
  be possible.
- But this system retrieved those events 13.4 to 17.7 hours after publication.
  Windows key off availability (section 4), so immediate (0–1 h) and short
  (1–6 h) reactions to publication fall before this system knew anything. Only
  windows opening at availability — daily and longer — could be studied from
  this corpus. A collector running at its three-hour cadence would narrow that
  gap; this one ran twice.

B4 already builds future-only targets at 1, 3 and 7 days, and BTC hourly bars
exist from 2024-09-01 in its data inventory. B5 would reuse both.

## 7. Controls

**Not drawn.** B4's placebo and matched-control machinery — timestamps shifted
within blocks, labels permuted within blocks, non-event origins matched on prior
realised volatility — is implemented and validated. It would supply the
unconditional, time-matched and pre-event baselines. None was run, because no
event was eligible.

## 8. Statistical methodology

**Not applied.** B4's inference layer — robust descriptive statistics, moving-
and stationary-block bootstraps with a declared block length, cluster-level
counting so forty articles are not forty events — is what a B5 study would use,
for the reasons B4's `METHOD.md` gives: returns are serially dependent and event
windows overlap.

## 9. Multiple-testing methodology

**Not applied.** The family would be every (event family × horizon × quality
stratum) test run, corrected together with Benjamini-Hochberg through B4's
`benjamini_hochberg`, at the q ≤ 0.10 B4 preregistered, with raw p, q, effect
size, practical threshold and sample size reported for each. No test was run.

## 10. Practical significance thresholds

B4's, preregistered from development-period dispersion before its validation
window was opened: **0.91% at one day, 1.55% at three, 2.45% at seven** (a quarter
of the development-period standard deviation). At the minimum sample, a study
could detect only effects of 2.27%, 3.89% and 6.14% (section 3).

No intraday threshold exists, because no intraday study was reachable. A future B5
run that includes intraday windows must preregister one before it looks.

## 11. Stability methodology

**Not measured.** B4's period stability and leave-one-period-out would split the
history into early, middle and late blocks. This corpus spans one day of event
time; there are no blocks to split.

## 12. Publisher and entity confounding

Measured descriptively, because nothing reached a study. Six documents from three
publishers — the SEC (3), the CFTC (2) and the Bureau of Labor Statistics (1) —
through one provider. Of the four events corrections leave, three are the SEC's,
and all three fall in one cluster. After the quality floor, no event and no
publisher remains. Leave-one-publisher-out needs a result to leave one out of.

## 13. Synthetic validation

**Of the Gate 1 audit** — twenty-two corpora with known answers, built through
the store's own API (`tests/test_intelligence_b5_audit.py`):

| World | Expected | Holds |
|---|---|---|
| Sufficient: 40 occurrences, 3 publishers, 312 days, daily collection | Gate 1 passes | yes |
| Empty store | every clause unmet | yes |
| Replica of the collected corpus | insufficient, only `ready_families` unmet | yes |
| One occurrence reported by three publishers, ×40 | 40 events, not 120 | yes |
| Every document rediscovered twice | sightings rise; the count does not move | yes |
| One publisher | publisher clause unmet | yes |
| Extraction confidence 0.35 | nothing passes the quality floor | yes |
| 15 of 40 events corrected | 25 counted; corrected events stay in the catalog | yes |
| 10 events known before they happened, beside a ready family | integrity clause closes the gate | yes |
| 40 retrieval-aligned event times, beside a ready family | event-time clause closes the gate | yes |
| An event moved past the audit instant | absent, then present later | yes |
| An event time moved, availability fixed | alignment changes, nothing else | yes |
| A future article | earlier audit byte-identical | yes |
| A later re-extraction | earlier audit byte-identical | yes |

**Of an event-study engine, not built.** Noise, known positive and negative
effects, an effect that disappears halfway, duplicates, future leaks and
publisher confounding are the worlds an event study must pass before a canonical
run. The stop rule prevents that run, so B5 builds no engine to validate. B4's
engine was validated this way — in its synthetic world with a real 2% jump, the
placebo exceedance rate stays at or below 5% — and a future B5 study would
extend that validation, not replace it.

## 14. Historical results

From the committed Gate 1 run in
[`research/market_intelligence/b5/gate1/`](../research/market_intelligence/b5/gate1/),
audited as of 2026-09-11T00:00:00Z.

**The historical corpus is empty.** B4's store holds no documents, and it cannot
be filled after the fact (section 4).

**The forward corpus:**

| Measure | Value |
|---|---|
| Documents | 6: SEC 3, CFTC 2, BLS 1; one provider |
| Retrieved copies | 15 — a 60% rediscovery rate |
| Quarantined | 0 |
| Events collected | 7 |
| Invalidated by corrections | 3 (43%), all `REDISCOVERY_DUPLICATE_PRE_FIX` |
| Event types, after corrections | `REGULATION` 3, `MONETARY_POLICY` 1 |
| Clusters, after corrections | 2 |
| Event time | 2026-09-01 14:57 → 2026-09-02 16:04 UTC |
| Collection | 5 runs, all `DEGRADED`, on 2 days; the last 8 days before the audit instant |
| Documents with a publication time | 100% |
| Point-in-time-valid events | 100% (4 of 4) |
| Event times from publication, intraday | 4 of 4 |
| Retrieval after publication | 13.4 h minimum, 16.0 h median, 17.7 h maximum |
| Documents with a disclosure stream recorded | 0% — collected before the field existed |

**The funnel**, from what was collected to what a study could count:

| Stage | Events |
|---|---:|
| Collected | 7 |
| Not invalidated | 4 |
| Sources resolved | 4 |
| Point-in-time valid | 4 |
| Relevance ≥ 0.5 and confidence ≥ 0.5 | **0** |
| Aligned to publication | 0 |
| Independent events | 0 |
| Effective events at 168 h | 0 |

**Gate 1:** `ready_families` **unmet** — 0 of 15 families ready.
`point_in_time_integrity` met (0 of 4 impossible). `event_time_from_publication`
met (4 of 4).

## 15. Negative findings

- **No defensible event study can be performed.** No family holds a single
  countable event against a bar of 30, and no family spans more than a day
  against a bar of 180.
- **The historical question remains unanswerable,** as B4 found. Nothing
  collected later can become point-in-time evidence for an earlier instant.
- **The collector ran for two days and stopped.** It is not deployed. The corpus
  it left is a pilot, not a sample.
- **The deployed extractor cannot produce a countable event.** `RuleBasedExtractor`
  assigns every event relevance 0.55, confidence 0.35, novelty 0.5 and sentiment
  0.0 by construction — it describes itself as a conservative routing fallback.
  Confidence 0.35 is below B4's floor of 0.5, so however long it runs, Gate 1
  would count nothing. And with those scores constant, every quality comparison B5
  asks about — high against low relevance, novelty, sentiment or confidence — is
  a comparison of a constant with itself.
- **The readiness report cannot open** (see the forensics above): `corpus-status`
  never supplies collection coverage.

## 16. Positive findings

**About BTC: none.** Nothing was estimated.

**About the corpus and the instrument:**

- Point-in-time integrity holds for everything that was collected: no impossible
  timestamp, every event time from a publication timestamp, every one intraday.
- The rediscovery defect the collector had is on the record and handled as
  designed: its three duplicate events remain in the store, carry their
  corrections, and are excluded from every count.
- The audit is validated: it passes a corpus that meets the bar, fails one that
  does not for each reason separately, resists the four point-in-time attacks,
  and never writes to what it reads.

## 17. Fragile findings

None. A finding needs an estimate, and none was made.

## 18. Limitations

- **One corpus, from one machine,** identified by the sha256 of its store file.
  The store is the collector's local state and is not committed.
- **An audit instant, not a moving present.** The result describes the corpus as
  of 2026-09-11. A later audit of a growing corpus is a new result under the same
  preregistration.
- **Coverage is computed per family,** over the family's own span, from days with a
  completed run — the readiness gate's own definition, applied because the report
  that should apply it does not.
- **Gate 1 reads no market data.** It decides whether a study is possible, not
  what one would find.
- **The thresholds are a floor on power** (section 3). A corpus that just clears
  them supports only a study able to see large effects.
- **The rule-based extractor's scores are constants**, so the quality floor here
  measures the extractor's design rather than a judgement about each event.

## 19. Decision

**`INTELLIGENCE_CORPUS_INSUFFICIENT`.**

The track stops at Gate 1, as preregistered. No reaction was computed, no
category tested, no control drawn, no signal mined. This is not evidence that
external intelligence carries no information about BTC; it is evidence that this
repository cannot yet test whether it does.

What would change it — without lowering any threshold:

1. **A collector running on a persistent host,** at its declared cadence, long
   enough for a family to span 180 days with collection on 80% of them.
2. **An extractor whose confidence can clear B4's floor** and whose scores vary by
   event. With the rule-based fallback, Gate 1 can never pass. Any LLM involved
   may classify and extract structured, timestamped events; it must never emit a
   trading instruction, and the study must never consume one.
3. **At least one family reaching 30 independent events from three publishers,**
   then Gate 1 re-run under the same preregistration at a new audit instant.
4. **Repairing `corpus-status`'s coverage accounting,** as a B4.1 change of its
   own, so the readiness report can say when that moment arrives.

Nothing is promoted, and live trading remains disabled. Quantitative research
stays frozen ([`quant-research-status.md`](quant-research-status.md)).

## 20. Reproducibility

```bash
# Check the committed result: every file's sha256 and the result digest.
python -m market_intelligence.b5 verify research/market_intelligence/b5/gate1

# Re-run the audit on a store, read-only, under the committed preregistration.
python -m market_intelligence.b5 audit --db <intelligence.duckdb> \
    --as-of 2026-09-11T00:00:00+00:00 --input-label collector-state:default-root \
    --output <scratch directory>
```

A re-run on the store whose sha256 the manifest records, at the same audit
instant, writes the same bytes and the same result digest. The canonical files
carry no timestamp of the run, no timing and no path, and none contains a newline,
so a checkout's line-ending conversion cannot change them. CI verifies the
committed result and checks that the committed preregistration is exactly what
the code declares.
