# B5.1 — collection readiness

B5 stopped at its first gate: there is no point-in-time corpus an event study
could use. That result stands and is not reinterpreted here. B5.1 is the
engineering that has to be true before Gate 1 can ever be passed honestly — so
that when the corpus has grown, the gate measures it correctly rather than
failing for reasons that have nothing to do with the evidence.

It is not research. No event study was run, no category was tested, nothing was
mined from the corpus, and nothing here produces a trading signal. Live trading
remains disabled.

**The corpus is not sufficient.** Section 11 says so without qualification.

---

## 1. B5 outcome

B5 audited the collected store as of 2026-09-11T00:00:00Z under a preregistered
policy (`70777f0e…`) and decided **`INTELLIGENCE_CORPUS_INSUFFICIENT`** (result
digest `80702011…`, committed in
[`research/market_intelligence/b5/gate1/`](../research/market_intelligence/b5/gate1/)).
Six documents and seven events from two days of collection; three events already
invalidated as pre-fix rediscovery duplicates; none of the four left cleared the
quality floor; no family of fifteen was ready. See
[`b5-event-study.md`](b5-event-study.md).

Besides elapsed time, B5's forensics named engineering defects that would have
kept Gate 1 shut however long collection ran:

| Defect | Effect |
|---|---|
| The rule-based extractor gave every event confidence 0.35 by construction | no event could ever clear the 0.5 floor |
| `corpus-status` never passed a coverage figure | every family read "collection covered 0% of days" |
| `coverage_fraction()` counted days outside the span and returned 0.0 for one-day spans | coverage could be fabricated in both directions |
| A CFTC clearing rule was typed `MONETARY_POLICY` | a substring match on "interest rate" |
| No measure of publication-to-collection lag | provider delay was invisible |
| The collector ran for two days and is not deployed | no corpus is accumulating |

B5.1 repairs the first five. The sixth is not a repository problem (section 8).

## 2. Gate 1 policy

**Unchanged.** The preregistration B5 applied is the one Gate 1 still applies,
byte for byte; CI fails if the re-run's preregistration hash differs from B5's.

| Clause | Required |
|---|---|
| events per family | ≥ 30 |
| effective (non-overlapping at 168 h) | ≥ 20 |
| publishers (union across the family) | ≥ 3 |
| providers | ≥ 1 |
| span | ≥ 180 days |
| collection coverage over the family's span | ≥ 80% |
| event relevance / confidence | each ≥ 0.5 |
| impossible timestamps | ≤ 5% of resolved, uncorrected events |
| event time from a publication time | ≥ 90% of point-in-time-valid events |
| ready families | ≥ 1, each meeting every clause alone |

The adequacy numbers are B4's, held once in the readiness policy object
(`collection/readiness.py`, `DEFAULT_POLICY`) and read by both `corpus-status`
and the Gate 1 audit. Families are never pooled.

What B5.1 changed is what gets *measured* against these numbers: coverage is
now measured instead of assumed zero; confidence is derived from evidence
instead of fixed; and an audit counts exactly one extractor version.

## 3. Extractor confidence contract

`rules-v1` is kept exactly as it was. Its events are in the corpus, and event
identity carries the extractor version so that old events keep their old
interpretation. Every `rules-v1` event carries relevance 0.55, confidence 0.35,
novelty 0.5 and sentiment 0.0, and it types a document by the first rule whose
term occurs as a raw substring of its title.

`rules-v2` (`EvidenceRuleExtractor`) is what the collector now runs. Its
confidence is a bounded, deterministic weighted sum of five factors, each read
from the document and the declared watchlist:

| Factor | Weight | Values |
|---|---:|---|
| type evidence | 0.40 | one type, two or more terms 1.0; one type, one term 0.8; several types, one singled out by context 0.6; several types, nothing singling one out 0.0; capped at 0.2 when the entity is declared never to produce the type |
| context agreement | 0.20 | the share of the available contexts — the entity's declared event types, the feed's disclosure stream — that include the type; 0.5 when neither says anything |
| entity certainty | 0.20 | one entity named in the title 1.0; found only in the publisher 0.7; several named in the title 0.4; none 0.0 |
| source reliability | 0.10 | official and primary 1.0; one of them 0.75; known publisher 0.5; otherwise 0.25 |
| timestamp certainty | 0.10 | the document's timestamp quality when a publication time exists and does not follow retrieval; otherwise 0.0 |

A document whose title matches no type term yields no event at all; an entity's
name is not type evidence. Terms match whole words and phrases, plurals
allowed, so "sec" no longer matches inside "second", and "interest rate" in a
CFTC title is weighed against "rule" and against the CFTC's declared
`REGULATION` instead of winning by position. `explain()` returns every factor,
so a confidence can always be traced to the evidence that produced it.

The contract, each clause pinned by `tests/test_intelligence_b51_extractor.py`:

- **evidence-derived, not constant** — confidence varies with type evidence,
  entity, source and timestamp, and a well-supported event scores above one that
  is contested, ambiguous or malformed;
- **bounded and deterministic** — in [0, 1], identical for identical input and
  version, independent of the other documents in a batch;
- **no future information** — its inputs are the document and the watchlist. An
  import scan fails if the extractor module imports anything beyond the standard
  library, pydantic and the data models; it has never seen a price, a return, a
  label, a later correction or a later document;
- **not tuned to pass** — the weights and values were fixed in the commit that
  introduced them, before the Gate 1 re-run, and the re-run counts `rules-v1`
  anyway (section 10).

Confidence expresses confidence in the *extracted event*, never that the event
will move BTC. Relevance, novelty and sentiment stay separate routing values
(0.55, 0.5, 0.0); nothing combines them into a score; direction is `UNKNOWN`;
there is no BUY, SELL, LONG or SHORT anywhere in the output.

Versions are never pooled. A document re-extracted under `rules-v2` gains a
second event beside its `rules-v1` one, so the Gate 1 audit counts exactly one
version — the only one present, or the one named with `--extractor-version` —
and refuses a mixed store until one is named.

Read-only, nothing written: `rules-v2` applied to the six collected documents
would produce four events — the SEC transfer-agent proposal (0.91), the SEC
charges (0.91), the CFTC clearing rule as `REGULATION` (0.83) and the CFTC
no-action letter (0.91) — and none for the SEC roundtable agenda or the BLS
indicator page. Those events are not in the store and are not counted anywhere.
They show the extractor can now produce countable confidence; they do not make
the corpus larger than four events over two days.

## 4. Readiness coverage semantics

One definition, `collection/coverage.py`, contract
`coverage-v2-provider-success-days`, read by `corpus-status`, `ops-status` and
the Gate 1 audit.

**A covered day is a UTC calendar day with at least one successful provider
attempt**, read from `provider_attempts`, which records every attempt of every
cycle.

| Case | Covered? |
|---|---|
| a `DEGRADED` run whose queries succeeded (a quality check failed) | yes — it collected |
| a run whose every attempt failed | no |
| a provider that did not answer, beside one that did | yes, through the one that did |
| a retired feed returning 404 | only if another feed was read |
| a syndication search in which no feed could be read | no — since B5.1 this is a failed attempt, not an empty success |
| a cycle that could not take the run lock | nothing recorded, so no |

Days are UTC whatever zone the database session is in; an attempt at 23:30Z is
on its own day, not the next one in Manila.

The record keeps four things apart: **elapsed** days (first attempt to the audit
instant, inclusive), **expected** days (every elapsed day — the collector runs
every three hours, so a day without a success is a missed day), **successful**
days, and the **fraction**. Beside them, at the three-hour cadence: expected,
attempted and successful cycles, failed attempts, first and last success, and
days since the last success.

A store with no provider attempt is **`NO_COLLECTION`**, with coverage `None`,
and a family whose coverage could not be measured fails the clause with
"collection coverage not measured" — never "0% of days". Coverage is never
fabricated upward either: a family's coverage is its covered days *within its
own span* (first to last event availability, inclusive) over the days in that
span, and a day outside the span never counts.

| | Before B5.1 | After |
|---|---|---|
| `corpus-status` coverage | never computed; every family 0% | per family, over its own span |
| days outside a span | counted, capped at 100% | never counted |
| a one-day span with collection | 0% | 100% |
| an empty store | 0% | `NO_COLLECTION`, not measurable |
| the collection service's day list | each watermark's latest retrieval only | the canonical successful days |
| the Gate 1 audit | days with a non-`FAILED` run | the canonical successful days |
| all-feeds-down syndication search | a successful attempt | a failed attempt |

Pinned by `tests/test_intelligence_b51_coverage.py` and
`tests/test_intelligence_b51_feed_failure.py`, including zero-run startup,
partial and full coverage, degraded and failed runs, a missing provider
response, a retired feed, a UTC boundary read under a Manila session, and the
deployed scheduled path.

## 5. Collection operations

The operational machinery already existed (B4.1-Ops) and is reused, not
duplicated:

| Need | Where | Pinned by |
|---|---|---|
| repeated scheduled runs | `collect-scheduled`, `btc-intel-collect.timer` (every 3 h, `Persistent=true`) | `test_intelligence_deployment.py`, `test_intelligence_b41ops_runtime.py` |
| run lock | `ops/runlock.py`; stale lock broken and reported | `test_intelligence_b41ops_runtime.py` |
| exit statuses | 0 ran, 2 failed, 3 lock held, 4 nothing due; `SuccessExitStatus=3 4` | `test_intelligence_deployment.py` |
| watermark persistence | watermarks advance only after atomic storage, only on success | `test_intelligence_operations.py`, `test_intelligence_b41ops_lifecycle.py` |
| retries and backoff | classified failures; only transient and rate-limit retried | `test_intelligence_b41_corpus.py`, `test_intelligence_operations.py`, `test_intelligence_b51_feed_failure.py` |
| provider failure isolation | one unreadable feed costs only itself | `test_intelligence_silent_empty.py`, `test_intelligence_b51_feed_failure.py` |
| health and status | `ops-status`, `corpus-status` | `test_intelligence_b41ops_lifecycle.py`, `test_intelligence_b51_ops.py` |
| stale-run detection | `ops-watch`: `COLLECTION_STALE`, `PROVIDER_STALE`, `ALL_PROVIDERS_FAILED`, `LOCK_STALE_BROKEN` | `test_intelligence_b41ops_lifecycle.py` |
| collection summary | daily summaries; no return statistic may reach them | `test_intelligence_b41ops_lifecycle.py` |
| backup and restore | `corpus-backup`, `corpus-restore` into a new location only | `test_intelligence_b41ops_lifecycle.py` |
| corpus verification | `corpus-verify`, fails closed; daily timer | `test_intelligence_b41ops_lifecycle.py`, `test_intelligence_cli_commands.py` |
| corrections | `corpus-correct`, append-only | `test_intelligence_corrections.py`, `test_intelligence_b51_correction.py` |

B5.1 added two things. A syndication search that could read no feed is now a
failed attempt, so an outage fails the run, trips the watchdog and does not
count as a covered day. And `ops-status` — the report an operator is told to
read — now carries the canonical collection record and the lag; before, its
only coverage figure counted gaps between the first and last document, so a
collector that had stopped showed no gap at all.

### Operational readiness checklist

In the repository, done and tested:

- [x] one scheduled cycle, idempotent, with exit codes a scheduler can act on
- [x] one collector at a time; a dead one never wedges the next
- [x] an outage is a failed run, a quiet day is not
- [x] coverage, staleness and lag stated where the operator looks
- [x] integrity verification, verifiable backups, restore into a new location
- [x] append-only corrections, committed as literal ids
- [x] an installer and systemd units for all three timers

Outside the repository, required, and **not in place**:

- [ ] an always-on Linux host with systemd (a laptop is not a host)
- [ ] `deploy/install.sh` run on it, with `BTC_INTEL_CONTACT` set
- [ ] the collect, verify and backup timers enabled and firing
- [ ] backups copied off the host, and a restore rehearsed
- [ ] someone reading `ops-watch` exit codes and alerts
- [ ] the committed corrections applied to that host's store
- [ ] 180 days of elapsed collection at ≥ 80% daily coverage per family

The software being able to schedule itself is not a corpus that has been
collected. Nothing is collecting now; the last successful cycle was on
2026-09-03.

## 6. Point-in-time rules

Unchanged, and now pinned against the new lag metric:

- `available_at` is the moment of **retrieval**, never publication; first write
  wins; rediscovery is a sighting, never a new availability;
- an event's `event_time` is its source's publication time where there is one
  and its first retrieval where there is not; its `available_time` is its
  source's first retrieval;
- every point-in-time read admits exactly what was available **at or before**
  the origin — `available_time <= origin`, inclusive at the origin, nothing a
  microsecond earlier.

**Collection lag** (`collection/lag.py`,
`collection-lag-v1-first-seen-minus-published`):

    collection_lag = first_seen_at - published_at

`first_seen_at` is the store's first sighting, falling back to first retrieval;
which basis was used is recorded. A document with no publication time has no
lag and is counted as such. A publication after first sight is an impossible
timestamp, reported and never clipped. The summary — median, nearest-rank p90,
minimum, maximum — is in `corpus-status`, `ops-status` and the Gate 1 audit.

It is descriptive. It is computed from the fields above and written nowhere any
of them is read; it never moves `event_time` or `available_time`.
`tests/test_intelligence_b51_pit.py` pins that computing it writes nothing, that
a late-seen document is never back-dated to its publication, that a re-crawl
reporting an earlier clock cannot move availability earlier, that later
sightings cannot qualify an event at an earlier origin, that a correction never
rewrites availability, and that a provider's delay stays visible.

## 7. Correction semantics

The ledger is append-only. A correction is a new row saying what is now known
about an observation; the observation stays exactly as it was written. The raw
view keeps everything, for audit and integrity; the eligible view is what
research counts. Ids are written literally, so a correction cannot widen. See
[`corrections/README.md`](../corrections/README.md).

B5.1 adds one correction,
[`2026-09-11-rules-v1-cftc-misclassification.json`](../corrections/2026-09-11-rules-v1-cftc-misclassification.json):
it invalidates the single `rules-v1` event (`6579615b…`) that typed *CFTC Issues
Final Rule to Modify Clearing Requirement for Canadian Dollar- and Mexican
Peso-Denominated Interest Rate Swaps* as `MONETARY_POLICY`.

The system distinguishes the two things the misclassification involves:

- **the original extraction** — the `rules-v1` event, still in the store exactly
  as written, still `MONETARY_POLICY`, excluded from research once the
  correction is applied;
- **the corrected classification** — a *separate event*, `rules-v2`'s
  `REGULATION` event for the same document (`2262b65a…`), with its own id
  because identity carries the extractor version. It exists only where an
  operator re-extracts under `rules-v2`, and no audit pools the two.

The correction is committed, not applied: the collected store was only read. On
a copy of it, a dry run recorded nothing; applying recorded one row; eligible
`rules-v1` events went from four to three while the raw view kept all seven;
the original store's sha256 was unchanged. The first correction file is
unchanged.

## 8. Known deployment requirements

From [`deploy/DEPLOYMENT.md`](../deploy/DEPLOYMENT.md): a Linux host that stays
on (1 vCPU, 1 GB RAM, 10 GB disk is ample), systemd, Python 3.11+, one-time
root to install the units, and no credentials — every feed is public — but a
contact address in `BTC_INTEL_CONTACT`, without which the collector refuses to
start rather than advertise a placeholder.

What the corpus then needs is time, and nothing in the repository can supply
it. Gate 1 requires a family spanning 180 days at 80% coverage; collection that
started on 2026-09-11 could not satisfy the span before 2027-03-10, and the
event count depends on how often the watched agencies publish, not on anything
the collector controls. Cadence stays at three hours: it is never changed to
increase document count.

An operator deciding to re-extract the existing store under `rules-v2` writes
new events beside the old ones. That is permitted and is never pooled with
`rules-v1`, but it is an operator's decision about their store, not something
this repository did.

## 9. Current corpus state

From the Gate 1 re-run as of 2026-09-11T00:00:00Z — the store B5 audited, same
sha256 (`fe9ae0a0…`) and same content fingerprint (`c298391b…`) — and from
`corpus-status` run on a copy of it:

| | |
|---|---|
| documents | 6 — SEC 3, CFTC 2, BLS 1 — from one provider, `syndication` |
| events | 7, all `rules-v1`; 3 invalidated as rediscovery duplicates; 4 eligible |
| once the CFTC correction is applied | 3 eligible |
| events above the quality floor | 0 — every `rules-v1` event has confidence 0.35 |
| document span | 2026-09-02 08:39Z to 2026-09-03 09:15Z |
| collection | 5 cycles, all `DEGRADED`; 65 provider attempts, all successful |
| coverage | 2 of 10 elapsed days (20%); 5 of 70 expected cycles; last success 2026-09-03, 8 days before the audit |
| collection lag | median 16.5 h, p90 19.2 h, range 13.4 h to 19.2 h, over 6 of 6 documents |
| families ready | 0 of 15 |

Funnel: 7 collected, 4 not invalidated, 4 with resolvable sources, 4
point-in-time valid, 0 above the quality floor, and so 0 aligned, 0 independent
and 0 effective.

What reads differently in the same corpus. B5's audit reported every family as
"collection covered 0% of days"; the re-run reports the fifteen Gate 1 families —
all empty once the quality floor is applied — as "collection coverage not
measured", and carries the collection record and the lag beside them. In
`corpus-status` under `rules-v1`, the three families that do hold an event
(`REGULATION`, `MONETARY_POLICY` — the uncorrected CFTC event — and the SEC
entity) each span a single covered day and read 100% coverage, where they read
0% before. Coverage was never what held them back: each is one event from one
publisher over a zero-day span, and each is `NOT_READY`.

## 10. Re-running Gate 1

```sh
python -m market_intelligence.b5 audit     --db <store.duckdb>     --as-of 2026-09-11T00:00:00+00:00     --input-label collector-state:default-root     --extractor-version rules-v1     --output research/market_intelligence/b51/gate1-rerun
python -m market_intelligence.b5 verify research/market_intelligence/b51/gate1-rerun
```

The store is opened read-only and refused if its schema differs. The
preregistration is read from `research/market_intelligence/b5/preregistration.json`
and checked against its own hash. `--extractor-version` is required once a store
holds more than one version; this store holds only `rules-v1`.

The result is committed in
[`research/market_intelligence/b51/gate1-rerun/`](../research/market_intelligence/b51/gate1-rerun/):

- decision **`INTELLIGENCE_CORPUS_INSUFFICIENT`**; Gate 1 not passed;
- preregistration `70777f0e…`, B5's own;
- result digest `44842364…`;
- a second audit of the same store at the same instant was byte-identical, and
  the store's sha256 was the same before and after.

CI recomputes every hash, checks that the preregistration is B5's, and asserts
the decision, the one version counted and the coverage and lag contracts. B5's
result in `research/market_intelligence/b5/gate1/` is unchanged.

On a grown corpus, re-run with a new `--as-of` and `--output` and nothing else.
The thresholds are not a parameter.

## 11. The corpus is not sufficient

**The collected corpus does not satisfy Gate 1**, and B5.1 does not claim
otherwise. No family meets the unchanged preregistered thresholds; the corpus
spans two days, not 180; six documents, not thirty events a family; and nothing
is collecting.

B5.1 repaired the instruments. It did not produce evidence. The next step is
long-running collection on a persistent host — an operations task, not a
research sprint — and no Track B5.2 is opened by this one. Until a re-run of the
unchanged Gate 1 passes on a real corpus, there is no event study, no event
reaction, no category test, no predictive model and no quant–intelligence
fusion, and "ready" is not a word this project uses about the corpus.
