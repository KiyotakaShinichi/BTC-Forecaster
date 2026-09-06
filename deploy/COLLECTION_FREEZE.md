# Collection semantics are frozen

**Frozen at deployment candidate `8203dc382ee575108ddf13d53aa2a9d694f8bd05`** -- the
first commit on this branch to pass remote CI
([run 34039803369](https://github.com/KiyotakaShinichi/BTC-Forecaster/actions/runs/34039803369)).
Recorded in [`DEPLOYMENT_CANDIDATE.json`](DEPLOYMENT_CANDIDATE.json).

This document exists because of a specific risk. The corpus this collector
builds is only useful if the rules that built it held constant while it was
being built. A study over six months of collection is a study of one measuring
instrument; change the instrument in month three and the result is two partial
studies that cannot be pooled — and nothing in the data will say so. The counts
still add up. The gap is invisible.

So the contracts below are settled. They are not sacred, and a genuine defect is
always worth fixing. What is forbidden is the casual edit: the loosened filter
that improves yield, the extra keyword, the "harmless" tidy-up.

---

## The frozen contracts

### 1. Point-in-time timing

`available_at` is the moment of **retrieval**, never publication. A three-week-old
press release first seen today became usable today; treating its publication date
as availability would fabricate three weeks of hindsight.

The planner's lookback window is applied to **publication**, as a lower bound
only. Comparing it against availability can never pass, because retrieval
necessarily happens after the planning instant.

**No backdating, ever.** Historical intelligence cannot be manufactured by
collecting it late.

### 2. Rediscovery

A document rediscovered in a later cycle **keeps the availability it was first
seen with**. First write wins. Rediscovery is recorded as a sighting, not as a
new document and not as a new availability.

Everything derived from a document — event identity, event availability, cluster
membership — reads the store's availability, not the freshly retrieved copy's.

### 3. Raw evidence is immutable

Nothing deletes or rewrites a collected record. Not a defective one, not a
duplicate, not one whose source has since been retired.

### 4. Corrections are append-only

An observation that turns out to be an artefact of a defect is **invalidated,
never deleted**. A correction is a new row stating what is now known; the
observation stays exactly where it was written.

Two views follow, and they are the load-bearing distinction:

| View | Contains | Read by |
|---|---|---|
| **Raw** | everything ever collected | integrity, audit, provenance |
| **Eligible** | everything no correction excludes | readiness, event studies, features, replay |

*Preserved physically, excluded scientifically.*

Corrections name event ids **literally**. A correction expressed as a rule keeps
matching new events forever; that is a filter, not a correction.

### 5. Snapshots

Snapshots are immutable and are never reinterpreted. Each records the contract
it was built under:

- **`raw-v0`** — membership is every event in the store. Every snapshot frozen
  before the correction ledger existed carries this, and is integrity-checked
  against the raw view, so a corrected corpus is never reported as a damaged one.
- **`event-eligibility-v1`** — membership excludes invalidated events. A
  corrected snapshot is a *distinct artefact* from the raw one at the same
  instant, not a replacement.

`AS_BELIEVED` reconstructs what a past run counted, applying only corrections
recorded by that instant. `CORRECTED` is the research default and applies all of
them, which introduces no lookahead: a correction says software wrote a row it
should not have, which tells an analysis nothing about prices.

### 6. Disclosure streams

`disclosure-stream-v1`. A document records which official stream it came from,
because `publisher` says who and `source_type` says how trustworthy, and neither
says what kind of disclosure it is.

The three SEC streams stay distinct: `REGULATORY_ANNOUNCEMENT`,
`ADMINISTRATIVE_PROCEEDINGS`, `CIVIL_LITIGATION`. **They must not be collapsed**,
however similar the publisher, title or text — an administrative proceeding and
the press release announcing it are different instruments.

Documents collected before this field existed read back `UNCLASSIFIED` and are
not relabelled. That classification is one nobody made.

Adding a stream member is a contract change, not a tidy-up.

### 6b. Read order

Point-in-time reads -- `documents_as_of` and `signals_as_of` -- return a **total
order**: availability first, id as the tiebreaker. Not merely repeatable, but
the same order on every machine.

This is load-bearing rather than tidy. `sum()` is not associative, so records
sharing an instant returning in arbitrary order moved every float derived from
them. A feature matrix built at one chunk size disagreed with the same matrix at
another in the last ULP, which made `dataset_id` -- whose entire job is to say
two datasets are identical -- depend on an implementation detail, and let a
corpus restored from a backup produce a different id than the corpus it came
from.

### 7. Deduplication

An identical canonical URL **is** identity and still merges. Identical text or
identical titles are only *evidence* of identity, and that evidence is overridden
when the two documents came from different disclosure streams.

### 8. Candidate matching

`matching-v1-word-start`. Terms are matched at a **word start**, suffixes
allowed: `regulation` matches "regulations", `bitcoin` matches "bitcoin's", and
`us` does not match inside "Rebus" or "Announces".

Match fields are **declared per stream**. `subject_bearing=False` records a
stream whose fields cannot carry subject matter — the administrative-proceedings
feed, whose title and description are both the respondent's name. **Zero topical
matches from such a stream is a property of the source, not a fault.**

The watchlist is frozen. New lexical aliases require source evidence, explicit
review, and independence from market outcomes. **Never added for yield.**

### 9. Admission

A document passing the topical filter is a **candidate**. It still has to survive
normalisation, provenance, extraction, schema validation, relevance scoring,
dedup and clustering. A matched document is not a research event.

### 10. Readiness

B4 adequacy thresholds are untouched: 30 events, 20 effective, 3 publishers, 1
provider, 180-day span, 80% coverage. Publisher diversity counts the **union of
publishers across a family**, not the best-covered event.

Readiness cannot be opened by repetition, and cannot be held shut by broken
diversity accounting. Both failure directions have regression tests.

### 11. XML repair

Strict parse **first**. Only when that fails, escape bare ampersands that cannot
begin a valid reference, and retry. Valid XML is never rewritten. Anything the
narrow repair cannot fix raises, and the failure is recorded as `SCHEMA`.

Every repair is reported, so a publisher whose feed is degrading becomes visible
rather than compensated for indefinitely. **No general HTML/XML cleanup.**

### 12. Cadence

Three hours. `available_at` is retrieval time, so cadence is part of the research
timing resolution and is decided once: at 12 hours the mean retrieval lag is 6
hours, a quarter of a 24-hour horizon; at 3 hours it is 1.5 hours, 6.2%. A corpus
gathered coarsely cannot be re-gathered finely.

Never changed to increase document count.

---

## Changing any of this

A change to a frozen contract requires all four:

1. **A new contract version.** Not an edit to the existing one — the corpus has
   to be able to say which rule produced which part of it.
2. **Behavioural tests** proving the new rule and the old one differ where they
   are supposed to and nowhere else.
3. **A migration analysis** stating what happens to data already collected. The
   default answer is "nothing" — old data keeps its old interpretation.
4. **A research-impact statement**: which studies are affected, and whether data
   from before and after can still be pooled.

If you cannot write the fourth one, the change is not ready.

## What is not frozen

Fixing a genuine defect. Every contract above was itself the result of finding
one, and fourteen silent-empty failures are enumerated in
`tests/test_intelligence_silent_empty.py`. If something here is wrong, correct it
under the process above — but read that file first, because most plausible
"improvements" to this collector have already been tried and were bugs.
