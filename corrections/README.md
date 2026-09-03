# Corrections

A correction records that an observation in the corpus is an artefact of the
software that collected it rather than a fact about the world.

**Nothing here deletes or edits anything.** A correction is a new row in an
append-only ledger; the observation it describes stays exactly where it was
written, with the availability it was written with. Two views follow:

| View | Contains | Read by |
|---|---|---|
| **Raw** — `signals_as_of` | everything ever collected | integrity checks, audit, provenance |
| **Eligible** — `eligible_signals_as_of` | everything a correction has not excluded | readiness, event studies, features, replay |

*Preserved physically, excluded scientifically.*

## Why not just delete them

Three reasons, in increasing order of how long they take to hurt:

1. **A deleted row leaves no trace.** Nobody later can distinguish a corpus that
   never held an observation from one that quietly dropped it. Both look like an
   absence of news, which is exactly the thing this corpus exists to measure.
2. **The things that counted them still exist.** Manifests, clusters and
   snapshots recorded those events. Deleting the rows leaves those artefacts
   referencing nothing, so integrity checks that were passing start failing —
   and the likely response is to relax the check until it passes again.
3. **Deleting is itself an unrecorded edit.** A corpus whose contents can change
   without leaving evidence is not evidence. Once one silent deletion is
   acceptable, no count in the corpus can be defended.

## Applying one

```sh
btc-intel --db <corpus> corpus-correct --file corrections/<file>.json --dry-run
btc-intel --db <corpus> corpus-correct --file corrections/<file>.json
btc-intel --db <corpus> corpus-corrections          # what is excluded, and why
```

Applying the same file twice records nothing the second time: correction ids are
content-addressed, so re-running a correction script cannot produce a second,
subtly different row for the same decision. A correction naming an event the
corpus does not hold is refused rather than recorded — a ledger row pointing at
nothing is indistinguishable from a typo and would sit there looking
authoritative.

## Writing one

Affected ids are **written out literally**, never expressed as a rule. A
correction defined as "events that look like duplicates" keeps matching new
events forever; that is a filter, not a correction, and it would silently widen
its own scope every time it ran.

```json
{
  "reason": "REDISCOVERY_DUPLICATE_PRE_FIX",
  "invalidated_at": "2026-09-03T00:00:00Z",
  "invalidated_by_version": "<the commit that made this knowable>",
  "source_bug": "<what produced the bad observation>",
  "notes": "<why this set is complete>",
  "corrections": [{ "event_id": "..." }]
}
```

`invalidated_at` is fixed in the file rather than taken from the clock, so the
correction ids are identical on every machine that applies it.

## The ledger is append-only

A correction is never modified. If an invalidation turns out to be wrong, append
a `REINSTATED` correction; both rows survive and the later one stands. "Later"
is a total order — `(invalidated_at, correction_id)` — because two corrections
recorded in the same instant would otherwise leave eligibility depending on row
order, which is the inconsistent state an append-only design exists to prevent.

## Snapshots and replay

Snapshots are immutable and are never reinterpreted. Each records the
eligibility contract it was built under:

- **`raw-v0`** — membership is every event in the store. Every snapshot frozen
  before the ledger existed carries this, and integrity checks it against the
  raw view, so a corrected corpus is not reported as a damaged one.
- **`event-eligibility-v1`** — membership excludes invalidated events. New
  snapshots are built under this contract, and a corrected snapshot is a
  *distinct artefact* from the raw one at the same instant — different
  `corpus_id`, different `membership_hash` — not a silent replacement of it.

Two modes exist for reading:

- **`CORRECTED`** (default for research) applies every correction on record,
  regardless of when it was made. This introduces no lookahead, and the reason
  is worth stating rather than assuming: a `REDISCOVERY_DUPLICATE_PRE_FIX`
  correction carries no information about the market. It says a piece of
  software wrote a row it should not have written. Removing that row tells an
  analysis nothing about prices it could not have known — it removes something
  that was never an observation of the world.
- **`AS_BELIEVED`** applies only corrections recorded at or before the instant
  being reconstructed. This answers "what did the run that day actually count?",
  which is an audit question, and it must not be answered with hindsight.

---

## Recorded corrections

### `2026-09-03-rediscovery-duplicates.json` — 3 events

**Defect.** Event identity was derived from the freshly retrieved document's
`available_at` rather than the availability the store already held for it. The
document layer had first-write-wins from B4.1; the event layer never read it. So
re-reading an unchanged feed produced a brand-new event, with a later
availability, on every cycle.

**Fixed in** `ec4d1c85980174f005deeed91b07257d600c5a95`, by
`IntelligenceStore.canonical_availability`, which settles a rediscovered
document against the store before anything is derived from it.

**Affected events.**

| Event | Source document | Written |
|---|---|---|
| `230b1e47079af960…` | `152465365612…` | 2026-09-02 09:01Z |
| `5ca33cfeebea156f…` | `5033b8afabbc…` | 2026-09-02 09:01Z |
| `cdc4e1728b1cc537…` | `459cfdb0dc9e…` | 2026-09-02 09:01Z |

**Why the set is complete.** One duplicate per source document, all three
written by the 09:01Z cycle. The 08:39Z cycle produced the three genuine
observations; the 09:18Z cycle, the first to run with the fix, produced none.
The set is closed, and it was identified by grouping on
`(source_ids, event_type, extractor_version)` and keeping the earliest
availability in each group — then frozen as literal ids so it cannot widen.

**Effect on research.** The corpus holds 6 events and offers 3. B4 readiness,
event studies, feature aggregation and replay count 3. `corpus-verify` still
sees 6 and still passes, because an invalidated event must go on satisfying
every invariant it always satisfied.

**Why deletion was rejected.** See above — and specifically here, the three rows
are the only remaining evidence of what the defect did. They are data about the
collector as well as artefacts of it.
