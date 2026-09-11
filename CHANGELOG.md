# Changelog

**There are no releases.** Nothing here has been tagged as one, published to an
index, or promised to anybody as stable, and this file does not pretend
otherwise. `pyproject.toml` declares `version = "0.2.0"`; that number
distinguishes the package layout from the scripts that preceded it and is not a
release.

One tag exists — `forward-collection-deployment-candidate-2026-09-06` — and it
marks a *deployment candidate for forward collection*, not a release and not a
trading system. What it certifies is written in
[`deploy/DEPLOYMENT_CANDIDATE.json`](deploy/DEPLOYMENT_CANDIDATE.json).

So what follows is milestones, in the order they happened, taken from the commit
history rather than written from memory. Dates are the dates of the work.

The load-bearing entries are the negative ones. A changelog that records only
what was added describes a project that never learned anything.

---

## 2026-09-11 — Production collection packaged; deployment blocked externally (Track B5.2)

- **`DEPLOYMENT_BLOCKED_EXTERNALLY`.** No always-on host, SSH access or operator
  contact address was available, so nothing was deployed and no production
  collection is running. The package was dry-run instead, offline and against a
  copy of the collected store. The 180-day accumulation Gate 1 needs has not
  started; Gate 1 is unchanged and still `INTELLIGENCE_CORPUS_INSUFFICIENT`.
- A cycle that ran and read nothing from any provider exited 0; it now exits 2,
  as `deploy/DEPLOYMENT.md` always said, and the health check calls a failed
  last run `ALL_PROVIDERS_FAILED` at once.
- The health check gained low disk, an unloadable configuration and a failed
  last backup, and runs hourly under systemd; every unit that can fail raises
  an alert, to the journal and to an operator-configured command.
- Backups carry a content fingerprint and a `.sha256`, are proved by a restore
  rehearsal before anything is pruned, keep the newest 30, and never prune an
  archive named by hand.
- New operator commands: `ops-config-check`, `ops-probe`, `ops-backup`,
  `corpus-backup-verify`, `ops-alert`, `ops-smoke`, and `python -m
  market_intelligence.b5 status` -- read-only, deterministic, and restating no
  Gate 1 threshold.
- Measured, and corrected in the docs: each cycle makes 78 requests (every feed
  once per query), 624 a day, not the 56 earlier documentation claimed. Still
  far inside every publisher's limits; fetching once per cycle is recommended,
  not done.
- Nothing trades, live trading remains disabled, and no research was run.

## 2026-09-11 — Collection readiness repaired; the corpus is still insufficient (Track B5.1)

- **Gate 1 re-run: `INTELLIGENCE_CORPUS_INSUFFICIENT`, again.** Same corpus, same
  preregistration (`70777f0e…`), same thresholds. Nothing was enriched, no event
  study was run, and B5's result is not reinterpreted.
- Extraction confidence is evidence-derived. `rules-v2` weighs type evidence,
  context agreement, entity certainty, source reliability and timestamp
  certainty — deterministic, bounded, and blind to market data by construction.
  `rules-v1` and its events are unchanged; versions are never pooled, and the
  Gate 1 audit counts exactly one.
- Collection coverage is computed once, from successful provider attempts per
  UTC day. It had read 0% for every family in `corpus-status`, could count days
  outside a span, and read an empty store as 0% rather than unmeasured. A
  syndication search that could read no feed was recorded as a successful
  attempt; it is now a failed one.
- Collection lag — first seen minus published — is reported by `corpus-status`,
  `ops-status` and the audit, and moves no timestamp.
- The `rules-v1` event that typed a CFTC swaps-clearing rule as monetary policy
  has an append-only correction. The original stays as written; the corrected
  classification is `rules-v2`'s separate event. Committed, not applied.
- The collector is operationally capable and not deployed: two of ten elapsed
  days collected, nothing since 2026-09-03. The next step is a persistent host.
  No Track B5.2.

## 2026-09-11 — Point-in-time event study stops at Gate 1 (Track B5)

- **`INTELLIGENCE_CORPUS_INSUFFICIENT`.** B5 asked whether independently
  timestamped external information is associated with BTC behaviour beyond price.
  Its first, preregistered gate — is there a point-in-time corpus to ask it of —
  failed, so no event study was run, no category tested and no signal mined.
- The corpus that exists: six documents and seven events from two days of
  collection in September, three of the events invalidated as pre-fix
  rediscovery duplicates. None of the four left clears B4's extraction-confidence
  floor. The historical corpus is empty, as B4 found, and cannot be filled after
  the fact.
- The thresholds are B4's own — 30 events, 20 effective, 3 publishers, 180 days,
  80% coverage — held as the readiness gate's policy object and preregistered
  (hash `70777f0e…`) before the audit ran, plus clauses that only tighten. At the
  threshold a study could detect only effects 2.5 times B4's practical floors;
  the bar is a floor on power, not a comfortable level.
- Found in forensics: the collector's rule-based extractor gives every event
  confidence 0.35 by construction, so it can never produce a countable event; and
  `corpus-status` never supplies collection coverage, so the readiness report can
  never open. Both are recorded; neither is changed here.
- `python -m market_intelligence.b5 audit` measures a store read-only and writes a
  canonical, verifiable result, and CI verifies the committed one. Nothing trades,
  live trading remains disabled, and quantitative research stays frozen.

## 2026-09-11 — Quantitative research frozen (Track A7.1)

- **`QUANT_RESEARCH_FROZEN`.** A2, A6 and A7 each found, under its own design,
  that no model beats the naive forecast. The project does not currently possess
  validated evidence of a tradable predictive edge. No model is promoted until it
  clears `NO_MODEL_PROMOTION_UNTIL` — eight criteria, from out-of-sample skill
  after multiple-testing correction to an independent, preregistered
  justification. No A8.
- A6 and A7 reach `main` by fast-forward, their history unchanged.
- `docs/quant-research-status.md` records the lineage as three results, not a
  leaderboard: three designs, three targets, no combined score.
  `docs/quant-research-handoff.md` says what was learned, where the evidence
  stops, and what would justify reopening — and that the next direction, if any,
  is a market-intelligence question with enough point-in-time evidence, not
  another model family.
- The freeze is enforced. `tests/test_quant_freeze.py` fails if live trading
  becomes eligible, a research module can reach the paper engine, A7 gains a
  promoting decision, or a committed A6 or A7 result changes. A6's results table
  is now recomputed from its committed CSV, at full float precision: pandas'
  default parser is not exact, and re-serialising what it reads changes the
  bytes.
- A fresh clone can verify the committed A7 evidence with `walk_forward verify
  --committed`, and the fresh-clone CI job does. Reproducing the whole run still
  needs the pinned snapshot, which is not redistributed.
- Found during integration: A6's results digest depended on the platform's line
  ending. pandas ends CSV lines with `os.linesep`; the canonical run was produced
  on Windows, so a faithful rerun on Linux hashed to `c7d1aa18…` instead of the
  recorded `b266a50d…`. The digest is now taken over the recorded CRLF form on
  every platform. Nothing committed changed, and on Windows nothing computed
  changes either. A7 pins its line endings and never had the defect.
- Provenance audit: five committed files carry realised values derived from
  Yahoo Finance data — four already on `main` (two legacy runs and A2), and A6's
  predictions arriving with this integration. The repository's no-redistribution
  position is recorded as an unreviewed operating rule, not a legal
  determination. Nothing was deleted or rewritten; the question is left to the
  maintainer.

## 2026-09-10 — Walk-forward robustness (Track A7)

- **Nothing beats the random walk, from any origin, at any horizon, with any
  amount of history.** A6's question re-asked from 1,417 daily origins at 1, 3,
  7 and 30 bars, with rolling windows of 250 to 2,000 rows and an expanding one:
  0 of 200 configurations significantly better than naive after
  Benjamini-Hochberg, 103 significantly worse. The largest positive skill,
  +0.37%, is not significant and loses in the most recent period. Decision
  `ROBUSTLY_UNINTERESTING`; A8 not proposed.
- The design, the input hash, the schedule and every gate threshold were
  committed before the run (`docs/walk-forward.md`, sections 1-11). One gate was
  relaxed afterwards -- the resource budget, which made the digest depend on
  machine load -- and the deviation is recorded beside the preregistration with
  the check that it could not change the decision.
- The engine was validated on synthetic worlds first: it finds an AR signal and
  a trend, refuses noise, a signal that stops half way, and a leaky oracle with
  perfect skill. One preregistered expectation was not met as written: a
  feature-driven signal was detected but, positive in four of six folds, called
  fragile rather than a candidate. Recorded, not tuned away.
- Byte-reproducible output: canonical files carry no timestamps, timings or run
  ids; the result digest covers them and nothing else; one worker and two give
  identical bytes. Inputs are hash-pinned, and
  `python -m btc_forecaster.research.snapshot` refuses a changed one.
- Series models gained an iterated multi-step forecast -- the only change to A6
  code; A6's result digest reproduces unchanged. A2's Diebold-Mariano gained an
  optional HAC lag count; its default is unchanged.
- Found and fixed while building it: a resource status that moved with machine
  load inside the digest; the first fold charged for a library import, a defect
  A6 had fixed in its own runner; a leakage adversary to which the drift
  baseline is mathematically blind; a partition check that rejected every
  schedule because it dropped a timezone.

## 2026-09-08 — Repository reproducibility and maintainability

- `requirements.lock`: the full transitive closure, 83 distributions pinned with
  hashes, universal across Linux/macOS/Windows, resolved at Python 3.11.
  Generated by `scripts/lock.sh` from `pyproject.toml` under `constraints.txt`.
  Previously 25 packages were pinned and 58 arrived at whatever version was
  current that day.
- `.env.example`: every one of the 41 environment variables the code reads,
  categorised, checked against the source by a test that fails in both
  directions.
- `market_intelligence.storage` split into `schema` / `store` / `queries`. Public
  API unchanged, 48 methods still present, reads delegate rather than
  reimplement.
- `market_intelligence.cli` split into `commands/`, replacing a 230-line
  `if args.command == ...` chain with a registry the parser is checked against.
  The three shared reports moved to `reports.py`, so the HTTP API no longer
  imports the argument parser to answer a question about the corpus.
- Structured logs given a destination. They had been emitted as JSON since
  August and discarded, because nothing ever attached a handler.
- `market_intelligence` added to `packages.find`. It had never been installed by
  `pip install -e .`; everything worked only because every caller started in the
  repository root.
- `CONTRIBUTING.md`, this file, and Dependabot.

## 2026-09-07 — One repository, one contract (Track A4)

- The quantitative core and the market-intelligence platform merged into a
  single mainline. No code was reimplemented; the two lineages already shared an
  ancestor.
- One dependency contract covering both halves. `pytz` had been reaching the
  collector only transitively through `yfinance` — a quant dependency with no
  relation to collection — so a clean deployment could not read its own corpus.
- Each subsystem given a CI job scoped to the lint and coverage contract it was
  written against, instead of one job checking both under the wrong config.
- The coverage floor recalibrated to the environment that enforces it. It had
  been set from a machine with every optional extra installed.

## 2026-09-06 — Point-in-time reads given a total order

`documents_as_of` and `signals_as_of` ordered by timestamp alone, so records
sharing an instant came back in whatever order the engine chose. `sum()` is not
associative: a feature matrix built at one chunk size disagreed with the same
matrix at another in the last ULP, which made `dataset_id` — whose entire job is
to say two datasets are identical — depend on an implementation detail. Fixed by
ordering on `(timestamp, id)`.

The collection deployment candidate was frozen at the first commit to pass
remote CI.

## 2026-09-04 — Pre-committed shadow forecasting (Track A3)

A forecast is registered before the outcome exists, in an append-only
hash-chained ledger that refuses to record a second forecast for the same
origin. Retrospective evidence cannot be laundered into forward evidence.

## 2026-09-03 — Corpus integrity, and freezing what collects it

- **Corrections, not deletions.** Three events had been manufactured by a
  rediscovery defect. They were invalidated by an append-only ledger rather than
  removed: preserved physically, excluded scientifically. Raw and eligible
  became two distinct views, and snapshots record which contract they were built
  under.
- **Two dead feeds retired.** Both answered 404; one took 35 seconds doing it,
  which is why it had presented as a timeout.
- **SEC administrative proceedings added as its own stream**, not as a
  replacement for the retired civil-litigation feed. The three SEC streams stay
  distinct.
- **Candidate matching anchored at word starts.** `us` had been matching inside
  "Rebus" and "Announces" — 4.7% of matches were substring false positives.
- **Every silent-empty failure named.** Fifteen ways a collection run could
  report success having collected nothing, and a diagnosis that tells a quiet
  feed from a broken one.
- **Collection semantics frozen**, with the contracts written down in
  [`deploy/COLLECTION_FREEZE.md`](deploy/COLLECTION_FREEZE.md). A study over six
  months of collection is a study of one measuring instrument.
- **Fail-closed paper trading (Track C0).** It reads A2's evidence, finds no
  promoted model, and refuses to trade. `LIVE_TRADING_ELIGIBLE: False` is the
  correct answer, not a placeholder.
- `pytz` declared. Without it a fresh install could not read a `TIMESTAMPTZ`,
  which is every timestamp in the corpus.
- CI failures made legible without admin rights on the repository. Six
  consecutive red runs had gone undiagnosed because job logs need `actions:read`
  and step summaries are not exposed by the API.

## 2026-09-01 – 2026-09-02 — Forward collection and operations (Track B4.1)

- A lawful collection path: declared providers, documented terms, per-provider
  cadence floors, and no credential required for the public government feeds.
- **Availability is retrieval time, never publication.** A three-week-old press
  release first seen today became usable today.
- **Rediscovery stopped manufacturing an event every cycle.** Event identity had
  used the freshly-retrieved availability, so the same document produced a new
  event on every pass — and B4's 30-event adequacy gate would have opened on
  repetition alone.
- A readiness gate that can stay shut, a watchdog that can fire, run locking,
  crash recovery, integrity verification and verifiable backups.
- One command for a scheduler to call, with exit codes as its whole interface:
  0 ran, 2 failed, 3 another collector holds the lock, 4 nothing was due.
- Deduplication stopped being reported as quarantine.
- Operators answered with a sentence rather than a stack trace.

## 2026-08-31 – 2026-09-01 — Historical signal validation (Track B4)

The study ran, and **nothing survived**. Most candidate signals were not merely
unsupported but untestable from the history that exists — which is a result, and
is recorded as one. The event-study engine was built to be capable of returning
a negative, and proved it on fixtures before it was pointed at real data.

## 2026-08-29 – 2026-08-31 — Replay optimisation (Track B3.1)

Profiled first, then optimised: bulk point-in-time replay proven equivalent to
the reference implementation, historical feature materialisation with chunking
and resumption, a dataset catalog, and a per-origin extractor-version lookup a
benchmark caught scanning.

## 2026-08-29 — The honest benchmark (Track A2)

**Nothing beats the random walk.** Eight models over 36 identical walk-forward
folds; zero promotions. Random-walk-drift scored +0.0040, ARIMA −0.0021, the
retuned causal XGBoost −0.0801, the legacy hybrid −1.4057.

Getting there required fixing the measurement twice: direction was being scored
at literally the first step rather than the first *scored* step, and the
directional test was a two-sided binomial against a coin rather than a
dependence-aware one-sided test against the base rate.

## 2026-08-28 — The quantitative core, rebuilt (Track A)

A 700-line script that executed at import became a typed package with a UTC time
contract, point-in-time features, and the leakage tests that motivated them —
which verify causality by perturbing the future and asserting the past does not
move, and which also prove the guard itself fires.

The first leak-free run was promoted with its verdict intact: **the hybrid does
not work.** The original 68.97% directional accuracy was an artefact of
evaluating a model on data it had been fitted on.

Also: the point-in-time market-intelligence subsystem, replay datasets, the
model registry, evaluation metrics, one reusable walk-forward engine,
diagnostics with multiple-testing control, and the artifact manifest that makes
a run reproducible.

## 2026-03-13 – 2026-04-14 — The original scripts

Four commits: Bayesian forecasting scripts, a Monte Carlo simulation and a
requirements file. Preserved unmaintained under
[`research/legacy/`](research/legacy/README.md), because the Optuna output in
them is still the provenance of hyperparameters the platform uses.
