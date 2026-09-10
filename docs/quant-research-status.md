# Quantitative research status

**`QUANT_RESEARCH_FROZEN`** — since 2026-09-11, Track A7.1.

The project does not currently possess validated evidence of a tradable
predictive edge. Three studies — A2, A6 and A7 — each asked, under its own
design, whether any forecasting model beats the naive random-walk forecast of
BTC-USD. None found one.

This document is the canonical record of that lineage, of the policy that now
governs any model promotion, and of the invariant that keeps live trading off.
[`quant-research-handoff.md`](quant-research-handoff.md) is the narrative: what
was learned, where the evidence stops, and what would justify reopening.

**Frozen means** no model is promoted, no generic model-shopping sprint is
started, and no study's thresholds are revisited to rescue a result. It does not
mean abandoned: maintenance, reproduction and fixes to the evidence tooling
continue under the usual rules. Market-intelligence research is a separate layer
with its own freeze, [`deploy/COLLECTION_FREEZE.md`](../deploy/COLLECTION_FREEZE.md),
and this one does not touch it.

`tests/test_quant_freeze.py` makes the freeze load-bearing: it fails if live
trading becomes eligible, if a research module can reach the paper engine, if A7
gains a promoting decision, or if a committed A6 or A7 result changes.

---

## The lineage

| | A2 | A6 | A7 |
|---|---|---|---|
| **Question** | Does any challenger earn promotion over the naive forecast? | Does anything in a broad, resource-constrained zoo beat it? | Is A6's answer robust across origins, horizons, windows and periods? |
| **Design** | 36 expanding walk-forward folds, about 73 bars apart, 2019–2026; 30-bar embargo; tuning nested inside each fold | One fixed partition; every model fitted once on 1,000 rows; one 705-bar holdout | 1,417 daily origins, 2022-09-21 to 2026-08-07; twelve refits; early, middle and late blocks |
| **Target** | USD price level along each forecast path, scored at steps 31–60 from the origin | Next-bar log return | Cumulative log return over 1, 3, 7 and 30 bars |
| **History** | Expanding, at least 900 bars | 1,000 rows | Rolling 250, 500, 1,000 and 2,000 rows, and expanding |
| **Models** | 8: the naive forecast and 7 challengers | 43 attempted, 40 active, 3 unsuitable for the constrained lab | 11 curated from A6, at A6's configurations, untuned |
| **Tested** | Each challenger against a five-criterion promotion policy | 39 models against naive, Benjamini-Hochberg | 200 model × horizon × window configurations against naive, one Benjamini-Hochberg family |
| **Result** | Every challenger `REJECT` | 28 of 39 significantly different after correction, all 28 worse; none of 40 positive in every temporal block | 0 of 200 significantly better, 103 significantly worse; no candidate, no fragile signal |
| **Decision** | Nothing promoted | Nothing promoted; `EXPLORATORY` by construction | `ROBUSTLY_UNINTERESTING`; nothing promoted |
| **Input** | Snapshot `056b866b…`, 3,527 rows | Snapshot `39b93e34…`, 3,536 rows | Snapshot `39b93e34…`, 3,536 rows |
| **Method** | [`benchmark.md`](benchmark.md) | [`model-zoo.md`](model-zoo.md) | [`walk-forward.md`](walk-forward.md) |
| **Evidence** | [`research/runs/2026-08-29-a2-benchmark/`](../research/runs/2026-08-29-a2-benchmark/) | [`research/runs/a6-model-zoo/`](../research/runs/a6-model-zoo/) | [`research/runs/a7-walk-forward/`](../research/runs/a7-walk-forward/) |

The evidence is pinned by digest, and the suite reads each one back:

- **A6** results table: `b266a50dc411be2452c683fca7a0f1a6a1ae80fb2e8929e6b2b99a52e5328a8a`
- **A7** result digest: `b3fba7227e82e13ab2c112cb44169857aeb567cf7cb2a6f375bd7792f0d13790`,
  configuration digest `ba369d2f65153de73fa7b94fbd9d37fb7d7bb9477b0a90b6cfc501624c39ec5c`,
  reproduced byte for byte on two workers and on four
- **A6 and A7** input: frame sha256
  `39b93e34520a496692ccc2156b8fe5c88c88b31bd9e9903a6fad74f282004bd3`

A2's steps 31–60 are the 30 bars after its 30-bar embargo. Scoring starts at
step 31, which is why the A6 and A7 documents call A2's target "a 31-step price
path".

## These are three results, not one score

The studies share a question and not a scale. Their numbers are not comparable,
and this document does not combine them.

- **A2 scores USD price level along a 30-bar path** after an embargo, over 36
  expanding folds. Its errors are dominated by the price level. It used a
  different snapshot — yfinance revised history between A2's pull and A6's — and a
  different model set.
- **A6 scores next-bar log returns** — one step — on one fixed partition, after a
  single fit on 1,000 rows. [`model-zoo.md`](model-zoo.md) classifies it against
  A2 as `NOT_COMPARABLE`.
- **A7 widens A6's question** to four horizons, five amounts of history, 1,417
  origins and twelve refits. At one bar its target is A6's and its input bytes
  are A6's, but its unit of evaluation is not: [`walk-forward.md`](walk-forward.md)
  section 15 compares A6 and A7 model by model and classifies the agreement as
  `CONSISTENT`, not the numbers as interchangeable.

**What they share is the answer to one question:** under each design, no model
beats the naive forecast. That is agreement on a conclusion. So do not average
skills across studies, count a model's wins across them, rank a model by its
place in more than one, set an A2 price error beside an A7 skill as if they were
on one axis, or pool their comparisons into one multiple-testing family after the
fact.

## Promotion policy: `NO_MODEL_PROMOTION_UNTIL`

No model is promoted — into the paper-trading engine, into any decision path,
into live trading — until **every** one of these holds:

1. **Positive out-of-sample skill against the naive forecast,** on data that
   played no part in choosing the model, its features or its settings.
2. **Statistical significance after multiple-testing correction** across every
   comparison the study made (Benjamini-Hochberg or stricter), wherever it made
   more than one.
3. **Practical significance:** skill above a floor preregistered before the run —
   A7 used 1% MAE skill — not merely above zero.
4. **Temporal stability:** positive in the most recent period and in a
   preregistered majority of refit folds. A7's rules were a positive late block and
   at least 75% of folds positive.
5. **Robustness across the horizons and training windows relevant to the claimed
   use,** not one favourable configuration.
6. **Leakage tests passed:** poisoning future data leaves the forecast unchanged,
   and perturbing valid past data changes it.
7. **A reproducible, immutable data snapshot:** the input is hash-pinned, the
   result digest recomputes, and a reproduction has been verified.
8. **Independent research justification:** a preregistered study with its own
   proposal stating why this model or signal should work, written before its
   result existed. The output of a search does not qualify, however good the
   number.

**No model enters live trading merely because it wins one benchmark.** Meeting all
eight makes a model a candidate for a separate confirmation study and a separate
decision about paper trading. It enables nothing by itself, and live trading
additionally needs engineering that does not exist.

Where each criterion is already enforced: A2's promotion policy
([`benchmark.md`](benchmark.md)); A7's preregistered gates and their Benjamini-
Hochberg family ([`walk-forward.md`](walk-forward.md), section 11); the leakage
adversaries (`btc_forecaster/research/walk_forward/leakage.py`); input manifests
and `python -m btc_forecaster.research.snapshot`.

## Live trading: the hard gate

**Live trading is disabled.** It is not implemented, and the repository connects
to no broker. The invariant holds in layers, each pinned by a test:

- `btc_forecaster/paper/decision.py` sets `LIVE_TRADING_ELIGIBLE = False`.
- `btc_forecaster/paper/a2.py::current_live_permission` reads only A2's
  `promotion.csv` and returns no live candidates, zero leverage and
  `live_trading_eligible: False` whatever that file says. A2 promotes nothing.
- A6 and A7 manifests record `promoted_models: []` and
  `live_trading_enabled: false`, and both runners refuse to write a manifest that
  says otherwise.
- A7's decision states are `ROBUSTLY_UNINTERESTING`, `FRAGILE_SIGNAL` and
  `ROBUST_RESEARCH_CANDIDATE`. None promotes or trades; the strongest means
  "merits a separate preregistered study".
- No module under `btc_forecaster/research/` imports the paper engine, so nothing
  A6 or A7 computes can reach it.

`tests/test_quant_freeze.py` pins all five; `tests/test_paper_a2_regression.py`
and `tests/test_shadow_a3.py` pin the paper side as well.

## Evidence and provenance

Three kinds of artefact, kept apart:

- **Historical protected evidence** — the committed run directories under
  `research/runs/` and `research/market_intelligence/`. Frozen by
  [`CONTRIBUTING.md`](../CONTRIBUTING.md) section 2: never edited, never deleted,
  pinned by tests. A2 through `tests/test_evidence.py` and
  `tests/test_paper_a2_regression.py`; A6 and A7 through
  `tests/test_quant_freeze.py`; A7's files also through
  `verify --committed` in the suite and in the fresh-clone CI job.
- **Regenerable research outputs** — not committed, recorded by hash: market
  snapshots under `data/snapshots/`, A7's row-level `predictions.csv.gz`, A6's
  serialized model artefacts. Regenerating one and comparing hashes is the check.
- **Future release artefacts** — none exist. There has been no release, and the
  repository has no licence file; the README says to add one before
  open-sourcing. What a release may carry is a decision for that release.

### The data policy, and the files that predate it

Every study reads Yahoo Finance data through yfinance. The repository's stated
position — in [`model-zoo.md`](model-zoo.md), A6's run README and A7's input
manifest — is that Yahoo's terms do not grant redistribution, and it is why
snapshots and A7's predictions are not committed. **That position has not been
reviewed against Yahoo's current terms by anyone qualified to interpret them.**
This document records it as the project's conservative operating rule, not as a
legal determination.

Some committed evidence carries realised market values anyway. Scanning the
header of every tracked CSV under `research/` for realised-value columns
(`actual`, `close`, `open`, `high`, `low`, `volume`, `price` and similar) finds
five files:

| File | Rows | Realised values | Source it records |
|---|---|---|---|
| `research/runs/2026-04-02/historical_prices.csv` | 365 | daily closes | none |
| `research/runs/2026-08-28-track-a-baseline/historical_prices.csv` | 365 | daily closes | yfinance, in its run manifest |
| `research/runs/2026-08-29-a2-benchmark/predictions.csv` | 8,640 | realised USD closes: `actual`, `origin_close` | yfinance 1.2.0 |
| `research/runs/2026-08-29-a2-benchmark/largest_failures.csv` | 25 | realised USD closes | yfinance 1.2.0 |
| `research/runs/a6-model-zoo/predictions.csv` | 705 | realised next-bar log returns: `actual` | yfinance 1.2.0, snapshot `39b93e34…` |

The first four were on `main` before this integration. The fifth arrives with it.

**The A6 file, audited.** `research/runs/a6-model-zoo/predictions.csv` has a
`target_bar`, the realised next-bar log return in `actual`, and one forecast
column for each of A6's 40 active models — 705 bars, 2024-10-02 to 2026-09-07.
It was committed with A6's canonical run in `76927fd` and regenerated in
`7647856`. A6's run README lists it as "every forecast, per target bar, for
re-analysis". The `actual` values are computed from Yahoo Finance daily closes
retrieved on 2026-09-07; the forecasts are model output computed from the same
data.

The repository is public on GitHub, and this file has been on the published
`track-a6-model-zoo` branch since A6. Integrating A6 into `main` puts it on the
default branch; it does not newly publish it.

**It is not changed here.** It is protected evidence, and deleting or rewriting
it would edit published history, which the contribution rules forbid, without
unpublishing anything. Whether it — and the four older files — should stay in
future snapshots of the default branch, and what licence the repository adopts
before any release, are open questions for the maintainer. They need their own
commit and their own reasoning, not a side effect of an integration.

### A6's results digest and the platform's line ending

`b266a50d…` is taken over A6's results table as CSV text. Until A7.1 that text
ended its lines with the operating system's `os.linesep`, because pandas does
unless told otherwise. The canonical A6 run was produced on Windows, so the
recorded digest covers CRLF line endings, and the same table hashes to
`c7d1aa18…` with LF: a faithful A6 rerun anywhere but Windows could not match its
own published digest. CI found it when `tests/test_quant_freeze.py` recomputed
the digest on Linux.

The fix pins the line ending to the recorded CRLF form
(`RESULTS_DIGEST_LINETERMINATOR` in `btc_forecaster/research/runner.py`), so
every platform computes `b266a50d…` from the committed table, and on Windows
nothing changes. The committed evidence is untouched, and A6's numbers were
never affected — only the bytes the hash was taken over. A7 does not share the
defect: its canonical CSV and the snapshot digest both pin `\n`, and CI verifies
the committed A7 files on Linux.

## Reproducing from a fresh clone

1. **Install**, exactly as the fresh-clone CI job does:

   ```bash
   python -m pip install --require-hashes -r requirements.lock
   python -m pip install --no-deps -e .
   ```

2. **Verify the committed evidence.** The full suite does this for A2, A6 and A7
   (`pytest`). For A7 there is also a command:

   ```bash
   python -m btc_forecaster.research.walk_forward verify --committed research/runs/a7-walk-forward
   ```

   It exits 0 when every committed canonical file matches its manifest and, with
   the recorded hash of the absent predictions, reproduces the recorded result
   digest — and says that the predictions themselves were not verified. Plain
   `verify` exits 2 on a clone and names `predictions.csv.gz` as absent. That is
   correct: nobody has regenerated them yet.

3. **Reproduce the complete A7 benchmark.** This needs the pinned snapshot, which
   is not in the repository:

   ```bash
   python -m btc_forecaster.research.snapshot verify data/snapshots/BTC-USD \
       --expect 39b93e34520a496692ccc2156b8fe5c88c88b31bd9e9903a6fad74f282004bd3
   python -m btc_forecaster.research.walk_forward run --data data/snapshots/BTC-USD \
       --expect-input 39b93e34520a496692ccc2156b8fe5c88c88b31bd9e9903a6fad74f282004bd3 \
       --output <scratch directory> --workers 4
   python -m btc_forecaster.research.walk_forward verify <scratch directory>
   ```

   A reproduction is a run whose result digest equals `b3fba722…0d13790`. The
   committed run took 2,214 s on four workers of an Intel Core i5-10210U and
   3,819 s on two.

   **About the snapshot.** `btc-forecast snapshot` fetches BTC-USD from 2017-01-01
   to today. It will not produce these bytes: its range is longer, and yfinance
   revises history silently. `snapshot verify` then exits 3 and `run
   --expect-input` refuses to start. That is the design working. A run on other
   bytes is a different experiment, not a reproduction. The pinned bytes are not
   redistributed; a reproduction needs a copy of the original snapshot, whose
   sha256 above says whether you have it.

## Changing this status

`QUANT_RESEARCH_FROZEN` changes only by a commit that edits this document, states
which of the reasons in [`quant-research-handoff.md`](quant-research-handoff.md)
section 7 applies, and points to the preregistered proposal. A new model, a new
library or a better-looking number is not one of those reasons.
