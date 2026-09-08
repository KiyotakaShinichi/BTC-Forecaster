# Contributing

**This repository has one maintainer and has had one contributor.** Saying so
up front is more useful than a governance section describing a process nobody
has run. There is no review rota, no triage schedule and no CODEOWNERS; if you
open a pull request it will be read by the person who wrote everything else.

What follows is what the build actually enforces, and the four rules that are
about the science rather than the code. The second group is the important one:
a change that breaks a lint rule fails loudly, and a change that quietly
invalidates the evidence does not.

---

## Setup

Python **3.11 or newer**. Both CI jobs, both Dockerfiles and the deployment host
run 3.11; development happens on 3.14.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on POSIX

# Reproducible: the exact closure, every artifact verified by hash.
pip install --require-hashes -r requirements.lock
pip install --no-deps -e .
```

`--no-deps` on the second line is not a shortcut: the lock already installed
every dependency at a verified hash, and letting pip resolve again would let it
substitute something the lock did not choose. An editable install cannot itself
be hash-checked, which is why it is a separate line.

To iterate against the declared bounds instead — faster, not byte-reproducible:

```bash
pip install -e ".[all]" -c constraints.txt
```

Copy `.env.example` to `.env` if you are running a service. Nothing loads it
automatically; that is deliberate, and the file explains why.

## The lockfile workflow

`pyproject.toml` is the authority. `constraints.txt` records the versions the
committed evidence under `research/runs/` was produced with. `requirements.lock`
is the closure of both, and is **generated, never edited**:

```bash
pip install uv          # resolves the lock; pip still installs it
bash scripts/lock.sh
```

To add a dependency:

1. add it to the right extra in `pyproject.toml`;
2. pin the tested version in `constraints.txt`;
3. mirror it into `requirements-market-intelligence.txt` if the collector needs
   it — that file is a frozen deployment artifact;
4. run `bash scripts/lock.sh` and commit `requirements.lock` **in the same
   commit** as the change that caused it.

A dependency that only ever arrives transitively is not declared, and will be
missing on the one machine that matters. See
[`DEPENDENCIES.md`](DEPENDENCIES.md).

## Tests

The two suites have different dependencies and different lint contracts, so they
run separately.

```bash
pytest                                       # everything: 1,400+ tests
pytest -m "not slow"                         # skip the heavy model fits
pytest tests/test_leakage.py -v              # the causality guard
pytest tests/test_intelligence_*.py -q       # the collector

bash scripts/check.sh                        # the full local gate
bash scripts/coverage.sh                     # quantitative core, floor 78%
```

Coverage floors are per subsystem and deliberately not blended: 78% for the
quantitative core as CI installs it, 75% for the collector. A single number
across two suites with different dependency sets measures nothing.

### The offline contract

**The default suite makes no network calls and reads no live data.** Every
series comes from the seeded generators in `btc_forecaster/testing.py`; every
document comes from `market_intelligence/collection/fixtures.py`. A test that
needs the network must be marked `@pytest.mark.network`, and there are currently
none.

This is not a preference. A suite that reaches the network is a suite that goes
red when a third party has an outage, and one that reads live prices is a suite
whose result depends on the day it ran.

## Static checks

```bash
ruff check btc_forecaster tests
mypy btc_forecaster --ignore-missing-imports
python -m compileall -q btc_forecaster tests api_server.py

ruff check --config market_intelligence/ruff.toml market_intelligence scripts tests/test_intelligence_*.py
mypy --config-file market_intelligence/mypy.ini market_intelligence scripts
```

Two configurations on purpose. The collector is checked under **mypy strict**;
the quantitative core is not, because the scientific stack is largely untyped
and demanding full annotations there produces noise rather than caught defects.
Each subsystem is checked once, under the contract it was written against.

## CI

Two workflows, five jobs, all required to be green:

| Job | Checks |
|---|---|
| `static-quality` | ruff, mypy and compileall for the whole repository |
| `quant-core` | the quantitative suite and its coverage floor |
| `dependency-contract` | installs from the declared contract; asserts the lock agrees with it |
| `fresh-clone` | installs from `requirements.lock` with `--require-hashes` and runs everything |
| `intelligence` | the collector's suite, its lint contract, and its offline smoke tests |

Push to a branch and read the result before assuming it passed. Six consecutive
red runs once went undiagnosed here because nobody looked; the workflows now
publish failures as annotations, which are readable without `actions:read`.

---

## The rules that are not about code

### 1. Point-in-time correctness

A forecast issued at `forecast_origin` may use an observation **iff**
`available_time <= forecast_origin`. That inequality is the whole leakage rule.

For collected evidence, `available_at` is **the moment of retrieval, never
publication**. A three-week-old press release first seen today became usable
today; treating its publication date as availability would fabricate three weeks
of hindsight. There is no exception, including for backfill: a backfill runs the
same cycle over a window that has passed, and is not a licence to date evidence
earlier.

Features are causal but that is not sufficient. Row `D` still contains bar `D`'s
close, so `to_supervised(step=1)` shifts them; `step=0` is rejected outright,
because it is precisely the original defect.

If you change anything in this area, `tests/test_leakage.py` is the file to read
first. It verifies causality by **perturbing the future and asserting the past
does not move**, which catches centred windows, negative shifts and whole-sample
normalisation that reading a column name will not.

### 2. Research evidence is preserved, never edited

Everything under `research/runs/` and `research/market_intelligence/` is frozen.
Runs are hash-manifested; a promoted run without its manifest is an anecdote.

The corpus follows the same rule with a sharper edge: **nothing is deleted.** An
observation later found to be an artefact of a defect is invalidated by
appending a correction row that says so, and the observation stays exactly where
it was written. Raw and eligible are two views of one record — preserved
physically, excluded scientifically. Corrections name event ids literally; a
correction expressed as a rule keeps matching new events forever, and that is a
filter, not a correction.

`deploy/COLLECTION_FREEZE.md` lists the frozen collection contracts and what
changing one requires. It is a short list and none of it is casual.

### 3. Negative results stay negative

The headline finding is that **nothing beats the random walk** — eight models,
36 folds, zero promotions. Do not retune a model to improve a published metric,
do not adjust an evaluation to make a result look better, and do not widen a
research question until something passes.

The paper-trading engine returns `LIVE_TRADING_ELIGIBLE: False` because it reads
A2's evidence and finds no promoted model. That is the correct answer and not a
placeholder. **Live trading is not implemented and this repository connects to
no broker.**

### 4. Commits

One change per commit, with the reasoning in the message. The history here is
long-form on purpose: a message that says *why* is the only place that survives,
because the diff already says what.

Every behaviour change ships with the test that would have caught the defect.
Not a test that exercises the new code — a test that fails without the fix.

Do not amend, rebase or force-push a published branch. Never bypass branch
protections.

---

## Adding a model

1. Implement the `ForecastModel` contract in `btc_forecaster/models/`. Heavy
   dependencies are imported inside the adapter, not at module scope, so the
   package stays importable without them and degrades to a clear error.
2. Register it, and add it to the `models` extra plus `constraints.txt`.
3. Run it through **the same walk-forward folds as everything else** — the
   benchmark composes one engine for exactly this reason — and let the promotion
   policy decide. A model that does not beat the baseline is a result to record,
   not a problem to fix.
4. Commit the evidence with its manifest.

## Adding a provider or a feed

1. Declare it in `market_intelligence/collection/`, with its terms, its cadence
   floor, its retention policy and whether it needs a credential. Naming a
   credential variable does not enable a provider; the profile has to name it
   too, and an unset variable is reported as "not enabled", never guessed at.
2. Give it a `DisclosureStream`. The three SEC streams stay distinct however
   similar the publisher — an administrative proceeding and the press release
   announcing it are different instruments — and **they are never merged to
   increase an event count**.
3. Declare its match fields. A stream whose fields cannot carry subject matter
   is marked `subject_bearing=False`, and zero topical matches from it is a
   property of the source, not a fault to work around.
4. Never add a lexical alias for yield. New aliases require source evidence,
   explicit review, and independence from market outcomes.
5. Add a fixture and prove the path offline. Every provider is exercised in CI
   without touching the network.
6. If the feed is public and government-operated, it needs no credential — and
   keeping it that way is worth something: a collector with no secret cannot
   leak one.
