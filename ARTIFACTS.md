# Generated vs. committed artifacts

This repository produces a lot of files that look like source but are not.
Without a rule, they accumulate in version control and stop being trustworthy:
nobody can tell which PNG corresponds to which code.

The rule is: **a file is committed only if it cannot be regenerated, or if it is
deliberately frozen as evidence of a specific historical run.**

## Three classes of file

### 1. Source — always committed

`btc_forecaster/`, `tests/`, `api_server.py`, `frontend/`, `Dockerfile*`,
`pyproject.toml`, `constraints.txt`, docs.

### 2. Runtime output — never committed

Everything written into `OUTPUT_DIR` (default `./out/`) by a forecast run:

| File | Producer |
| --- | --- |
| `forecast_results.csv` | forecast run |
| `forecast_summary.json` | forecast run |
| `backtest_results.csv` | walk-forward engine |
| `historical_prices.csv` | forecast run |
| `manifest.json` | artifact writer |
| `*.png` | diagnostics / plots |
| `forecast_run.log` | API subprocess wrapper |

`out/` is gitignored except for `out/.gitkeep`. Local market-data snapshots in
`data/snapshots/` are likewise ignored — they are large and reproducible from
the manifest that records their provenance (see `btc_forecaster/data/snapshot.py`).

A fresh clone therefore has an empty `out/`. That is intended. The dashboard
reports "no forecast yet" until a run happens, which is honest.

### 3. Research evidence — committed, frozen, dated

`research/runs/<date>/` holds artifacts that have been **deliberately promoted**
because they document a result worth keeping. They are read-only history: never
overwritten by a run, never regenerated in place.

| Directory | Contents |
| --- | --- |
| `research/runs/2025-03-13-initial/` | Artifacts from the initial commit (`1cb1396`): the original hybrid + Monte Carlo research output, produced by the scripts now in `research/legacy/`. |
| `research/runs/2026-04-02/` | The last `bayesianCutoff.py` run before the Track A refactor (`952b304`). Baseline for before/after comparison. Its headline numbers are **not** clean out-of-sample — see `research/runs/2026-04-02/README.md`. |

`research/legacy/` holds superseded implementations. They are kept because they
contain research decisions (notably the Optuna hyperparameter search that
produced the hardcoded XGBoost parameters still in use) that are not recorded
anywhere else. They are not maintained and are excluded from the test suite.

## Promoting a run

```bash
mkdir -p research/runs/$(date -u +%Y-%m-%d)
cp out/* research/runs/$(date -u +%Y-%m-%d)/
# write a README.md next to it saying what the run was for and what it showed
git add research/runs/$(date -u +%Y-%m-%d)
```

Include `manifest.json` — it carries the data snapshot hash and config that make
the run reproducible. A promoted run without its manifest is an anecdote.

## Negative results

Runs that showed a model *failing* are as worth promoting as runs that showed one
working, and should say so plainly in their README. A repository that only
contains its successes is not a research record.
