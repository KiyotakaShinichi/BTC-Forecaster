# Dependencies

**`pyproject.toml` is the canonical contract.** Everything else in the
repository is either a pinned subset of it or a historical artifact, and this
file says which is which — because the repository briefly had two contracts that
did not know about each other, and the gap between them was invisible until a
clean machine tried to install one and run the other.

## Installing

```sh
python -m pip install -e ".[all]" -c constraints.txt
```

That gives the whole repository: the quantitative core, the market-intelligence
collector, the API, plots, cloud sync and the dev tooling.

For a narrower install, pick extras:

| Extra | Brings | Needed for |
|---|---|---|
| *(base)* | numpy, pandas, scipy, statsmodels | `btc_forecaster` core, feature and evaluation logic |
| `models` | prophet, xgboost, scikit-learn, arch | fitting anything; adapters degrade to a clear error without it |
| `data` | yfinance | live market pulls only — never imported by unit tests |
| `plots` | matplotlib | figure artifacts |
| `api` | fastapi, uvicorn, pydantic | `api_server.py` |
| `cloud` | boto3 | S3 artifact sync |
| `market-intelligence` | duckdb, pytz, fastapi, uvicorn, pydantic | `market_intelligence/`, `btc-intel` |
| `dev` | pytest, coverage, ruff, mypy, httpx | running the suites |

### Why `pytz` is declared explicitly

duckdb needs it to hand a `TIMESTAMPTZ` column back as a Python datetime, and
does not declare it itself. Every timestamp in the intelligence corpus is
`TIMESTAMPTZ`, so without it the collector raises on the first read of anything
it has written. It used to reach developer machines only transitively, through
`yfinance` — a quantitative dependency with no relation to collection — so every
machine that had run the quant tracks had it and a clean deployment did not.
That is the shape of bug this file exists to prevent.

## The other files

**`constraints.txt`** — the exact versions the platform was developed and tested
against. Not a lockfile: no transitive closure, no hashes. Its job is
reproducing a run or bisecting a regression, and the committed evidence under
`research/runs/` was produced under these pins. Bumping them is a separate
change that produces its own evidence.

**`requirements-market-intelligence.txt`** and **`-dev.txt`** — the pinned subset
the market-intelligence CI job installs. They are kept, not folded into
`pyproject.toml`, because that job is part of a frozen deployment contract
(`deploy/COLLECTION_FREEZE.md`) and its green history is evidence about the
collector. They must stay consistent with the `market-intelligence` extra; a
test asserts they do.

**`requirements.txt`** — historical. It predates the package layout and is still
referenced by `DEPLOYMENT.md` and `AWS_BACKEND_API.md`. Prefer the extras above.

## Adding a dependency

Add it to the appropriate extra in `pyproject.toml`, pin the tested version in
`constraints.txt`, and — if the collector needs it — mirror it into
`requirements-market-intelligence.txt`. A dependency that only ever arrives
transitively is not declared, and will be missing on the one machine that
matters.
