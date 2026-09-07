# Deploying the quantitative forecaster

This is the **batch forecaster** — the quantitative core that fits models and
writes a run. For the market-intelligence collector, which is a different
service with a different schedule and its own frozen contracts, see
[`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md).

> This produces research artifacts. It places no trades, connects to no broker,
> and its headline finding is that nothing here beats a random walk.

## What runs

`btc-forecast run` — backtest every model on identical walk-forward folds, then
forecast forward with the primary model. It replaced `bayesianCutoff.py`, which
is preserved unmaintained under
[`research/legacy/`](research/legacy/README.md) because its Optuna output is
still the provenance of hyperparameters the platform uses.

## Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     POSIX: source .venv/bin/activate
pip install -e ".[models,data,plots]" -c constraints.txt

btc-forecast run --output-dir ./out
```

`.[models,data,plots]` is the narrowest set that can fit and plot a real run;
see [`DEPENDENCIES.md`](DEPENDENCIES.md) for the full extras table.

Artifacts land in `--output-dir` (default `./out`) alongside a `manifest.json`
recording the data snapshot's SHA-256, the configuration, hyperparameters, fold
boundaries, package versions and the git commit. A run without its manifest is
an anecdote — see [`ARTIFACTS.md`](ARTIFACTS.md).

## Configuration

The current entrypoint is **flag-driven, not environment-driven**. The legacy
script read `TICKER`, `HORIZON_DAYS`, `BAYESIAN_TEMPERATURE`, `PLOT_SHOW` and
friends from the environment; `btc-forecast` reads none of them.

```
--ticker            instrument (default BTC-USD)
--start             earliest bar to load, YYYY-MM-DD
--output-dir        where the run and its manifest are written
--snapshot-dir      pinned data snapshots
--refresh-data      re-fetch instead of reusing the pinned snapshot
--models            comma-separated model names
--baseline          the model skill is measured against
--folds             walk-forward fold count
--horizon           forecast horizon in bars
--no-plots          skip figures
```

`btc-forecast run --help` is authoritative; `btc-forecast --help` lists the other
subcommands (`benchmark`, `backtest`, `diagnose`, `snapshot`).

Data is read from a **hash-verified snapshot** rather than re-fetched, because
`yfinance` returns a slightly different series every day and silently revises
history. `--refresh-data` is the deliberate opt-in to a new pull.

## Docker

```bash
docker build -t btc-forecaster .
docker run --rm -v "$PWD/out:/app/out" btc-forecaster
```

## Scheduling

Any scheduler that can run a container or a command on a timer works: a GitHub
Actions cron, an ECS scheduled task, a systemd timer. There is nothing
long-running to supervise — a run starts, writes its artifacts and exits.

If you schedule it, persist `--output-dir`. Promote a run worth keeping into
`research/runs/<date>/` deliberately, with its manifest; everything left in
`out/` is regenerable and is not committed.

## Known gaps

Real and unaddressed, listed so a deployment does not assume otherwise:

- No retry or timeout around the market-data fetch.
- No structured logging, and no non-zero exit on a fatal data problem.
- No alerting on accuracy drift — and given the benchmark result, no accuracy
  worth alerting on.
