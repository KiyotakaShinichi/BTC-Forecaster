# BTC Forecaster Deployment (Bayesian Version)

## Model selected
This deploys `bayesianCutoff.py` (strongest research version in this repo).

## 1) Local run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OUTPUT_DIR = ".\out"
$env:PLOT_SHOW = "0"
python .\bayesianCutoff.py
```

Artifacts are written to `OUTPUT_DIR`:
- `nextgen_hybrid_forecast_results_montecarlo.csv`
- `pacf_diagnostic.png`
- `hybrid_forecast_montecarlo.png`
- `learning_curve.png`

## 2) Docker build + run

```powershell
docker build -t btc-bayesian-forecaster .
docker run --rm -e OUTPUT_DIR=/app/out -v ${PWD}/out:/app/out btc-bayesian-forecaster
```

## 3) Runtime environment variables

- `TICKER` (default: `BTC-USD`)
- `HORIZON_DAYS` (default: `365`)
- `TEST_LAST_DAYS` (default: `90`)
- `MONTE_CARLO_RUNS` (default: `1000`)
- `BAYESIAN_TEMPERATURE` (default: `2.0`)
- `RANDOM_STATE` (default: `42`)
- `MAX_LAG` (default: `60`)
- `OUTPUT_DIR` (default: current working dir)
- `PLOT_SHOW` (`0/1`, default: `0`)

## 4) Cloud scheduling patterns

- **GitHub Actions (cron):** build image, run daily, upload `out/` as artifact.
- **Azure Container Apps Job / AWS ECS Scheduled Task:** run container on schedule, push output CSV/PNGs to storage.
- **VM + cron/task scheduler:** run Python command daily and persist `out/`.

## 5) Production hardening (next)

- Add retries/timeouts around `yfinance` downloads.
- Add structured logging and non-zero exits on fatal data issues.
- Add walk-forward validation and tracking of live-vs-backtest drift.
- Add alerting when directional accuracy degrades.
