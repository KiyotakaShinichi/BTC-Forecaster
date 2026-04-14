# BTC Forecaster (Hybrid Time-Series + Uncertainty + API)

A small end-to-end time-series forecasting project built while learning applied time-series analysis.

It generates **BTC-USD forecasts** using a hybrid approach:
- **Prophet** for the baseline trend/seasonality forecast
- **XGBoost** to model and correct Prophet residuals using engineered lag/rolling/EMA/SMA features
- **GARCH(1,1) + Monte Carlo** to produce uncertainty bands around the hybrid forecast

It also includes a **FastAPI** server that can trigger runs and serve the latest outputs, plus a lightweight static dashboard.

> Note: This is a research/demo project. Forecasts are not financial advice.

---

## What’s in this repo

- `bayesianCutoff.py` — main forecasting pipeline + diagnostics + artifacts
- `api_server.py` — FastAPI service to trigger forecasts and serve latest artifacts
- `frontend/` — static dashboard served by the API
- `Dockerfile` — container to run the batch forecaster
- `Dockerfile.api` — container to run the API service
- `docker-compose.yml` — local API container helper
- `out/` — default output folder (created automatically)

---

## Modeling overview (high level)

### 1) Data
- Pulls daily BTC prices and volume using `yfinance`.

### 2) Cutoff selection (“Bayesian cutoff”)
- Evaluates multiple historical start dates (“cutoffs”) and scores them via a strict train/test directional-accuracy evaluation.
- Uses a softmax (temperature-controlled) weighting over cutoff scores to select a MAP cutoff.

### 3) Feature engineering
- Returns + log price
- PACF-driven lag selection (train-only)
- Rolling mean/std of returns and rolling volume
- EMA/SMA candidates chosen using train-only correlation with next-day returns

### 4) Hybrid forecast
- Fit Prophet on **training data only**.
- Train XGBoost on training residuals.
- Predict test residuals for evaluation.
- Predict future residuals and combine with Prophet future predictions.

### 5) Uncertainty
- Fit GARCH(1,1) on training residuals.
- Use Monte Carlo sampling from conditional variance to generate 95% intervals.

### 6) Evaluation
- Holdout evaluation on the last `TEST_LAST_DAYS` days (directional accuracy + binomial test).
- Walk-forward (expanding window) directional-accuracy evaluation across multiple folds.

---

## Outputs

After a run, artifacts are written to `OUTPUT_DIR` (default: `./out`):

- `nextgen_hybrid_forecast_results_montecarlo.csv` — forecast dataframe (baseline, residual, combined, CI)
- `historical_prices.csv` — last 365 days of historical prices used by the dashboard plot
- `forecast_summary.json` — run metadata + metrics (incl. walk-forward results)
- `pacf_diagnostic.png` — PACF plot
- `hybrid_forecast_montecarlo.png` — forecast plot with 95% interval
- `learning_curve.png` — learning curve (directional accuracy vs training size)

---

## Quickstart (local)

### 1) Create venv + install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Run the batch forecaster

```powershell
$env:OUTPUT_DIR = ".\out"
$env:PLOT_SHOW = "0"
python .\bayesianCutoff.py
```

---

## Run the API + dashboard locally

Start the server:

```powershell
.\.venv\Scripts\Activate.ps1
$env:APP_HOST = "0.0.0.0"
$env:APP_PORT = "8010"
$env:APP_AUTO_OPEN = "1"
python .\api_server.py
```

Open:
- Dashboard: http://localhost:8010/
- Swagger docs: http://localhost:8010/docs

### API endpoints

- `GET /health`
- `GET /status`
- `POST /run` — starts a background forecast run
- `GET /latest` — returns latest summary + last row
- `GET /artifacts` — lists output files

---

## Docker

### Batch container

```powershell
docker build -t btc-bayesian-forecaster .
docker run --rm -e OUTPUT_DIR=/app/out -v ${PWD}/out:/app/out btc-bayesian-forecaster
```

### API container

```powershell
docker build -f Dockerfile.api -t btc-forecast-api .
docker run --rm -p 8010:8010 -e OUTPUT_DIR=/app/out -v ${PWD}/out:/app/out btc-forecast-api
```

Or with compose:

```powershell
docker compose up --build
```

---

## Configuration

### Forecast env vars (`bayesianCutoff.py`)

- `TICKER` (default: `BTC-USD`)
- `HORIZON_DAYS` (default: `365`)
- `TEST_LAST_DAYS` (default: `90`)
- `MONTE_CARLO_RUNS` (default: `1000`)
- `BAYESIAN_TEMPERATURE` (default: `2.0`)
- `RANDOM_STATE` (default: `42`)
- `MAX_LAG` (default: `60`)
- `OUTPUT_DIR` (default: current working dir)
- `PLOT_SHOW` (`0/1`, default: `0`)

### API env vars (`api_server.py`)

- `APP_HOST` (default: `0.0.0.0`)
- `APP_PORT` (default: `8010`)
- `APP_RELOAD` (`0/1`, default: `0`)
- `APP_AUTO_OPEN` (`0/1`, default: `1`)
- `OUTPUT_DIR` (default: `./out`)

Security + storage options:
- `API_TOKEN` — if set, `POST /run` requires header `X-API-Key: <API_TOKEN>`
- `ALLOWED_OUTPUT_ROOT` — restricts any `output_dir` override to a safe root
- `S3_ARTIFACT_BUCKET` — if set, successful runs upload artifacts to S3
- `S3_ARTIFACT_PREFIX` — optional prefix within the bucket
- `S3_REGION` — optional AWS region

---

## Notes / limitations

- Forecast quality depends heavily on market regime changes; crypto markets are non-stationary.
- The binomial test on direction assumes independent trials; treat p-values as a rough signal, not a guarantee.
- Uncertainty bands are conditional on the fitted GARCH model and do not capture all tail risks.
- This project is intended for learning, demos, and iteration (not production trading decisions).

---

## Deployment (cloud)

See:
- `DEPLOYMENT.md` — local + Docker + scheduling patterns
- `AWS_BACKEND_API.md` — recommended AWS ECS Fargate patterns (API + scheduled runs)

---

## License

Add a license if you plan to open-source this.
