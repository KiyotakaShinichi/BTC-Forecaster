FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml constraints.txt README.md ./
COPY btc_forecaster ./btc_forecaster
RUN pip install --no-cache-dir -e ".[models,data,plots,cloud]" -c constraints.txt

ENV OUTPUT_DIR=/app/out     SNAPSHOT_DIR=/app/data/snapshots     PLOT_SHOW=0     TICKER=BTC-USD     HORIZON_DAYS=365     MONTE_CARLO_RUNS=1000     WF_FOLDS=6     WF_HORIZON=30

RUN mkdir -p /app/out /app/data/snapshots

# Backtests every model through identical walk-forward folds, then forecasts.
CMD ["python", "-m", "btc_forecaster.cli", "run"]
