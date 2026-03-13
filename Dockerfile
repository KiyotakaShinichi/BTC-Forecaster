FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY bayesianCutoff.py ./

ENV OUTPUT_DIR=/app/out \
    PLOT_SHOW=0 \
    TICKER=BTC-USD \
    HORIZON_DAYS=365 \
    TEST_LAST_DAYS=90 \
    MONTE_CARLO_RUNS=1000 \
    BAYESIAN_TEMPERATURE=2.0

RUN mkdir -p /app/out

CMD ["python", "bayesianCutoff.py"]
