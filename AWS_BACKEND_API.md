# AWS Deployment Options (Backend + API)

## TL;DR recommendation
Use **AWS ECS Fargate**.

- **Batch forecast (scheduled):** EventBridge Scheduler -> ECS Task (runs `bayesianCutoff.py`) -> writes CSV/PNGs to S3.
- **On-demand API:** API Gateway or ALB -> ECS Fargate Service (runs `api_server.py`) -> triggers forecast run and serves latest output.

This is the cleanest fit for your current Python stack (`prophet`, `xgboost`, `arch`) without forcing Lambda limits.

---

## Option 1: Batch only (lowest cost/ops)

### Architecture
- EventBridge Scheduler (cron)
- ECS Fargate Task using `Dockerfile`
- S3 bucket for outputs
- CloudWatch Logs

### Best for
- Daily/weekly forecast generation
- No public API needed

### What you need
1. ECR repo for image
2. ECS task definition (CPU/memory + env vars)
3. Task execution role + S3 write permission
4. EventBridge schedule

---

## Option 2: API + Batch (most flexible)

### Architecture
- ECS Fargate Service using `Dockerfile.api`
- ALB (or API Gateway + VPC Link) exposing `/health`, `/run`, `/status`, `/latest`
- Optional EventBridge trigger calling `/run` on schedule
- S3 for artifacts

### Best for
- Dashboard/mobile frontend needs API
- Manual + scheduled runs

### Backend/API endpoints
- `GET /health`
- `GET /status`
- `POST /run`
- `GET /latest`
- `GET /artifacts`

---

## Option 3: Lambda container (possible but not ideal)

Works only if runtime/memory/time fits. Heavy ML libs and long forecast runs often exceed practical Lambda limits.

---

## Quick local API test

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python api_server.py
```

Then:

```powershell
Invoke-RestMethod http://localhost:8010/health
Invoke-RestMethod -Method Post -Uri http://localhost:8010/run -Body (@{
  ticker = "BTC-USD"
  horizon_days = 365
  test_last_days = 90
  monte_carlo_runs = 1000
  bayesian_temperature = 2.0
  max_lag = 60
  random_state = 42
} | ConvertTo-Json) -ContentType "application/json"
Invoke-RestMethod http://localhost:8010/status
Invoke-RestMethod http://localhost:8010/latest
```

---

## Minimal production checklist

- Store outputs in S3 (not only local container disk).
- Add auth for API (`/run` should not be public/open).
- Add retry and timeout around data downloads.
- Add CloudWatch alarms for failed runs.
- Add versioned image tags and rollback policy.

### New environment variables supported by `api_server.py`

- `API_TOKEN`: if set, requests to `POST /run` must include header `X-API-Key: <API_TOKEN>`.
- `ALLOWED_OUTPUT_ROOT`: restricts `output_dir` to a safe root path.
- `S3_ARTIFACT_BUCKET`: enables post-run artifact upload to S3.
- `S3_ARTIFACT_PREFIX`: optional key prefix inside the S3 bucket.
- `S3_REGION`: optional explicit AWS region for S3 client.

Recommended for ECS task definition:

- Store `API_TOKEN` in AWS Secrets Manager and inject as secret.
- Set `ALLOWED_OUTPUT_ROOT=/app/out`.
- Mount writable volume at `/app/out` if you need local retention in addition to S3.

---

## Suggested next build

1. Add S3 artifact upload after each run.
2. Add token auth for API endpoints.
3. Add Terraform/CDK for one-click AWS provisioning.
