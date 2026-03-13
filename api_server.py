import os
import sys
import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DEFAULT_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "out"))).resolve()
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = "nextgen_hybrid_forecast_results_montecarlo.csv"
SUMMARY_JSON = "forecast_summary.json"
RUN_LOG = "forecast_run.log"

app = FastAPI(title="BTC Bayesian Forecaster API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/out", StaticFiles(directory=str(DEFAULT_OUTPUT_DIR)), name="out")

_state_lock = threading.Lock()
_state = {
    "running": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_exit_code": None,
    "last_error": None,
}


class ForecastRunRequest(BaseModel):
    ticker: str = Field(default="BTC-USD")
    horizon_days: int = Field(default=365, ge=30, le=3650)
    test_last_days: int = Field(default=90, ge=30, le=1000)
    monte_carlo_runs: int = Field(default=1000, ge=100, le=20000)
    bayesian_temperature: float = Field(default=2.0, gt=0.0, le=10.0)
    max_lag: int = Field(default=60, ge=5, le=120)
    random_state: int = Field(default=42)
    output_dir: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_forecast_job(req: ForecastRunRequest) -> None:
    env = os.environ.copy()
    out_dir = Path(req.output_dir).resolve() if req.output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "TICKER": req.ticker,
            "HORIZON_DAYS": str(req.horizon_days),
            "TEST_LAST_DAYS": str(req.test_last_days),
            "MONTE_CARLO_RUNS": str(req.monte_carlo_runs),
            "BAYESIAN_TEMPERATURE": str(req.bayesian_temperature),
            "MAX_LAG": str(req.max_lag),
            "RANDOM_STATE": str(req.random_state),
            "OUTPUT_DIR": str(out_dir),
            "PLOT_SHOW": "0",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )

    cmd = [sys.executable, "bayesianCutoff.py"]
    log_path = out_dir / RUN_LOG

    with _state_lock:
        _state["running"] = True
        _state["last_started_at"] = _now_iso()
        _state["last_finished_at"] = None
        _state["last_exit_code"] = None
        _state["last_error"] = None

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n===== RUN {datetime.now().isoformat()} =====\n")
            f.write(result.stdout or "")
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        with _state_lock:
            _state["last_exit_code"] = result.returncode
            if result.returncode != 0:
                _state["last_error"] = f"Forecast process exited with code {result.returncode}"
    except Exception as exc:
        with _state_lock:
            _state["last_error"] = str(exc)
            _state["last_exit_code"] = -1
    finally:
        with _state_lock:
            _state["running"] = False
            _state["last_finished_at"] = _now_iso()


@app.get("/health")
def health():
    return {"ok": True, "service": "btc-bayesian-forecaster-api"}


@app.get("/")
def frontend_home():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))


@app.get("/status")
def status():
    with _state_lock:
        return dict(_state)


@app.post("/run")
def run_forecast(req: ForecastRunRequest):
    with _state_lock:
        if _state["running"]:
            raise HTTPException(status_code=409, detail="Forecast job already running")

    thread = threading.Thread(target=_run_forecast_job, args=(req,), daemon=True)
    thread.start()

    return {
        "started": True,
        "message": "Forecast job started",
        "status_url": "/status",
        "artifacts_url": "/artifacts",
    }


@app.get("/artifacts")
def artifacts(output_dir: Optional[str] = None):
    out_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR
    if not out_dir.exists():
        return {"files": []}

    files = []
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "url": f"/out/{path.name}",
                }
            )

    return {"files": files}


@app.get("/latest")
def latest(output_dir: Optional[str] = None):
    out_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR
    csv_path = out_dir / RESULT_CSV
    summary_path = out_dir / SUMMARY_JSON

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"No forecast file found at {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        raise HTTPException(status_code=404, detail="Forecast file is empty")

    last_row = df.iloc[-1].to_dict()
    first_row = df.iloc[0].to_dict()
    stats = {}
    if summary_path.exists():
        try:
            stats = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            stats = {}

    response = {
        "csvList": [f"/out/{RESULT_CSV}", "/out/historical_prices.csv", f"/out/{SUMMARY_JSON}"],
        "rows": int(len(df)),
        "start_date": str(df.index[0]),
        "end_date": str(df.index[-1]),
        "first_row": first_row,
        "last_row": last_row,
        "stats": stats,
    }

    return json.loads(json.dumps(response, default=str))


if __name__ == "__main__":
    import time
    import webbrowser
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8010"))
    reload_enabled = os.getenv("APP_RELOAD", "0").strip().lower() in {"1", "true", "yes"}

    auto_open = os.getenv("APP_AUTO_OPEN", "1").strip().lower() in {"1", "true", "yes"}
    if auto_open and not reload_enabled:
        def _open_dashboard() -> None:
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open_dashboard, daemon=True).start()

    uvicorn.run("api_server:app", host=host, port=port, reload=reload_enabled)
