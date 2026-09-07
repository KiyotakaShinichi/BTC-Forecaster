#!/usr/bin/env bash
# Canonical project-wide coverage command. A missing/empty data file is fatal.
set -euo pipefail

PY="${PYTHON:-./.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="python"

rm -f .coverage coverage.json
# The collector's tests belong to the market-intelligence workflow: they are
# measured against market_intelligence, not btc_forecaster, and running them
# here would need duckdb/pytz and would run the same suite twice.
"$PY" -m pytest --cov=btc_forecaster --cov-branch --cov-report=term-missing --cov-fail-under=85 --ignore-glob="tests/test_intelligence_*.py" --ignore="tests/test_market_intelligence.py" -q
test -s .coverage
"$PY" -m coverage json -o coverage.json
"$PY" - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("coverage.json").read_text(encoding="utf-8"))
if not report.get("files"):
    raise SystemExit("coverage failed closed: zero source files measured")
print(f"coverage measured {len(report['files'])} source files")
PY
"$PY" -m coverage report --include='btc_forecaster/paper/*'
"$PY" -m coverage report --include='btc_forecaster/shadow/*'
