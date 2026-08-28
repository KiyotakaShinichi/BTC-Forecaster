#!/usr/bin/env bash
# The pre-commit gate for Track A2 (A2.26). Every commit must pass all of it.
#
#   bash scripts/check.sh            # full gate
#   bash scripts/check.sh --fast     # skip the slow model tests
set -u

PY="${PYTHON:-./.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="python"

fail=0
run() {
  echo ""
  echo "=== $1 ==="
  shift
  "$@" || fail=1
}

run "ruff" "$PY" -m ruff check btc_forecaster tests
run "mypy" "$PY" -m mypy btc_forecaster --ignore-missing-imports
run "compileall" "$PY" -m compileall -q btc_forecaster tests api_server.py

echo ""
echo "=== git diff --check ==="
git diff --check || fail=1

if [ "${1:-}" = "--fast" ]; then
  run "pytest (fast)" "$PY" -m pytest -q -m "not slow"
else
  run "pytest" "$PY" -m pytest -q
fi

echo ""
if [ "$fail" -eq 0 ]; then
  echo "GATE PASSED"
else
  echo "GATE FAILED"
fi
exit "$fail"
