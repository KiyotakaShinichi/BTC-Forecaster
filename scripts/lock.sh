#!/usr/bin/env bash
# Regenerate requirements.lock -- the repository's transitive dependency closure.
#
# The lock is DERIVED, never hand-edited. Two inputs decide it:
#
#   pyproject.toml   what the repository declares it needs (the authority)
#   constraints.txt  the exact versions the committed research evidence was
#                    produced under (passed as constraints, so the closure
#                    cannot silently resolve away from the tested set)
#
# The output is universal -- one file with environment markers covering Linux,
# macOS and Windows -- and resolved for Python 3.11, which is the floor in
# `requires-python`, the version both CI jobs use and the version in the
# Dockerfiles. Resolving at the floor is deliberate: a closure resolved at 3.14
# can contain a distribution that has no 3.11 artifact, and the failure would
# surface on the deployment host rather than here.
#
# Usage:
#   bash scripts/lock.sh
#
# Then commit requirements.lock together with whatever change caused it.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif python -m uv --version >/dev/null 2>&1; then
  UV=(python -m uv)
else
  echo "uv is required to regenerate the lock: pip install uv" >&2
  echo "(uv resolves the lock; pip installs it. The repository is a pip project.)" >&2
  exit 1
fi

"${UV[@]}" pip compile pyproject.toml \
  --all-extras \
  --universal \
  --python-version 3.11 \
  --constraints constraints.txt \
  --generate-hashes \
  --output-file requirements.lock

echo "requirements.lock regenerated: $(grep -c '^[a-zA-Z0-9]' requirements.lock) pinned distributions"
