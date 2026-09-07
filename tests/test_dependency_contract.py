"""The repository has one dependency contract, and these tests keep it one.

For a while it had two that did not know about each other: `pyproject.toml`
described the quantitative core, `requirements-market-intelligence.txt`
described the collector, and neither mentioned the other's packages. That gap
was invisible on any developer machine, because between them the two lineages
had installed everything. It was visible only on a clean host — where the
collector was installed without `pytz` and raised on the first read of its own
corpus, since duckdb needs pytz for `TIMESTAMPTZ` and does not declare it.

So the invariant worth guarding is not "the lists are tidy". It is that a single
declared install produces a repository where both halves actually run.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MI_REQUIREMENTS = REPO_ROOT / "requirements-market-intelligence.txt"
MI_DEV_REQUIREMENTS = REPO_ROOT / "requirements-market-intelligence-dev.txt"
CONSTRAINTS = REPO_ROOT / "constraints.txt"

#: `duckdb>=1.0,<2` -> `duckdb`; `uvicorn[standard]>=0.30` -> `uvicorn`.
_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def requirement_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = _NAME.match(line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def extras(pyproject: dict) -> dict[str, set[str]]:
    optional = pyproject["project"]["optional-dependencies"]
    return {name: requirement_names("\n".join(reqs)) for name, reqs in optional.items()}


class TestOneContractCoversBothHalves:
    def test_the_collector_has_an_extra_of_its_own(self, extras: dict[str, set[str]]) -> None:
        assert "market-intelligence" in extras

    def test_installing_everything_installs_the_collector(self, pyproject: dict) -> None:
        """`all` is what a newcomer installs. If it omits half the repository,
        the omission surfaces as something failing at runtime.

        Read raw rather than parsed: `all` is a self-reference of the form
        `btc-forecaster[models,...]`, and the extras it pulls in live inside the
        brackets a name parser throws away.
        """
        raw = " ".join(pyproject["project"]["optional-dependencies"]["all"])
        assert "market-intelligence" in raw, raw

    def test_the_extra_covers_what_the_collector_ci_installs(
        self, extras: dict[str, set[str]]
    ) -> None:
        """The pinned CI list is a frozen deployment artifact and stays. What it
        must never do is drift away from the canonical contract."""
        pinned = requirement_names(MI_REQUIREMENTS.read_text(encoding="utf-8"))
        missing = pinned - extras["market-intelligence"]
        assert not missing, f"declared for CI but not in the extra: {sorted(missing)}"

    def test_pytz_is_declared_rather_than_inherited(
        self, extras: dict[str, set[str]]
    ) -> None:
        """The specific defect. duckdb needs pytz to return a TIMESTAMPTZ and
        does not declare it; every timestamp in the corpus is TIMESTAMPTZ. It
        used to arrive only through yfinance, so it was present on every machine
        that had run the quant tracks and absent on a clean deployment."""
        assert "pytz" in extras["market-intelligence"]
        assert "pytz" in requirement_names(MI_REQUIREMENTS.read_text(encoding="utf-8"))

    def test_duckdb_is_declared(self, extras: dict[str, set[str]]) -> None:
        assert "duckdb" in extras["market-intelligence"]


class TestConstraintsPinWhatIsDeclared:
    def test_constraints_pin_the_collector_packages(self) -> None:
        """`constraints.txt` reproduces a run. A package it does not pin is a
        package whose version the evidence does not record."""
        pinned = requirement_names(CONSTRAINTS.read_text(encoding="utf-8"))
        assert {"duckdb", "pytz"} <= pinned

    def test_every_constraint_is_an_exact_pin(self) -> None:
        """A range in a constraints file records nothing."""
        loose = [
            line.strip()
            for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#") and "==" not in line
        ]
        assert not loose, f"not exact pins: {loose}"

    def test_constraints_cover_the_declared_extras(
        self, extras: dict[str, set[str]], pyproject: dict
    ) -> None:
        pinned = requirement_names(CONSTRAINTS.read_text(encoding="utf-8"))
        declared = set(requirement_names("\n".join(pyproject["project"]["dependencies"])))
        for name, packages in extras.items():
            if name == "all":
                continue
            declared |= packages
        # `starlette` is pinned as a transitive of fastapi; that is deliberate.
        missing = declared - pinned
        assert not missing, f"declared but unpinned: {sorted(missing)}"


class TestTheDevContractIsUsable:
    def test_the_collector_dev_list_builds_on_its_runtime_list(self) -> None:
        text = MI_DEV_REQUIREMENTS.read_text(encoding="utf-8")
        assert "-r requirements-market-intelligence.txt" in text

    def test_dev_tooling_is_declared(self, extras: dict[str, set[str]]) -> None:
        assert {"pytest", "ruff", "mypy"} <= extras["dev"]
