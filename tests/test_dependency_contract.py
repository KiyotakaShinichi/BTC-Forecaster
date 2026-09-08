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


# --------------------------------------------------------------------------
# The lock. `constraints.txt` pins what the repository declares; it says nothing
# about what those packages themselves pull in. 25 direct pins expanded to 83
# distributions on resolution, so 58 packages were reaching every install at
# whatever version happened to be current that day -- including the ones that
# actually parse the data (`python-dateutil`), sign the requests (`certifi`,
# `urllib3`) and back the numerics (`joblib`, `threadpoolctl`).
#
# `requirements.lock` closes that. It is generated by `scripts/lock.sh` from
# pyproject + constraints, never edited by hand, and these tests exist so a
# hand edit or a stale regeneration is a red build rather than a surprise on a
# deployment host.
# --------------------------------------------------------------------------

LOCK = REPO_ROOT / "requirements.lock"

#: `duckdb==1.5.5 \` and `tomli==2.4.1 ; python_full_version <= '3.11' \`.
_LOCK_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)")


def lock_pins() -> dict[str, str]:
    """Distribution -> version, for every entry in the lock."""
    pins: dict[str, str] = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "\t", "#")):
            continue
        match = _LOCK_PIN.match(line)
        if match:
            pins[match.group("name").lower().replace("_", "-")] = match.group("version")
    return pins


def lock_entries() -> list[list[str]]:
    """The lock split into one block of lines per pinned distribution."""
    blocks: list[list[str]] = []
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if _LOCK_PIN.match(line):
            blocks.append([line])
        elif blocks and line.startswith((" ", "\t")):
            blocks[-1].append(line)
    return blocks


class TestTheLockIsARealLock:
    def test_the_lock_exists(self) -> None:
        assert LOCK.exists(), "requirements.lock is the reproducible install contract"

    def test_it_records_the_command_that_produced_it(self) -> None:
        """A lock nobody can regenerate is a snapshot, not a contract."""
        header = LOCK.read_text(encoding="utf-8").splitlines()[:2]
        joined = " ".join(header)
        assert "uv pip compile" in joined, joined
        assert "--generate-hashes" in joined, joined
        assert "--universal" in joined, joined

    def test_it_is_resolved_at_the_supported_python_floor(self, pyproject: dict) -> None:
        """`requires-python` is >=3.11, both CI jobs and both Dockerfiles use
        3.11, and the deployment host is 3.11. A closure resolved at 3.14 can
        contain a distribution with no 3.11 artifact, and that failure would
        surface on the host rather than here."""
        header = " ".join(LOCK.read_text(encoding="utf-8").splitlines()[:2])
        assert "--python-version 3.11" in header, header
        assert pyproject["project"]["requires-python"] == ">=3.11"

    def test_every_entry_is_an_exact_pin(self) -> None:
        loose = [
            line
            for line in LOCK.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith((" ", "\t", "#")) and "==" not in line
        ]
        assert not loose, f"not exact pins: {loose}"

    def test_every_entry_carries_hashes(self) -> None:
        """Without hashes, `pip install --require-hashes` is refused and the
        lock pins a name and a number rather than an artifact."""
        unhashed = [
            block[0].split()[0]
            for block in lock_entries()
            if not any("--hash=sha256:" in line for line in block[1:])
        ]
        assert not unhashed, f"pinned without a hash: {unhashed}"

    def test_it_is_a_closure_and_not_a_copy_of_the_constraints(self) -> None:
        """The point of the file. If it only contained what is declared, it
        would be `constraints.txt` under a different name."""
        pins = lock_pins()
        declared = requirement_names(CONSTRAINTS.read_text(encoding="utf-8"))
        transitive = set(pins) - declared
        assert len(transitive) > 30, f"only {len(transitive)} transitive packages resolved"
        # A representative few, each of which genuinely reaches runtime.
        assert {"certifi", "urllib3", "python-dateutil", "joblib"} <= transitive


class TestTheLockAgreesWithTheDeclaredContract:
    def test_it_covers_every_declared_package(
        self, extras: dict[str, set[str]], pyproject: dict
    ) -> None:
        pins = lock_pins()
        declared = set(requirement_names("\n".join(pyproject["project"]["dependencies"])))
        for name, packages in extras.items():
            if name == "all":
                continue
            declared |= packages
        missing = declared - set(pins)
        assert not missing, f"declared but not locked: {sorted(missing)}"

    def test_it_does_not_drift_from_the_tested_versions(self) -> None:
        """`constraints.txt` records what the committed research evidence under
        `research/runs/` was produced with, and is passed to the resolver as a
        constraint. A disagreement means the lock was regenerated without it --
        so the evidence and the installable environment have parted company."""
        pins = lock_pins()
        drift = []
        for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            name, _, version = line.partition("==")
            key = name.strip().lower().replace("_", "-")
            if key in pins and pins[key] != version.strip():
                drift.append(f"{key}: constraints {version.strip()} != lock {pins[key]}")
        assert not drift, drift

    def test_the_collector_can_be_installed_from_it(self) -> None:
        """The deployment host installs the collector and nothing else. Its
        packages -- including the pytz that duckdb needs and does not declare --
        have to be in the closure or the lock cannot serve that host."""
        pins = lock_pins()
        collector = requirement_names(MI_REQUIREMENTS.read_text(encoding="utf-8"))
        assert collector <= set(pins), sorted(collector - set(pins))
        assert "pytz" in pins


class TestThereIsOnePackagingAuthority:
    def test_no_stale_root_requirements_file(self) -> None:
        """A second, looser list at the root outlived the package layout by
        several tracks. It declared neither duckdb nor pytz, so anything that
        installed from it got a repository that could not read its own corpus,
        and nothing in the repository referenced it."""
        assert not (REPO_ROOT / "requirements.txt").exists()

    def test_the_declared_extras_are_the_authority(self, pyproject: dict) -> None:
        assert pyproject["project"]["dependencies"], "pyproject declares the base"
        assert "all" in pyproject["project"]["optional-dependencies"]

    def test_the_lock_is_regenerable_by_a_committed_script(self) -> None:
        script = (REPO_ROOT / "scripts" / "lock.sh").read_text(encoding="utf-8")
        assert "pip compile pyproject.toml" in script
        assert "--constraints constraints.txt" in script
        assert "--generate-hashes" in script
        assert "--output-file requirements.lock" in script
