"""Guards for repository hygiene invariants.

These exist because the invariants were all violated at once before Track A:
`.gitignore` was UTF-16 encoded (git silently ignores such a file entirely), so
nothing was actually being ignored, and compiled bytecode plus every regenerable
artifact had been committed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git unavailable or not a repository: {result.stderr.strip()}")
    return result.stdout


@pytest.fixture(scope="module")
def tracked_files() -> list[str]:
    return [line for line in _git("ls-files").splitlines() if line]


def test_gitignore_is_utf8_and_therefore_honoured_by_git() -> None:
    """A UTF-16 .gitignore parses as garbage and silently ignores nothing.

    This is the root cause of every other hygiene problem in the pre-Track-A
    repository, so it gets its own test.
    """
    raw = (REPO_ROOT / ".gitignore").read_bytes()

    assert not raw.startswith(b"\xff\xfe"), ".gitignore has a UTF-16 LE BOM"
    assert not raw.startswith(b"\xfe\xff"), ".gitignore has a UTF-16 BE BOM"
    assert not raw.startswith(b"\xef\xbb\xbf"), ".gitignore has a UTF-8 BOM"
    assert b"\x00" not in raw, ".gitignore contains NUL bytes (not UTF-8)"

    text = raw.decode("utf-8")
    assert "__pycache__/" in text


def test_no_compiled_bytecode_is_tracked(tracked_files: list[str]) -> None:
    offenders = [f for f in tracked_files if f.endswith((".pyc", ".pyo")) or "__pycache__" in f]
    assert offenders == [], f"compiled bytecode is tracked: {offenders}"


def test_runtime_output_directory_is_not_tracked(tracked_files: list[str]) -> None:
    """`out/` is OUTPUT_DIR. Everything in it is regenerable (see ARTIFACTS.md)."""
    offenders = [f for f in tracked_files if f.startswith("out/") and f != "out/.gitkeep"]
    assert offenders == [], f"regenerable run output is tracked: {offenders}"


def test_no_generated_artifacts_at_repository_root(tracked_files: list[str]) -> None:
    """Plots and result CSVs belong under research/runs/<date>/, never at the root."""
    offenders = [
        f
        for f in tracked_files
        if "/" not in f and f.endswith((".png", ".csv", ".log"))
    ]
    assert offenders == [], f"generated artifacts at repo root: {offenders}"


@pytest.mark.parametrize(
    "path",
    [
        "out",
        "data/snapshots",
        "__pycache__",
    ],
)
def test_generated_directories_are_ignored(path: str) -> None:
    probe = f"{path}/probe-that-does-not-exist.tmp"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"{probe} is not ignored by .gitignore"


def test_research_evidence_is_not_ignored() -> None:
    """Frozen research artifacts must stay committable despite the *.png rules."""
    probe = "research/runs/2026-04-02/hybrid_forecast_montecarlo.png"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, "research/runs/ evidence is being ignored"


def test_promoted_research_runs_document_themselves() -> None:
    runs = sorted((REPO_ROOT / "research" / "runs").glob("*"))
    assert runs, "no promoted research runs found"
    for run in runs:
        if run.is_dir():
            assert (run / "README.md").exists(), f"{run.name} has no README.md explaining it"


def test_legacy_scripts_are_quarantined_not_deleted() -> None:
    """Superseded research must be preserved, but out of the import path."""
    legacy = REPO_ROOT / "research" / "legacy"
    expected = {
        "BTC_v1.py",
        "BTC_Predictor.py",
        "predefined_optuna.py",
        "MC_Automation.py",
        "cutoffOptimization.py",
    }
    present = {p.name for p in legacy.glob("*.py")}
    assert expected <= present, f"legacy research lost: {expected - present}"
    assert (legacy / "README.md").exists()
    assert not (legacy / "__init__.py").exists(), "legacy must not be an importable package"
