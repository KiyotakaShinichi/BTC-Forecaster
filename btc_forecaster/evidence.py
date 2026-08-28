"""Immutable research evidence: results that must not be quietly improved.

Two results in this repository are load-bearing for its credibility, and both
are inconvenient. Recording them as data -- with the archive path, the exact
headline numbers, and the reason each is what it is -- makes them tamper-evident:
:mod:`tests.test_evidence` reads the archived JSON back and fails if the numbers
have drifted.

The reason for the mechanism is specific. The natural failure mode of a
forecasting project is to keep tuning until the number looks acceptable and then
report the last number. Nothing catches that, because each individual retune is
defensible. A frozen reference that a test checks against is what makes the
drift visible.

Neither result may be deleted, overwritten, re-tuned until it looks better, or
re-scored on a metric chosen to make it competitive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_RUNS = REPO_ROOT / "research" / "runs"

#: The pre-Track-A headline. Reported as significant; produced by leakage.
LEGACY_INVALID_RESULT = "LEGACY_INVALID_RESULT"

#: The same model re-scored without the leaks. Negative, and the honest number.
LEAKAGE_CORRECTED_REFERENCE = "LEAKAGE_CORRECTED_REFERENCE"


class EvidenceTampering(AssertionError):
    """Raised when an archived result no longer matches its recorded value."""


@dataclass(frozen=True)
class PreservedResult:
    """A frozen research result, its location, and why it is preserved."""

    label: str
    run_dir: str
    source_commit: str
    status: str
    headline: dict[str, float]
    why: str
    do_not: tuple[str, ...] = field(default=())

    @property
    def path(self) -> Path:
        return RESEARCH_RUNS / self.run_dir

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "run_dir": self.run_dir,
            "source_commit": self.source_commit,
            "status": self.status,
            "headline": dict(self.headline),
            "why": self.why,
            "do_not": list(self.do_not),
        }


LEGACY_INVALID = PreservedResult(
    label=LEGACY_INVALID_RESULT,
    run_dir="2026-04-02",
    source_commit="952b304",
    status="INVALID -- do not cite as a result",
    headline={
        "directional_accuracy": 0.6896551724137931,
        "p_value": 0.03071417286992073,
        "walk_forward_mean_accuracy": 0.5862068965517241,
    },
    why=(
        "Produced by bayesianCutoff.py with three independent defects: features "
        "and target on the same bar (rolling/EMA/SMA columns at row D contain "
        "close[D], the target was the residual of log_close[D]); the cutoff "
        "selected by scoring 48 candidates on the same last-90-days window the "
        "holdout was then reported on; and a directional metric that compared "
        "the forecast path against itself via np.diff. The walk-forward number "
        "is also contaminated -- features were selected once, over a window "
        "overlapping every fold's test period."
    ),
    do_not=(
        "cite 0.6897 or p=0.031 as evidence of forecasting skill",
        "compare a new model against it -- it is not a valid baseline",
        "re-run bayesianCutoff.py and overwrite this directory",
    ),
)

LEAKAGE_CORRECTED = PreservedResult(
    label=LEAKAGE_CORRECTED_REFERENCE,
    run_dir="2026-08-28-track-a-baseline",
    source_commit="c415b03",
    status="VALID -- negative result, preserved deliberately",
    headline={
        "hybrid_directional_accuracy": 0.24444444444444444,
        "hybrid_mae_skill_vs_random_walk": -1.6395517444453533,
        "hybrid_interval_coverage": 0.2333333333333333,
        "hybrid_mae": 10158.505895137203,
        "random_walk_mae": 3848.572363286555,
    },
    why=(
        "The same Prophet+XGBoost hybrid, scored through identical walk-forward "
        "folds with the leaks removed. It is the worst of seven models tested: "
        "2.6x the random walk's error, directional accuracy materially below a "
        "coin, and a stated 95% interval containing 23% of outcomes. Nothing "
        "about the model changed -- only that it can no longer see the price it "
        "is predicting. This is the reference every A2 challenger is measured "
        "against, and the number the project must not quietly walk back."
    ),
    do_not=(
        "re-tune the legacy hybrid until this improves -- build a named "
        "challenger instead (XGBOOST_CAUSAL_RETUNED)",
        "swap the metric for one on which the hybrid looks better",
        "overwrite this directory with a later run",
    ),
)

PRESERVED: tuple[PreservedResult, ...] = (LEGACY_INVALID, LEAKAGE_CORRECTED)


def by_label(label: str) -> PreservedResult:
    for result in PRESERVED:
        if result.label == label:
            return result
    raise KeyError(f"unknown evidence label {label!r}; known: {[r.label for r in PRESERVED]}")


def _read_summary(result: PreservedResult) -> dict:
    path = result.path / "forecast_summary.json"
    if not path.exists():
        raise EvidenceTampering(
            f"{result.label}: archived summary missing at {path}. "
            "Preserved research runs must not be deleted."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_observed(summary: dict) -> dict[str, float]:
    return {
        "directional_accuracy": float(summary["directional_accuracy"]),
        "p_value": float(summary["p_value"]),
        "walk_forward_mean_accuracy": float(summary["walk_forward_mean_accuracy"]),
    }


def _corrected_observed(summary: dict) -> dict[str, float]:
    comparison = {row["model"]: row for row in summary["model_comparison"]}
    skill = {row["model"]: row for row in summary["skill_vs_baseline"]}
    hybrid = comparison["prophet_xgb_hybrid"]
    return {
        "hybrid_directional_accuracy": float(hybrid["directional_accuracy"]),
        "hybrid_mae_skill_vs_random_walk": float(
            skill["prophet_xgb_hybrid"]["mae_skill_vs_random_walk"]
        ),
        "hybrid_interval_coverage": float(hybrid["interval_coverage"]),
        "hybrid_mae": float(hybrid["mae"]),
        "random_walk_mae": float(comparison["random_walk"]["mae"]),
    }


_READERS = {
    LEGACY_INVALID_RESULT: _legacy_observed,
    LEAKAGE_CORRECTED_REFERENCE: _corrected_observed,
}


def verify(result: PreservedResult, *, rel_tolerance: float = 1e-9) -> dict[str, float]:
    """Re-read the archived run and confirm its headline numbers are unchanged.

    Raises :class:`EvidenceTampering` naming the drifted metric, so a failure
    says what moved rather than merely that something did.
    """
    observed = _READERS[result.label](_read_summary(result))

    for metric, expected in result.headline.items():
        actual = observed.get(metric)
        if actual is None:
            raise EvidenceTampering(f"{result.label}: {metric!r} is no longer present in the archive")
        if abs(actual - expected) > abs(expected) * rel_tolerance + 1e-12:
            raise EvidenceTampering(
                f"{result.label}: {metric} changed from {expected!r} to {actual!r}. "
                "Preserved results are immutable; archive a new dated run instead."
            )
    return observed


def verify_all() -> dict[str, dict[str, float]]:
    return {result.label: verify(result) for result in PRESERVED}


def summary_table() -> str:
    """A short human-readable statement of both results, for docs and reports."""
    lines = []
    for result in PRESERVED:
        lines.append(f"{result.label}  [{result.status}]")
        lines.append(f"  archive: research/runs/{result.run_dir}/  (from {result.source_commit})")
        for metric, value in result.headline.items():
            lines.append(f"    {metric:<38} {value:.6g}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "LEAKAGE_CORRECTED",
    "LEAKAGE_CORRECTED_REFERENCE",
    "LEGACY_INVALID",
    "LEGACY_INVALID_RESULT",
    "PRESERVED",
    "EvidenceTampering",
    "PreservedResult",
    "by_label",
    "summary_table",
    "verify",
    "verify_all",
]
