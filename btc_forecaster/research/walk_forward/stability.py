"""The robustness gate, and the one decision A7 is allowed to reach.

A (model, horizon, window) is a ``ROBUST_RESEARCH_CANDIDATE`` only if it clears
**every** gate below, with the thresholds fixed in
:class:`~btc_forecaster.research.walk_forward.config.GateThresholds` before any
result existed:

``aggregate_skill``   MAE skill against naive strictly above the floor
``bh_better``         significantly better after Benjamini-Hochberg
``late_block``        positive skill in the last chronological block
``fold_majority``     positive in at least the preregistered share of folds
``practical``         skill at least the practical floor
``leakage``           the model passed the walk-forward adversaries
``resources``         every fold ran, within its declared resource class

A leakage check that was not run is a failure, not a pass: an unchecked model
has not shown it cannot see the future.

The benchmark then reaches one of three decisions:

``ROBUST_RESEARCH_CANDIDATE``
    At least one configuration cleared every gate. That merits a separate,
    preregistered study -- not a trade.

``FRAGILE_SIGNAL``
    Nothing cleared every gate, but something looked like a signal: positive
    skill with a one-sided raw p-value under the preregistered level, or
    significance that did not survive the other gates. Recorded with the gates
    it failed, so the fragility is specific.

``ROBUSTLY_UNINTERESTING``
    Nothing looked like a signal at all.

There is no fourth state. Nothing here can promote a model or enable trading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GateThresholds
from .scoring import ConfigScores
from .significance import Comparison

ROBUSTLY_UNINTERESTING = "ROBUSTLY_UNINTERESTING"
FRAGILE_SIGNAL = "FRAGILE_SIGNAL"
ROBUST_RESEARCH_CANDIDATE = "ROBUST_RESEARCH_CANDIDATE"
DECISIONS: tuple[str, ...] = (ROBUSTLY_UNINTERESTING, FRAGILE_SIGNAL, ROBUST_RESEARCH_CANDIDATE)

LEAKAGE_PASSED = "PASSED"
LEAKAGE_FAILED = "FAILED"
LEAKAGE_NOT_CHECKED = "NOT_CHECKED"

GATE_NAMES: tuple[str, ...] = (
    "aggregate_skill",
    "bh_better",
    "late_block",
    "fold_majority",
    "practical",
    "leakage",
    "resources",
)


@dataclass(frozen=True)
class GateResult:
    model_id: str
    horizon: int
    window: str
    checks: dict[str, bool]
    raw_signal: bool
    statistically_different: bool
    statistically_better: bool
    practically_useful: bool

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.model_id, self.horizon, self.window)

    @property
    def passed(self) -> bool:
        return all(self.checks[name] for name in GATE_NAMES)

    @property
    def failed(self) -> list[str]:
        return [name for name in GATE_NAMES if not self.checks[name]]

    @property
    def fragile(self) -> bool:
        return not self.passed and (self.raw_signal or self.statistically_better)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "horizon": self.horizon,
            "window": self.window,
            "checks": {name: self.checks[name] for name in GATE_NAMES},
            "failed": self.failed,
            "passed": self.passed,
            "raw_signal": self.raw_signal,
            "statistically_different": self.statistically_different,
            "statistically_better": self.statistically_better,
            "practically_useful": self.practically_useful,
        }


def evaluate_gates(
    scores: list[ConfigScores],
    comparisons: list[Comparison],
    *,
    gates: GateThresholds,
    late_block: str,
    leakage: dict[str, str],
    clean: dict[tuple[str, int, str], bool],
    baseline: str,
) -> list[GateResult]:
    """Apply every gate to every non-baseline configuration, in key order."""
    by_key = {c.key: c for c in comparisons}
    results: list[GateResult] = []
    for score in scores:
        if score.model_id == baseline:
            continue
        key = (score.model_id, score.horizon, score.window)
        comparison = by_key.get(key)
        skill = score.skill_vs_naive
        late = score.late_block_skill(late_block)
        different = bool(comparison and comparison.statistically_different(gates.alpha))
        better = bool(comparison and comparison.statistically_better(gates.alpha))
        raw_signal = bool(
            comparison
            and comparison.testable
            and np.isfinite(skill)
            and skill > 0
            and comparison.p_value_better is not None
            and comparison.p_value_better < gates.raw_signal_alpha
        )
        checks = {
            "aggregate_skill": bool(np.isfinite(skill) and skill > gates.min_aggregate_skill),
            "bh_better": better,
            "late_block": bool(np.isfinite(late) and late > gates.min_late_block_skill),
            "fold_majority": bool(
                np.isfinite(score.positive_fold_fraction)
                and score.positive_fold_fraction >= gates.min_positive_fold_fraction
            ),
            "practical": bool(np.isfinite(skill) and skill >= gates.practical_min_skill),
            "leakage": leakage.get(score.model_id, LEAKAGE_NOT_CHECKED) == LEAKAGE_PASSED,
            "resources": bool(clean.get(key, False)),
        }
        results.append(
            GateResult(
                model_id=score.model_id,
                horizon=score.horizon,
                window=score.window,
                checks=checks,
                raw_signal=raw_signal,
                statistically_different=different,
                statistically_better=better,
                practically_useful=better and checks["practical"],
            )
        )
    return results


def decide(results: list[GateResult]) -> dict:
    """The benchmark's single decision, with the configurations behind it."""
    candidates = [r for r in results if r.passed]
    fragile = [r for r in results if r.fragile]
    if candidates:
        decision = ROBUST_RESEARCH_CANDIDATE
    elif fragile:
        decision = FRAGILE_SIGNAL
    else:
        decision = ROBUSTLY_UNINTERESTING
    return {
        "decision": decision,
        "configurations_evaluated": len(results),
        "candidates": [r.as_dict() for r in candidates],
        "fragile_signals": [r.as_dict() for r in fragile],
        "gate_failure_counts": {
            name: sum(1 for r in results if not r.checks[name]) for name in GATE_NAMES
        },
        "statistically_different": sum(1 for r in results if r.statistically_different),
        "statistically_better": sum(1 for r in results if r.statistically_better),
        "practically_useful": sum(1 for r in results if r.practically_useful),
        "promoted_models": [],
        "live_trading_enabled": False,
    }


__all__ = [
    "DECISIONS",
    "FRAGILE_SIGNAL",
    "GATE_NAMES",
    "LEAKAGE_FAILED",
    "LEAKAGE_NOT_CHECKED",
    "LEAKAGE_PASSED",
    "ROBUSTLY_UNINTERESTING",
    "ROBUST_RESEARCH_CANDIDATE",
    "GateResult",
    "decide",
    "evaluate_gates",
]
