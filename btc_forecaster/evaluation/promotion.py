"""Champion / challenger promotion policy (A2.21).

**This policy is defined before the outer-fold aggregates are inspected.** That
ordering is the whole point. A rule written after seeing the numbers is not a
rule, it is a description of the outcome you already preferred, and it will be
adjusted again next time. Writing it first means a challenger can fail.

The criteria, and why each is present
-------------------------------------
A challenger is PROMOTED only if it clears every one of these:

1. **Positive MAE skill against the naive baseline.** Not against the incumbent:
   beating a model that itself loses to a random walk is worthless. This is the
   gate the legacy hybrid fails at -1.64.
2. **Skill that survives its own uncertainty.** The bootstrap interval on the
   loss differential must exclude zero. A point estimate of +0.02 with an
   interval spanning [-0.15, +0.19] is noise.
3. **Stability.** The worst fold must not be catastrophic relative to the
   typical one. A model with the best mean and one blow-up is not deployable.
4. **Interval calibration.** Coverage within tolerance of nominal. A model whose
   95% band covers 23% of outcomes is not merely imprecise, it is misleading.
5. **Proportionate cost.** A large compute multiple demands a correspondingly
   large improvement, not a marginal one.

Anything that clears (1) but fails a later criterion is INCONCLUSIVE, not
REJECTED: the distinction between "we showed it does not work" and "we could not
show it works" is worth keeping. Failing (1) outright is REJECTED.

**A2 promotes nothing into production.** The decision this produces is a
research verdict; there is no production model to replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Decision(str, Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class PromotionPolicy:
    """Thresholds, fixed before results are inspected."""

    #: Minimum mean MAE skill against the naive baseline. Deliberately not zero:
    #: an improvement of 0.001 is not worth a model.
    min_mae_skill: float = 0.02

    #: The bootstrap interval on skill must exclude zero.
    require_interval_excludes_zero: bool = True

    #: Worst fold may not exceed this multiple of the median fold.
    max_worst_to_median: float = 3.0

    #: Observed interval coverage must be within this of nominal.
    max_coverage_error: float = 0.15

    #: Above this compute multiple relative to the cheapest model, the skill
    #: requirement is raised to `expensive_min_mae_skill`.
    expensive_cost_multiple: float = 20.0
    expensive_min_mae_skill: float = 0.10

    #: Minimum folds before any promotion is considered at all.
    min_folds: int = 10

    def to_dict(self) -> dict:
        return {
            "min_mae_skill": self.min_mae_skill,
            "require_interval_excludes_zero": self.require_interval_excludes_zero,
            "max_worst_to_median": self.max_worst_to_median,
            "max_coverage_error": self.max_coverage_error,
            "expensive_cost_multiple": self.expensive_cost_multiple,
            "expensive_min_mae_skill": self.expensive_min_mae_skill,
            "min_folds": self.min_folds,
            "declared": "before outer-fold aggregates were inspected (A2.21)",
        }


@dataclass(frozen=True)
class PromotionVerdict:
    """One model's decision, with every criterion's outcome recorded."""

    model: str
    decision: Decision
    mae_skill: float
    skill_ci: tuple[float, float] | None
    worst_to_median: float
    coverage_error: float
    cost_multiple: float
    n_folds: int
    passed: tuple[str, ...] = field(default=())
    failed: tuple[str, ...] = field(default=())
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "decision": self.decision.value,
            "mae_skill": self.mae_skill,
            "skill_ci_lower": None if self.skill_ci is None else self.skill_ci[0],
            "skill_ci_upper": None if self.skill_ci is None else self.skill_ci[1],
            "worst_to_median": self.worst_to_median,
            "coverage_error": self.coverage_error,
            "cost_multiple": self.cost_multiple,
            "n_folds": self.n_folds,
            "passed": list(self.passed),
            "failed": list(self.failed),
            "rationale": self.rationale,
        }


def _finite(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def evaluate_promotion(
    model: str,
    *,
    mae_skill: float,
    n_folds: int,
    skill_ci: tuple[float, float] | None = None,
    worst_to_median: float = float("nan"),
    coverage_error: float = float("nan"),
    cost_multiple: float = 1.0,
    policy: PromotionPolicy | None = None,
) -> PromotionVerdict:
    """Apply the policy to one challenger. Criteria are evaluated in order."""
    policy = policy or PromotionPolicy()
    passed: list[str] = []
    failed: list[str] = []

    skill = _finite(mae_skill)
    worst_ratio = _finite(worst_to_median)
    coverage = _finite(coverage_error)
    cost = _finite(cost_multiple, 1.0)

    # Gate 0: enough folds to say anything.
    if n_folds < policy.min_folds:
        return PromotionVerdict(
            model=model,
            decision=Decision.INCONCLUSIVE,
            mae_skill=skill,
            skill_ci=skill_ci,
            worst_to_median=worst_ratio,
            coverage_error=coverage,
            cost_multiple=cost,
            n_folds=n_folds,
            failed=("min_folds",),
            rationale=(
                f"{n_folds} folds is below the {policy.min_folds} required to consider "
                "promotion at all. Not a statement about the model."
            ),
        )
    passed.append("min_folds")

    # Gate 1: skill against the naive baseline. Failing this is a REJECT.
    required_skill = (
        policy.expensive_min_mae_skill
        if cost > policy.expensive_cost_multiple
        else policy.min_mae_skill
    )
    if not np.isfinite(skill) or skill < required_skill:
        return PromotionVerdict(
            model=model,
            decision=Decision.REJECT,
            mae_skill=skill,
            skill_ci=skill_ci,
            worst_to_median=worst_ratio,
            coverage_error=coverage,
            cost_multiple=cost,
            n_folds=n_folds,
            passed=tuple(passed),
            failed=("mae_skill",),
            rationale=(
                f"MAE skill {skill:.4f} does not clear the {required_skill:.2f} required"
                + (
                    f" at a {cost:.0f}x compute multiple. "
                    if cost > policy.expensive_cost_multiple
                    else ". "
                )
                + "A model that does not beat the naive baseline is rejected regardless "
                "of its other properties."
            ),
        )
    passed.append("mae_skill")

    # Gates 2-4: failing any of these is INCONCLUSIVE, not REJECT.
    if policy.require_interval_excludes_zero:
        if skill_ci is None:
            failed.append("skill_interval_missing")
        elif skill_ci[0] <= 0.0 <= skill_ci[1]:
            failed.append("skill_interval_includes_zero")
        else:
            passed.append("skill_interval_excludes_zero")

    if not np.isfinite(worst_ratio):
        failed.append("stability_unknown")
    elif worst_ratio > policy.max_worst_to_median:
        failed.append("stability")
    else:
        passed.append("stability")

    if not np.isfinite(coverage):
        failed.append("calibration_unknown")
    elif coverage > policy.max_coverage_error:
        failed.append("interval_calibration")
    else:
        passed.append("interval_calibration")

    if failed:
        return PromotionVerdict(
            model=model,
            decision=Decision.INCONCLUSIVE,
            mae_skill=skill,
            skill_ci=skill_ci,
            worst_to_median=worst_ratio,
            coverage_error=coverage,
            cost_multiple=cost,
            n_folds=n_folds,
            passed=tuple(passed),
            failed=tuple(failed),
            rationale=(
                f"Beat the baseline on mean MAE (skill {skill:.4f}) but did not clear "
                f"{', '.join(failed)}. Not shown to fail, not shown to work."
            ),
        )

    return PromotionVerdict(
        model=model,
        decision=Decision.PROMOTE,
        mae_skill=skill,
        skill_ci=skill_ci,
        worst_to_median=worst_ratio,
        coverage_error=coverage,
        cost_multiple=cost,
        n_folds=n_folds,
        passed=tuple(passed),
        rationale=(
            f"Cleared every criterion: MAE skill {skill:.4f} with an interval excluding "
            f"zero, worst fold {worst_ratio:.2f}x the median, coverage error "
            f"{coverage:.3f}, {cost:.1f}x compute. Promoted as a research result only -- "
            "A2 replaces no production model."
        ),
    )


def decide_all(
    summary: pd.DataFrame,
    skill_table: pd.DataFrame,
    stability: pd.DataFrame,
    costs: pd.DataFrame,
    *,
    baseline: str,
    n_folds: int,
    skill_intervals: dict[str, tuple[float, float]] | None = None,
    policy: PromotionPolicy | None = None,
) -> pd.DataFrame:
    """Apply the policy to every model in the benchmark."""
    policy = policy or PromotionPolicy()
    skill_column = f"mae_skill_vs_{baseline}"
    intervals = skill_intervals or {}

    verdicts = []
    for model in summary.index:
        if model == baseline:
            continue

        verdicts.append(
            evaluate_promotion(
                str(model),
                mae_skill=(
                    skill_table.loc[model, skill_column]
                    if skill_column in skill_table.columns and model in skill_table.index
                    else float("nan")
                ),
                n_folds=n_folds,
                skill_ci=intervals.get(str(model)),
                worst_to_median=(
                    stability.loc[model, "worst_to_median"]
                    if not stability.empty and model in stability.index
                    else float("nan")
                ),
                coverage_error=(
                    summary.loc[model, "coverage_error"]
                    if "coverage_error" in summary.columns
                    else float("nan")
                ),
                cost_multiple=(
                    costs.loc[model, "cost_multiple_vs_cheapest"]
                    if not costs.empty
                    and "cost_multiple_vs_cheapest" in costs.columns
                    and model in costs.index
                    else 1.0
                ),
                policy=policy,
            ).to_dict()
        )

    if not verdicts:
        return pd.DataFrame()
    return pd.DataFrame(verdicts).set_index("model")


__all__ = [
    "Decision",
    "PromotionPolicy",
    "PromotionVerdict",
    "decide_all",
    "evaluate_promotion",
]
