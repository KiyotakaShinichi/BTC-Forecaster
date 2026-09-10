"""Scoring walk-forward records: point, direction, and skill across time.

Every number is computed on **paired origins**: a model is scored only where
the naive baseline was scored too, at the same origins, against the same
realised return. A model whose fold failed loses those origins from its score
rather than being compared on a different, possibly kinder, stretch.

Point metrics are the ones A6 reported, at every horizon: MAE, RMSE, MASE and
skill against the naive forecast, and bias. Skill is the comparable quantity
across horizons; MAE is not, because a 30-bar return is far more variable than
a 1-bar one.

Direction reuses A6's scorers and its conventions: a forecast of exactly zero
is a wrong call, not half a right one, and the null is the constant direction
the training targets favoured -- not a coin.

None of the curated models declares quantiles, a direction probability or a
conditional variance, so pinball loss, Brier score, calibration error and QLIKE
are recorded as not declared rather than computed from something invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..metrics import balanced_accuracy, directional_accuracy, forecast_bias, matthews_correlation
from .origins import OriginSchedule

#: Scores that need a capability none of the curated models declares.
NOT_DECLARED = {
    "pinball_loss": "no curated model declares QUANTILES",
    "brier_score": "no curated model declares DIRECTION_PROBABILITY",
    "calibration_error": "no curated model declares DIRECTION_PROBABILITY",
    "qlike": "no curated model declares VARIANCE",
}

KEY: tuple[str, ...] = ("model_id", "horizon", "window")


def mae_skill(model_errors: np.ndarray, baseline_errors: np.ndarray) -> float:
    """``1 - MAE(model) / MAE(baseline)`` over the same origins."""
    denominator = float(np.sum(np.abs(baseline_errors)))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(np.abs(model_errors))) / denominator


def paired_with_baseline(records: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Each model's rows beside the baseline's, on (horizon, window, origin).

    Inner join, so an origin either model is missing is dropped from both.
    """
    base = records.loc[records["model_id"] == baseline, ["horizon", "window", "origin", "predicted"]]
    base = base.rename(columns={"predicted": "baseline_predicted"})
    merged = records.merge(base, on=["horizon", "window", "origin"], how="inner", validate="many_to_one")
    return merged.sort_values([*KEY, "origin"], kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class ConfigScores:
    """One (model, horizon, window), scored."""

    model_id: str
    horizon: int
    window: str
    n_origins: int
    mae: float
    rmse: float
    mase: float
    skill_vs_naive: float
    bias: float
    directional_accuracy: float
    balanced_accuracy: float
    mcc: float
    constant_direction_accuracy: float
    fold_skill: dict[int, float]
    block_skill: dict[str, float]

    @property
    def beats_constant_direction(self) -> bool:
        return bool(self.directional_accuracy > self.constant_direction_accuracy)

    @property
    def positive_fold_fraction(self) -> float:
        values = [v for v in self.fold_skill.values() if np.isfinite(v)]
        return float(np.mean([v > 0 for v in values])) if values else float("nan")

    def late_block_skill(self, late: str) -> float:
        return self.block_skill.get(late, float("nan"))

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "horizon": self.horizon,
            "window": self.window,
            "n_origins": self.n_origins,
            "point": {
                "mae": self.mae,
                "rmse": self.rmse,
                "mase": self.mase,
                "skill_vs_naive": self.skill_vs_naive,
                "bias": self.bias,
            },
            "direction": {
                "accuracy": self.directional_accuracy,
                "balanced_accuracy": self.balanced_accuracy,
                "mcc": self.mcc,
                "constant_direction_accuracy": self.constant_direction_accuracy,
                "beats_constant_direction": self.beats_constant_direction,
            },
            "stability": {
                "fold_skill": {str(k): v for k, v in sorted(self.fold_skill.items())},
                "block_skill": dict(self.block_skill),
                "positive_fold_fraction": self.positive_fold_fraction,
            },
            "not_declared": NOT_DECLARED,
        }


def score_configs(records: pd.DataFrame, schedule: OriginSchedule, *, baseline: str) -> list[ConfigScores]:
    """Score every (model, horizon, window) against the baseline, in key order."""
    paired = paired_with_baseline(records, baseline)
    fold_of = schedule.fold_of()
    block_of = schedule.block_of()
    paired["block"] = paired["origin"].map(block_of)
    if paired["block"].isna().any():
        raise ValueError("a record has an origin outside the schedule")

    scores: list[ConfigScores] = []
    for key, group in paired.groupby(list(KEY), sort=True):
        model_id, horizon, window = key
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        base = group["baseline_predicted"].to_numpy(dtype=float)
        errors, base_errors = actual - predicted, actual - base
        base_mae = float(np.mean(np.abs(base_errors)))
        up = group["train_direction_up"].to_numpy(dtype=bool)

        fold_skill: dict[int, float] = {}
        for fold, rows in group.groupby(group["origin"].map(fold_of), sort=True):
            fold_skill[int(fold)] = mae_skill(
                rows["actual"].to_numpy() - rows["predicted"].to_numpy(),
                rows["actual"].to_numpy() - rows["baseline_predicted"].to_numpy(),
            )
        block_skill: dict[str, float] = {}
        for name in schedule.block_names:
            rows = group[group["block"] == name]
            block_skill[name] = (
                mae_skill(
                    rows["actual"].to_numpy() - rows["predicted"].to_numpy(),
                    rows["actual"].to_numpy() - rows["baseline_predicted"].to_numpy(),
                )
                if len(rows)
                else float("nan")
            )

        scores.append(
            ConfigScores(
                model_id=str(model_id),
                horizon=int(horizon),
                window=str(window),
                n_origins=len(group),
                mae=float(np.mean(np.abs(errors))),
                rmse=float(np.sqrt(np.mean(errors**2))),
                mase=float(np.mean(np.abs(errors)) / base_mae) if base_mae > 0 else float("nan"),
                skill_vs_naive=mae_skill(errors, base_errors),
                bias=forecast_bias(actual, predicted),
                directional_accuracy=directional_accuracy(actual, predicted),
                balanced_accuracy=balanced_accuracy(actual, predicted),
                mcc=matthews_correlation(actual, predicted),
                constant_direction_accuracy=float(np.mean((actual > 0) == up)),
                fold_skill=fold_skill,
                block_skill=block_skill,
            )
        )
    return scores


__all__ = [
    "KEY",
    "NOT_DECLARED",
    "ConfigScores",
    "mae_skill",
    "paired_with_baseline",
    "score_configs",
]
