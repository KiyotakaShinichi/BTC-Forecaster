"""Scoring, with the null hypotheses stated rather than assumed.

The point metrics reuse `btc_forecaster.evaluation.metrics` -- MAE, RMSE, MASE,
sMAPE, pinball, Winkler, coverage all already exist and are already tested, and
a second implementation of MAE is a second thing that can disagree.

What is added here is the part specific to a direction forecast on a series with
an upward drift, where the obvious null is wrong. A2 made that mistake once: it
tested directional accuracy against 0.5, as though the question were whether a
coin is fair. It is not. On this series the base rate of up-days is around 53%,
so a model that always says "up" scores 53% and beats the coin without
containing any information at all.

So three baselines are reported beside every directional accuracy:

``base_rate``               the empirical fraction of up-days in the evaluation
                            block. What "always up" would score.
``train_constant_baseline`` what a model that always predicts the majority
                            direction **of the training rows** scores on the
                            evaluation block. The honest null, because it is
                            chosen without seeing the answer.
``random_walk``             the naive zero-return forecast, whose direction is
                            a tie at every step.

Balanced accuracy and MCC are reported for the same reason: both are invariant
to the class imbalance that makes raw accuracy flattering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..evaluation.metrics import (
    interval_coverage,
    mae,
    mean_interval_width,
    pinball_loss,
    rmse,
    smape,
    winkler_score,
)


def forecast_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean signed error. A model can have a good MAE and a persistent tilt."""
    return float(np.mean(predicted - actual))


def mase_against_naive(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MASE where the naive forecast is zero change.

    The zoo's target is a log return, so the naive one-step forecast is 0 and
    the scaling denominator is the mean absolute return itself.
    """
    denominator = float(np.mean(np.abs(actual)))
    if denominator <= 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / denominator)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of bars whose sign was called correctly.

    A prediction of exactly zero is scored as wrong rather than as half-right.
    Three of the baselines forecast a constant, and one of them forecasts
    exactly zero; crediting a tie would hand the naive model a free 50%.
    """
    called = np.sign(predicted)
    truth = np.sign(actual)
    return float(np.mean((called == truth) & (called != 0)))


def balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean of per-class recall. Invariant to the up-day imbalance."""
    up = actual > 0
    down = ~up
    predicted_up = predicted > 0
    if not up.any() or not down.any():
        return float("nan")
    return float(0.5 * (np.mean(predicted_up[up]) + np.mean(~predicted_up[down])))


def matthews_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MCC: +1 perfect, 0 no better than chance, -1 perfectly wrong.

    Reported because it is the one directional statistic that goes to zero for
    a constant predictor regardless of the class balance -- which is exactly
    what most of this zoo is.
    """
    truth = actual > 0
    called = predicted > 0
    tp = float(np.sum(truth & called))
    tn = float(np.sum(~truth & ~called))
    fp = float(np.sum(~truth & called))
    fn = float(np.sum(truth & ~called))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denominator)


def base_rate(actual: np.ndarray) -> float:
    """Fraction of up-days. What "always up" would score."""
    return float(np.mean(actual > 0))


def train_constant_baseline(train_actual: np.ndarray, actual: np.ndarray) -> float:
    """What always predicting the training majority direction scores.

    The honest null: the direction is chosen on the training rows, so it is
    available before the evaluation block is seen. Testing against the
    evaluation base rate instead would be testing against a number the model
    could not have known.
    """
    majority_up = float(np.mean(train_actual > 0)) >= 0.5
    return float(np.mean((actual > 0) == majority_up))


def brier_score(actual: np.ndarray, probability_up: np.ndarray) -> float:
    """Mean squared error of a probability forecast. A constant 0.5 scores 0.25."""
    return float(np.mean((probability_up - (actual > 0).astype(float)) ** 2))


def expected_calibration_error(
    actual: np.ndarray, probability_up: np.ndarray, *, bins: int = 10
) -> dict:
    """Reliability: does a stated 70% happen 70% of the time?

    Returns the error and the populated bins, because an ECE computed from two
    occupied bins is not a reliability curve and the reader has to be able to
    see that.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    outcome = (actual > 0).astype(float)
    total = 0.0
    populated = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (probability_up >= low) & (
            probability_up < high if i < bins - 1 else probability_up <= high
        )
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probability_up[mask].mean())
        frequency = float(outcome[mask].mean())
        total += count / len(actual) * abs(confidence - frequency)
        populated.append(
            {"bin": [float(low), float(high)], "n": count,
             "mean_probability": confidence, "observed_rate": frequency}
        )
    return {
        "expected_calibration_error": float(total),
        "populated_bins": len(populated),
        "bins": populated,
    }


#: Phase 17 asks for sMAPE "where meaningful", and on this target it is not.
#: sMAPE divides by |actual| + |forecast|, and the target is a log return that
#: crosses zero constantly: the denominator vanishes on quiet days, so the
#: statistic is dominated by the calmest bars and saturates near its 200%
#: ceiling for every model. Measured, then excluded, with the measurement kept
#: so the exclusion is checkable rather than asserted.
SMAPE_IS_MEANINGFUL = False
SMAPE_EXCLUSION_REASON = (
    "sMAPE is not reported as a headline metric: the target is a signed log "
    "return that crosses zero, so its denominator |actual| + |forecast| vanishes "
    "on quiet days and the statistic saturates near 200% for every model in the "
    "zoo -- including the naive baseline, which scores exactly 200%. It is "
    "computed and retained per model for inspection, not ranked on."
)


@dataclass(frozen=True)
class PointScores:
    mae: float
    rmse: float
    mase: float
    smape: float
    bias: float
    skill_vs_naive: float

    def as_dict(self) -> dict:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mase": self.mase,
            "bias": self.bias,
            "skill_vs_naive": self.skill_vs_naive,
            "smape": self.smape,
            "smape_is_meaningful": SMAPE_IS_MEANINGFUL,
        }


@dataclass(frozen=True)
class DirectionScores:
    accuracy: float
    balanced_accuracy: float
    mcc: float
    base_rate: float
    train_constant_baseline: float

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
            "base_rate": self.base_rate,
            "train_constant_baseline": self.train_constant_baseline,
            "beats_train_constant": self.accuracy > self.train_constant_baseline,
        }


@dataclass(frozen=True)
class ProbabilisticScores:
    mean_pinball: float
    pinball_by_level: dict[float, float]
    coverage: dict[str, float]
    mean_interval_width: float
    winkler: float
    brier: float | None
    calibration: dict | None

    def as_dict(self) -> dict:
        return {
            "mean_pinball": self.mean_pinball,
            "pinball_by_level": {str(k): v for k, v in self.pinball_by_level.items()},
            "coverage": self.coverage,
            "mean_interval_width": self.mean_interval_width,
            "winkler_90": self.winkler,
            "brier": self.brier,
            "calibration": self.calibration,
        }


def score_point(
    actual: np.ndarray, predicted: np.ndarray, *, naive_mae: float
) -> PointScores:
    model_mae = mae(actual, predicted)
    return PointScores(
        mae=model_mae,
        rmse=rmse(actual, predicted),
        mase=mase_against_naive(actual, predicted),
        smape=smape(actual, predicted),
        bias=forecast_bias(actual, predicted),
        skill_vs_naive=float(1.0 - model_mae / naive_mae) if naive_mae > 0 else float("nan"),
    )


def score_direction(
    actual: np.ndarray, predicted: np.ndarray, *, train_actual: np.ndarray
) -> DirectionScores:
    return DirectionScores(
        accuracy=directional_accuracy(actual, predicted),
        balanced_accuracy=balanced_accuracy(actual, predicted),
        mcc=matthews_correlation(actual, predicted),
        base_rate=base_rate(actual),
        train_constant_baseline=train_constant_baseline(train_actual, actual),
    )


def score_probabilistic(
    actual: np.ndarray,
    quantiles: dict[float, np.ndarray] | None,
    probability_up: np.ndarray | None,
) -> ProbabilisticScores | None:
    """None when the model declared neither capability. Never a filled-in default."""
    if quantiles is None and probability_up is None:
        return None

    by_level: dict[float, float] = {}
    coverage: dict[str, float] = {}
    width = float("nan")
    winkler = float("nan")
    if quantiles is not None:
        for level, values in sorted(quantiles.items()):
            by_level[level] = pinball_loss(actual, values, quantile=level)
        for low, high, label in ((0.05, 0.95, "90"), (0.1, 0.9, "80"), (0.25, 0.75, "50")):
            if low in quantiles and high in quantiles:
                coverage[f"nominal_{label}"] = interval_coverage(
                    actual, quantiles[low], quantiles[high]
                )
        if 0.05 in quantiles and 0.95 in quantiles:
            width = mean_interval_width(quantiles[0.05], quantiles[0.95])
            winkler = winkler_score(actual, quantiles[0.05], quantiles[0.95], level=0.90)

    return ProbabilisticScores(
        mean_pinball=float(np.mean(list(by_level.values()))) if by_level else float("nan"),
        pinball_by_level=by_level,
        coverage=coverage,
        mean_interval_width=width,
        winkler=winkler,
        brier=None if probability_up is None else brier_score(actual, probability_up),
        calibration=(
            None
            if probability_up is None
            else expected_calibration_error(actual, probability_up)
        ),
    )


def score_variance(actual: np.ndarray, variance: np.ndarray | None) -> dict | None:
    """QLIKE and MSE against the squared-return proxy.

    QLIKE is reported alongside MSE because the proxy is extremely noisy -- one
    observation per day -- and QLIKE is the standard loss that stays consistent
    under that noise where MSE does not.
    """
    if variance is None:
        return None
    realised = actual**2
    safe = np.maximum(variance, 1e-16)
    ratio = np.maximum(realised, 1e-16) / safe
    return {
        "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
        "mse_vs_squared_return": float(np.mean((realised - variance) ** 2)),
        "mean_forecast_volatility": float(np.mean(np.sqrt(variance))),
        "realised_volatility": float(np.sqrt(np.mean(realised))),
        "proxy_note": (
            "the squared return is an unbiased but extremely noisy proxy for "
            "realised variance; QLIKE is reported because it stays consistent "
            "under that noise and MSE does not"
        ),
    }


@dataclass(frozen=True)
class ModelScores:
    """Everything scored for one model, with absent capabilities left absent."""

    model_id: str
    point: PointScores
    direction: DirectionScores
    probabilistic: ProbabilisticScores | None = None
    variance: dict | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload: dict = {
            "model_id": self.model_id,
            "point": self.point.as_dict(),
            "direction": self.direction.as_dict(),
        }
        payload["probabilistic"] = (
            None if self.probabilistic is None else self.probabilistic.as_dict()
        )
        payload["variance"] = self.variance
        if self.extra:
            payload["extra"] = self.extra
        return payload


__all__ = [
    "SMAPE_EXCLUSION_REASON",
    "SMAPE_IS_MEANINGFUL",
    "DirectionScores",
    "ModelScores",
    "PointScores",
    "ProbabilisticScores",
    "balanced_accuracy",
    "base_rate",
    "brier_score",
    "directional_accuracy",
    "expected_calibration_error",
    "forecast_bias",
    "mase_against_naive",
    "matthews_correlation",
    "score_direction",
    "score_point",
    "score_probabilistic",
    "score_variance",
    "train_constant_baseline",
]
