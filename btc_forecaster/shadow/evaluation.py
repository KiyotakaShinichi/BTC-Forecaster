"""Paired forward metrics and conservative promotion readiness."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..paper.calibration import diagnostics
from .contracts import Readiness
from .ledger import EvidenceLedger

MIN_FORWARD_FORECASTS = 100
MIN_INTERVAL_FORECASTS = 50


def _mcc(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / denominator if denominator else None


def forward_report(ledger: EvidenceLedger) -> dict[str, Any]:
    forecasts = ledger.forecast_payloads()
    outcomes = {item["forecast_id"]: item for item in ledger.outcome_payloads()}
    by_model: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    issued: dict[str, int] = defaultdict(int)
    for forecast in forecasts:
        issued[forecast["model_id"]] += 1
        if forecast["forecast_id"] in outcomes:
            by_model[forecast["model_id"]].append((forecast, outcomes[forecast["forecast_id"]]))
    baselines = {
        forecast["forecast_origin"]: outcomes[forecast["forecast_id"]]
        for forecast in forecasts
        if forecast["model_id"] == "random_walk" and forecast["forecast_id"] in outcomes
    }
    models: dict[str, Any] = {}
    for model_id in sorted(issued):
        pairs = by_model[model_id]
        errors = np.asarray([abs(outcome["point_error"]) for _, outcome in pairs], dtype=float)
        squared = np.asarray([outcome["squared_error"] for _, outcome in pairs], dtype=float)
        actual_direction = np.asarray([outcome["actual_direction"] > 0 for _, outcome in pairs])
        predicted_direction = np.asarray([forecast["expected_return"] > 0 for forecast, _ in pairs])
        paired = [
            (forecast, outcome, baselines[forecast["forecast_origin"]])
            for forecast, outcome in pairs
            if forecast["forecast_origin"] in baselines
        ]
        baseline_mae = (
            float(np.mean([abs(base["point_error"]) for _, _, base in paired])) if paired else None
        )
        candidate_mae = (
            float(np.mean([abs(outcome["point_error"]) for _, outcome, _ in paired]))
            if paired
            else None
        )
        probabilities = [
            (forecast["raw_direction_probability"], outcome["actual_direction"] > 0)
            for forecast, outcome in pairs
            if forecast["raw_direction_probability"] is not None
        ]
        calibration = None
        if probabilities:
            report = diagnostics(
                [float(item[0]) for item in probabilities], [int(item[1]) for item in probabilities]
            )
            calibration = {
                "brier_score": report.brier_score,
                "ece": report.expected_calibration_error,
                "bins": report.bins,
            }
        n = len(pairs)
        if n < MIN_FORWARD_FORECASTS:
            readiness = Readiness.INSUFFICIENT_FORWARD_EVIDENCE
        elif baseline_mae is None or candidate_mae is None or candidate_mae >= baseline_mae:
            readiness = Readiness.FAILED_FORWARD_VALIDATION
        else:
            readiness = Readiness.READY_FOR_PROMOTION_REVIEW
        models[model_id] = {
            "issued": issued[model_id],
            "scored": n,
            "pending": issued[model_id] - n,
            "coverage": n / issued[model_id] if issued[model_id] else 0.0,
            "mae": float(errors.mean()) if n else None,
            "rmse": float(np.sqrt(squared.mean())) if n else None,
            "paired_baseline_n": len(paired),
            "paired_mae": candidate_mae,
            "baseline_mae": baseline_mae,
            "mae_skill_vs_baseline": None
            if baseline_mae is None or baseline_mae == 0 or candidate_mae is None
            else 1 - candidate_mae / baseline_mae,
            "directional_accuracy": float(np.mean(actual_direction == predicted_direction))
            if n
            else None,
            "base_rate": float(np.mean(actual_direction)) if n else None,
            "balanced_accuracy": _balanced_accuracy(actual_direction, predicted_direction)
            if n
            else None,
            "mcc": _mcc(actual_direction.astype(int), predicted_direction.astype(int))
            if n
            else None,
            "forecast_bias": float(np.mean([outcome["point_error"] for _, outcome in pairs]))
            if n
            else None,
            "calibration": calibration,
            "interval_state": "AVAILABLE"
            if n >= MIN_INTERVAL_FORECASTS
            else "INSUFFICIENT_N_FOR_INTERVAL",
            "promotion_readiness": readiness.value,
        }
    return {"minimum_forward_forecasts": MIN_FORWARD_FORECASTS, "models": models}


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    rates = []
    for label in (False, True):
        mask = actual == label
        if mask.any():
            rates.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(rates)) if len(rates) == 2 else None


def block_bootstrap_skill(
    candidate_errors: list[float],
    baseline_errors: list[float],
    *,
    block_size: int = 5,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float] | str:
    if len(candidate_errors) != len(baseline_errors):
        raise ValueError("paired errors required")
    if len(candidate_errors) < MIN_INTERVAL_FORECASTS:
        return "INSUFFICIENT_N_FOR_INTERVAL"
    differences = np.asarray(baseline_errors) - np.asarray(candidate_errors)
    rng = np.random.default_rng(seed)
    starts = np.arange(len(differences) - block_size + 1)
    means = []
    for _ in range(samples):
        chosen = rng.choice(starts, int(np.ceil(len(differences) / block_size)), replace=True)
        sample = np.concatenate([differences[index : index + block_size] for index in chosen])[
            : len(differences)
        ]
        means.append(float(sample.mean()))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
