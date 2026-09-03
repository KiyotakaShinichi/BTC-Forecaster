"""Empirical probability calibration and frozen confidence bands."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp

import numpy as np

from .contracts import ConfidenceBand


@dataclass(frozen=True)
class CalibrationDiagnostics:
    brier_score: float
    expected_calibration_error: float
    bins: tuple[tuple[float, float, int], ...]


@dataclass(frozen=True)
class PlattCalibrator:
    intercept: float
    slope: float
    version: str
    sample_size: int

    def predict(self, raw_probability: float) -> float:
        if not 0 <= raw_probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        return 1.0 / (1.0 + exp(-(self.intercept + self.slope * raw_probability)))


def fit_platt(
    probabilities: Sequence[float], outcomes: Sequence[int], *, version: str, min_samples: int = 50
) -> PlattCalibrator:
    """Fit a regularized two-parameter logistic map; reject tiny samples."""
    if len(probabilities) != len(outcomes) or len(probabilities) < min_samples:
        raise ValueError(f"calibration requires at least {min_samples} paired samples")
    x = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if np.any((x < 0) | (x > 1)) or np.any((y != 0) & (y != 1)) or len(np.unique(y)) < 2:
        raise ValueError("invalid or single-class calibration sample")
    beta = np.zeros(2)
    design = np.column_stack([np.ones(len(x)), x])
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(design @ beta, -30, 30)))
        weights = np.maximum(p * (1 - p), 1e-8)
        hessian = design.T @ (weights[:, None] * design) + np.diag([1e-6, 1e-2])
        step = np.linalg.solve(hessian, design.T @ (y - p) - np.array([0, 1e-2 * beta[1]]))
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return PlattCalibrator(float(beta[0]), float(beta[1]), version, len(x))


def diagnostics(
    probabilities: Iterable[float], outcomes: Iterable[int], *, n_bins: int = 10
) -> CalibrationDiagnostics:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if len(p) == 0 or len(p) != len(y):
        raise ValueError("non-empty paired observations required")
    rows: list[tuple[float, float, int]] = []
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for index in range(n_bins):
        mask = (p >= edges[index]) & (p < edges[index + 1] if index < n_bins - 1 else p <= 1)
        if mask.any():
            predicted, observed, count = (
                float(p[mask].mean()),
                float(y[mask].mean()),
                int(mask.sum()),
            )
            rows.append((predicted, observed, count))
            ece += count / len(p) * abs(predicted - observed)
    return CalibrationDiagnostics(float(np.mean((p - y) ** 2)), ece, tuple(rows))


def confidence_band(probability: float) -> ConfidenceBand:
    """Frozen participation bands based on distance from chance."""
    edge = abs(probability - 0.5)
    if edge < 0.10:
        return ConfidenceBand.LOW
    if edge < 0.20:
        return ConfidenceBand.MODERATE
    return ConfidenceBand.HIGH
