"""Stability, failure analysis and resource cost (A2.16-A2.19).

Three questions a mean cannot answer:

* **Is the average representative?** A model with the best mean MAE and one
  catastrophic fold is not better than a slightly worse model that never breaks.
  Reporting mean alone hides exactly the behaviour that matters for deployment.
* **Where does it fail?** The largest errors are not evenly distributed. Knowing
  they cluster in high-volatility regimes, or at long horizons, or on specific
  dates, is what turns a bad number into a diagnosis.
* **What did it cost?** A 1% error reduction bought with a 50x fit time and a
  hyperparameter search is a different proposition from one that is free.

Deliberately quantitative only. Track B will later supply point-in-time event
context for the failure dates; A2 stops at identifying them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StabilityProfile:
    """Distribution of a metric across folds for one model."""

    model: str
    metric: str
    n_folds: int
    mean: float
    median: float
    std: float
    best: float
    worst: float
    best_fold: int
    worst_fold: int
    iqr: float
    worst_to_median: float
    lower_is_better: bool

    @property
    def coefficient_of_variation(self) -> float:
        """Dispersion relative to level. Comparable across models and metrics."""
        if not np.isfinite(self.mean) or abs(self.mean) < 1e-12:
            return float("nan")
        return float(self.std / abs(self.mean))

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "metric": self.metric,
            "n_folds": self.n_folds,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "iqr": self.iqr,
            "best": self.best,
            "worst": self.worst,
            "best_fold": self.best_fold,
            "worst_fold": self.worst_fold,
            "worst_to_median": self.worst_to_median,
            "coefficient_of_variation": self.coefficient_of_variation,
        }


def stability_profile(
    per_fold: pd.DataFrame,
    *,
    metric: str = "mae",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Per-model distribution of ``metric`` across folds.

    ``worst_to_median`` is the headline stability number: how much worse the
    worst fold was than the typical one. A model at 1.2 degrades gracefully; one
    at 8 has a failure mode that a mean will not show you.
    """
    required = {"model", "fold", metric}
    missing = required - set(per_fold.columns)
    if missing:
        raise ValueError(f"per-fold table is missing column(s): {sorted(missing)}")

    scored = per_fold[per_fold["error"].isna()] if "error" in per_fold.columns else per_fold

    profiles: list[StabilityProfile] = []
    for model, group in scored.groupby("model", sort=True):
        values = group[metric].astype(float)
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue

        best_idx = finite.idxmin() if lower_is_better else finite.idxmax()
        worst_idx = finite.idxmax() if lower_is_better else finite.idxmin()
        median = float(finite.median())
        worst = float(finite.loc[worst_idx])

        profiles.append(
            StabilityProfile(
                model=str(model),
                metric=metric,
                n_folds=len(finite),
                mean=float(finite.mean()),
                median=median,
                std=float(finite.std(ddof=0)),
                best=float(finite.loc[best_idx]),
                worst=worst,
                best_fold=int(group.loc[best_idx, "fold"]),
                worst_fold=int(group.loc[worst_idx, "fold"]),
                iqr=float(finite.quantile(0.75) - finite.quantile(0.25)),
                worst_to_median=(
                    float(worst / median) if abs(median) > 1e-12 else float("nan")
                ),
                lower_is_better=lower_is_better,
            )
        )

    if not profiles:
        return pd.DataFrame()
    return pd.DataFrame([p.to_dict() for p in profiles]).set_index("model")


def stability_by_period(
    per_fold: pd.DataFrame,
    *,
    metric: str = "mae",
    freq: str = "YE",
) -> pd.DataFrame:
    """The same metric grouped by calendar period of the test window.

    Catches a model that was fine until a regime change and never recovered --
    a pattern invisible in a fold-indexed view when folds are evenly spaced.
    """
    if "test_start" not in per_fold.columns:
        raise ValueError("per-fold table needs a 'test_start' column")

    scored = per_fold[per_fold["error"].isna()] if "error" in per_fold.columns else per_fold
    if scored.empty:
        return pd.DataFrame()

    working = scored.copy()
    # to_period() drops tz information and warns about it. The bars are UTC by
    # contract, so converting explicitly first is lossless and silences a
    # warning that would otherwise appear in every benchmark run.
    naive = pd.DatetimeIndex(working["test_start"]).tz_convert(None)
    working["period"] = naive.to_period(freq[0] if freq.endswith("E") else freq)

    return (
        working.groupby(["model", "period"], sort=True)[metric]
        .agg(["size", "mean", "median", "std", "min", "max"])
        .rename(columns={"size": "n_folds"})
    )


def largest_failures(
    records: pd.DataFrame,
    *,
    model: str | None = None,
    top_n: int = 20,
    origin_regimes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The worst individual predictions, with the context to interpret them.

    Reports forecast origin, target date, horizon step, actual and predicted
    move, and -- when ``origin_regimes`` is supplied -- the regime that was
    identifiable at the origin. That last column is what makes a miss
    diagnosable rather than merely large.
    """
    required = {"actual", "predicted", "origin_close"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records is missing column(s): {sorted(missing)}")

    data = records if model is None else records[records["model"] == model]
    if data.empty:
        return pd.DataFrame()

    working = data.copy()
    working["abs_error"] = (working["actual"] - working["predicted"]).abs()
    working["actual_move"] = working["actual"] / working["origin_close"] - 1.0
    working["predicted_move"] = working["predicted"] / working["origin_close"] - 1.0
    working["direction_hit"] = (working["predicted_move"] > 0) == (working["actual_move"] > 0)
    working["relative_error"] = working["abs_error"] / working["actual"].abs()

    if origin_regimes is not None and "origin" in working.columns:
        keep = [c for c in ("origin", "regime", "realised_vol") if c in origin_regimes.columns]
        working = working.merge(origin_regimes[keep], on="origin", how="left")

    columns = [
        c
        for c in (
            "model", "fold", "origin", "date", "step",
            "origin_close", "actual", "predicted",
            "actual_move", "predicted_move", "direction_hit",
            "abs_error", "relative_error", "regime", "realised_vol",
        )
        if c in working.columns
    ]
    return working.nlargest(top_n, "abs_error")[columns].reset_index(drop=True)


def failure_summary(records: pd.DataFrame, *, model: str, quantile: float = 0.95) -> dict:
    """How concentrated a model's error is in its worst predictions.

    ``share_of_total_error`` above roughly 0.3 means the headline MAE is mostly
    a few disasters rather than typical behaviour, which changes what the number
    means and what would fix it.
    """
    data = records[records["model"] == model]
    if data.empty:
        return {}

    errors = (data["actual"] - data["predicted"]).abs().to_numpy(dtype=float)
    threshold = float(np.quantile(errors, quantile))
    tail = errors[errors >= threshold]

    actual_move = (data["actual"] / data["origin_close"] - 1.0).to_numpy(dtype=float)
    predicted_move = (data["predicted"] / data["origin_close"] - 1.0).to_numpy(dtype=float)
    tail_mask = errors >= threshold

    return {
        "model": model,
        "n_predictions": int(len(errors)),
        "quantile": quantile,
        "error_threshold": threshold,
        "n_in_tail": int(len(tail)),
        "tail_mean_error": float(tail.mean()) if len(tail) else float("nan"),
        "overall_mean_error": float(errors.mean()),
        "share_of_total_error": float(tail.sum() / errors.sum()) if errors.sum() > 0 else float("nan"),
        "tail_directional_accuracy": (
            float(np.mean((predicted_move[tail_mask] > 0) == (actual_move[tail_mask] > 0)))
            if tail_mask.any()
            else float("nan")
        ),
        "overall_directional_accuracy": float(
            np.mean((predicted_move > 0) == (actual_move > 0))
        ),
    }


def residual_diagnostics(records: pd.DataFrame, *, model: str, max_lag: int = 10) -> dict:
    """Autocorrelation, heteroskedasticity, bias and shape of a model's residuals.

    Interpreted, not dispatched on (A2.18). A single rejected test is not grounds
    for discarding a model: financial forecast residuals essentially always
    reject normality and usually show ARCH effects, and neither is a defect in
    the mean model. What matters is the pattern across all four.
    """
    from ..diagnostics.suite import (
        arch_lm_test,
        jarque_bera_test,
        ljung_box_test,
    )

    data = records[records["model"] == model]
    if data.empty:
        return {}

    # Step-1 residuals only: pooling horizons would mix h-step errors whose
    # autocorrelation is mechanical rather than informative.
    step_one = data[data["step"] == 1] if "step" in data.columns else data
    if len(step_one) < 30:
        return {"model": model, "n": int(len(step_one)), "note": "too few step-1 residuals"}

    ordered = step_one.sort_values("origin") if "origin" in step_one.columns else step_one
    residual = pd.Series(
        (ordered["actual"] - ordered["predicted"]).to_numpy(dtype=float)
        / ordered["origin_close"].to_numpy(dtype=float)
    )

    mean = float(residual.mean())
    std_error = float(residual.std(ddof=1) / np.sqrt(len(residual)))

    ljung = ljung_box_test(residual, lags=min(max_lag, len(residual) // 5))
    arch = arch_lm_test(residual, lags=min(max_lag, len(residual) // 5))
    normality = jarque_bera_test(residual)

    return {
        "model": model,
        "n": int(len(residual)),
        "mean_residual": mean,
        "mean_residual_t": float(mean / std_error) if std_error > 0 else float("nan"),
        "biased": bool(std_error > 0 and abs(mean / std_error) > 2.0),
        "ljung_box": ljung.to_dict(),
        "arch_lm": arch.to_dict(),
        "jarque_bera": normality.to_dict(),
        "interpretation": _interpret_residuals(ljung, arch, normality, mean, std_error),
    }


def _interpret_residuals(ljung, arch, normality, mean: float, std_error: float) -> str:
    parts: list[str] = []
    if ljung.rejects_null:
        parts.append(
            "residuals are serially correlated, so the mean model left structure behind"
        )
    else:
        parts.append("no remaining serial correlation in the mean model")

    if arch.rejects_null:
        parts.append(
            "ARCH effects remain, which is expected and is what the volatility model is for"
        )
    if normality.rejects_null:
        excess = normality.detail.get("excess_kurtosis", float("nan"))
        parts.append(
            f"residuals are non-normal (excess kurtosis {excess:.1f}), so Gaussian "
            "intervals understate tail risk"
        )
    if std_error > 0 and abs(mean / std_error) > 2.0:
        direction = "over" if mean < 0 else "under"
        parts.append(f"forecasts are systematically {direction}-predicting (biased mean residual)")

    return "; ".join(parts)


def resource_costs(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Per-model fit and inference time (A2.19).

    A small forecast improvement bought with a large compute bill is a different
    proposition from a free one, and the promotion policy consults this.
    """
    scored = per_fold[per_fold["error"].isna()] if "error" in per_fold.columns else per_fold
    available = [c for c in ("fit_seconds", "predict_seconds") if c in scored.columns]
    if scored.empty or not available:
        return pd.DataFrame()

    table = scored.groupby("model", sort=True)[available].agg(
        ["mean", "median", "sum", "max"]
    )
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]

    if {"fit_seconds_median", "predict_seconds_median"} <= set(table.columns):
        # The cost multiple uses the MEDIAN fold, not the mean. Lazy third-party
        # imports (scipy.stats, arch, xgboost) are paid once, on the first fold
        # that needs them, and are charged to whichever model happened to run
        # first. Averaged over folds that made RandomWalk -- the cheapest model
        # there is -- look 18x more expensive than the drift model in an early
        # run. The median is immune, and the promotion policy consults this
        # number, so the distortion would have had consequences.
        table["total_seconds_median"] = (
            table["fit_seconds_median"] + table["predict_seconds_median"]
        )
        cheapest = table["total_seconds_median"].min()
        table["cost_multiple_vs_cheapest"] = (
            table["total_seconds_median"] / cheapest if cheapest > 0 else np.nan
        )
    return table


__all__ = [
    "StabilityProfile",
    "failure_summary",
    "largest_failures",
    "residual_diagnostics",
    "resource_costs",
    "stability_by_period",
    "stability_profile",
]
