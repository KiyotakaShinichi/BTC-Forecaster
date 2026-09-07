"""The walk-forward engine: one set of folds, every model scored through it.

Guarantees the engine enforces, so no individual model has to:

* Every model sees identical folds, identical forecast origins and identical
  scored bars. Differences between models are then differences between models.
* A model is handed a ``TrainingWindow`` built from the fold's training slice
  and nothing else, so it cannot see the test window. Fitting -- including any
  feature selection or order selection a model does internally -- happens inside
  the fold.
* A model that fails on one fold is recorded as failing on that fold, not
  silently dropped and not fatal to the run. A comparison table with a hole in
  it is honest; one that quietly omits the failures is not.
* Skill scores are computed against a named baseline on the same folds, so
  "better than a random walk" is measured rather than assumed.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.contracts import validate_market_frame
from ..evaluation.metrics import (
    LOWER_IS_BETTER,
    SUMMARY_METRICS,
    binomial_direction_test,
    evaluate_forecast,
    naive_scale_from_training,
    skill_score,
)
from ..models.base import ForecastModel, TrainingWindow
from ..timebase import BAR_DURATION
from .splits import Fold, WalkForwardSplitter, assert_folds_are_disjoint

#: Columns in the per-fold table that are wall-clock measurements. They are
#: informative but not reproducible, so any comparison of two runs must exclude
#: them. Defined once here rather than re-listed at each call site.
TIMING_COLUMNS: tuple[str, ...] = ("fit_seconds", "predict_seconds")


@dataclass(frozen=True)
class FoldOutcome:
    """One model's result on one fold."""

    model: str
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_bars: int
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    #: One row per scored bar: step, date, actual, predicted, lower, upper.
    #: Kept so metrics can be decomposed by horizon step after the fact, and so
    #: the largest misses can be inspected without re-running the backtest.
    predictions: pd.DataFrame | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_row(self) -> dict:
        return {
            "model": self.model,
            "fold": self.fold,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_bars": self.train_bars,
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
            "error": self.error,
            **self.metrics,
        }


@dataclass
class BacktestResult:
    """Everything a walk-forward run produced, and how it was configured."""

    outcomes: list[FoldOutcome]
    folds: list[Fold]
    splitter: WalkForwardSplitter
    model_descriptions: dict[str, dict] = field(default_factory=dict)
    skipped_models: dict[str, str] = field(default_factory=dict)
    baseline: str | None = None

    # -- tabular views ----------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """One row per (model, fold)."""
        return pd.DataFrame([outcome.to_row() for outcome in self.outcomes])

    def summary(self, metrics: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Mean and standard deviation per model across folds.

        Failed folds are excluded from the aggregate but counted in
        ``n_folds_failed``, so a model that only worked once cannot look good.
        """
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame()

        succeeded = frame[frame["error"].isna()]
        chosen = metrics or tuple(c for c in succeeded.columns if c in SUMMARY_METRICS)
        chosen = tuple(c for c in chosen if c in succeeded.columns)

        rows = []
        for model, group in succeeded.groupby("model", sort=False):
            row: dict = {
                "model": model,
                "n_folds_ok": len(group),
                "n_folds_failed": int((frame["model"] == model).sum() - len(group)),
            }
            for metric in chosen:
                values = group[metric].astype(float)
                row[metric] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
            rows.append(row)

        for model in frame.loc[frame["error"].notna(), "model"].unique():
            if model not in {r["model"] for r in rows}:
                rows.append(
                    {
                        "model": model,
                        "n_folds_ok": 0,
                        "n_folds_failed": int((frame["model"] == model).sum()),
                    }
                )

        return pd.DataFrame(rows).set_index("model")

    def skill_table(self, baseline: str | None = None) -> pd.DataFrame:
        """Skill of every model against a baseline, on identical folds.

        Positive skill means the model reduced that error metric relative to the
        baseline. This is the table that decides whether a model earns its
        complexity.
        """
        base = baseline or self.baseline
        summary = self.summary()
        if base is None or base not in summary.index or summary.empty:
            return pd.DataFrame()

        rows = []
        for model in summary.index:
            row = {"model": model}
            for metric in ("mae", "rmse", "mase", "winkler_score"):
                if metric in summary.columns:
                    row[f"{metric}_skill_vs_{base}"] = skill_score(
                        float(summary.loc[model, metric]), float(summary.loc[base, metric])
                    )
            if "directional_accuracy" in summary.columns:
                row["directional_accuracy"] = float(summary.loc[model, "directional_accuracy"])
                row["directional_accuracy_edge"] = float(
                    summary.loc[model, "directional_accuracy"] - 0.5
                )
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    def direction_test(self, model: str) -> dict:
        """Pooled binomial direction test across folds, with its caveat attached."""
        frame = self.to_frame()
        rows = frame[(frame["model"] == model) & frame["error"].isna()]
        if rows.empty or "directional_accuracy" not in rows.columns:
            return {}

        n_total = int(rows["n"].sum())
        n_correct = int(round(float((rows["directional_accuracy"] * rows["n"]).sum())))
        return binomial_direction_test(n_correct, n_total).to_dict()

    def failures(self) -> list[FoldOutcome]:
        return [outcome for outcome in self.outcomes if outcome.failed]

    def prediction_records(self, model: str | None = None) -> pd.DataFrame:
        """One row per (model, fold, horizon step) with actual and predicted.

        The raw material for per-step metrics, dependence-aware inference and
        failure analysis. Aggregating first and asking questions later loses
        exactly the structure those need.
        """
        frames = [
            outcome.predictions
            for outcome in self.outcomes
            if outcome.predictions is not None and (model is None or outcome.model == model)
        ]
        if not frames:
            return pd.DataFrame(
                columns=[
                    "model", "fold", "origin", "date", "step",
                    "actual", "predicted", "lower", "upper", "origin_close",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def to_manifest(self) -> dict:
        summary = self.summary()
        return {
            "splitter": self.splitter.describe(),
            "n_folds": len(self.folds),
            "folds": [fold.to_dict() for fold in self.folds],
            "models": self.model_descriptions,
            "skipped_models": self.skipped_models,
            "baseline": self.baseline,
            "summary": (
                {} if summary.empty else summary.reset_index().to_dict(orient="records")
            ),
            "n_failures": len(self.failures()),
        }


def run_walk_forward(
    frame: pd.DataFrame,
    models: list[ForecastModel],
    splitter: WalkForwardSplitter,
    *,
    exog: pd.DataFrame | None = None,
    baseline: str | None = "random_walk",
    on_fold=None,
) -> BacktestResult:
    """Score every model through identical walk-forward folds.

    ``models`` are refitted from scratch on every fold. Passing already-fitted
    models is fine -- their state is overwritten -- but the same instance is
    reused across folds, so a model must not accumulate state outside ``_fit``.
    """
    validate_market_frame(frame)
    folds = splitter.split(frame.index)
    assert_folds_are_disjoint(folds)

    if exog is not None and not exog.index.equals(frame.index):
        raise ValueError("exog must be indexed identically to the market frame")

    outcomes: list[FoldOutcome] = []

    for fold in folds:
        train_frame = fold.train_slice(frame)
        test_frame = fold.test_slice(frame)
        train_exog = None if exog is None else fold.train_slice(exog)

        window = TrainingWindow(frame=train_frame, exog=train_exog)
        naive_scale = naive_scale_from_training(train_frame["close"])
        origin_close = float(train_frame["close"].iloc[-1])
        actual = test_frame["close"]

        for model in models:
            started = time.perf_counter()
            try:
                # Fit and inference are timed separately: a model can be cheap to
                # fit and expensive to forecast (the hybrid's recursive rollout
                # rebuilds its whole feature frame once per step), and the two
                # costs matter in different deployments.
                model.fit(window)
                fitted_at = time.perf_counter()

                # Forecast across any embargo gap, then score only the test window.
                result = model.predict(fold.steps_to_test_end)

                fit_seconds = fitted_at - started
                predict_seconds = time.perf_counter() - fitted_at

                metrics = evaluate_forecast(
                    actual,
                    result.point,
                    reference=origin_close,
                    lower=result.lower,
                    upper=result.upper,
                    interval_level=result.interval_level,
                    naive_scale=naive_scale,
                )
                records = _prediction_records(
                    model_name=model.name,
                    fold=fold,
                    actual=actual,
                    result=result,
                    origin_close=origin_close,
                )
                outcome = FoldOutcome(
                    model=model.name,
                    fold=fold.index,
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    train_bars=len(train_frame),
                    metrics=metrics.to_dict(),
                    fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,
                    predictions=records,
                )
            except Exception as exc:
                outcome = FoldOutcome(
                    model=model.name,
                    fold=fold.index,
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    train_bars=len(train_frame),
                    error=f"{type(exc).__name__}: {exc}",
                    fit_seconds=time.perf_counter() - started,
                )
                outcome.metrics.setdefault("traceback", traceback.format_exc(limit=3))

            outcomes.append(outcome)
            if on_fold is not None:
                on_fold(outcome)

    descriptions: dict[str, dict] = {}
    for model in models:
        try:
            descriptions[model.name] = model.describe()
        except Exception:  # a model that cannot describe itself is not fatal
            descriptions[model.name] = {"name": model.name}

    return BacktestResult(
        outcomes=outcomes,
        folds=folds,
        splitter=splitter,
        model_descriptions=descriptions,
        baseline=baseline,
    )


def _prediction_records(
    *,
    model_name: str,
    fold: Fold,
    actual: pd.Series,
    result,
    origin_close: float,
) -> pd.DataFrame:
    """Build the per-bar record frame for one model on one fold.

    ``step`` counts from the forecast origin, so with an embargo the first
    *scored* bar is step ``embargo_bars + 1``. That is deliberate: step is the
    forecast distance, not the position within the scored window, and pooling
    "step 1" across folds must mean the same forecast distance every time.
    """
    aligned = result.point.index.intersection(actual.index)
    if len(aligned) == 0:
        return pd.DataFrame()

    origin_bar = fold.origin.last_observed_bar
    steps = ((aligned - origin_bar) / BAR_DURATION).astype(int)

    records = pd.DataFrame(
        {
            "model": model_name,
            "fold": fold.index,
            "origin": origin_bar,
            "date": aligned,
            "step": steps,
            "actual": actual.loc[aligned].to_numpy(dtype=float),
            "predicted": result.point.loc[aligned].to_numpy(dtype=float),
            "origin_close": float(origin_close),
        }
    )
    records["lower"] = (
        result.lower.loc[aligned].to_numpy(dtype=float) if result.lower is not None else np.nan
    )
    records["upper"] = (
        result.upper.loc[aligned].to_numpy(dtype=float) if result.upper is not None else np.nan
    )
    return records


def rank_models(result: BacktestResult, metric: str = "mae") -> pd.DataFrame:
    """Models ordered best-first on one metric."""
    summary = result.summary()
    if summary.empty or metric not in summary.columns:
        return summary
    ascending = metric in LOWER_IS_BETTER
    return summary.sort_values(metric, ascending=ascending)


def has_useful_skill(result: BacktestResult, model: str, *, baseline: str = "random_walk") -> bool:
    """Whether ``model`` beat ``baseline`` on MAE across the folds.

    Deliberately blunt. A single boolean is easy to report honestly and hard to
    talk around, which is the point: a negative answer is a result, not a bug.
    """
    table = result.skill_table(baseline)
    column = f"mae_skill_vs_{baseline}"
    if table.empty or column not in table.columns or model not in table.index:
        return False
    value = table.loc[model, column]
    return bool(np.isfinite(value) and value > 0)


__all__ = [
    "TIMING_COLUMNS",
    "BacktestResult",
    "FoldOutcome",
    "has_useful_skill",
    "rank_models",
    "run_walk_forward",
]
