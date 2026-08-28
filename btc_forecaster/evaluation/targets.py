"""What is actually being forecast, and how each choice is scored.

An audit, encoded rather than left as a convention.

The problem
-----------
Log price is I(1) -- Track A's diagnostics confirm a unit root (ADF does not
reject; KPSS does). Two consequences follow, and both distort a benchmark built
on price-level error:

1. **MAE on price is dominated by the level.** BTC ranged from ~$3.5k to ~$100k
   over the sample. A 5% error costs $175 in one regime and $5,000 in another,
   so a price-level MAE averaged across folds is mostly a statement about which
   folds happened to sit at high prices. Fold dispersion in the A2 baseline run
   (``mae_std`` ~ 1,355 against a mean of 3,849) is largely this.

2. **The random walk is near-optimal by construction.** If log price is a
   martingale, ``P[T+h] = P[T]`` minimises expected squared error. Beating it on
   price-level MAE requires predicting the *drift*, which over 30 days is a tiny
   signal buried in a large variance. Small skill numbers on this task are
   expected, not evidence of a broken model.

Returns are the stationary object. Direction is defined on them. "No skill"
has a meaningful value (zero) rather than a data-dependent one.

The decision
------------
**Both tasks are reported, as separate tasks, from the same forecasts.**

Price level is retained as the primary reported task because the preserved
LEAKAGE_CORRECTED_REFERENCE was scored on it, and silently switching to a metric
on which the hybrid looks better is exactly what
:mod:`btc_forecaster.evidence` forbids. Return-space metrics are added
alongside, not instead.

No model changes. A price path and a return path from a fixed origin are the
same object under an invertible transform:

    cumulative_return[h] = point[h] / origin_close - 1
    step_return[h]       = point[h] / point[h-1] - 1

so both views are derived from the existing :class:`ForecastResult`, and the two
tasks cannot disagree about what the model predicted.

Directional evaluation
----------------------
The tradeable definition is ``sign(predicted return) == sign(realised return)``
measured from the forecast origin -- the last price that could actually have
been transacted at. That is the forecast-origin formulation A2.7 asks for, and
it is what :func:`btc_forecaster.evaluation.metrics.directional_accuracy`
computes given ``reference=origin_close``.

One caveat is load-bearing. Scoring 30 bars against a single origin gives 30
*correlated* observations, not 30 independent ones: if the market trends away
from the origin, every bar in the fold agrees. The 0.889 directional accuracy
``random_walk_drift`` posted in the A2 baseline run is exactly this artefact.
Per-step decomposition (:func:`evaluate_by_step`) is the fix -- step 1 across
many origins is far closer to an independent sample than steps 1..30 from one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

EPSILON = 1e-12


class ForecastTask(str, Enum):
    """The distinct research tasks a forecast can be scored on."""

    #: Absolute price. Primary reported task; comparable to the preserved reference.
    PRICE_LEVEL = "price_level"

    #: Return from the forecast origin to bar h. Scale-free; the tradeable view.
    CUMULATIVE_RETURN = "cumulative_return"

    #: Return from bar h-1 to bar h along the forecast path.
    STEP_RETURN = "step_return"


@dataclass(frozen=True)
class ReturnView:
    """A price path re-expressed in return space, relative to a fixed origin."""

    task: ForecastTask
    predicted: pd.Series
    actual: pd.Series
    origin_close: float

    def __post_init__(self) -> None:
        if not self.predicted.index.equals(self.actual.index):
            raise ValueError("predicted and actual return series must share an index")

    @property
    def n(self) -> int:
        return len(self.predicted)

    def direction_agreement(self) -> pd.Series:
        """Per-bar boolean: did the sign of the predicted return match the realised one?

        Zero counts as "not up", so a flat forecast (the random walk) predicts
        down everywhere. That is the honest reading -- a forecast of no change is
        not a bet on an increase.
        """
        return (self.predicted > 0) == (self.actual > 0)


def cumulative_return_view(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    origin_close: float,
) -> ReturnView:
    """Return from the origin close to each forecast bar.

    This is the tradeable quantity: at the origin you could transact at
    ``origin_close``, and the question is whether the model called the move away
    from it correctly.
    """
    if abs(origin_close) < EPSILON:
        raise ValueError("origin_close must be non-zero to define a return")

    aligned = predicted.index.intersection(actual.index)
    return ReturnView(
        task=ForecastTask.CUMULATIVE_RETURN,
        predicted=predicted.loc[aligned] / origin_close - 1.0,
        actual=actual.loc[aligned] / origin_close - 1.0,
        origin_close=float(origin_close),
    )


def step_return_view(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    origin_close: float,
) -> ReturnView:
    """Bar-to-bar return along each path, with the origin close prepended.

    Step 1's return is measured from the origin; later steps from the previous
    bar of their own series. Note this compares the *predicted path's* internal
    moves against the actual path's -- informative about path shape, but only
    step 1 is tradeable from a real price, which is why
    :func:`cumulative_return_view` is the default for directional scoring.
    """
    if abs(origin_close) < EPSILON:
        raise ValueError("origin_close must be non-zero to define a return")

    aligned = predicted.index.intersection(actual.index)
    p = predicted.loc[aligned]
    a = actual.loc[aligned]

    def steps(series: pd.Series) -> pd.Series:
        prior = np.concatenate([[origin_close], series.to_numpy()[:-1]])
        return pd.Series(series.to_numpy() / prior - 1.0, index=series.index)

    return ReturnView(
        task=ForecastTask.STEP_RETURN,
        predicted=steps(p),
        actual=steps(a),
        origin_close=float(origin_close),
    )


@dataclass(frozen=True)
class StepMetrics:
    """Metrics at one horizon step, pooled across many forecast origins."""

    step: int
    n_origins: int
    mae: float
    rmse: float
    directional_accuracy: float
    n_correct: int
    return_correlation: float
    interval_coverage: float

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "n_origins": self.n_origins,
            "mae": self.mae,
            "rmse": self.rmse,
            "directional_accuracy": self.directional_accuracy,
            "n_correct": self.n_correct,
            "return_correlation": self.return_correlation,
            "interval_coverage": self.interval_coverage,
        }


def evaluate_by_step(
    records: pd.DataFrame,
    *,
    max_step: int | None = None,
) -> list[StepMetrics]:
    """Decompose metrics by horizon step, pooling across origins.

    ``records`` must carry one row per (origin, step) with columns
    ``step``, ``actual``, ``predicted``, ``origin_close`` and optionally
    ``lower``/``upper``.

    Why this matters more than the pooled number: 30 bars from one origin are
    30 correlated observations. Step 1 from 60 origins is close to 60
    independent ones, and step 1 is also the only horizon at which the "next
    return" is tradeable without an intervening forecast. Reading directional
    accuracy at step 1 across origins is therefore a far better estimate of
    predictability than the pooled figure, and the two routinely disagree.
    """
    required = {"step", "actual", "predicted", "origin_close"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records is missing column(s): {sorted(missing)}")

    out: list[StepMetrics] = []
    for step, group in records.groupby("step", sort=True):
        if max_step is not None and int(step) > max_step:
            continue

        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        reference = group["origin_close"].to_numpy(dtype=float)

        actual_return = actual / reference - 1.0
        predicted_return = predicted / reference - 1.0
        agreement = (predicted_return > 0) == (actual_return > 0)

        if (
            len(actual_return) >= 3
            and np.std(actual_return) > EPSILON
            and np.std(predicted_return) > EPSILON
        ):
            correlation = float(np.corrcoef(actual_return, predicted_return)[0, 1])
        else:
            correlation = float("nan")

        if {"lower", "upper"} <= set(group.columns):
            lower = group["lower"].to_numpy(dtype=float)
            upper = group["upper"].to_numpy(dtype=float)
            coverage = float(np.mean((actual >= lower) & (actual <= upper)))
        else:
            coverage = float("nan")

        out.append(
            StepMetrics(
                step=int(step),
                n_origins=len(group),
                mae=float(np.mean(np.abs(actual - predicted))),
                rmse=float(np.sqrt(np.mean((actual - predicted) ** 2))),
                directional_accuracy=float(np.mean(agreement)),
                n_correct=int(np.sum(agreement)),
                return_correlation=correlation,
                interval_coverage=coverage,
            )
        )
    return out


def step_metrics_frame(metrics: list[StepMetrics]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame([m.to_dict() for m in metrics]).set_index("step")


def one_step_direction_sample(records: pd.DataFrame) -> np.ndarray:
    """Boolean hit/miss at step 1, one entry per origin, in origin order.

    This is the series the block bootstrap consumes. Restricting to step 1
    removes the within-fold correlation that comes from scoring many horizons
    against a single origin; what remains is serial dependence *between* origins,
    which is exactly what a block method is designed to handle.
    """
    if "step" not in records.columns:
        raise ValueError("records must carry a 'step' column")

    first = records[records["step"] == 1]
    if first.empty:
        raise ValueError("no step-1 rows in records")

    ordered = first.sort_values("origin") if "origin" in first.columns else first
    actual_return = ordered["actual"].to_numpy(float) / ordered["origin_close"].to_numpy(float) - 1.0
    predicted_return = (
        ordered["predicted"].to_numpy(float) / ordered["origin_close"].to_numpy(float) - 1.0
    )
    return (predicted_return > 0) == (actual_return > 0)


#: Written into every benchmark manifest so a reader knows what was scored.
TARGET_AUDIT = {
    "primary_task": ForecastTask.PRICE_LEVEL.value,
    "secondary_tasks": [ForecastTask.CUMULATIVE_RETURN.value, ForecastTask.STEP_RETURN.value],
    "why_price_level_is_primary": (
        "The preserved LEAKAGE_CORRECTED_REFERENCE was scored on price-level MAE. "
        "Switching the primary metric would make the challenger incomparable to it, "
        "and choosing a metric on which the incumbent looks better is what "
        "btc_forecaster.evidence forbids."
    ),
    "why_returns_are_also_reported": (
        "Log price is I(1), so price-level MAE is dominated by the price level and "
        "the random walk is near-optimal by construction. Returns are stationary, "
        "'no skill' has a meaningful value of zero, and direction is defined on them."
    ),
    "directional_definition": (
        "sign(predicted return from forecast origin) == sign(realised return from "
        "forecast origin). Zero counts as not-up, so a flat forecast predicts down."
    ),
    "known_limitation": (
        "Scoring h bars against one origin yields h correlated observations, not h "
        "independent ones. Per-step decomposition and step-1 pooling across origins "
        "is the mitigation; block bootstrap handles the residual serial dependence "
        "between origins."
    ),
}


__all__ = [
    "TARGET_AUDIT",
    "ForecastTask",
    "ReturnView",
    "StepMetrics",
    "cumulative_return_view",
    "evaluate_by_step",
    "one_step_direction_sample",
    "step_metrics_frame",
    "step_return_view",
]
