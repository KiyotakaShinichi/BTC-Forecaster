"""Fit and forecast one model, at one horizon, over one window -- fold by fold.

A **job** is a (model, horizon, window) triple. It walks the refit folds in
order: at each fold's first origin it estimates the model on that fold's
training rows, then forecasts every daily origin in the fold with those
parameters frozen. The output is one row per origin with the forecast and the
return that actually followed.

How a model reaches a horizon is decided by what it declares, not by a list:

``ITERATED``
    Models declaring ``MULTI_STEP`` -- the baselines and the series models --
    estimate on one-bar returns, exactly as in A6, and iterate their own
    recursion ``h`` bars with frozen parameters.

``DIRECT``
    Everything else is trained on the ``h``-bar target itself: the features at
    each origin, and the return realised ``h`` bars later.

Both are scored against the same realised target at the same origins, so the
strategy is a property of the model, not of the comparison.

One fold's fit-and-forecast is :func:`forecast_fold`, and it is the only place
it happens: the benchmark calls it, and so do the leakage adversaries -- through
an optional hook on the prediction context -- so what the adversaries test is
the code path the results came from, not a copy of it.

Failures are recorded, never dropped. A fold whose fit raises is ``FAILED``
with its exception, and the gate refuses any configuration with one.

A fold that costs more than the model's A6 resource class allows is recorded
too -- in the timings, not in the canonical result. Whether a fold overran its
budget depends on how many processes shared the machine: the first canonical
BTC run marked fifteen LSTM folds over their 120 s budget at 121-240 s under
four-way parallel load, folds that take about 48 s alone. A status that moves
with machine load cannot sit inside a digest that claims byte-identical
reproduction, so it is reported beside it. So is any deep-model fold that hit
A6's own training time cap, the one case in which the forecasts themselves
would depend on timing.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import registry
from ..contracts import (
    RESOURCE_BUDGET_SECONDS,
    Capability,
    EvaluationContext,
    ModelStatus,
    ZooModel,
)
from ..runner import warm_imports
from .config import MIN_ROLLING_ROWS, WalkForwardConfig, WindowSpec
from .origins import OriginSchedule, RefitFold, build_schedule, first_origin_position
from .windows import InformationSet, information_set


def schedule_for(frame: pd.DataFrame, config: WalkForwardConfig) -> OriginSchedule:
    """The one origin schedule every job in a benchmark on ``frame`` shares.

    The first origin is the first bar at which the largest fixed window can be
    filled with settled rows whose longest-horizon targets are realised, so no
    window is ever scored with less history than it declares.
    """
    return build_schedule(
        pd.DatetimeIndex(frame.index),
        first_position=first_origin_position(
            warmup_bars=config.warmup_bars,
            training_rows=config.largest_fixed_window or MIN_ROLLING_ROWS,
            max_horizon=config.max_horizon,
        ),
        max_horizon=config.max_horizon,
        n_refits=config.n_refits,
        n_blocks=config.n_blocks,
    )


ITERATED = "ITERATED"
DIRECT = "DIRECT"

#: Canonical column order of a job's records.
RECORD_COLUMNS: tuple[str, ...] = (
    "model_id",
    "horizon",
    "window",
    "fold",
    "origin",
    "target_bar",
    "actual",
    "predicted",
    "train_direction_up",
)

#: Transforms the context a model predicts from. ``None`` in the benchmark; the
#: leakage adversaries use it to poison what the model is handed.
ContextHook = Callable[[EvaluationContext], EvaluationContext]


def strategy_for(model: ZooModel) -> str:
    return ITERATED if model.supports(Capability.MULTI_STEP) else DIRECT


@dataclass(frozen=True)
class JobSpec:
    model_id: str
    horizon: int
    window: WindowSpec

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.model_id, self.horizon, self.window.label)


@dataclass(frozen=True)
class FoldForecast:
    """One fold's forecasts, aligned by origin to the returns that followed."""

    origins: pd.DatetimeIndex
    target_bars: pd.DatetimeIndex
    actual: np.ndarray
    predicted: np.ndarray
    train_rows: int
    train_fingerprint: str | None
    train_direction_up: bool
    fit_seconds: float
    predict_seconds: float
    #: A6's deep models stop training at a wall-clock cap. If one did, this
    #: fold's forecasts depend on machine speed, and the run says so.
    hit_training_time_cap: bool = False


def forecast_fold(
    spec: JobSpec,
    info: InformationSet,
    config: WalkForwardConfig,
    *,
    strategy: str,
    context_hook: ContextHook | None = None,
) -> FoldForecast:
    """Estimate on the fold's training rows, then forecast its origins. May raise."""
    fit_horizon = 1 if strategy == ITERATED else spec.horizon
    train = info.training_set(fit_horizon)
    train_end = train.X.index.max()
    scored = info.evaluation_context(spec.horizon, train_end=train_end)
    # The constant-direction baseline: whichever direction the realised h-bar
    # training targets favoured in this window. Chosen from training data, not
    # assumed to be a coin. Ties count as up, as in A6's
    # `train_constant_baseline`, so the two never disagree on a tie.
    up = bool((info.training_set(spec.horizon).y > 0).mean() >= 0.5)

    started = time.perf_counter()
    model = registry.build(spec.model_id)
    if model.needs_calibration:
        fit_rows, dev = info.early_stopping_split(
            train, horizon=fit_horizon, dev_fraction=config.dev_fraction
        )
        model.fit(fit_rows)
        model.calibrate(dev)
    else:
        model.fit(train)
    fit_seconds = time.perf_counter() - started

    predict_started = time.perf_counter()
    if strategy == ITERATED:
        one_bar = info.evaluation_context(1, train_end=train_end)
        if not one_bar.origins.equals(scored.origins):
            raise AssertionError("iterated and scored origins disagree")
        context = context_hook(one_bar) if context_hook else one_bar
        predicted = model.predict_cumulative(context, spec.horizon)
    else:
        context = context_hook(scored) if context_hook else scored
        predicted = model.predict(context).point
    predict_seconds = time.perf_counter() - predict_started

    return FoldForecast(
        origins=scored.origins,
        target_bars=scored.target_bars,
        actual=scored.y.to_numpy(dtype=float),
        predicted=np.asarray(predicted, dtype=float),
        train_rows=len(train),
        train_fingerprint=model.describe().get("train_fingerprint"),
        train_direction_up=up,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        hit_training_time_cap=bool(
            (model.hyperparameters().get("training") or {}).get("hit_time_budget", False)
        ),
    )


@dataclass(frozen=True)
class FoldOutcome:
    """What happened at one refit, including how long it took.

    The timings are real and are reported -- in a non-canonical file. They are
    never part of a result digest, because they describe the machine.
    """

    fold: int
    status: str
    train_rows: int = 0
    train_fingerprint: str | None = None
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    failure: dict | None = None
    #: Timing facts: reported in run_info.json, never in the canonical result.
    over_budget: bool = False
    budget_seconds: float | None = None
    hit_training_time_cap: bool = False

    @property
    def total_seconds(self) -> float:
        return self.fit_seconds + self.predict_seconds

    def canonical(self) -> dict:
        return {
            "fold": self.fold,
            "status": self.status,
            "train_rows": self.train_rows,
            "train_fingerprint": self.train_fingerprint,
            "failure": self.failure,
        }

    def timing(self) -> dict:
        return {
            "fold": self.fold,
            "fit_seconds": round(self.fit_seconds, 4),
            "predict_seconds": round(self.predict_seconds, 4),
            "budget_seconds": self.budget_seconds,
            "over_budget": self.over_budget,
            "hit_training_time_cap": self.hit_training_time_cap,
        }


@dataclass
class JobResult:
    spec: JobSpec
    strategy: str | None
    records: pd.DataFrame
    folds: list[FoldOutcome] = field(default_factory=list)

    @property
    def statuses(self) -> list[str]:
        return [f.status for f in self.folds]

    @property
    def clean(self) -> bool:
        """Every fold completed. Deterministic: wall-clock overruns are not part of it."""
        return bool(self.folds) and all(s == ModelStatus.ACTIVE.value for s in self.statuses)


class InformationCache:
    """One information set per (fold, window), shared by every job in a process.

    Building it is deterministic and a function of the frame, the fold and the
    window only, so sharing it cannot couple one model's result to another's.
    """

    def __init__(self, frame: pd.DataFrame, config: WalkForwardConfig) -> None:
        self._frame = frame
        self._config = config
        self._cache: dict[tuple[int, str], InformationSet] = {}

    def get(self, fold: RefitFold, window: WindowSpec) -> InformationSet:
        key = (fold.index, window.label)
        if key not in self._cache:
            self._cache[key] = information_set(
                self._frame,
                fold,
                window,
                max_horizon=self._config.max_horizon,
                warmup_bars=self._config.warmup_bars,
            )
        return self._cache[key]


def _empty_records() -> pd.DataFrame:
    return pd.DataFrame({column: [] for column in RECORD_COLUMNS})


def run_job(
    spec: JobSpec,
    schedule: OriginSchedule,
    config: WalkForwardConfig,
    cache: InformationCache,
) -> JobResult:
    """Every fold of one (model, horizon, window), in order."""
    registration = registry.get(spec.model_id)
    declared = registration.effective_status()
    if declared is not ModelStatus.ACTIVE:
        reason = registration.unsuitable_reason or registration.missing_dependency()
        return JobResult(
            spec=spec,
            strategy=None,
            records=_empty_records(),
            folds=[
                FoldOutcome(fold=f.index, status=declared.value, failure={"reason": reason})
                for f in schedule.folds
            ],
        )

    budget = RESOURCE_BUDGET_SECONDS[registration.resource_class]
    strategy = strategy_for(registry.build(spec.model_id))
    # Before any fold's clock starts. The first fold to touch statsmodels would
    # otherwise be charged for loading it -- the defect A6 found in its own
    # runner, which showed up here as `ar_p` over its 5 s budget in fold 0 and
    # at 0.14 s in every fold after.
    warm_imports([spec.model_id])

    frames: list[pd.DataFrame] = []
    outcomes: list[FoldOutcome] = []
    for fold in schedule.folds:
        info = cache.get(fold, spec.window)
        started = time.perf_counter()
        try:
            result = forecast_fold(spec, info, config, strategy=strategy)
        except Exception as exc:  # noqa: BLE001 -- one fold must not end the job
            outcomes.append(
                FoldOutcome(
                    fold=fold.index,
                    status=ModelStatus.FAILED.value,
                    fit_seconds=time.perf_counter() - started,
                    failure={
                        "exception": type(exc).__name__,
                        "message": str(exc)[:500],
                        "traceback": traceback.format_exc(limit=4)[-1200:],
                    },
                )
            )
            continue

        outcomes.append(
            FoldOutcome(
                fold=fold.index,
                status=ModelStatus.ACTIVE.value,
                train_rows=result.train_rows,
                train_fingerprint=result.train_fingerprint,
                fit_seconds=result.fit_seconds,
                predict_seconds=result.predict_seconds,
                over_budget=result.fit_seconds + result.predict_seconds > budget,
                budget_seconds=budget,
                hit_training_time_cap=result.hit_training_time_cap,
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "model_id": spec.model_id,
                    "horizon": spec.horizon,
                    "window": spec.window.label,
                    "fold": fold.index,
                    "origin": result.origins,
                    "target_bar": result.target_bars,
                    "actual": result.actual,
                    "predicted": result.predicted,
                    "train_direction_up": result.train_direction_up,
                }
            )
        )

    records = pd.concat(frames, ignore_index=True) if frames else _empty_records()
    return JobResult(spec=spec, strategy=strategy, records=records[list(RECORD_COLUMNS)], folds=outcomes)


__all__ = [
    "DIRECT",
    "ITERATED",
    "RECORD_COLUMNS",
    "ContextHook",
    "FoldForecast",
    "FoldOutcome",
    "InformationCache",
    "JobResult",
    "JobSpec",
    "forecast_fold",
    "run_job",
    "schedule_for",
    "strategy_for",
]
