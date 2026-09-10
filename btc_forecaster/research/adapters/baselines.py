"""The baselines. Not decoration, and not to be hidden when they win.

A2's headline finding is that nothing beat the random walk over 36 folds. The
purpose of putting the baselines in the zoo is so that finding has a chance to
repeat -- or not -- against forty models rather than eight, under a sample size
where it is even harder to beat them.

Three of them, and the distinctions are real rather than cosmetic:

``naive_last_value``       price is a martingale: the forecast log return is 0.
``random_walk_drift``      the mean training log return, which is *identically*
                           the classical endpoint drift
                           ``(log P_end - log P_start) / (n - 1)``, because the
                           log returns telescope. Registered once, not twice.
``historical_mean_return`` ``log(1 + mean simple return)``. Distinct from the
                           above by Jensen's inequality -- the arithmetic mean
                           of simple returns exceeds the mean log return by
                           roughly half the variance, which on daily BTC is not
                           a rounding error.

All three are constant forecasts. That is the point: a constant is the
hypothesis every conditional model has to disprove, and at n=1,000 it is a
genuinely hard one, because a constant has one parameter and no variance to pay
for it.
"""

from __future__ import annotations

import numpy as np

from ..contracts import (
    Capability,
    EvaluationContext,
    Family,
    Preprocessing,
    ResourceClass,
    TrainingSet,
    ZooModel,
)
from ..registry import ZooRegistration, register


class ConstantForecast(ZooModel):
    """A single number, repeated. Subclasses decide which number."""

    family = Family.BASELINE
    resource_class = ResourceClass.TRIVIAL
    preprocessing = Preprocessing.NONE
    capabilities = frozenset({Capability.POINT, Capability.SERIALIZE, Capability.MULTI_STEP})

    def __init__(self) -> None:
        super().__init__()
        self._value = 0.0

    def _fit(self, train: TrainingSet) -> None:
        self._value = float(self._constant(train))

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        return np.full(len(context), self._value, dtype=float)

    def _predict_cumulative(self, context: EvaluationContext, horizon: int) -> np.ndarray:
        # A constant log return per bar is h of them over h bars: the naive
        # forecast stays zero, the drift forecast is h times the mean.
        return np.full(len(context), self._value * horizon, dtype=float)

    def _constant(self, train: TrainingSet) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def hyperparameters(self) -> dict:
        return {"constant_log_return": self._value}

    def parameter_count(self) -> int | None:
        return 1


class NaiveLastValue(ConstantForecast):
    """P[T+1] = P[T]. Zero parameters, and the hypothesis to disprove."""

    model_id = "naive_last_value"

    def _constant(self, train: TrainingSet) -> float:
        return 0.0

    def parameter_count(self) -> int | None:
        return 0


class RandomWalkDrift(ConstantForecast):
    """The mean training log return, i.e. the classical endpoint drift."""

    model_id = "random_walk_drift"

    def _constant(self, train: TrainingSet) -> float:
        returns = train.y.to_numpy(dtype=float)
        return float(np.mean(returns)) if len(returns) else 0.0


class HistoricalMeanReturn(ConstantForecast):
    """log(1 + mean simple return). Jensen-distinct from the drift above."""

    model_id = "historical_mean_return"

    def _constant(self, train: TrainingSet) -> float:
        log_returns = train.y.to_numpy(dtype=float)
        if len(log_returns) == 0:
            return 0.0
        mean_simple = float(np.mean(np.expm1(log_returns)))
        # A mean simple return at or below -100% is not representable in log
        # space; fall back to the log mean rather than emitting -inf.
        return float(np.log1p(mean_simple)) if mean_simple > -0.999999 else float(
            np.mean(log_returns)
        )


register(
    ZooRegistration(
        model_id="naive_last_value",
        factory=NaiveLastValue,
        family=Family.BASELINE,
        resource_class=ResourceClass.TRIVIAL,
        description="P[T+1] = P[T]; the forecast log return is exactly zero.",
        notes=(
            "Zero fitted parameters. Every conditional model in the zoo is "
            "measured against this one.",
        ),
    )
)

register(
    ZooRegistration(
        model_id="random_walk_drift",
        factory=RandomWalkDrift,
        family=Family.BASELINE,
        resource_class=ResourceClass.TRIVIAL,
        description="Constant drift: the mean training log return.",
        notes=(
            "Mathematically identical to the classical endpoint drift "
            "(log P_end - log P_start) / (n - 1), because log returns telescope. "
            "Registered once rather than twice.",
            "A2's strongest model over 36 folds, at MAE skill +0.003957 -- which "
            "was still a REJECT.",
        ),
    )
)

register(
    ZooRegistration(
        model_id="historical_mean_return",
        factory=HistoricalMeanReturn,
        family=Family.BASELINE,
        resource_class=ResourceClass.TRIVIAL,
        description="log(1 + mean training simple return).",
        notes=(
            "Distinct from random_walk_drift by Jensen's inequality: the "
            "arithmetic mean of simple returns exceeds the mean log return by "
            "approximately half the return variance.",
        ),
    )
)


__all__ = [
    "ConstantForecast",
    "HistoricalMeanReturn",
    "NaiveLastValue",
    "RandomWalkDrift",
]
