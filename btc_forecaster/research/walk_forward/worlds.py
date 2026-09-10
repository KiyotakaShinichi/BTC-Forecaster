"""Worlds where the right answer is known -- for validating the engine itself.

On BTC, a benchmark that reports "nothing works" is indistinguishable from a
benchmark that is broken. These worlds are how the two are told apart. A7
validates the **research engine**, not only the BTC result: it must stay silent
on noise, and it must find a signal that is genuinely there.

``NOISE``
    Independent returns. The engine must not produce a candidate here; if it
    does, the finding is a bug report.

``TREND``
    A drift large relative to the noise. The drift baseline should beat the
    naive forecast -- which is exactly why drift is a required baseline.

``AUTOCORRELATION``
    AR(1) returns. An autoregressive model, reaching longer horizons by
    iterating its own recursion, should find it.

``KNOWN_SIGNAL``
    Returns driven by the previous week's mean return, a feature the tabular
    models are given. A directly trained model should find it. Together with
    ``AUTOCORRELATION`` this exercises both of the engine's forecasting paths.

``REGIME_SHIFT``
    The AR(1) signal of ``AUTOCORRELATION`` for the first part of the series,
    then nothing. Aggregate skill can survive; the late-block gate must not.

And one adversary that is a model rather than a world: :class:`LeakyOracle`
reads the realised target and so forecasts perfectly. It is never registered
except inside :func:`registered_leaky_oracle`. Its purpose is to lose -- to
show that a model whose numbers are spectacular is still refused when the
leakage adversaries catch it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np

from .. import registry
from ..contracts import EvaluationContext, Family, ResourceClass, TrainingSet, ZooModel
from ..registry import ZooRegistration
from ..synthetic import World, _frame, _prices, noise_only, trend


def autocorrelation(periods: int = 900, *, phi: float = 0.35, noise: float = 0.01, seed: int = 41) -> World:
    rng = np.random.default_rng(seed)
    returns = np.zeros(periods)
    shocks = rng.normal(0.0, noise, periods)
    for t in range(1, periods):
        returns[t] = phi * returns[t - 1] + shocks[t]
    return World(
        name="AUTOCORRELATION",
        frame=_frame(_prices(returns)),
        expectation=f"AR(1) returns with phi={phi}; an iterated AR model should find it",
        predictable_mean=True,
    )


def known_signal(
    periods: int = 900, *, beta: float = 0.6, noise: float = 0.01, seed: int = 43
) -> World:
    """``r[t] = beta * mean(r[t-7 .. t-1]) + e[t]``.

    Stationary for ``beta < 1`` (seven equal AR coefficients summing to beta).
    The regressor is, up to the log/simple-return distinction, the
    ``roll_mean_ret_7`` feature every tabular model receives.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1) for the process to be stationary")
    rng = np.random.default_rng(seed)
    returns = np.zeros(periods)
    shocks = rng.normal(0.0, noise, periods)
    for t in range(7, periods):
        returns[t] = beta * returns[t - 7 : t].mean() + shocks[t]
    return World(
        name="KNOWN_SIGNAL",
        frame=_frame(_prices(returns)),
        expectation="returns follow last week's mean return; a model given that feature should find it",
        predictable_mean=True,
    )


def regime_shift(
    periods: int = 900, *, phi: float = 0.45, shift_at: float = 0.5, noise: float = 0.01, seed: int = 47
) -> World:
    """AR(1) until ``shift_at`` of the way through, independent noise after."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(periods)
    shocks = rng.normal(0.0, noise, periods)
    cut = int(periods * shift_at)
    for t in range(1, periods):
        returns[t] = (phi * returns[t - 1] if t < cut else 0.0) + shocks[t]
    return World(
        name="REGIME_SHIFT",
        frame=_frame(_prices(returns)),
        expectation="a real signal that stops; aggregate skill may survive, the late block must not",
        predictable_mean=True,
    )


WORLDS: dict[str, Callable[[], World]] = {
    "NOISE": lambda: noise_only(900, noise=0.02, seed=7),
    "TREND": lambda: trend(900, drift=0.003, noise=0.004, seed=1),
    "AUTOCORRELATION": autocorrelation,
    "KNOWN_SIGNAL": known_signal,
    "REGIME_SHIFT": regime_shift,
}


def build(name: str) -> World:
    if name not in WORLDS:
        raise KeyError(f"unknown world {name!r}; known: {sorted(WORLDS)}")
    return WORLDS[name]()


LEAKY_ORACLE_ID = "leaky_oracle_adversary"


class LeakyOracle(ZooModel):
    """Forecasts the realised target by reading it. A deliberate leak."""

    model_id = LEAKY_ORACLE_ID
    family = Family.BASELINE
    resource_class = ResourceClass.TRIVIAL

    def _fit(self, train: TrainingSet) -> None:
        return None

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        return context.y.to_numpy(dtype=float)


@contextmanager
def registered_leaky_oracle() -> Iterator[str]:
    """Register the oracle for the duration of a block, then remove it.

    Scoped so it cannot outlive the check that needs it and appear in a real
    benchmark's registry.
    """
    registry.get("naive_last_value")  # load the adapters before touching the registry
    registry.register(
        ZooRegistration(
            model_id=LEAKY_ORACLE_ID,
            factory=LeakyOracle,
            family=Family.BASELINE,
            resource_class=ResourceClass.TRIVIAL,
            description="reads the realised target; exists only to be caught",
        )
    )
    try:
        yield LEAKY_ORACLE_ID
    finally:
        registry._REGISTRY.pop(LEAKY_ORACLE_ID, None)


__all__ = [
    "LEAKY_ORACLE_ID",
    "WORLDS",
    "LeakyOracle",
    "autocorrelation",
    "build",
    "known_signal",
    "registered_leaky_oracle",
    "regime_shift",
]
