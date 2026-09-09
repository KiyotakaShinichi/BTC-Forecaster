"""Seven synthetic worlds where the right answer is known.

On real BTC returns nothing works, which means the benchmark cannot distinguish
a correctly-wired model from a broken one: both score approximately zero. That
is the whole problem with validating a model zoo on a series with no signal.

So these are worlds with a *known* structure, and each one asks a question a
real series cannot:

``TREND``                 a deterministic drift. Anything that cannot find it is
                          mis-wired.
``STATIONARY``            i.i.d. noise around a constant. Nothing should beat the
                          mean.
``MEAN_REVERTING``        an Ornstein-Uhlenbeck level. Negative autocorrelation
                          in returns; a model that exploits it is working.
``AR_PROCESS``            AR(1) returns with a known coefficient. An AR model
                          should recover roughly that coefficient.
``VOLATILITY_CLUSTERING`` GARCH-generated. The variance is predictable and the
                          direction is not -- which separates the two claims.
``NONLINEAR``             a threshold process. Linear models should be beaten by
                          something that can bend.
``NOISE_ONLY``            the control, and the most important one.

**The NOISE_ONLY control carries the load.** It is unpredictable by
construction, so any model scoring materially better than the mean on it has
found something that is not there -- a leak, an alignment error, or a scaler
that saw the future. It is the one world where a *good* result is a bug report.

Rankings are deliberately not asserted anywhere. Phase 28 says these verify
behaviour, not order, and demanding that an LSTM beat a Ridge on a synthetic
AR(1) would be encoding an expectation rather than testing one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..testing import daily_index


@dataclass(frozen=True)
class World:
    """One synthetic series and what is known to be true about it."""

    name: str
    frame: pd.DataFrame
    #: What a correctly-wired model should be able to do here, in one sentence.
    expectation: str
    #: True when the series carries genuinely predictable conditional mean.
    predictable_mean: bool
    #: True when the conditional variance is predictable even if the mean is not.
    predictable_variance: bool = False


def _frame(close: np.ndarray, start: str = "2019-01-01") -> pd.DataFrame:
    index = daily_index(len(close), start)
    frame = pd.DataFrame(
        {
            "close": pd.Series(close, index=index),
            "volume": pd.Series(np.full(len(close), 1_000.0), index=index),
        }
    )
    frame.index.name = "date"
    return frame


def _prices(returns: np.ndarray, initial: float = 100.0) -> np.ndarray:
    return initial * np.exp(np.cumsum(returns))


def trend(periods: int = 600, *, drift: float = 0.003, noise: float = 0.004, seed: int = 1) -> World:
    """A deterministic drift, deliberately large relative to the noise.

    At a drift of 0.2 sigma the best possible MAE skill is under 2%, which is
    too small a margin for a wiring check to fail on: a broken model and a
    correct one would look the same. At 0.75 sigma the gap is unmistakable,
    which is the point of a world where the answer is known.
    """
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0.0, noise, periods)
    return World(
        name="TREND",
        frame=_frame(_prices(returns)),
        expectation="a drift term is recoverable; a model that forecasts zero is wrong by the drift",
        predictable_mean=True,
    )


def stationary(periods: int = 600, *, noise: float = 0.01, seed: int = 2) -> World:
    rng = np.random.default_rng(seed)
    return World(
        name="STATIONARY",
        frame=_frame(_prices(rng.normal(0.0, noise, periods))),
        expectation="i.i.d. around zero; nothing should beat the mean",
        predictable_mean=False,
    )


def mean_reverting(
    periods: int = 600, *, theta: float = 0.5, noise: float = 0.01, seed: int = 3
) -> World:
    """Ornstein-Uhlenbeck in log price: returns are negatively autocorrelated.

    theta is 0.5 rather than something gentler because the induced return
    autocorrelation is far weaker than the level's reversion rate. At
    theta = 0.08 the returns correlate at only -0.04 -- indistinguishable from
    noise at n=200, so a test built on it would assert nothing. At 0.5 the
    correlation is about -0.25 and a model that uses lag 1 measurably gains.
    """
    rng = np.random.default_rng(seed)
    level = np.zeros(periods)
    for t in range(1, periods):
        level[t] = level[t - 1] * (1 - theta) + rng.normal(0.0, noise)
    return World(
        name="MEAN_REVERTING",
        frame=_frame(100.0 * np.exp(level)),
        expectation="returns are negatively autocorrelated; a model that uses lag 1 should gain",
        predictable_mean=True,
    )


def ar_process(
    periods: int = 600, *, phi: float = 0.35, noise: float = 0.01, seed: int = 4
) -> World:
    """AR(1) returns with a known coefficient an AR model should recover."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(periods)
    innovations = rng.normal(0.0, noise, periods)
    for t in range(1, periods):
        returns[t] = phi * returns[t - 1] + innovations[t]
    world = World(
        name="AR_PROCESS",
        frame=_frame(_prices(returns)),
        expectation=f"AR(1) with phi={phi}; an autoregressive model should recover roughly that",
        predictable_mean=True,
    )
    return world


def volatility_clustering(
    periods: int = 600,
    *,
    omega: float = 1e-6,
    alpha: float = 0.1,
    beta: float = 0.85,
    seed: int = 5,
) -> World:
    """GARCH(1,1) returns: predictable variance, unpredictable direction.

    The world that separates the two claims a volatility model is allowed to
    make. A GARCH fit here should beat a constant-variance forecast; nothing
    should beat the mean on direction.
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(periods)
    variance = omega / (1 - alpha - beta)
    for t in range(periods):
        variance = omega + alpha * returns[t - 1] ** 2 + beta * variance if t else variance
        returns[t] = rng.normal(0.0, np.sqrt(variance))
    return World(
        name="VOLATILITY_CLUSTERING",
        frame=_frame(_prices(returns)),
        expectation="variance is predictable and direction is not",
        predictable_mean=False,
        predictable_variance=True,
    )


def nonlinear(periods: int = 600, *, noise: float = 0.006, seed: int = 6) -> World:
    """A threshold process: the mapping bends at zero.

    Yesterday's sign flips the sensitivity, so a straight line through the lag
    cannot represent it and something that can bend should do better.
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(periods)
    innovations = rng.normal(0.0, noise, periods)
    for t in range(1, periods):
        previous = returns[t - 1]
        coefficient = 0.6 if previous < 0 else -0.2
        returns[t] = coefficient * previous + innovations[t]
    return World(
        name="NONLINEAR",
        frame=_frame(_prices(returns)),
        expectation="threshold dependence on the sign of the previous return",
        predictable_mean=True,
    )


def noise_only(periods: int = 600, *, noise: float = 0.02, seed: int = 7) -> World:
    """The control. Unpredictable by construction.

    A model scoring materially better than the mean here has found something
    that is not there. This is the one world where a good result is a bug
    report, and it is the reason the other six can be trusted.
    """
    rng = np.random.default_rng(seed)
    return World(
        name="NOISE_ONLY",
        frame=_frame(_prices(rng.normal(0.0, noise, periods))),
        expectation="nothing is predictable; a model that beats the mean has a leak",
        predictable_mean=False,
    )


#: Built lazily by name so a test can ask for one without generating seven.
BUILDERS = {
    "TREND": trend,
    "STATIONARY": stationary,
    "MEAN_REVERTING": mean_reverting,
    "AR_PROCESS": ar_process,
    "VOLATILITY_CLUSTERING": volatility_clustering,
    "NONLINEAR": nonlinear,
    "NOISE_ONLY": noise_only,
}

WORLD_NAMES = tuple(BUILDERS)


def build_world(name: str, **kwargs: object) -> World:
    if name not in BUILDERS:
        raise KeyError(f"unknown world {name!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[name](**kwargs)  # type: ignore[operator]


def all_worlds() -> list[World]:
    return [build_world(name) for name in WORLD_NAMES]


__all__ = [
    "BUILDERS",
    "WORLD_NAMES",
    "World",
    "all_worlds",
    "ar_process",
    "build_world",
    "mean_reverting",
    "noise_only",
    "nonlinear",
    "stationary",
    "trend",
    "volatility_clustering",
]
