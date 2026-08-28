"""Deterministic synthetic market data for tests and examples.

No unit test in this repository touches the network. Everything that needs a
price series builds one here, seeded, so failures are reproducible and CI does
not depend on Yahoo Finance being up or on what BTC did last night.

The generators are deliberately *not* realistic BTC. They are constructed so
that specific properties are known in advance -- a pure random walk has no
predictable direction, a trending series has a known drift, a series with an
injected AR(1) term has a known autocorrelation -- which is what makes them
useful for asserting that a model or metric behaves correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .timebase import UTC


def daily_index(periods: int, start: str = "2019-01-01") -> pd.DatetimeIndex:
    """A gapless run of UTC daily bar labels."""
    return pd.date_range(start=start, periods=periods, freq="D", tz=UTC, name="date")


def random_walk_prices(
    periods: int = 800,
    *,
    start: str = "2019-01-01",
    initial_price: float = 10_000.0,
    daily_vol: float = 0.03,
    drift: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Geometric random walk. Direction is unpredictable by construction."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=drift, scale=daily_vol, size=periods)
    log_price = np.log(initial_price) + np.cumsum(shocks)
    return pd.Series(np.exp(log_price), index=daily_index(periods, start), name="close")


def ar1_return_prices(
    periods: int = 800,
    *,
    start: str = "2019-01-01",
    initial_price: float = 10_000.0,
    phi: float = 0.4,
    daily_vol: float = 0.02,
    seed: int = 0,
) -> pd.Series:
    """Prices whose *returns* follow AR(1) with coefficient ``phi``.

    Gives the diagnostics and autoregressive models something with a known,
    detectable signal, unlike the pure random walk.
    """
    rng = np.random.default_rng(seed)
    innovations = rng.normal(scale=daily_vol, size=periods)
    returns = np.zeros(periods)
    for t in range(1, periods):
        returns[t] = phi * returns[t - 1] + innovations[t]
    log_price = np.log(initial_price) + np.cumsum(returns)
    return pd.Series(np.exp(log_price), index=daily_index(periods, start), name="close")


def synthetic_market_frame(
    periods: int = 800,
    *,
    start: str = "2019-01-01",
    kind: str = "random_walk",
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """A contract-satisfying market frame: UTC daily index, close and volume.

    ``kind`` selects the price process: ``"random_walk"``, ``"trend"`` (random
    walk with positive drift) or ``"ar1"``.
    """
    if kind == "random_walk":
        close = random_walk_prices(periods, start=start, seed=seed, **kwargs)
    elif kind == "trend":
        kwargs.setdefault("drift", 0.0015)
        close = random_walk_prices(periods, start=start, seed=seed, **kwargs)
    elif kind == "ar1":
        close = ar1_return_prices(periods, start=start, seed=seed, **kwargs)
    else:
        raise ValueError(f"unknown synthetic series kind: {kind!r}")

    rng = np.random.default_rng(seed + 9_973)
    volume = pd.Series(
        rng.lognormal(mean=20.0, sigma=0.4, size=periods),
        index=close.index,
        name="volume",
    )
    frame = pd.concat([close, volume], axis=1)
    frame.index.name = "date"
    return frame


def constant_growth_frame(
    periods: int = 400,
    *,
    start: str = "2019-01-01",
    initial_price: float = 100.0,
    daily_growth: float = 0.001,
) -> pd.DataFrame:
    """A noiseless exponential series.

    Every reasonable forecaster should nail this. Used to assert that model
    adapters are wired up correctly before asking anything harder of them.
    """
    index = daily_index(periods, start)
    close = pd.Series(initial_price * np.exp(daily_growth * np.arange(periods)), index=index, name="close")
    volume = pd.Series(np.full(periods, 1_000.0), index=index, name="volume")
    frame = pd.concat([close, volume], axis=1)
    frame.index.name = "date"
    return frame


__all__ = [
    "ar1_return_prices",
    "constant_growth_frame",
    "daily_index",
    "random_walk_prices",
    "synthetic_market_frame",
]
