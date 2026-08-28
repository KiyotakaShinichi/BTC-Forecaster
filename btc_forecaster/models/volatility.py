"""GARCH volatility and Monte Carlo prediction intervals.

Contains the fix for the most consequential modelling bug in the original
pipeline.

The original built its 95% band like this::

    garch_forecast = garch_fitted.forecast(horizon=HORIZON_DAYS)
    cond_var = garch_forecast.variance.values[-1] / 10000
    cond_std = np.sqrt(cond_var)

    for i in range(MONTE_CARLO_RUNS):
        z = np.random.normal(size=HORIZON_DAYS)
        mc_matrix[i] = np.exp(combined_log + z * cond_std)

Each horizon step receives one independent shock scaled by that step's
conditional standard deviation. But a price is the *accumulation* of its
shocks: uncertainty about ``P[T+365]`` includes every shock between ``T`` and
``T+365``, not just the one on the final day. Because GARCH conditional variance
mean-reverts to its unconditional level within a few weeks, ``cond_std`` is
nearly flat past the short term -- so the resulting band was roughly constant
width a year out, when it should have grown like ``sqrt(h)``.

For BTC at ~3% daily volatility the understatement is large: a correct 365-day
band spans a factor of several, the original spanned a few percent. Any interval
coverage statistic computed against it would be catastrophically miscalibrated,
which is why coverage is now a first-class metric (see
:mod:`btc_forecaster.evaluation.metrics`).

The fix is one ``cumsum``: shocks accumulate along each simulated path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityForecast:
    """Per-step conditional volatility of log returns over a horizon."""

    sigma: np.ndarray
    model: str
    params: dict

    def __len__(self) -> int:
        return len(self.sigma)


def constant_volatility(log_returns: pd.Series, steps: int) -> VolatilityForecast:
    """Flat volatility at the sample standard deviation.

    The fallback when ``arch`` is unavailable, and a useful control: if a GARCH
    band and a constant-volatility band score the same on coverage, the GARCH
    machinery is not earning its complexity.
    """
    sigma = float(log_returns.dropna().std(ddof=1))
    return VolatilityForecast(
        sigma=np.full(steps, sigma, dtype=float),
        model="constant",
        params={"sigma": sigma},
    )


def garch_volatility(
    log_returns: pd.Series,
    steps: int,
    *,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> VolatilityForecast:
    """GARCH(p,q) conditional volatility forecast for log returns.

    Returns are rescaled by 100 before fitting and the result scaled back.
    ``arch`` warns about poorly-scaled data otherwise -- daily log returns are
    O(0.01), and the optimiser struggles at that magnitude.
    """
    from arch import arch_model

    series = log_returns.dropna() * 100.0
    fitted = arch_model(series, vol="GARCH", p=p, q=q, mean="Zero", dist=dist).fit(disp="off")
    variance = np.asarray(fitted.forecast(horizon=steps, reindex=False).variance.values[-1])
    sigma = np.sqrt(variance) / 100.0

    return VolatilityForecast(
        sigma=sigma.astype(float),
        model=f"GARCH({p},{q})",
        params={
            key: float(value)
            for key, value in fitted.params.items()
            if np.isfinite(value)
        },
    )


def simulate_price_paths(
    mean_log_path: np.ndarray,
    volatility: VolatilityForecast,
    *,
    n_paths: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Monte Carlo price paths around a mean log-price path.

    Shocks **accumulate**: the deviation of a path at step ``h`` is the sum of
    its shocks over steps ``1..h``. Under constant sigma this reproduces the
    analytic ``sigma * sqrt(h)`` widening; under GARCH it also captures the
    conditional variance term structure in the short term.

    Returns an ``(n_paths, horizon)`` array of prices.
    """
    mean_log_path = np.asarray(mean_log_path, dtype=float)
    steps = len(mean_log_path)
    if len(volatility) != steps:
        raise ValueError(
            f"volatility forecast covers {len(volatility)} steps but the mean path has {steps}"
        )
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, steps)) * volatility.sigma[None, :]
    # The one-line fix: a price accumulates its shocks along the path.
    cumulative = np.cumsum(shocks, axis=1)
    return np.exp(mean_log_path[None, :] + cumulative)


def monte_carlo_interval(
    mean_log_path: np.ndarray,
    volatility: VolatilityForecast,
    *,
    level: float = 0.95,
    n_paths: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(point, lower, upper)`` from simulated paths.

    The point forecast is the exponentiated mean log path -- the median of the
    simulated distribution, not its mean. For a lognormal the mean sits above
    the median by ``exp(sigma^2 h / 2)``, which at long horizons is a large and
    entirely artefactual upward bias. Reporting the median keeps the point
    forecast equal to the underlying model's own prediction.
    """
    paths = simulate_price_paths(mean_log_path, volatility, n_paths=n_paths, seed=seed)
    tail = (1.0 - level) / 2.0
    lower = np.percentile(paths, 100.0 * tail, axis=0)
    upper = np.percentile(paths, 100.0 * (1.0 - tail), axis=0)
    return np.exp(np.asarray(mean_log_path, dtype=float)), lower, upper


def analytic_interval(
    mean_log_path: np.ndarray,
    volatility: VolatilityForecast,
    *,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form equivalent of :func:`monte_carlo_interval`.

    Variance accumulates as the sum of per-step conditional variances. Used to
    cross-check the simulation: if the two disagree beyond sampling error, the
    simulation is wrong.
    """
    from scipy.stats import norm

    mean_log_path = np.asarray(mean_log_path, dtype=float)
    cumulative_var = np.cumsum(volatility.sigma**2)
    z = float(norm.ppf(0.5 + level / 2.0))
    sd = np.sqrt(cumulative_var)

    point = np.exp(mean_log_path)
    return point, np.exp(mean_log_path - z * sd), np.exp(mean_log_path + z * sd)


__all__ = [
    "VolatilityForecast",
    "analytic_interval",
    "constant_volatility",
    "garch_volatility",
    "monte_carlo_interval",
    "simulate_price_paths",
]
