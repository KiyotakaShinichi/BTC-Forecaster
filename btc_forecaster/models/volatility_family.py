"""Comparing volatility model variants, instead of assuming GARCH(1,1).

The legacy pipeline fitted GARCH(1,1) with Gaussian innovations unconditionally
and never checked whether that specification earned its place. Two questions go
unanswered by that:

1. **Does the conditional variance model help at all?** A constant-volatility
   band is the control. If GARCH and constant score the same on interval
   calibration, the GARCH machinery is decoration.
2. **Are Gaussian innovations defensible?** Jarque-Bera rejects normality on BTC
   returns decisively. A Student-t innovation has fatter tails and should
   produce better-calibrated intervals at the extremes -- or, if it does not,
   that is itself worth knowing.

This module fits a small, CPU-friendly set of variants and compares them on
information criteria *and* on out-of-sample interval calibration, because those
two can disagree: AIC rewards in-sample likelihood, which is not what a
prediction interval is for.

Deliberately not exhaustive (A2.13 says so explicitly). Six specifications, not
every combination of every option.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .volatility import InnovationDist, VolatilityForecast, constant_volatility

#: The comparison set: two asymmetry treatments crossed with two innovation
#: distributions, plus the symmetric baseline and the no-model control.
#:
#: EGARCH and GJR-GARCH both capture the leverage effect (negative returns raise
#: future volatility more than positive ones do), by different means -- EGARCH
#: models log variance so no positivity constraint is needed; GJR adds an
#: indicator term. Including both distinguishes "asymmetry matters" from "this
#: particular parameterisation of asymmetry matters".
VOLATILITY_SPECS: tuple[dict, ...] = (
    {"key": "constant", "vol": None, "dist": "normal"},
    {"key": "garch11_normal", "vol": "GARCH", "o": 0, "dist": "normal"},
    {"key": "garch11_t", "vol": "GARCH", "o": 0, "dist": "t"},
    {"key": "gjr_normal", "vol": "GARCH", "o": 1, "dist": "normal"},
    {"key": "gjr_t", "vol": "GARCH", "o": 1, "dist": "t"},
    {"key": "egarch_t", "vol": "EGARCH", "o": 1, "dist": "t"},
)


@dataclass(frozen=True)
class VolatilityFit:
    """One fitted volatility specification and what it cost."""

    key: str
    model: str
    distribution: str
    aic: float
    bic: float
    log_likelihood: float
    n_params: int
    fit_seconds: float
    converged: bool
    params: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "model": self.model,
            "distribution": self.distribution,
            "aic": self.aic,
            "bic": self.bic,
            "log_likelihood": self.log_likelihood,
            "n_params": self.n_params,
            "fit_seconds": self.fit_seconds,
            "converged": self.converged,
            "params": dict(self.params),
            "error": self.error,
        }


def fit_volatility_spec(
    log_returns: pd.Series,
    spec: dict,
    *,
    steps: int,
) -> tuple[VolatilityFit, VolatilityForecast | None]:
    """Fit one specification and forecast ``steps`` ahead.

    Returns the fit summary and its volatility forecast. A specification that
    fails to converge is reported as a non-converged fit rather than raising:
    non-convergence is evidence about that specification.
    """
    import time

    started = time.perf_counter()
    key = spec["key"]

    if spec["vol"] is None:
        forecast = constant_volatility(log_returns, steps)
        return (
            VolatilityFit(
                key=key,
                model="constant",
                distribution="n/a",
                aic=float("nan"),
                bic=float("nan"),
                log_likelihood=float("nan"),
                n_params=1,
                fit_seconds=time.perf_counter() - started,
                converged=True,
                params=dict(forecast.params),
            ),
            forecast,
        )

    from arch import arch_model

    series = log_returns.dropna() * 100.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                series,
                vol=spec["vol"],
                p=1,
                o=spec.get("o", 0),
                q=1,
                mean="Zero",
                dist=spec["dist"],
            )
            fitted = model.fit(disp="off", show_warning=False)
            variance = np.asarray(
                fitted.forecast(horizon=steps, reindex=False).variance.values[-1]
            )
        sigma = (np.sqrt(variance) / 100.0).astype(float)

        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise RuntimeError("non-finite or non-positive conditional volatility")

        label = f"{spec['vol']}(1,{spec.get('o', 0)},1)"
        return (
            VolatilityFit(
                key=key,
                model=label,
                distribution=str(spec["dist"]),
                aic=float(fitted.aic),
                bic=float(fitted.bic),
                log_likelihood=float(fitted.loglikelihood),
                n_params=int(len(fitted.params)),
                fit_seconds=time.perf_counter() - started,
                converged=True,
                params={
                    k: float(v) for k, v in fitted.params.items() if np.isfinite(v)
                },
            ),
            VolatilityForecast(
                sigma=sigma,
                model=f"{label}-{spec['dist']}",
                params={k: float(v) for k, v in fitted.params.items() if np.isfinite(v)},
            ),
        )
    except Exception as exc:
        return (
            VolatilityFit(
                key=key,
                model=str(spec["vol"]),
                distribution=str(spec["dist"]),
                aic=float("nan"),
                bic=float("nan"),
                log_likelihood=float("nan"),
                n_params=0,
                fit_seconds=time.perf_counter() - started,
                converged=False,
                error=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )


def compare_volatility_specs(
    log_returns: pd.Series,
    *,
    steps: int = 30,
    specs: tuple[dict, ...] = VOLATILITY_SPECS,
) -> pd.DataFrame:
    """Fit every specification and rank by AIC.

    In-sample only. AIC says which specification describes the training returns
    best; it does not say which produces the best prediction interval. Use
    :func:`score_interval_calibration` for that, and expect the two to disagree.
    """
    rows = []
    for spec in specs:
        fit, _ = fit_volatility_spec(log_returns, spec, steps=steps)
        rows.append(fit.to_dict())

    frame = pd.DataFrame(rows).set_index("key")
    converged = frame[frame["converged"] & frame["aic"].notna()]
    if not converged.empty:
        frame["delta_aic"] = frame["aic"] - converged["aic"].min()
    return frame


def score_interval_calibration(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    level: float = 0.95,
) -> dict:
    """Out-of-sample interval quality for one specification.

    Reports coverage, its distance from nominal, mean relative width and the
    Winkler score. Coverage alone cannot rank specifications -- a band of
    infinite width covers everything -- so the proper scoring rule is what
    decides, and the width is reported so the trade is visible.
    """
    from ..evaluation.metrics import (
        interval_coverage,
        relative_interval_width,
        winkler_score,
    )

    coverage = interval_coverage(actual, lower, upper)
    return {
        "interval_coverage": coverage,
        "coverage_error": abs(coverage - level),
        "relative_interval_width": relative_interval_width(actual, lower, upper),
        "winkler_score": winkler_score(actual, lower, upper, level=level),
        "n": int(len(actual)),
        "nominal_level": level,
    }


def interval_width_growth(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    monotonic_tolerance: float = 0.02,
    n_checkpoints: int = 10,
) -> dict:
    """How the band widens with horizon. The A2.12 accumulation check.

    Under accumulating shocks the log width grows like ``sqrt(h)``, so the ratio
    of the last width to the first should be about ``sqrt(horizon)``. A ratio
    near 1 means the shocks were drawn independently per step rather than
    accumulated -- the defect Track A fixed, restated as a monitorable quantity.

    ``monotonically_widening`` is checked across ``n_checkpoints`` evenly spaced
    positions rather than between adjacent steps. That is deliberate, and it is
    the right question: "does the band widen with horizon" is a claim about the
    trend, not about every consecutive pair. A Monte Carlo band's endpoints are
    empirical percentiles whose sampling error is O(1/sqrt(n_paths)), so a
    correctly accumulating band routinely dips a few percent between adjacent
    steps -- at 5,000 paths the observed dip runs to ~6%. Checking adjacent
    steps would therefore test the random number generator, and would need a
    tolerance so loose it could no longer detect a genuinely flat band.
    Averaging over checkpoint spacing removes the noise while leaving the flat
    case easy to detect.

    ``strictly_monotonic`` keeps the exact adjacent-step check, which analytic
    intervals satisfy because they have no sampling noise.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if len(lower) < 2:
        return {
            "growth_ratio": float("nan"),
            "expected_sqrt_ratio": float("nan"),
            "monotonically_widening": True,
            "strictly_monotonic": True,
            "monotonic_tolerance": monotonic_tolerance,
            "n_checkpoints": 0,
            "accumulates": False,
            "first_width": float("nan"),
            "last_width": float("nan"),
            "max_relative_dip": 0.0,
            "max_adjacent_relative_dip": 0.0,
        }

    log_width = np.log(upper) - np.log(lower)
    horizon = len(lower)
    expected = float(np.sqrt(horizon))
    ratio = float(log_width[-1] / log_width[0]) if log_width[0] > 0 else float("nan")

    scale = float(log_width[0]) if log_width[0] > 0 else 1.0

    checkpoints = np.unique(
        np.linspace(0, horizon - 1, min(n_checkpoints, horizon), dtype=int)
    )
    checkpoint_steps = np.diff(log_width[checkpoints])
    max_checkpoint_dip = (
        float(-min(checkpoint_steps.min(), 0.0) / scale) if len(checkpoint_steps) else 0.0
    )

    adjacent_steps = np.diff(log_width)
    max_adjacent_dip = float(-min(adjacent_steps.min(), 0.0) / scale)

    return {
        "growth_ratio": ratio,
        "expected_sqrt_ratio": expected,
        "monotonically_widening": bool(max_checkpoint_dip <= monotonic_tolerance),
        "strictly_monotonic": bool(np.all(adjacent_steps >= -1e-9)),
        "monotonic_tolerance": monotonic_tolerance,
        "n_checkpoints": int(len(checkpoints)),
        "max_relative_dip": max_checkpoint_dip,
        "max_adjacent_relative_dip": max_adjacent_dip,
        "accumulates": bool(np.isfinite(ratio) and ratio > 1.5),
        "first_width": float(log_width[0]),
        "last_width": float(log_width[-1]),
    }


__all__ = [
    "VOLATILITY_SPECS",
    "InnovationDist",
    "VolatilityFit",
    "compare_volatility_specs",
    "fit_volatility_spec",
    "interval_width_growth",
    "score_interval_calibration",
]
