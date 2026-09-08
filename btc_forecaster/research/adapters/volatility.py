"""GARCH, EGARCH and GJR-GARCH -- and what they do and do not forecast.

A volatility model is not a direction model, and the most common way to
overstate one is to score its conditional variance and quietly present the
result as forecasting skill. These three models are registered with an explicit
output contract instead:

``POINT``      the conditional **mean**, which for these specifications is a
               fitted constant. It will score approximately like the naive
               baseline, and that is the honest answer -- a GARCH model makes no
               claim about tomorrow's direction.
``VARIANCE``   the one-step-ahead conditional variance. This is the thing these
               models are actually for, and it is scored with variance metrics,
               never as though it were a return forecast.
``QUANTILES``  from the fitted conditional distribution. Legitimate here in a
               way it is not for a point regressor: a Student-t GARCH *is* a
               distributional model, so its quantiles are the model's own
               statement rather than a Gaussian assumption bolted on afterwards.

Parameters are estimated once on the 1,000 training rows, then applied to the
longer series with ``fix()`` -- the arch package's equivalent of
``apply(refit=False)``. The GARCH recursion computes sigma_t from information
through t-1, so the conditional volatility at the target bar is a genuine
one-step-ahead forecast made from the origin.

Returns are scaled by 100 before estimation. That is not cosmetic: daily log
returns are order 1e-2, the variance is order 1e-4, and the optimiser's default
tolerances are not built for a likelihood surface at that scale.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..contracts import (
    Capability,
    EvaluationContext,
    Family,
    ModelFitError,
    Preprocessing,
    ResourceClass,
    TrainingSet,
    ZooModel,
)
from ..registry import ZooRegistration, register

#: arch estimates far more reliably on percent returns than on decimals.
RETURN_SCALE = 100.0

#: The quantiles every distributional model in the zoo reports, so that pinball
#: loss and interval coverage are computed on the same grid for all of them.
QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class GarchFamilyModel(ZooModel):
    """One specification of the GARCH family, with parameters frozen after training."""

    family = Family.VOLATILITY
    preprocessing = Preprocessing.MODEL_NATIVE
    capabilities = frozenset(
        {Capability.POINT, Capability.VARIANCE, Capability.QUANTILES, Capability.SERIALIZE}
    )
    requires = ("arch",)
    resource_class = ResourceClass.LIGHT

    #: arch's volatility process name, and the asymmetry order. Both typed as
    #: the literals arch accepts, so a typo is a type error rather than a
    #: runtime one three hundred model-fits into a benchmark.
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH"
    p: int = 1
    o: int = 0
    q: int = 1
    #: Standardised Student-t. Daily crypto returns are not Gaussian, and a
    #: normal GARCH produces intervals that are too narrow exactly when it
    #: matters. Typed as the literal arch accepts rather than as `str`, so a
    #: typo is a type error instead of a runtime one.
    dist: Literal["normal", "t", "skewt", "ged"] = "t"

    def __init__(self) -> None:
        super().__init__()
        self._params: pd.Series | None = None
        self._distribution: Any = None

    def _build(self, values: np.ndarray) -> Any:
        from arch import arch_model

        return arch_model(
            values, mean="Constant", vol=self.vol, p=self.p, o=self.o, q=self.q, dist=self.dist
        )

    def _fit(self, train: TrainingSet) -> None:
        values = train.y.to_numpy(dtype=float) * RETURN_SCALE
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self._build(values).fit(disp="off", show_warning=False)
        except Exception as exc:  # noqa: BLE001 -- recorded against the model
            raise ModelFitError(f"{self.model_id} failed to estimate: {exc}") from exc
        self._params = result.params
        self._distribution = result.model.distribution

    def _conditional(self, context: EvaluationContext) -> tuple[np.ndarray, np.ndarray]:
        """(mean, sigma) at each target bar, from the frozen parameters."""
        if self._params is None:
            raise ModelFitError(f"{self.model_id} has no parameters")
        series = context.series.dropna()
        values = series.to_numpy(dtype=float) * RETURN_SCALE
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fixed = self._build(values).fix(self._params)
            sigma = np.asarray(fixed.conditional_volatility, dtype=float)
        except Exception as exc:  # noqa: BLE001
            raise ModelFitError(f"{self.model_id} failed to roll forward: {exc}") from exc

        # sigma[k] is the conditional volatility OF bar k given information
        # through k-1 -- a one-step-ahead forecast made at the origin.
        sigma_series = pd.Series(sigma, index=series.index).reindex(context.target_bars)
        mu = float(self._params["mu"])
        n = len(context)
        return np.full(n, mu, dtype=float), sigma_series.to_numpy(dtype=float)

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        mean, _ = self._conditional(context)
        return mean / RETURN_SCALE

    def _predict_variance(self, context: EvaluationContext) -> np.ndarray:
        _, sigma = self._conditional(context)
        return (sigma / RETURN_SCALE) ** 2

    def _predict_quantiles(self, context: EvaluationContext) -> dict[float, np.ndarray]:
        mean, sigma = self._conditional(context)
        params = self._params
        if params is None:
            raise ModelFitError(f"{self.model_id} has no parameters")
        names = self._distribution.parameter_names()
        shape = params[names].to_numpy(dtype=float) if names else None
        standard = np.asarray(self._distribution.ppf(list(QUANTILE_LEVELS), shape), dtype=float)
        return {
            level: (mean + sigma * standard[i]) / RETURN_SCALE
            for i, level in enumerate(QUANTILE_LEVELS)
        }

    def parameter_count(self) -> int | None:
        return None if self._params is None else int(len(self._params))

    def hyperparameters(self) -> dict:
        estimated = (
            {} if self._params is None else {k: float(v) for k, v in self._params.items()}
        )
        return {
            "vol": self.vol,
            "p": self.p,
            "o": self.o,
            "q": self.q,
            "dist": self.dist,
            "return_scale": RETURN_SCALE,
            "estimated": estimated,
        }


class Garch11(GarchFamilyModel):
    """Symmetric GARCH(1,1): yesterday's shock moves variance regardless of sign."""

    model_id = "garch_11"
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH"
    p, o, q = 1, 0, 1


class EGarch11(GarchFamilyModel):
    """EGARCH(1,1): log-variance, so positivity is structural rather than constrained.

    Its asymmetry term lets a negative shock raise variance more than a positive
    one of the same size -- the leverage effect, which is the standard reason to
    prefer it over symmetric GARCH.
    """

    model_id = "egarch_11"
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "EGARCH"
    p, o, q = 1, 1, 1
    resource_class = ResourceClass.MODERATE


class GjrGarch11(GarchFamilyModel):
    """GJR-GARCH(1,1,1): the same asymmetry in levels rather than logs.

    Registered alongside EGARCH rather than instead of it because they impose
    the leverage effect differently -- an indicator term on negative shocks
    versus a sign term in the log-variance recursion -- and disagree in the
    tails, which is where a variance forecast is worth having.
    """

    model_id = "gjr_garch_11"
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH"
    p, o, q = 1, 1, 1
    resource_class = ResourceClass.MODERATE


for _cls, _description in (
    (Garch11, "Symmetric GARCH(1,1) with Student-t innovations on percent log returns."),
    (
        EGarch11,
        "EGARCH(1,1,1) with Student-t innovations: log-variance with a leverage term.",
    ),
    (
        GjrGarch11,
        "GJR-GARCH(1,1,1) with Student-t innovations: threshold asymmetry in levels.",
    ),
):
    register(
        ZooRegistration(
            model_id=_cls.model_id,
            factory=_cls,
            family=Family.VOLATILITY,
            resource_class=_cls.resource_class,
            description=_description,
            requires=("arch",),
            notes=(
                "Forecasts conditional variance, not direction. Its POINT output "
                "is the fitted constant mean and will score like the naive "
                "baseline; that is the honest answer, not a failure.",
                "Quantiles come from the fitted Student-t conditional "
                "distribution, so they are the model's own statement rather "
                "than a Gaussian assumption added afterwards.",
            ),
        )
    )


__all__ = [
    "QUANTILE_LEVELS",
    "RETURN_SCALE",
    "EGarch11",
    "Garch11",
    "GarchFamilyModel",
    "GjrGarch11",
]
