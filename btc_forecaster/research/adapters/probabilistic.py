"""Models that forecast a distribution, and the words that may be used for it.

Phase 8 says not to call an interval a "confidence" interval unless the
statistical meaning is correct, and that restraint is the whole design here.
Three models, three different guarantees, none of them overstated:

**Quantile gradient boosting** and **quantile regression** estimate conditional
quantiles directly, by minimising pinball loss at each level. What they produce
is a conditional quantile estimate. It is not a confidence interval, it carries
no coverage guarantee, and its calibration is an empirical question that the
benchmark answers rather than assumes.

**Split-conformal Ridge** is the one with a finite-sample guarantee -- and the
guarantee requires **exchangeability**, which a time series does not satisfy.
That is not a technicality to bury: conformal prediction promises marginal
coverage when calibration and test points are exchangeable, and consecutive
daily returns from a volatility-clustered series are not. So the interval is
reported as a *split-conformal interval calibrated on DEV*, its empirical
coverage on HOLDOUT is measured, and the gap between nominal and empirical is
treated as the interesting quantity rather than as an error to explain away.

All three also report a **direction probability**, and it is genuinely derived
rather than invented: given a set of conditional quantiles, P(return > 0) is
read off the estimated quantile function at zero by interpolation. A point
regressor cannot do this, which is why the point regressors do not claim it.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

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
from ..preprocessing import Scaler, fit_scaler
from ..registry import ZooRegistration, register
from .volatility import QUANTILE_LEVELS


def direction_probability_from_quantiles(
    quantiles: dict[float, np.ndarray],
) -> np.ndarray:
    """P(next log return > 0), read off the estimated quantile function.

    The quantile function is a step estimate on a coarse grid, so the value at
    zero is interpolated between the two bracketing levels. Where zero falls
    outside the estimated range the probability is clipped to the outermost
    level rather than extrapolated -- an extrapolated tail probability from
    seven quantiles is a number with no evidence behind it.
    """
    levels = np.array(sorted(quantiles), dtype=float)
    values = np.vstack([quantiles[level] for level in levels])  # (n_levels, n_rows)
    out = np.empty(values.shape[1], dtype=float)
    for i in range(values.shape[1]):
        column = values[:, i]
        # P(Y <= 0) by interpolating the level at which the quantile crosses 0.
        if 0.0 <= column[0]:
            below = float(levels[0])
        elif 0.0 >= column[-1]:
            below = float(levels[-1])
        else:
            below = float(np.interp(0.0, column, levels))
        out[i] = 1.0 - below
    return np.clip(out, 0.0, 1.0)


class QuantileModel(ZooModel):
    """Shared surface for the two direct quantile estimators."""

    family = Family.PROBABILISTIC
    preprocessing = Preprocessing.STANDARDIZED
    capabilities = frozenset(
        {
            Capability.POINT,
            Capability.QUANTILES,
            Capability.DIRECTION_PROBABILITY,
            Capability.SERIALIZE,
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._models: dict[float, Any] = {}
        self._scaler: Scaler | None = None
        self._feature_names: tuple[str, ...] = ()

    def _fit_one(self, X: np.ndarray, y: np.ndarray, level: float) -> Any:
        raise NotImplementedError

    def _predict_one(self, model: Any, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _fit(self, train: TrainingSet) -> None:
        self._scaler = fit_scaler(train.X, self.preprocessing)
        self._feature_names = tuple(train.X.columns)
        X = self._scaler.transform(train.X).to_numpy(dtype=float)
        y = train.y.to_numpy(dtype=float)
        for level in QUANTILE_LEVELS:
            try:
                self._models[level] = self._fit_one(X, y, level)
            except Exception as exc:  # noqa: BLE001
                raise ModelFitError(
                    f"{self.model_id} failed to fit the {level} quantile: {exc}"
                ) from exc

    def _design(self, context: EvaluationContext) -> np.ndarray:
        if self._scaler is None:
            raise ModelFitError(f"{self.model_id} has no fitted scaler")
        if tuple(context.X.columns) != self._feature_names:
            raise ModelFitError(f"{self.model_id} was fitted on a different feature set")
        return self._scaler.transform(context.X).to_numpy(dtype=float)

    def _raw_quantiles(self, context: EvaluationContext) -> dict[float, np.ndarray]:
        X = self._design(context)
        return {
            level: np.asarray(self._predict_one(model, X), dtype=float)
            for level, model in self._models.items()
        }

    def _predict_quantiles(self, context: EvaluationContext) -> dict[float, np.ndarray]:
        raw = self._raw_quantiles(context)
        levels = sorted(raw)
        stacked = np.vstack([raw[level] for level in levels])
        # Independently fitted quantile models can cross. Sorting each column is
        # the standard rearrangement fix, and it cannot make calibration worse:
        # a crossed pair is a statement the model could not have meant.
        stacked = np.sort(stacked, axis=0)
        return {level: stacked[i] for i, level in enumerate(levels)}

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        """The conditional median, which is what a pinball-fitted model estimates.

        Not the mean. A quantile model has no mean to report, and substituting
        one would be reporting a quantity it never estimated.
        """
        return self._predict_quantiles(context)[0.5]

    def _predict_direction_probability(self, context: EvaluationContext) -> np.ndarray:
        return direction_probability_from_quantiles(self._predict_quantiles(context))

    def hyperparameters(self) -> dict:
        return {
            "quantile_levels": list(QUANTILE_LEVELS),
            "n_sub_models": len(self._models),
            "point_forecast_is": "conditional median",
        }


class QuantileGradientBoosting(QuantileModel):
    """One gradient-boosted model per quantile level, each on pinball loss."""

    model_id = "quantile_gbr"
    resource_class = ResourceClass.MODERATE
    requires = ("sklearn",)
    n_estimators = 200
    learning_rate = 0.05

    def _fit_one(self, X: np.ndarray, y: np.ndarray, level: float) -> Any:
        from sklearn.ensemble import GradientBoostingRegressor

        from ._tabular import SEED

        return GradientBoostingRegressor(
            loss="quantile",
            alpha=level,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=3,
            min_samples_leaf=20,
            random_state=SEED,
        ).fit(X, y)

    def _predict_one(self, model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def hyperparameters(self) -> dict:
        return {
            **super().hyperparameters(),
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": 3,
        }


class LinearQuantileRegression(QuantileModel):
    """Koenker-Bassett quantile regression, one fit per level.

    statsmodels rather than scikit-learn's `QuantileRegressor`: the latter
    solves a linear program that is slow at this width, and this family already
    costs seven fits per model.
    """

    model_id = "quantile_linear"
    resource_class = ResourceClass.LIGHT
    requires = ("statsmodels",)

    def _fit_one(self, X: np.ndarray, y: np.ndarray, level: float) -> Any:
        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm.QuantReg(y, sm.add_constant(X, has_constant="add")).fit(q=level)

    def _predict_one(self, model: Any, X: np.ndarray) -> np.ndarray:
        import statsmodels.api as sm

        return model.predict(sm.add_constant(X, has_constant="add"))

    def parameter_count(self) -> int | None:
        if not self._models:
            return None
        per_level = len(self._feature_names) + 1
        return per_level * len(self._models)


class ConformalRidge(ZooModel):
    """Ridge with split-conformal intervals calibrated on DEV.

    The interval is built from the empirical quantiles of the **absolute
    residuals on a block the model never saw**, which is what makes it
    conformal rather than a resubstitution interval.

    Its guarantee is stated carefully because it does not hold here. Split
    conformal delivers marginal coverage at least ``1 - alpha`` when the
    calibration and test points are *exchangeable*. Daily returns from a
    volatility-clustered series are not exchangeable: a calm calibration block
    and a turbulent evaluation block break the assumption in the direction that
    makes intervals too narrow. So the nominal level is reported as nominal, the
    empirical coverage is measured on HOLDOUT, and the gap is the result.
    """

    model_id = "conformal_ridge"
    family = Family.PROBABILISTIC
    resource_class = ResourceClass.TRIVIAL
    preprocessing = Preprocessing.STANDARDIZED
    requires = ("sklearn",)
    needs_calibration = True
    capabilities = frozenset(
        {
            Capability.POINT,
            Capability.QUANTILES,
            Capability.DIRECTION_PROBABILITY,
            Capability.SERIALIZE,
        }
    )
    alpha = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._ridge: Any = None
        self._scaler: Scaler | None = None
        self._feature_names: tuple[str, ...] = ()
        self._residual_quantiles: dict[float, float] = {}
        self._calibration_rows = 0

    def _fit(self, train: TrainingSet) -> None:
        from sklearn.linear_model import Ridge as SkRidge

        from ._tabular import SEED

        self._scaler = fit_scaler(train.X, self.preprocessing)
        self._feature_names = tuple(train.X.columns)
        X = self._scaler.transform(train.X).to_numpy(dtype=float)
        self._ridge = SkRidge(alpha=self.alpha, random_state=SEED).fit(
            X, train.y.to_numpy(dtype=float)
        )

    def _design(self, context: EvaluationContext) -> np.ndarray:
        if self._scaler is None:
            raise ModelFitError(f"{self.model_id} has no fitted scaler")
        return self._scaler.transform(context.X).to_numpy(dtype=float)

    def _calibrate(self, context: EvaluationContext) -> None:
        residuals = context.y.to_numpy(dtype=float) - self._ridge.predict(self._design(context))
        self._calibration_rows = int(len(residuals))
        # Signed residual quantiles rather than absolute ones: an absolute
        # residual band is symmetric by construction, and a return distribution
        # with a heavier left tail is exactly the asymmetry worth keeping.
        self._residual_quantiles = {
            level: float(np.quantile(residuals, level)) for level in QUANTILE_LEVELS
        }

    def _require_calibration(self) -> None:
        if not self._residual_quantiles:
            raise ModelFitError(
                f"{self.model_id} has not been calibrated; a conformal interval "
                "without a held-out calibration block is a resubstitution "
                "interval wearing a conformal label"
            )

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        return np.asarray(self._ridge.predict(self._design(context)), dtype=float)

    def _predict_quantiles(self, context: EvaluationContext) -> dict[float, np.ndarray]:
        self._require_calibration()
        centre = self._predict_point(context)
        return {level: centre + offset for level, offset in self._residual_quantiles.items()}

    def _predict_direction_probability(self, context: EvaluationContext) -> np.ndarray:
        return direction_probability_from_quantiles(self._predict_quantiles(context))

    def parameter_count(self) -> int | None:
        if self._ridge is None:
            return None
        return int(np.size(self._ridge.coef_)) + 1

    def hyperparameters(self) -> dict:
        return {
            "alpha": self.alpha,
            "calibration_rows": self._calibration_rows,
            "calibration_block": "DEV",
            "residual_quantiles": dict(self._residual_quantiles),
            "guarantee": (
                "split-conformal marginal coverage requires exchangeability, "
                "which this series does not satisfy; nominal levels are nominal "
                "and empirical coverage is measured on HOLDOUT"
            ),
        }


register(
    ZooRegistration(
        model_id="quantile_gbr",
        factory=QuantileGradientBoosting,
        family=Family.PROBABILISTIC,
        resource_class=ResourceClass.MODERATE,
        description="Gradient boosting on pinball loss, one model per quantile level.",
        requires=("sklearn",),
        notes=(
            "Estimates conditional quantiles. Not a confidence interval and it "
            "carries no coverage guarantee; calibration is measured, not claimed.",
            "Its point forecast is the conditional median, because that is what "
            "pinball loss at 0.5 estimates. It has no mean to report.",
            "Independently fitted quantiles can cross; the columns are sorted, "
            "which cannot hurt calibration because a crossed pair is a statement "
            "the model could not have meant.",
        ),
    )
)

register(
    ZooRegistration(
        model_id="quantile_linear",
        factory=LinearQuantileRegression,
        family=Family.PROBABILISTIC,
        resource_class=ResourceClass.LIGHT,
        description="Koenker-Bassett linear quantile regression, one fit per level.",
        requires=("statsmodels",),
        notes=(
            "The linear control for quantile_gbr: same loss, same levels, no "
            "capacity to fit a non-linear conditional distribution.",
        ),
    )
)

register(
    ZooRegistration(
        model_id="conformal_ridge",
        factory=ConformalRidge,
        family=Family.PROBABILISTIC,
        resource_class=ResourceClass.TRIVIAL,
        description="Ridge with split-conformal intervals calibrated on the DEV block.",
        requires=("sklearn",),
        notes=(
            "The conformal coverage guarantee requires exchangeability, which a "
            "volatility-clustered daily return series does not satisfy. The "
            "nominal level is reported as nominal and the empirical coverage is "
            "measured; the gap between them is the result.",
            "Calibrated on DEV, never on HOLDOUT -- calibrating on the block it "
            "is about to be scored against would guarantee coverage by "
            "construction and measure nothing.",
            "Uses signed residual quantiles rather than absolute ones, so an "
            "asymmetric return distribution produces an asymmetric interval.",
        ),
    )
)


__all__ = [
    "ConformalRidge",
    "LinearQuantileRegression",
    "QuantileGradientBoosting",
    "QuantileModel",
    "direction_probability_from_quantiles",
]
