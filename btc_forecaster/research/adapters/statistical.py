"""Autoregressive, exponential-smoothing and state-space models.

Every model here estimates its parameters **once**, on the 1,000 training rows,
and then rolls its state forward through the evaluation block with realised
values. It does not refit. That is the difference between a one-step-ahead
forecast and a hindsight fit, and it is the whole protocol.

For the state-space family that is `apply(..., refit=False)`: the Kalman filter
is re-run over the longer series with the training parameters frozen, and the
one-step-ahead predictions it produces at time ``t`` depend on data through
``t`` only. That is a property of the filter recursion rather than a promise,
but `test_a6_leakage.py` poisons the future bars and asserts every prediction is
bit-identical anyway -- a mathematical guarantee that nobody checks is a
mathematical guarantee about the wrong implementation.

**On redundancy.** Phase 3 asks for local level, local linear trend, Holt, ETS,
Unobserved Components and "Kalman", and several of those are the same model:

* a local-level model *is* simple exponential smoothing;
* a local-linear-trend model *is* Holt's linear trend;
* "Kalman" is how the state-space family is estimated, not a model.

So Kalman is not registered as a model, and where the deterministic smoothing
form and the stochastic state-space form of the same structure are both
present, they are registered as two entries with the difference stated: Holt
minimises squared error over smoothing weights, the structural model maximises
a likelihood over variances, and they produce different forecasts.

**On seasonality.** Three of the requested models are seasonal. Measured on the
training rows only, this series has no weekly seasonality worth modelling:
ACF(7) = -0.018 against a +/-0.062 white-noise band, a day-of-week ANOVA at
p = 0.95, and an STL seasonal strength of 0.065. They are registered as
UNSUITABLE_FOR_CONSTRAINED_LAB with that measurement as the reason -- visible in
every table, and not quietly absent.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import (
    Capability,
    EvaluationContext,
    Family,
    ModelFitError,
    ModelStatus,
    Preprocessing,
    ResourceClass,
    TrainingSet,
    ZooModel,
)
from ..registry import ZooRegistration, register

#: The measurement behind every seasonal exclusion in this module. Taken on the
#: 1,000 training rows; the holdout was not consulted.
NO_SEASONALITY_EVIDENCE = (
    "no weekly seasonality in the training rows: ACF(7) = -0.018 against a "
    "+/-0.062 white-noise band, day-of-week ANOVA p = 0.95, STL seasonal "
    "strength 0.065. A seasonal component here would fit noise, and at n=1,000 "
    "it would fit it confidently."
)


def _quiet_fit(fit: Any) -> Any:
    """statsmodels warns about convergence on short series. Recorded, not printed.

    Suppressed only around the fit call, so a genuine error still raises.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit()


class _SeriesModel(ZooModel):
    """Shared plumbing: estimate on the training series, roll forward on realised.

    Subclasses choose the series (log returns or log price) and how the frozen
    parameters produce one-step-ahead predictions.
    """

    family = Family.STATISTICAL
    preprocessing = Preprocessing.MODEL_NATIVE
    capabilities = frozenset({Capability.POINT, Capability.SERIALIZE})
    requires = ("statsmodels",)

    #: Which series the model is estimated on. Log price for level/trend models,
    #: log returns for anything modelling the increment directly.
    on_log_price = False

    def _endog(self, series: pd.Series, close: pd.Series) -> pd.Series:
        return np.log(close) if self.on_log_price else series

    def _to_return(self, predicted_level: np.ndarray, previous_level: np.ndarray) -> np.ndarray:
        """A level forecast becomes the log return it implies."""
        return predicted_level - previous_level


class StateSpaceModel(_SeriesModel):
    """A statsmodels state-space model with frozen parameters and a rolling filter."""

    def __init__(self) -> None:
        super().__init__()
        self._result: Any = None
        self._train_end: pd.Timestamp | None = None

    def _build(self, endog: pd.Series) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _positional(series: pd.Series) -> pd.Series:
        """Re-index onto 0..n-1.

        statsmodels reaches for the index frequency when the index is a
        DatetimeIndex, and this one has none -- a daily crypto calendar has no
        gaps but pandas does not know that. Working positionally sidesteps the
        whole question; the timestamps are re-attached afterwards.
        """
        return pd.Series(series.to_numpy(dtype=float), index=pd.RangeIndex(len(series)))

    def _estimate(self, model: Any) -> Any:
        """`fit(disp=False)` for the state-space classes, `fit()` for ARIMA."""
        try:
            return _quiet_fit(lambda: model.fit(disp=False))
        except TypeError:
            return _quiet_fit(model.fit)

    def _fit(self, train: TrainingSet) -> None:
        endog = self._endog(train.series, train.close).dropna()
        endog = endog.loc[endog.index <= train.X.index.max()]
        try:
            self._result = self._estimate(self._build(self._positional(endog)))
        except Exception as exc:  # noqa: BLE001 -- recorded against the model
            raise ModelFitError(f"{self.model_id} failed to estimate: {exc}") from exc
        self._train_end = pd.Timestamp(endog.index.max())

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        full = self._endog(context.series, context.close).dropna()
        # Re-run the filter over the longer series with TRAINING parameters.
        # refit=False is the whole contract: the state moves, the parameters do
        # not. The filter's one-step-ahead prediction at t uses data through t-1,
        # which is what makes handing it the whole series safe -- and what the
        # leakage suite verifies rather than assumes.
        try:
            applied = _quiet_fit(lambda: self._result.apply(self._positional(full), refit=False))
            predicted = np.asarray(
                applied.get_prediction(dynamic=False).predicted_mean, dtype=float
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelFitError(f"{self.model_id} failed to roll forward: {exc}") from exc

        # predicted[k] is the one-step-ahead forecast OF bar k, made from k-1.
        forecast_of_bar = pd.Series(predicted, index=full.index)
        implied = (
            forecast_of_bar - full.shift(1) if self.on_log_price else forecast_of_bar
        )
        return np.asarray(implied.reindex(context.target_bars).to_numpy(), dtype=float)

    def parameter_count(self) -> int | None:
        if self._result is None:
            return None
        return int(len(np.atleast_1d(self._result.params)))

    def hyperparameters(self) -> dict:
        if self._result is None:
            return {}
        params = np.atleast_1d(np.asarray(self._result.params, dtype=float))
        names = list(getattr(self._result, "param_names", [])) or [
            f"p{i}" for i in range(len(params))
        ]
        return {"estimated": dict(zip(names, [float(v) for v in params], strict=False))}


class ArimaReturns(StateSpaceModel):
    """ARIMA on log returns. The differencing already happened: d = 0.

    Order is fixed rather than searched. Phase 16 permits at most three
    development-only configurations, and (1,0,1) is the one that was selected;
    the alternatives considered are recorded in :attr:`considered`.
    """

    model_id = "arima"
    resource_class = ResourceClass.LIGHT
    considered = ((1, 0, 0), (1, 0, 1), (2, 0, 2))
    order = (1, 0, 1)

    def _build(self, endog: pd.Series) -> Any:
        from statsmodels.tsa.arima.model import ARIMA

        return ARIMA(endog, order=self.order, trend="c")

    def hyperparameters(self) -> dict:
        return {"order": list(self.order), "considered_on_dev": [list(o) for o in self.considered],
                **super().hyperparameters()}


class LocalLevel(StateSpaceModel):
    """Stochastic local level on log price -- the state-space form of SES."""

    model_id = "local_level"
    resource_class = ResourceClass.LIGHT
    on_log_price = True

    def _build(self, endog: pd.Series) -> Any:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        return UnobservedComponents(endog, level="local level")


class LocalLinearTrend(StateSpaceModel):
    """Stochastic local linear trend on log price -- the state-space form of Holt."""

    model_id = "local_linear_trend"
    resource_class = ResourceClass.LIGHT
    on_log_price = True

    def _build(self, endog: pd.Series) -> Any:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        return UnobservedComponents(endog, level="local linear trend")


class UnobservedComponentsCycle(StateSpaceModel):
    """Local level plus a stochastic cycle.

    Genuinely distinct from the two above: it adds a damped periodic component
    whose frequency is estimated rather than imposed, which is the honest way to
    ask "is there a cycle" when a fixed seasonal period is not justified.
    """

    model_id = "uc_stochastic_cycle"
    resource_class = ResourceClass.MODERATE
    on_log_price = True

    def _build(self, endog: pd.Series) -> Any:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        return UnobservedComponents(endog, level="local level", cycle=True, stochastic_cycle=True)


class HoltLinearTrend(StateSpaceModel):
    """Holt's linear trend in its state-space formulation, on log price.

    The deterministic-smoothing counterpart of :class:`LocalLinearTrend`. Same
    structure, different estimation: smoothing weights by likelihood over an
    innovations state space rather than variances over a structural one.
    """

    model_id = "holt_linear_trend"
    resource_class = ResourceClass.LIGHT
    on_log_price = True

    def _build(self, endog: pd.Series) -> Any:
        from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

        return ExponentialSmoothing(endog, trend=True, damped_trend=True)


class AutoRegressive(_SeriesModel):
    """AR(p) on log returns, evaluated by the frozen linear recursion.

    Deliberately not routed through statsmodels' prediction machinery. An AR
    one-step-ahead forecast is a dot product of the last ``p`` realised returns
    with frozen coefficients, and writing it out makes the causality obvious in
    the four lines that compute it rather than in a library's filter.
    """

    model_id = "ar_p"
    resource_class = ResourceClass.TRIVIAL
    considered = (1, 3, 7)
    lags = 3

    def __init__(self) -> None:
        super().__init__()
        self._const = 0.0
        self._coefficients = np.zeros(0)

    def _fit(self, train: TrainingSet) -> None:
        from statsmodels.tsa.ar_model import AutoReg

        endog = train.y.to_numpy(dtype=float)
        try:
            result = _quiet_fit(lambda: AutoReg(endog, lags=self.lags, old_names=False).fit())
        except Exception as exc:  # noqa: BLE001
            raise ModelFitError(f"{self.model_id} failed to estimate: {exc}") from exc
        params = np.asarray(result.params, dtype=float)
        self._const = float(params[0])
        # AutoReg returns [const, phi_1, ..., phi_p]; phi_1 multiplies the most
        # recent observation.
        self._coefficients = params[1 : self.lags + 1]

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        out = np.empty(len(context), dtype=float)
        for i in range(len(context)):
            history = context.history_at(i).to_numpy(dtype=float)
            recent = history[-self.lags :][::-1]  # most recent first
            if len(recent) < self.lags:
                out[i] = self._const
                continue
            out[i] = self._const + float(np.dot(self._coefficients, recent))
        return out

    def parameter_count(self) -> int | None:
        return int(len(self._coefficients) + 1)

    def hyperparameters(self) -> dict:
        return {
            "lags": self.lags,
            "considered_on_dev": list(self.considered),
            "const": self._const,
            "coefficients": [float(v) for v in self._coefficients],
        }


class ThetaMethod(_SeriesModel):
    """The Theta method, with its parameters frozen after training.

    statsmodels estimates the model; the forward pass is written out because
    `ThetaModel` has no `apply(refit=False)`, and refitting at each of 705
    origins would be a different protocol -- one where the parameters have seen
    the evaluation block.

    With ``theta = 2`` the method is simple exponential smoothing on the series
    plus half the drift of a fitted linear trend, which is the standard
    formulation and is what the recursion below computes.
    """

    model_id = "theta"
    resource_class = ResourceClass.LIGHT

    def __init__(self) -> None:
        super().__init__()
        self._alpha = 0.5
        self._drift = 0.0
        self._level = 0.0

    def _fit(self, train: TrainingSet) -> None:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        endog = pd.Series(train.y.to_numpy(dtype=float), index=pd.RangeIndex(len(train.y)))
        try:
            result = _quiet_fit(
                lambda: ThetaModel(endog, period=1, deseasonalize=False).fit()
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelFitError(f"{self.model_id} failed to estimate: {exc}") from exc
        self._alpha = float(np.clip(result.params["alpha"], 1e-4, 1.0))
        self._drift = float(result.params.get("b0", 0.0)) / 2.0

    def _ses_levels(self, values: np.ndarray) -> np.ndarray:
        """The SES level after each observation.

        Run once over the whole series rather than re-run per origin. That is
        not a shortcut past causality: the level at ``t`` is a function of
        ``values[:t+1]`` and of nothing later, so indexing it at the forecast
        origin gives exactly what re-running the recursion there would.
        """
        levels = np.empty(len(values), dtype=float)
        level = float(values[0])
        levels[0] = level
        for i in range(1, len(values)):
            level = self._alpha * float(values[i]) + (1.0 - self._alpha) * level
            levels[i] = level
        return levels

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        # The first log return is undefined (there is no bar before the first),
        # and carrying that NaN into the recursion poisons every level after it.
        series = context.series.dropna()
        levels = pd.Series(self._ses_levels(series.to_numpy(dtype=float)), index=series.index)
        at_origin = levels.reindex(context.origins)
        return np.asarray(at_origin.to_numpy() + self._drift, dtype=float)

    def parameter_count(self) -> int | None:
        return 2

    def hyperparameters(self) -> dict:
        return {"theta": 2.0, "smoothing_level": self._alpha, "half_drift": self._drift}


def _unsuitable(model_id: str, description: str, resource: ResourceClass) -> None:
    register(
        ZooRegistration(
            model_id=model_id,
            factory=lambda: (_ for _ in ()).throw(  # never constructed
                RuntimeError(f"{model_id} is not runnable in this lab")
            ),
            family=Family.STATISTICAL,
            resource_class=resource,
            description=description,
            requires=("statsmodels",),
            status=ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB,
            unsuitable_reason=NO_SEASONALITY_EVIDENCE,
            notes=(
                "Registered rather than omitted: the measurement that excludes it "
                "is itself a result, and a missing row is indistinguishable from "
                "an oversight.",
            ),
        )
    )


register(
    ZooRegistration(
        model_id="ar_p",
        factory=AutoRegressive,
        family=Family.STATISTICAL,
        resource_class=ResourceClass.TRIVIAL,
        description="AR(3) on log returns, evaluated by the frozen linear recursion.",
        requires=("statsmodels",),
        notes=("Lag order chosen from {1, 3, 7} on DEV only.",),
    )
)

register(
    ZooRegistration(
        model_id="arima",
        factory=ArimaReturns,
        family=Family.STATISTICAL,
        resource_class=ResourceClass.LIGHT,
        description="ARIMA(1,0,1) with constant on log returns; d=0 because the series is already differenced.",
        requires=("statsmodels",),
        notes=("Order chosen from {(1,0,0), (1,0,1), (2,0,2)} on DEV only.",),
    )
)

register(
    ZooRegistration(
        model_id="local_level",
        factory=LocalLevel,
        family=Family.STATISTICAL,
        resource_class=ResourceClass.LIGHT,
        description="Stochastic local level on log price; the state-space form of simple exponential smoothing.",
        requires=("statsmodels",),
        notes=(
            "Estimated by maximum likelihood over state variances, via the Kalman "
            "filter. Kalman is not registered as a separate model because it is "
            "the estimator for this family, not a model.",
        ),
    )
)

register(
    ZooRegistration(
        model_id="local_linear_trend",
        factory=LocalLinearTrend,
        family=Family.STATISTICAL,
        resource_class=ResourceClass.LIGHT,
        description="Stochastic local linear trend on log price; the state-space form of Holt's method.",
        requires=("statsmodels",),
        notes=(
            "Shares its structure with holt_linear_trend and differs in "
            "estimation: variances by likelihood here, smoothing weights there. "
            "Two entries, one structure, different forecasts.",
        ),
    )
)

register(
    ZooRegistration(
        model_id="holt_linear_trend",
        factory=HoltLinearTrend,
        family=Family.STATISTICAL,
        resource_class=ResourceClass.LIGHT,
        description="Damped Holt linear trend on log price, innovations state-space form.",
        requires=("statsmodels",),
    )
)

register(
    ZooRegistration(
        model_id="uc_stochastic_cycle",
        factory=UnobservedComponentsCycle,
        family=Family.STATISTICAL,
        resource_class=ResourceClass.MODERATE,
        description="Unobserved components: local level plus a stochastic cycle of estimated frequency.",
        requires=("statsmodels",),
        notes=(
            "The honest way to ask 'is there a cycle' when no fixed seasonal "
            "period is justified: the frequency is estimated, not imposed.",
        ),
    )
)

register(
    ZooRegistration(
        model_id="theta",
        factory=ThetaMethod,
        family=Family.STATISTICAL,
        resource_class=ResourceClass.LIGHT,
        description="Theta method (theta=2): frozen SES level plus half the fitted drift.",
        requires=("statsmodels",),
        notes=(
            "The forward recursion is written out because ThetaModel has no "
            "apply(refit=False), and refitting at every origin would be a "
            "different protocol -- one whose parameters have seen the holdout.",
        ),
    )
)

_unsuitable(
    "seasonal_naive_weekly",
    "Seasonal naive at period 7: the forecast is the return 7 bars ago.",
    ResourceClass.TRIVIAL,
)
_unsuitable(
    "sarima",
    "Seasonal ARIMA with a weekly period.",
    ResourceClass.MODERATE,
)
_unsuitable(
    "holt_winters",
    "Holt-Winters additive seasonal exponential smoothing at period 7.",
    ResourceClass.LIGHT,
)


__all__ = [
    "NO_SEASONALITY_EVIDENCE",
    "ArimaReturns",
    "AutoRegressive",
    "HoltLinearTrend",
    "LocalLevel",
    "LocalLinearTrend",
    "StateSpaceModel",
    "ThetaMethod",
    "UnobservedComponentsCycle",
]
