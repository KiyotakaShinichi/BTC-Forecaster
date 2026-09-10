"""Iterated h-bar forecasts: the MULTI_STEP capability on the series models.

A7 forecasts 3, 7 and 30 bars ahead. Tabular and sequence models are trained
directly on the h-bar target and need nothing new. Series models estimate on
one-bar returns and can only step forward one bar, so they gained one method:
iterate their own recursion with the parameters frozen at fitting.

What that method must be, pinned here:

* **the A6 forecast at h = 1**, to machine precision -- the capability adds a
  horizon, it does not change a model;
* **statsmodels' own forecast** for every state-space model, at several origins
  and horizons -- the propagation is the library's, done for all origins at once;
* **blind to the future** -- poisoning bars after an origin cannot move the
  forecast made at it, and poisoning the origin itself can.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.adapters.statistical import StateSpaceModel, _time_invariant
from btc_forecaster.research.contracts import (
    Capability,
    CapabilityNotSupported,
    EvaluationContext,
)
from btc_forecaster.research.partition import PartitionSpec, build_dataset
from btc_forecaster.testing import synthetic_market_frame


def available(*model_ids: str) -> list[str]:
    """Only what this environment can build; `.[dev]` has statsmodels but not sklearn."""
    return [m for m in model_ids if registry.get(m).is_available()]


FRAME = synthetic_market_frame(periods=700, kind="ar1", seed=3)
DATASET = build_dataset(FRAME, spec=PartitionSpec(train_rows=300))
CONTEXT = DATASET.evaluation_context()

MULTI_STEP_MODELS = available(
    "naive_last_value",
    "random_walk_drift",
    "historical_mean_return",
    "ar_p",
    "arima",
    "theta",
    "local_level",
    "local_linear_trend",
    "holt_linear_trend",
    "uc_stochastic_cycle",
)
STATE_SPACE = available(
    "arima", "local_level", "local_linear_trend", "holt_linear_trend", "uc_stochastic_cycle"
)
DIRECT_ONLY = available("ridge", "xgboost", "mlp", "lstm")
MOVING = [m for m in MULTI_STEP_MODELS if m not in {"naive_last_value", "random_walk_drift", "historical_mean_return"}]


@pytest.fixture(scope="module")
def fitted():
    train = DATASET.training_set()
    return {model_id: registry.build(model_id).fit(train) for model_id in MULTI_STEP_MODELS}


def poisoned_after(context: EvaluationContext, cut_row: int) -> EvaluationContext:
    """Every realised bar after row ``cut_row``'s origin, grossly corrupted."""
    cut = context.origins[cut_row]
    series, close = context.series.copy(), context.close.copy()
    series[series.index > cut] = series[series.index > cut] * 40.0 + 0.3
    close[close.index > cut] = close[close.index > cut] * 7.0
    return EvaluationContext(
        X=context.X, y=context.y, series=series, close=close,
        feature_bar=context.feature_bar, train_end=context.train_end, design=context.design,
    )


class TestTheCapabilityIsDeclared:
    def test_the_series_models_are_under_test(self) -> None:
        pytest.importorskip("statsmodels")
        assert len(MULTI_STEP_MODELS) >= 9 and len(STATE_SPACE) >= 5

    @pytest.mark.parametrize("model_id", MULTI_STEP_MODELS)
    def test_every_series_model_declares_it(self, model_id) -> None:
        assert registry.build(model_id).supports(Capability.MULTI_STEP)

    @pytest.mark.parametrize("model_id", DIRECT_ONLY)
    def test_direct_models_do_not_and_asking_is_refused(self, model_id) -> None:
        """A tabular model has no one-step recursion to iterate. Faking one would
        manufacture a forecast; it is trained on the h-bar target instead."""
        model = registry.build(model_id)
        assert not model.supports(Capability.MULTI_STEP)
        with pytest.raises(CapabilityNotSupported, match="MULTI_STEP"):
            model.predict_cumulative(CONTEXT, 3)

    def test_a_nonpositive_horizon_is_refused(self, fitted) -> None:
        with pytest.raises(ValueError, match="horizon"):
            fitted["naive_last_value"].predict_cumulative(CONTEXT, 0)


class TestOneStepIsTheA6Forecast:
    @pytest.mark.parametrize("model_id", MULTI_STEP_MODELS)
    def test_h1_equals_the_point_forecast(self, fitted, model_id) -> None:
        model = fitted[model_id]
        assert np.allclose(
            model.predict_cumulative(CONTEXT, 1), model.predict(CONTEXT).point, rtol=0.0, atol=1e-12
        )


class TestStateSpaceEqualsStatsmodels:
    @pytest.mark.parametrize("model_id", STATE_SPACE)
    @pytest.mark.parametrize("horizon", [3, 7, 30])
    def test_against_forecast_from_the_end_of_a_sample(self, fitted, model_id, horizon) -> None:
        model = fitted[model_id]
        full = model._endog(CONTEXT.series, CONTEXT.close).dropna()
        ours = model.predict_cumulative(CONTEXT, horizon)
        for row in (0, len(CONTEXT) // 2, len(CONTEXT) - 1):
            position = full.index.get_loc(CONTEXT.origins[row])
            sample = StateSpaceModel._positional(full.iloc[: position + 1])
            path = np.asarray(model._result.apply(sample, refit=False).forecast(horizon), dtype=float)
            expected = path[-1] - full.iloc[position] if model.on_log_price else path.sum()
            assert ours[row] == pytest.approx(expected, abs=1e-10), (model_id, horizon, row)


class TestTheClosedForms:
    def test_the_ar_recursion_is_iterated_on_its_own_forecasts(self, fitted) -> None:
        pytest.importorskip("statsmodels")
        model = fitted["ar_p"]
        horizon = 5
        ours = model.predict_cumulative(CONTEXT, horizon)
        for row in (0, 40, len(CONTEXT) - 1):
            lags = list(CONTEXT.history_at(row).to_numpy(dtype=float)[-model.lags :][::-1])
            total = 0.0
            for _ in range(horizon):
                step = model._const + float(np.dot(model._coefficients, lags))
                total += step
                lags = [step, *lags[:-1]]
            assert ours[row] == pytest.approx(total, abs=1e-14)

    def test_theta_is_a_flat_level_plus_accruing_drift(self, fitted) -> None:
        pytest.importorskip("statsmodels")
        model = fitted["theta"]
        level = model.predict(CONTEXT).point - model._drift
        for horizon in (3, 30):
            expected = horizon * level + model._drift * horizon * (horizon + 1) / 2.0
            assert np.allclose(model.predict_cumulative(CONTEXT, horizon), expected, atol=1e-14)

    @pytest.mark.parametrize("horizon", [1, 3, 30])
    def test_baselines_scale_their_constant(self, fitted, horizon) -> None:
        assert np.all(fitted["naive_last_value"].predict_cumulative(CONTEXT, horizon) == 0.0)
        drift = fitted["random_walk_drift"]
        assert np.allclose(drift.predict_cumulative(CONTEXT, horizon), horizon * drift._value)


class TestTheFutureCannotReachAnIteratedForecast:
    CUT = 50

    @pytest.mark.parametrize("model_id", MULTI_STEP_MODELS)
    def test_poisoning_every_later_bar_changes_nothing_up_to_the_cut(self, fitted, model_id) -> None:
        model = fitted[model_id]
        clean = model.predict_cumulative(CONTEXT, 7)
        dirty = model.predict_cumulative(poisoned_after(CONTEXT, self.CUT), 7)
        assert np.array_equal(clean[: self.CUT + 1], dirty[: self.CUT + 1])

    @pytest.mark.parametrize("model_id", MOVING)
    def test_poisoning_the_origin_itself_does_change_it(self, fitted, model_id) -> None:
        """The sensitivity half. The bar at the origin is information the model
        is entitled to; if corrupting it moved nothing, the test above would be
        passing for the wrong reason."""
        model = fitted[model_id]
        clean = model.predict_cumulative(CONTEXT, 7)
        dirty = model.predict_cumulative(poisoned_after(CONTEXT, self.CUT - 1), 7)
        assert clean[self.CUT] != dirty[self.CUT]


class TestSystemMatrices:
    def test_a_time_invariant_matrix_loses_its_time_axis(self) -> None:
        assert _time_invariant(np.ones((1, 2, 1)), "m").shape == (1, 2)

    def test_a_constant_repeated_over_time_is_accepted(self) -> None:
        assert np.array_equal(_time_invariant(np.full((1, 5), 0.3), "m"), np.array([0.3]))

    def test_a_genuinely_time_varying_matrix_is_refused(self) -> None:
        varying = np.arange(5, dtype=float).reshape(1, 5)
        with pytest.raises(CapabilityNotSupported, match="time-varying"):
            _time_invariant(varying, "m")
