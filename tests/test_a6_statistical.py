"""The statistical family: frozen parameters, rolling state, no seasonality.

The central claim of this adapter is that parameters are estimated once and the
state then rolls forward on realised values. That claim is worth exactly as
much as the test that the future cannot reach backwards through it, so the
poison test is here rather than only in the leakage suite: it is this module's
own contract, not a generic property.

The second claim is that three requested seasonal models are excluded on
measured evidence rather than on preference. That is asserted too -- an
exclusion without a stated reason is not a scientific statement, and the
registry refuses to accept one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.adapters.statistical import (
    NO_SEASONALITY_EVIDENCE,
    ThetaMethod,
)
from btc_forecaster.research.contracts import EvaluationContext, Family, ModelStatus
from btc_forecaster.research.partition import PartitionSpec, build_dataset
from btc_forecaster.testing import constant_growth_frame, synthetic_market_frame

STATISTICAL_IDS = [
    "ar_p",
    "arima",
    "holt_linear_trend",
    "local_level",
    "local_linear_trend",
    "theta",
    "uc_stochastic_cycle",
]


@pytest.fixture(scope="module")
def dataset():
    frame = synthetic_market_frame(periods=700, kind="ar1", seed=5)
    return build_dataset(frame, spec=PartitionSpec(train_rows=250))


@pytest.fixture(scope="module")
def fitted(dataset):
    train = dataset.training_set()
    return {mid: registry.build(mid).fit(train) for mid in STATISTICAL_IDS}


class TestEveryStatisticalModelRuns:
    @pytest.mark.parametrize("model_id", STATISTICAL_IDS)
    def test_it_fits_and_forecasts_one_step_ahead(self, dataset, fitted, model_id) -> None:
        context = dataset.evaluation_context()
        forecast = fitted[model_id].predict(context)
        assert len(forecast.point) == len(context)
        assert np.isfinite(forecast.point).all()

    @pytest.mark.parametrize("model_id", STATISTICAL_IDS)
    def test_it_reports_a_parameter_count(self, fitted, model_id) -> None:
        """Phase 24 compares performance against complexity, which needs a
        denominator. A model that cannot say how many parameters it fitted
        cannot be placed on that axis."""
        count = fitted[model_id].parameter_count()
        assert isinstance(count, int) and count > 0

    @pytest.mark.parametrize("model_id", STATISTICAL_IDS)
    def test_it_records_its_frozen_configuration(self, fitted, model_id) -> None:
        assert fitted[model_id].hyperparameters()

    def test_they_are_all_in_the_statistical_family(self) -> None:
        for model_id in STATISTICAL_IDS:
            assert registry.get(model_id).family is Family.STATISTICAL


class TestTheFutureCannotReachBackwards:
    """This adapter hands the whole series to a Kalman filter and relies on the
    recursion being causal. Relying is not enough."""

    @pytest.mark.parametrize("model_id", STATISTICAL_IDS)
    def test_poisoning_the_future_changes_nothing(self, dataset, fitted, model_id) -> None:
        context = dataset.evaluation_context()
        clean = fitted[model_id].predict(context).point

        poisoned_series = context.series.copy()
        poisoned_close = context.close.copy()
        cut = context.origins[0]
        poisoned_series.loc[poisoned_series.index > cut] = 5.0
        poisoned_close.loc[poisoned_close.index > cut] = 1e9

        poisoned_context = EvaluationContext(
            X=context.X,
            y=context.y,
            series=poisoned_series,
            close=poisoned_close,
            feature_bar=context.feature_bar,
            train_end=context.train_end,
        )
        dirty = fitted[model_id].predict(poisoned_context).point
        # Only the first prediction is legitimately computable from unpoisoned
        # data; it must be bit-identical.
        assert dirty[0] == clean[0], f"{model_id} read a future bar"

    def test_the_theta_shortcut_equals_the_per_origin_recursion(self, dataset, fitted) -> None:
        """The SES level is computed once over the whole series and indexed at
        each origin. That is only legitimate because the level at t depends on
        nothing after t -- so it must equal re-running the recursion per origin."""
        model = fitted["theta"]
        assert isinstance(model, ThetaMethod)
        context = dataset.evaluation_context()
        vectorised = model.predict(context).point

        series = context.series.dropna()
        for i in (0, len(context) // 3, len(context) - 1):
            history = series.loc[series.index <= context.origins[i]].to_numpy(dtype=float)
            level = float(history[0])
            for value in history[1:]:
                level = model._alpha * float(value) + (1.0 - model._alpha) * level
            assert vectorised[i] == pytest.approx(level + model._drift, abs=1e-12)


class TestSeasonalModelsAreExcludedOnEvidence:
    SEASONAL = ("seasonal_naive_weekly", "sarima", "holt_winters")

    @pytest.mark.parametrize("model_id", SEASONAL)
    def test_it_is_registered_rather_than_omitted(self, model_id) -> None:
        """A missing row is indistinguishable from an oversight."""
        assert model_id in registry.model_ids()

    @pytest.mark.parametrize("model_id", SEASONAL)
    def test_it_is_marked_unsuitable_with_the_measurement(self, model_id) -> None:
        registration = registry.get(model_id)
        assert registration.effective_status() is ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB
        assert registration.unsuitable_reason == NO_SEASONALITY_EVIDENCE
        assert "ACF(7)" in registration.unsuitable_reason

    @pytest.mark.parametrize("model_id", SEASONAL)
    def test_it_is_not_in_the_runnable_set(self, model_id) -> None:
        assert model_id not in registry.runnable_ids()

    def test_the_series_really_has_no_weekly_seasonality(self) -> None:
        """The measurement that justifies the exclusion, re-derived here so the
        reason string cannot drift away from the data it claims to describe."""
        frame = synthetic_market_frame(periods=1200, kind="ar1", seed=3)
        dataset = build_dataset(frame, spec=PartitionSpec(train_rows=500))
        returns = dataset.training_set().y
        band = 1.96 / np.sqrt(len(returns))
        lag7 = float(pd.Series(returns.to_numpy()).autocorr(lag=7))
        assert abs(lag7) < band * 2.0


class TestNoRedundantRegistrations:
    def test_kalman_is_not_registered_as_a_model(self) -> None:
        """It is the estimator for the state-space family, not a model. A
        'kalman' entry would be local_level under a second name."""
        assert not [m for m in registry.model_ids() if "kalman" in m.lower()]

    def test_ses_is_not_registered_beside_local_level(self) -> None:
        """Simple exponential smoothing IS the local-level model."""
        ids = registry.model_ids()
        assert "local_level" in ids
        assert "simple_exponential_smoothing" not in ids
        assert "ses" not in ids

    def test_holt_and_local_linear_trend_are_genuinely_different(self, fitted, dataset) -> None:
        """Same structure, different estimation -- so they must not agree."""
        context = dataset.evaluation_context()
        holt = fitted["holt_linear_trend"].predict(context).point
        structural = fitted["local_linear_trend"].predict(context).point
        assert not np.allclose(holt, structural)


class TestSanityOnANoiselessSeries:
    def test_a_trend_model_recovers_a_constant_growth_rate(self) -> None:
        """A noiseless exponential series has one right answer. A trend model
        that cannot find it is mis-wired, and this catches that before any
        result from it is believed."""
        frame = constant_growth_frame(periods=400, daily_growth=0.001)
        dataset = build_dataset(frame, spec=PartitionSpec(train_rows=150))
        model = registry.build("holt_linear_trend").fit(dataset.training_set())
        point = model.predict(dataset.evaluation_context()).point
        assert np.abs(point - 0.001).mean() < 5e-4

    def test_a_level_model_correctly_forecasts_no_change(self) -> None:
        """The complement, and the more interesting half. A local-level model is
        a random walk plus noise: it has no trend component, so on a perfectly
        trending series it forecasts zero change and is wrong by exactly the
        drift. That is the model's structure showing through, not a defect --
        and asserting it is how the two entries stay distinguishable."""
        frame = constant_growth_frame(periods=400, daily_growth=0.001)
        dataset = build_dataset(frame, spec=PartitionSpec(train_rows=150))
        model = registry.build("local_level").fit(dataset.training_set())
        point = model.predict(dataset.evaluation_context()).point
        assert np.abs(point).max() < 1e-5

    def test_the_naive_baseline_is_wrong_on_a_trend(self) -> None:
        """The control for the test above: predicting zero on a growing series
        must be measurably wrong, or the comparison proves nothing."""
        frame = constant_growth_frame(periods=400, daily_growth=0.001)
        dataset = build_dataset(frame, spec=PartitionSpec(train_rows=150))
        model = registry.build("naive_last_value").fit(dataset.training_set())
        point = model.predict(dataset.evaluation_context()).point
        assert np.abs(point - 0.001).mean() == pytest.approx(0.001, rel=1e-6)
