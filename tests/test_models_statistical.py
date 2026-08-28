"""Statistical models and the registry.

statsmodels is a core dependency, so these always run. The contract battery from
test_models_contract.py is re-applied here to the statistical family, which is
the point of having a contract at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.models import registry
from btc_forecaster.models.base import MissingDependencyError, TrainingWindow
from btc_forecaster.models.statistical import (
    DEFAULT_ORDER_GRID,
    ArimaModel,
    EtsModel,
    SarimaxModel,
    select_arima_order,
)
from btc_forecaster.testing import constant_growth_frame, synthetic_market_frame


@pytest.fixture
def window() -> TrainingWindow:
    return TrainingWindow(synthetic_market_frame(periods=400, seed=5))


def statistical_models():
    return [ArimaModel(order=(1, 1, 1)), EtsModel(), SarimaxModel(order=(1, 1, 1))]


@pytest.mark.parametrize("model", statistical_models(), ids=lambda m: m.name)
class TestStatisticalModelsSatisfyTheContract:
    def test_forecast_covers_the_requested_bars(self, model, window):
        result = model.fit_predict(window, 30)
        assert result.index.equals(window.origin.target_bars(30))

    def test_forecast_is_finite_and_positive(self, model, window):
        result = model.fit_predict(window, 60)
        assert np.isfinite(result.point).all()
        assert (result.point > 0).all()

    def test_intervals_contain_the_point_forecast(self, model, window):
        result = model.fit_predict(window, 60)
        assert result.has_interval
        assert (result.lower <= result.point + 1e-9).all()
        assert (result.point <= result.upper + 1e-9).all()

    def test_intervals_widen_with_horizon(self, model, window):
        result = model.fit_predict(window, 120)
        width = np.log(result.upper) - np.log(result.lower)
        assert width.iloc[-1] > width.iloc[0]

    def test_forecast_never_overlaps_training_bars(self, model, window):
        result = model.fit_predict(window, 30)
        assert not set(result.index) & set(window.frame.index)

    def test_metadata_is_serialisable(self, model, window):
        import json

        json.dumps(model.fit_predict(window, 10).metadata)

    def test_describe_is_serialisable(self, model, window):
        import json

        json.dumps(model.fit(window).describe())

    def test_refitting_updates_the_origin(self, model, window):
        model.fit(window)
        first = model.origin
        model.fit(TrainingWindow(window.frame.iloc[:-40]))
        assert model.origin.last_observed_bar < first.last_observed_bar


class TestArima:
    def test_random_walk_order_reproduces_a_flat_forecast(self, window):
        """ARIMA(0,1,0) with no trend IS the random walk; a good cross-check."""
        result = ArimaModel(order=(0, 1, 0)).fit_predict(window, 20)
        last_close = float(window.close.iloc[-1])
        np.testing.assert_allclose(result.point.to_numpy(), last_close, rtol=1e-6)

    def test_random_walk_order_bands_scale_as_sqrt_horizon(self, window):
        result = ArimaModel(order=(0, 1, 0)).fit_predict(window, 100)
        width = np.log(result.upper) - np.log(result.lower)
        assert width.iloc[99] / width.iloc[0] == pytest.approx(10.0, rel=1e-3)

    def test_drift_term_extrapolates_a_noiseless_trend(self):
        frame = constant_growth_frame(periods=210, daily_growth=0.002)
        window = TrainingWindow(frame.iloc[:200])
        result = ArimaModel(order=(0, 1, 0), trend="t").fit_predict(window, 10)
        np.testing.assert_allclose(
            result.point.to_numpy(), frame["close"].iloc[200:210].to_numpy(), rtol=1e-3
        )

    def test_auto_order_selection_happens_inside_fit(self, window):
        model = ArimaModel(order=None)
        model.fit(window)
        assert model._selection is not None
        assert model._chosen_order in DEFAULT_ORDER_GRID
        assert model._selection.n_converged >= 1

    def test_auto_order_reselects_per_training_window(self, window):
        """What makes per-fold refitting non-leaky."""
        model = ArimaModel(order=None)
        model.fit(window)
        first = model._selection

        model.fit(TrainingWindow(window.frame.iloc[:200]))
        second = model._selection

        assert first is not second
        assert second.n_candidates == len(DEFAULT_ORDER_GRID)

    def test_order_grid_contains_the_random_walk_null(self):
        assert (0, 1, 0) in DEFAULT_ORDER_GRID

    def test_aic_search_reports_its_ranking(self, window):
        selection = select_arima_order(window.log_close.to_numpy())
        aics = [aic for _, aic in selection.ranking]
        assert aics == sorted(aics)
        assert selection.aic == aics[0]

    def test_aic_prefers_ar1_structure_on_a_genuine_ar1_series(self):
        """The search must be able to find real structure, not just noise."""
        ar1 = synthetic_market_frame(periods=600, kind="ar1", phi=0.6, seed=4)
        selection = select_arima_order(np.log(ar1["close"].to_numpy()))
        assert selection.order != (0, 1, 0), "AIC should reject the random walk here"
        assert selection.order[0] >= 1 or selection.order[2] >= 1

    def test_describe_records_the_chosen_order(self, window):
        model = ArimaModel(order=None).fit(window)
        described = model.describe()
        assert described["requested_order"] == "auto"
        assert described["order"] == list(model._chosen_order)


class TestEts:
    def test_damped_trend_is_the_default(self):
        assert EtsModel().damped_trend is True

    def test_damping_bounds_the_long_horizon_forecast(self):
        """An undamped trend extrapolates linearly forever; damping must not."""
        frame = constant_growth_frame(periods=300, daily_growth=0.004)
        window = TrainingWindow(frame)

        damped = EtsModel(damped_trend=True).fit_predict(window, 365).point.iloc[-1]
        undamped = EtsModel(damped_trend=False).fit_predict(window, 365).point.iloc[-1]
        assert damped < undamped

    def test_it_tracks_a_noiseless_trend_over_a_short_horizon(self):
        frame = constant_growth_frame(periods=310, daily_growth=0.002)
        window = TrainingWindow(frame.iloc[:300])
        result = EtsModel().fit_predict(window, 10)
        actual = frame["close"].iloc[300:310].to_numpy()
        assert np.abs(result.point.to_numpy() / actual - 1.0).max() < 0.02


class TestSarimaxExogSeam:
    """The declared attachment point for future exogenous signals."""

    @staticmethod
    def _calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
        """A regressor genuinely known in advance: day-of-week."""
        return pd.DataFrame({"is_weekend": (index.dayofweek >= 5).astype(float)}, index=index)

    def test_exog_model_fits_and_forecasts_with_future_regressors(self, window):
        exog = self._calendar_exog(window.frame.index)
        model = SarimaxModel(order=(1, 1, 1), exog_columns=("is_weekend",))
        model.fit(TrainingWindow(window.frame, exog=exog))

        target = window.origin.target_bars(30)
        result = model.predict(30, future_exog=self._calendar_exog(target))

        assert result.index.equals(target)
        assert np.isfinite(result.point).all()

    def test_missing_training_exog_is_a_clear_error(self, window):
        model = SarimaxModel(exog_columns=("is_weekend",))
        with pytest.raises(ValueError, match="carries no exog"):
            model.fit(window)

    def test_missing_future_exog_is_a_clear_error(self, window):
        exog = self._calendar_exog(window.frame.index)
        model = SarimaxModel(exog_columns=("is_weekend",))
        model.fit(TrainingWindow(window.frame, exog=exog))
        with pytest.raises(ValueError, match="known in advance"):
            model.predict(30)

    def test_future_exog_must_cover_exactly_the_target_bars(self, window):
        exog = self._calendar_exog(window.frame.index)
        model = SarimaxModel(exog_columns=("is_weekend",))
        model.fit(TrainingWindow(window.frame, exog=exog))

        short = self._calendar_exog(window.origin.target_bars(10))
        with pytest.raises(ValueError, match="exactly the target bars"):
            model.predict(30, future_exog=short)

    def test_wrong_exog_column_is_named_in_the_error(self, window):
        exog = pd.DataFrame({"other": np.zeros(len(window))}, index=window.frame.index)
        model = SarimaxModel(exog_columns=("is_weekend",))
        with pytest.raises(ValueError, match="is_weekend"):
            model.fit(TrainingWindow(window.frame, exog=exog))

    def test_seasonal_order_is_accepted(self, window):
        model = SarimaxModel(order=(1, 0, 0), seasonal_order=(1, 0, 0, 7))
        result = model.fit_predict(window, 14)
        assert np.isfinite(result.point).all()


class TestRegistry:
    def test_baselines_are_always_available(self):
        for name in ["random_walk", "random_walk_drift", "historical_mean_return"]:
            assert registry.available()[name]["available"]

    def test_statistical_models_are_available_here(self):
        for name in ["arima", "arima_auto", "sarimax", "ets"]:
            assert registry.available()[name]["available"]

    def test_build_returns_a_working_model(self, window):
        model = registry.build("random_walk")
        assert model.fit_predict(window, 5).horizon == 5

    def test_build_passes_keyword_arguments_through(self):
        model = registry.build("historical_mean_return", lookback=30)
        assert model.lookback == 30

    def test_built_models_are_named_after_their_registry_key(self):
        """Otherwise a lookup by configured name silently misses.

        ArimaModel names itself from its order ("arima(1, 1, 1)"), which would
        never match the "arima" key used to request it -- so primary_model,
        baseline_model and every skill-table row would fail to resolve.
        """
        for key in ("arima", "arima_auto", "ets", "sarimax", "random_walk"):
            assert registry.build(key).name == key

    def test_an_explicit_name_still_wins(self):
        assert registry.build("arima", name="custom").name == "custom"

    def test_unknown_model_lists_what_is_registered(self):
        with pytest.raises(KeyError, match="random_walk"):
            registry.build("no_such_model")

    def test_families_are_reported(self):
        assert {"baseline", "statistical"} <= set(registry.families())

    def test_names_can_be_filtered_by_family(self):
        assert set(registry.names("baseline")) == {
            "random_walk",
            "random_walk_drift",
            "historical_mean_return",
        }

    def test_registering_a_duplicate_name_is_refused(self):
        with pytest.raises(ValueError, match="already registered"):
            registry.register("random_walk", lambda: None, family="baseline")

    def test_availability_report_is_serialisable(self):
        import json

        json.dumps(registry.available())

    def test_build_available_skips_rather_than_raises(self):
        """A missing optional model must not abort a whole comparison run."""
        built, skipped = registry.build_available(["random_walk", "arima"])
        assert len(built) == 2
        assert skipped == {}

    def test_missing_dependency_names_the_extra_to_install(self, monkeypatch):
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "prophet":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

        with pytest.raises(MissingDependencyError, match=r"btc-forecaster\[models\]"):
            registry.build("prophet")

        built, skipped = registry.build_available(["random_walk", "prophet"])
        assert len(built) == 1
        assert "prophet" in skipped
