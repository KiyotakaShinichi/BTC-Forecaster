"""Prophet, the XGBoost hybrid, and the Monte Carlo interval fix.

The volatility tests need no optional dependency and always run. The Prophet and
hybrid tests are marked ``slow`` (each Prophet fit takes seconds) and skip
cleanly when the optional model extras are not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.models.base import TrainingWindow
from btc_forecaster.models.volatility import (
    VolatilityForecast,
    analytic_interval,
    constant_volatility,
    monte_carlo_interval,
    simulate_price_paths,
)
from btc_forecaster.testing import synthetic_market_frame

has_prophet = importlib.util.find_spec("prophet") is not None
has_xgboost = importlib.util.find_spec("xgboost") is not None
has_arch = importlib.util.find_spec("arch") is not None

needs_prophet = pytest.mark.skipif(not has_prophet, reason="prophet not installed")
needs_hybrid = pytest.mark.skipif(
    not (has_prophet and has_xgboost), reason="prophet/xgboost not installed"
)


@pytest.fixture
def window() -> TrainingWindow:
    return TrainingWindow(synthetic_market_frame(periods=500, seed=5))


class TestMonteCarloAccumulatesUncertainty:
    """The core correction: shocks accumulate along the path.

    Each test here fails against the original implementation, which drew one
    independent shock per horizon step instead of summing along the path.
    """

    def test_band_width_grows_as_sqrt_of_horizon(self):
        vol = VolatilityForecast(sigma=np.full(100, 0.03), model="constant", params={})
        _, lower, upper = monte_carlo_interval(
            np.zeros(100), vol, n_paths=20_000, seed=1
        )
        log_width = np.log(upper) - np.log(lower)
        ratio = log_width[99] / log_width[0]
        assert ratio == pytest.approx(10.0, rel=0.05), "sqrt(100)/sqrt(1) = 10"

    def test_the_original_non_accumulating_band_stays_flat(self):
        """Demonstrates the defect, so the fix is not merely asserted.

        This reproduces the original construction -- one iid shock per step, no
        cumsum -- and shows the band does not widen. The correct construction,
        on identical inputs, widens tenfold.
        """
        sigma = np.full(100, 0.03)
        rng = np.random.default_rng(0)

        original = np.exp(rng.standard_normal((20_000, 100)) * sigma[None, :])
        original_width = np.log(np.percentile(original, 97.5, axis=0)) - np.log(
            np.percentile(original, 2.5, axis=0)
        )
        assert original_width[99] / original_width[0] == pytest.approx(1.0, abs=0.05)

        vol = VolatilityForecast(sigma=sigma, model="constant", params={})
        _, lower, upper = monte_carlo_interval(np.zeros(100), vol, n_paths=20_000, seed=0)
        corrected_width = np.log(upper) - np.log(lower)
        assert corrected_width[99] / corrected_width[0] > 9.0

    def test_simulation_agrees_with_the_closed_form(self):
        vol = VolatilityForecast(sigma=np.full(60, 0.02), model="constant", params={})
        mean_log = np.linspace(9.0, 9.3, 60)

        _, mc_low, mc_high = monte_carlo_interval(mean_log, vol, n_paths=40_000, seed=3)
        _, an_low, an_high = analytic_interval(mean_log, vol)

        np.testing.assert_allclose(mc_low, an_low, rtol=0.03)
        np.testing.assert_allclose(mc_high, an_high, rtol=0.03)

    def test_analytic_variance_is_the_cumulative_sum_of_step_variances(self):
        sigma = np.array([0.05, 0.04, 0.03, 0.02])
        vol = VolatilityForecast(sigma=sigma, model="varying", params={})
        _, lower, upper = analytic_interval(np.zeros(4), vol)

        from scipy.stats import norm

        z = norm.ppf(0.975)
        expected_sd = np.sqrt(np.cumsum(sigma**2))
        np.testing.assert_allclose(np.log(upper), z * expected_sd, rtol=1e-12)
        np.testing.assert_allclose(np.log(lower), -z * expected_sd, rtol=1e-12)

    def test_point_forecast_is_the_median_not_the_lognormal_mean(self):
        """Reporting the mean would introduce an exp(sigma^2 h/2) upward drift."""
        vol = VolatilityForecast(sigma=np.full(200, 0.04), model="constant", params={})
        mean_log = np.full(200, np.log(50_000.0))

        point, _, _ = monte_carlo_interval(mean_log, vol, n_paths=40_000, seed=2)
        paths = simulate_price_paths(mean_log, vol, n_paths=40_000, seed=2)

        np.testing.assert_allclose(point, 50_000.0)

        # For a lognormal, mean/median = exp(cumulative_var / 2). Reporting the
        # mean would bake that ratio into the point forecast as pure artefact.
        cumulative_var = float(np.sum(vol.sigma**2))
        expected_ratio = float(np.exp(cumulative_var / 2.0))
        assert expected_ratio > 1.15, "sanity: the bias is material at this horizon"
        assert paths[:, -1].mean() / point[-1] == pytest.approx(expected_ratio, rel=0.05)

    def test_simulation_is_reproducible_under_a_seed(self):
        vol = constant_volatility(pd.Series(np.random.default_rng(0).normal(scale=0.02, size=500)), 30)
        a = simulate_price_paths(np.zeros(30), vol, n_paths=100, seed=7)
        b = simulate_price_paths(np.zeros(30), vol, n_paths=100, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_mismatched_volatility_length_is_an_error(self):
        vol = VolatilityForecast(sigma=np.full(10, 0.02), model="constant", params={})
        with pytest.raises(ValueError, match="covers 10 steps"):
            simulate_price_paths(np.zeros(30), vol, n_paths=10)

    def test_wider_level_gives_a_wider_band(self):
        vol = VolatilityForecast(sigma=np.full(30, 0.02), model="constant", params={})
        _, lo95, hi95 = monte_carlo_interval(np.zeros(30), vol, level=0.95, n_paths=20_000, seed=4)
        _, lo99, hi99 = monte_carlo_interval(np.zeros(30), vol, level=0.99, n_paths=20_000, seed=4)
        assert (hi99 - lo99 >= hi95 - lo95).all()


@pytest.mark.skipif(not has_arch, reason="arch not installed")
class TestGarchVolatility:
    def test_garch_produces_a_finite_positive_term_structure(self):
        from btc_forecaster.models.volatility import garch_volatility

        returns = pd.Series(np.random.default_rng(0).normal(scale=0.02, size=800))
        vol = garch_volatility(returns, steps=90)

        assert len(vol) == 90
        assert np.isfinite(vol.sigma).all()
        assert (vol.sigma > 0).all()
        assert vol.model == "GARCH(1,1)"

    def test_garch_conditional_volatility_is_roughly_flat_at_long_horizons(self):
        """Why per-step sigma alone cannot widen a band: it mean-reverts."""
        from btc_forecaster.models.volatility import garch_volatility

        returns = pd.Series(np.random.default_rng(1).normal(scale=0.02, size=1000))
        vol = garch_volatility(returns, steps=365)
        assert vol.sigma[-1] / vol.sigma[180] == pytest.approx(1.0, abs=0.05)


@needs_prophet
@pytest.mark.slow
class TestProphetAdapter:
    def test_it_satisfies_the_forecast_contract(self, window):
        from btc_forecaster.models.prophet_model import ProphetModel

        result = ProphetModel().fit_predict(window, 30)
        assert result.index.equals(window.origin.target_bars(30))
        assert (result.point > 0).all()
        assert result.has_interval
        assert (result.lower <= result.point).all()
        assert (result.point <= result.upper).all()

    def test_timezone_is_stripped_at_the_boundary_only(self, window):
        from btc_forecaster.models.prophet_model import to_prophet_frame

        frame = to_prophet_frame(window.index)
        assert frame["ds"].dt.tz is None
        assert frame["ds"].iloc[0] == window.index[0].tz_convert(None)

    def test_predict_log_covers_arbitrary_bars(self, window):
        from btc_forecaster.models.prophet_model import ProphetModel

        model = ProphetModel().fit(window)
        in_sample = model.predict_log(window.index)
        assert len(in_sample) == len(window)
        assert np.isfinite(in_sample).all()


@needs_hybrid
@pytest.mark.slow
class TestProphetXgboostHybrid:
    def test_it_satisfies_the_forecast_contract(self, window):
        from btc_forecaster.models.hybrid import ProphetXgboostHybrid

        model = ProphetXgboostHybrid(monte_carlo_runs=200)
        result = model.fit_predict(window, 20)

        assert result.index.equals(window.origin.target_bars(20))
        assert (result.point > 0).all()
        assert (result.lower <= result.point).all()
        assert (result.point <= result.upper).all()

    def test_intervals_widen_with_horizon(self, window):
        from btc_forecaster.models.hybrid import ProphetXgboostHybrid

        result = ProphetXgboostHybrid(monte_carlo_runs=500).fit_predict(window, 90)
        width = np.log(result.upper) - np.log(result.lower)
        assert width.iloc[-1] > width.iloc[0] * 2

    def test_feature_selection_happens_inside_fit(self, window):
        from btc_forecaster.models.hybrid import ProphetXgboostHybrid

        model = ProphetXgboostHybrid(monte_carlo_runs=100)
        model.fit(window)
        assert model._selection is not None
        assert model._selection.fitted_end == window.frame.index[-1]

    def test_refitting_reselects_features_on_the_new_window(self, window):
        """What makes the hybrid safe to score in a walk-forward loop."""
        from btc_forecaster.models.hybrid import ProphetXgboostHybrid

        model = ProphetXgboostHybrid(monte_carlo_runs=100)
        model.fit(window)
        first_end = model._selection.fitted_end

        model.fit(TrainingWindow(window.frame.iloc[:400]))
        assert model._selection.fitted_end < first_end

    def test_training_never_pairs_a_feature_with_its_own_bar(self, window):
        """The specific defect the original had, asserted on the fitted model."""
        from btc_forecaster.features.pipeline import build_feature_frame, to_supervised
        from btc_forecaster.models.hybrid import ProphetXgboostHybrid

        model = ProphetXgboostHybrid(monte_carlo_runs=100)
        model.fit(window)

        features = build_feature_frame(window.frame, list(model._selection.specs))
        residual = pd.Series(np.zeros(len(window)), index=window.index)
        data = to_supervised(features, residual, step=1)
        assert (data.feature_bar < data.X.index).all()

    def test_legacy_hyperparameters_are_preserved_verbatim(self):
        from btc_forecaster.models.hybrid import LEGACY_XGB_PARAMS

        assert LEGACY_XGB_PARAMS["max_depth"] == 3
        assert LEGACY_XGB_PARAMS["learning_rate"] == 0.010349570637285655
        assert LEGACY_XGB_PARAMS["subsample"] == 0.8021272578985711
        assert LEGACY_XGB_PARAMS["colsample_bytree"] == 0.7728798862419759

    def test_describe_records_the_selected_features(self, window):
        import json

        from btc_forecaster.models.hybrid import ProphetXgboostHybrid

        model = ProphetXgboostHybrid(monte_carlo_runs=100).fit(window)
        described = model.describe()
        assert described["features"]["fitted_on"]["n_observations"] == len(window)
        json.dumps(described)
