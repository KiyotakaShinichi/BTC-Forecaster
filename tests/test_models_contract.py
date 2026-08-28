"""The ForecastModel contract and the naive baselines.

The contract tests are parametrised over every always-available model, so a new
model gets the whole battery for free by being added to the registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.models.base import (
    ForecastResult,
    NotFittedError,
    TrainingWindow,
    random_walk_bands,
)
from btc_forecaster.models.baselines import (
    HistoricalMeanReturn,
    RandomWalk,
    RandomWalkWithDrift,
    default_baselines,
)
from btc_forecaster.testing import constant_growth_frame, synthetic_market_frame
from btc_forecaster.timebase import UTC, ForecastOrigin, HorizonSpec


@pytest.fixture
def window() -> TrainingWindow:
    return TrainingWindow(synthetic_market_frame(periods=400, seed=5))


def baseline_models():
    return default_baselines()


def baseline_ids(model):
    return model.name


class TestTrainingWindow:
    def test_window_validates_its_frame(self):
        bad = synthetic_market_frame(periods=50).iloc[::-1]
        with pytest.raises(Exception, match="sorted"):
            TrainingWindow(bad)

    def test_origin_is_the_last_observed_bar(self, window):
        assert window.origin == ForecastOrigin(window.frame.index[-1])

    def test_derived_series_are_consistent(self, window):
        assert len(window.close) == len(window)
        assert np.allclose(np.exp(window.log_close), window.close)
        assert window.returns.isna().iloc[0]

    def test_exog_must_share_the_market_index(self, window):
        misaligned = pd.DataFrame({"signal": [0.0]}, index=window.frame.index[:1])
        with pytest.raises(ValueError, match="indexed identically"):
            TrainingWindow(window.frame, exog=misaligned)

    def test_exog_aligned_to_the_index_is_accepted(self, window):
        """The Track B seam: aligned exogenous columns pass the contract."""
        exog = pd.DataFrame({"signal": np.zeros(len(window))}, index=window.frame.index)
        assert TrainingWindow(window.frame, exog=exog).exog is not None

    def test_tail_narrows_the_window(self, window):
        assert len(window.tail(50)) == 50


class TestForecastResultInvariants:
    def test_forecast_may_not_include_the_origin_bar(self):
        origin = ForecastOrigin("2024-01-05")
        with pytest.raises(ValueError, match="not after its origin"):
            ForecastResult(
                model="m",
                origin=origin,
                point=pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-05"], tz=UTC)),
            )

    def test_inverted_interval_is_rejected(self):
        origin = ForecastOrigin("2024-01-05")
        index = pd.DatetimeIndex(["2024-01-06"], tz=UTC)
        with pytest.raises(ValueError, match="inverted prediction interval"):
            ForecastResult(
                model="m",
                origin=origin,
                point=pd.Series([100.0], index=index),
                lower=pd.Series([110.0], index=index),
                upper=pd.Series([90.0], index=index),
            )

    def test_band_index_must_match_the_point_forecast(self):
        origin = ForecastOrigin("2024-01-05")
        index = pd.DatetimeIndex(["2024-01-06"], tz=UTC)
        with pytest.raises(ValueError, match="share the point forecast"):
            ForecastResult(
                model="m",
                origin=origin,
                point=pd.Series([100.0], index=index),
                lower=pd.Series([90.0], index=pd.DatetimeIndex(["2024-01-07"], tz=UTC)),
                upper=pd.Series([110.0], index=index),
            )

    def test_head_truncates_point_and_bands_together(self, window):
        result = RandomWalk().fit_predict(window, 30)
        short = result.head(7)
        assert short.horizon == 7
        assert short.lower is not None and len(short.lower) == 7


@pytest.mark.parametrize("model", baseline_models(), ids=baseline_ids)
class TestModelContract:
    """Every model must satisfy these, whatever it does internally."""

    def test_predict_before_fit_is_an_error(self, model, window):
        with pytest.raises(NotFittedError):
            model.predict(10)

    def test_fit_returns_self_for_chaining(self, model, window):
        assert model.fit(window) is model

    def test_forecast_covers_exactly_the_requested_bars(self, model, window):
        result = model.fit_predict(window, 30)
        expected = window.origin.target_bars(30)
        assert result.index.equals(expected)

    def test_forecast_starts_the_bar_after_the_origin(self, model, window):
        result = model.fit_predict(window, 5)
        assert result.index[0] == window.frame.index[-1] + pd.Timedelta(days=1)

    def test_forecast_never_includes_an_observed_bar(self, model, window):
        result = model.fit_predict(window, 30)
        assert not set(result.index) & set(window.frame.index)

    def test_horizon_spec_and_int_agree(self, model, window):
        model.fit(window)
        a = model.predict(12)
        b = model.predict(HorizonSpec(12))
        pd.testing.assert_series_equal(a.point, b.point)

    def test_forecast_is_finite_and_positive(self, model, window):
        result = model.fit_predict(window, 90)
        assert np.isfinite(result.point).all()
        assert (result.point > 0).all(), "price forecasts must be positive"

    def test_intervals_contain_the_point_forecast(self, model, window):
        result = model.fit_predict(window, 90)
        assert result.has_interval
        assert (result.lower <= result.point).all()
        assert (result.point <= result.upper).all()

    def test_intervals_widen_with_horizon(self, model, window):
        """Uncertainty about a price accumulates; a flat band is a broken model."""
        result = model.fit_predict(window, 180)
        width = np.log(result.upper) - np.log(result.lower)
        assert width.iloc[-1] > width.iloc[0] * 3, "log-width should scale ~sqrt(h)"
        assert width.is_monotonic_increasing

    def test_refitting_on_a_shorter_window_moves_the_origin(self, model, window):
        model.fit(window)
        first = model.origin
        model.fit(TrainingWindow(window.frame.iloc[:-50]))
        assert model.origin.last_observed_bar < first.last_observed_bar

    def test_forecast_is_deterministic(self, model, window):
        a = model.fit_predict(window, 30).point
        b = model.fit_predict(window, 30).point
        pd.testing.assert_series_equal(a, b)

    def test_too_little_data_is_a_clear_error(self, model, window):
        with pytest.raises(ValueError, match="at least"):
            model.fit(TrainingWindow(window.frame.iloc[:5]))

    def test_describe_is_serialisable(self, model, window):
        import json

        json.dumps(model.fit(window).describe())

    def test_future_exog_index_is_validated(self, model, window):
        model.fit(window)
        wrong = pd.DataFrame({"x": [0.0]}, index=pd.DatetimeIndex(["2030-01-01"], tz=UTC))
        with pytest.raises(ValueError, match="exactly the target bars"):
            model.predict(10, future_exog=wrong)


class TestRandomWalk:
    def test_point_forecast_is_flat_at_the_last_close(self, window):
        result = RandomWalk().fit_predict(window, 60)
        last_close = float(window.close.iloc[-1])
        assert np.allclose(result.point.to_numpy(), last_close)

    def test_band_width_scales_as_sqrt_of_horizon(self, window):
        result = RandomWalk().fit_predict(window, 100)
        width = np.log(result.upper) - np.log(result.lower)
        ratio = width.iloc[99] / width.iloc[0]
        assert ratio == pytest.approx(10.0, rel=1e-6), "sqrt(100)/sqrt(1) = 10"

    def test_it_ignores_drift_by_construction(self):
        trending = TrainingWindow(synthetic_market_frame(periods=400, kind="trend", seed=2))
        result = RandomWalk().fit_predict(trending, 30)
        assert result.point.nunique() == 1


class TestRandomWalkWithDrift:
    def test_drift_is_recovered_from_a_noiseless_series(self):
        window = TrainingWindow(constant_growth_frame(periods=300, daily_growth=0.002))
        model = RandomWalkWithDrift().fit(window)
        assert model._drift == pytest.approx(0.002, rel=1e-9)

    def test_it_extrapolates_a_noiseless_series_exactly(self):
        frame = constant_growth_frame(periods=310, daily_growth=0.002)
        window = TrainingWindow(frame.iloc[:300])
        result = RandomWalkWithDrift().fit_predict(window, 10)
        actual = frame["close"].iloc[300:310]
        np.testing.assert_allclose(result.point.to_numpy(), actual.to_numpy(), rtol=1e-9)

    def test_it_beats_the_random_walk_on_a_strongly_trending_series(self):
        """Sanity check that drift does something, on data where it should."""
        frame = constant_growth_frame(periods=340, daily_growth=0.003)
        window = TrainingWindow(frame.iloc[:300])
        actual = frame["close"].iloc[300:340].to_numpy()

        drift_err = np.abs(RandomWalkWithDrift().fit_predict(window, 40).point.to_numpy() - actual)
        naive_err = np.abs(RandomWalk().fit_predict(window, 40).point.to_numpy() - actual)
        assert drift_err.mean() < naive_err.mean()

    def test_drift_uses_only_the_endpoints(self, window):
        """The classical estimator is invariant to reordering the interior."""
        model_a = RandomWalkWithDrift().fit(window)
        shuffled = window.frame.copy()
        interior = shuffled["close"].to_numpy()[1:-1][::-1]
        shuffled.iloc[1:-1, shuffled.columns.get_loc("close")] = interior
        model_b = RandomWalkWithDrift().fit(TrainingWindow(shuffled))
        assert model_a._drift == pytest.approx(model_b._drift)


class TestHistoricalMeanReturn:
    def test_it_is_not_the_same_estimator_as_drift(self, window):
        """Arithmetic mean of simple returns exceeds the geometric log drift."""
        mean_model = HistoricalMeanReturn(lookback=None).fit(window)
        drift_model = RandomWalkWithDrift().fit(window)
        assert mean_model._mean_return > drift_model._drift

    def test_lookback_limits_the_estimation_window(self, window):
        model = HistoricalMeanReturn(lookback=90).fit(window)
        assert model._n_used == 90

    def test_full_history_is_used_when_lookback_is_none(self, window):
        model = HistoricalMeanReturn(lookback=None).fit(window)
        assert model._n_used == len(window) - 1

    def test_lookback_changes_the_forecast_under_a_regime_shift(self):
        """A recent-window estimate must track the recent regime."""
        flat = synthetic_market_frame(periods=400, kind="random_walk", daily_vol=0.001, seed=1)
        rising = flat.copy()
        rising.iloc[300:, rising.columns.get_loc("close")] *= np.exp(
            0.01 * np.arange(1, len(rising) - 299)
        )
        window = TrainingWindow(rising)

        recent = HistoricalMeanReturn(lookback=60).fit_predict(window, 30).point.iloc[-1]
        whole = HistoricalMeanReturn(lookback=None).fit_predict(window, 30).point.iloc[-1]
        assert recent > whole


class TestRandomWalkBands:
    def test_variance_accumulates_linearly_in_horizon(self):
        _, lower, upper = random_walk_bands(0.0, 0.0, 0.02, steps=4, level=0.95)
        log_width = np.log(upper) - np.log(lower)
        np.testing.assert_allclose(log_width / log_width[0], np.sqrt([1, 2, 3, 4]), rtol=1e-9)

    def test_drift_shifts_the_centre_not_the_width(self):
        _, lo_a, hi_a = random_walk_bands(0.0, 0.0, 0.02, steps=10)
        _, lo_b, hi_b = random_walk_bands(0.0, 0.001, 0.02, steps=10)
        np.testing.assert_allclose(np.log(hi_a) - np.log(lo_a), np.log(hi_b) - np.log(lo_b))

    def test_higher_confidence_gives_a_wider_band(self):
        _, lo95, hi95 = random_walk_bands(0.0, 0.0, 0.02, steps=5, level=0.95)
        _, lo99, hi99 = random_walk_bands(0.0, 0.0, 0.02, steps=5, level=0.99)
        assert (hi99 - lo99 > hi95 - lo95).all()

    def test_zero_steps_is_rejected(self):
        with pytest.raises(ValueError, match="steps"):
            random_walk_bands(0.0, 0.0, 0.02, steps=0)
