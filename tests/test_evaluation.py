"""Evaluation metrics, with emphasis on the ones that were wrong or absent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.evaluation.metrics import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    binomial_direction_test,
    directional_accuracy,
    evaluate_forecast,
    interval_coverage,
    mae,
    mape,
    mase,
    naive_scale_from_training,
    path_directional_accuracy,
    pinball_loss,
    relative_interval_width,
    return_forecast_correlation,
    rmse,
    skill_score,
    smape,
    winkler_score,
)
from btc_forecaster.timebase import UTC


def series(values, start="2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", tz=UTC, name="date")
    return pd.Series(np.asarray(values, dtype=float), index=index)


class TestMagnitudeMetrics:
    def test_perfect_forecast_scores_zero_error(self):
        actual = series([100, 101, 102, 103])
        assert mae(actual, actual) == 0.0
        assert rmse(actual, actual) == 0.0
        assert mape(actual, actual) == 0.0
        assert smape(actual, actual) == 0.0

    def test_rmse_penalises_large_errors_more_than_mae(self):
        actual = series([100, 100, 100, 100])
        spread = series([90, 110, 90, 110])
        concentrated = series([100, 100, 100, 140])
        assert mae(actual, spread) == mae(actual, concentrated)
        assert rmse(actual, concentrated) > rmse(actual, spread)

    def test_mape_is_scale_free(self):
        a, p = series([100, 200]), series([110, 220])
        assert mape(a, p) == pytest.approx(10.0)
        assert mape(a * 1000, p * 1000) == pytest.approx(10.0)

    def test_mape_is_undefined_at_zero(self):
        assert np.isnan(mape(series([0.0, 1.0]), series([1.0, 1.0])))

    def test_smape_is_bounded_where_mape_is_not(self):
        actual = series([1.0, 1.0])
        wild = series([1000.0, 1000.0])
        assert mape(actual, wild) > 1000
        assert smape(actual, wild) <= 200.0

    def test_mase_below_one_means_better_than_the_naive_scale(self):
        actual = series([100, 102, 104, 106])
        good = series([100.5, 102.5, 104.5, 106.5])
        scale = naive_scale_from_training(series([100, 102, 104, 106, 108]))
        assert mase(actual, good, naive_scale=scale) < 1.0

    def test_mase_is_nan_for_a_degenerate_scale(self):
        assert np.isnan(mase(series([1, 2]), series([1, 2]), naive_scale=0.0))

    def test_naive_scale_is_mean_absolute_first_difference(self):
        assert naive_scale_from_training(series([100, 110, 105])) == pytest.approx(7.5)

    def test_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="length mismatch"):
            mae(series([1, 2, 3]), series([1, 2]))


class TestDirectionalAccuracy:
    """The corrected metric, and the one the original actually computed."""

    def test_it_measures_movement_away_from_the_last_known_price(self):
        reference = series([100.0, 100.0, 100.0, 100.0])
        actual = series([101.0, 99.0, 102.0, 98.0])
        predicted = series([105.0, 95.0, 103.0, 97.0])
        assert directional_accuracy(actual, predicted, reference=reference) == 1.0

    def test_a_uniformly_wrong_call_scores_zero(self):
        reference = series([100.0, 100.0])
        actual = series([101.0, 102.0])
        predicted = series([99.0, 98.0])
        assert directional_accuracy(actual, predicted, reference=reference) == 0.0

    def test_the_two_definitions_genuinely_disagree(self):
        """A shape-correct but level-wrong forecast: perfect path score, zero
        tradeable score. This is the gap the original metric hid."""
        reference = series([100.0, 100.0, 100.0, 100.0])
        actual = series([101.0, 102.0, 103.0, 104.0])
        predicted = series([90.0, 91.0, 92.0, 93.0])

        assert path_directional_accuracy(actual, predicted) == 1.0
        assert directional_accuracy(actual, predicted, reference=reference) == 0.0

    def test_path_accuracy_is_invariant_to_a_constant_offset(self):
        actual = series([100, 101, 100, 103])
        predicted = series([50, 51, 50, 53])
        assert path_directional_accuracy(actual, predicted) == 1.0

    def test_path_accuracy_needs_at_least_two_points(self):
        assert np.isnan(path_directional_accuracy(series([1.0]), series([1.0])))

    def test_a_flat_forecast_calls_everything_down(self):
        """sign(0) is not positive, so a random-walk forecast never predicts up."""
        reference = series([100.0, 100.0])
        actual = series([101.0, 99.0])
        flat = series([100.0, 100.0])
        assert directional_accuracy(actual, flat, reference=reference) == 0.5


class TestBinomialTest:
    def test_a_strong_hit_rate_is_significant(self):
        result = binomial_direction_test(70, 100)
        assert result.accuracy == 0.7
        assert result.p_value < 0.001

    def test_a_coin_flip_is_not_significant(self):
        assert binomial_direction_test(50, 100).p_value > 0.4

    def test_the_independence_caveat_travels_with_the_result(self):
        """It cannot be quoted without its own limitation attached."""
        payload = binomial_direction_test(60, 100).to_dict()
        assert "independent Bernoulli trials" in payload["caveat"]
        assert "multiple comparisons" in payload["caveat"]

    def test_empty_input_is_nan_not_a_crash(self):
        assert np.isnan(binomial_direction_test(0, 0).p_value)


class TestReturnCorrelation:
    def test_a_perfectly_scaled_forecast_correlates_at_one(self):
        reference = series([100.0] * 10)
        actual = series(100.0 * (1.0 + np.linspace(-0.05, 0.05, 10)))
        predicted = series(100.0 * (1.0 + 0.5 * np.linspace(-0.05, 0.05, 10)))
        assert return_forecast_correlation(actual, predicted, reference=reference) == pytest.approx(1.0)

    def test_an_inverted_forecast_correlates_at_minus_one(self):
        reference = series([100.0] * 10)
        moves = np.linspace(-0.05, 0.05, 10)
        actual = series(100.0 * (1.0 + moves))
        predicted = series(100.0 * (1.0 - moves))
        assert return_forecast_correlation(actual, predicted, reference=reference) == pytest.approx(-1.0)

    def test_a_constant_forecast_has_undefined_correlation(self):
        reference = series([100.0] * 10)
        actual = series(100.0 * (1.0 + np.linspace(-0.05, 0.05, 10)))
        assert np.isnan(return_forecast_correlation(actual, series([100.0] * 10), reference=reference))


class TestIntervalMetrics:
    def test_coverage_counts_outcomes_inside_the_band(self):
        actual = series([100.0, 100.0, 100.0, 100.0])
        lower = series([99.0, 99.0, 101.0, 99.0])
        upper = series([101.0, 101.0, 102.0, 101.0])
        assert interval_coverage(actual, lower, upper) == 0.75

    def test_coverage_includes_the_boundary(self):
        assert interval_coverage(series([100.0]), series([100.0]), series([101.0])) == 1.0

    def test_relative_width_is_comparable_across_price_levels(self):
        narrow = relative_interval_width(series([100.0]), series([95.0]), series([105.0]))
        scaled = relative_interval_width(series([10000.0]), series([9500.0]), series([10500.0]))
        assert narrow == pytest.approx(scaled)

    def test_winkler_penalises_a_missed_outcome(self):
        actual = series([100.0])
        contained = winkler_score(actual, series([95.0]), series([105.0]))
        missed = winkler_score(actual, series([106.0]), series([116.0]))
        assert missed > contained

    def test_winkler_penalises_a_needlessly_wide_band(self):
        """A proper scoring rule cannot be gamed by widening the interval."""
        actual = series([100.0, 100.0, 100.0])
        tight = winkler_score(actual, series([98.0] * 3), series([102.0] * 3))
        huge = winkler_score(actual, series([1.0] * 3), series([10_000.0] * 3))
        assert huge > tight

    def test_winkler_prefers_calibration_to_either_extreme(self):
        actual = series([100.0, 105.0, 95.0, 102.0, 98.0])
        calibrated = winkler_score(actual, series([94.0] * 5), series([106.0] * 5))
        too_tight = winkler_score(actual, series([99.5] * 5), series([100.5] * 5))
        too_wide = winkler_score(actual, series([1.0] * 5), series([100_000.0] * 5))
        assert calibrated < too_tight
        assert calibrated < too_wide

    def test_pinball_loss_is_asymmetric(self):
        actual = series([100.0])
        below = pinball_loss(actual, series([90.0]), quantile=0.9)
        above = pinball_loss(actual, series([110.0]), quantile=0.9)
        assert below > above, "a 0.9 quantile should be penalised more for being too low"


class TestSkillScore:
    def test_positive_skill_means_beating_the_baseline(self):
        assert skill_score(model_error=5.0, baseline_error=10.0) == pytest.approx(0.5)

    def test_zero_skill_means_matching_the_baseline(self):
        assert skill_score(10.0, 10.0) == 0.0

    def test_negative_skill_means_losing_to_the_baseline(self):
        assert skill_score(20.0, 10.0) == pytest.approx(-1.0)

    def test_degenerate_baseline_is_nan(self):
        assert np.isnan(skill_score(1.0, 0.0))


class TestEvaluateForecast:
    def test_it_reports_every_metric_group(self):
        actual = series([100.0, 102.0, 101.0, 104.0])
        point = series([101.0, 101.5, 102.0, 103.0])
        lower = point * 0.95
        upper = point * 1.05

        metrics = evaluate_forecast(
            actual,
            point,
            reference=100.0,
            lower=lower,
            upper=upper,
            interval_level=0.95,
            naive_scale=1.5,
        )

        assert metrics.n == 4
        assert np.isfinite(metrics.mae)
        assert np.isfinite(metrics.rmse)
        assert np.isfinite(metrics.mase)
        assert np.isfinite(metrics.directional_accuracy)
        assert np.isfinite(metrics.interval_coverage)
        assert np.isfinite(metrics.winkler_score)
        assert metrics.interval_level == 0.95

    def test_a_scalar_reference_is_the_origin_close(self):
        actual = series([101.0, 99.0])
        point = series([102.0, 98.0])
        metrics = evaluate_forecast(actual, point, reference=100.0)
        assert metrics.directional_accuracy == 1.0

    def test_interval_metrics_are_nan_without_a_band(self):
        metrics = evaluate_forecast(series([100.0, 101.0]), series([100.0, 101.0]), reference=100.0)
        assert np.isnan(metrics.interval_coverage)
        assert np.isnan(metrics.winkler_score)

    def test_non_overlapping_series_is_an_error(self):
        actual = series([100.0], start="2024-01-01")
        point = series([100.0], start="2025-01-01")
        with pytest.raises(ValueError, match="do not overlap"):
            evaluate_forecast(actual, point, reference=100.0)

    def test_metrics_are_serialisable(self):
        import json

        metrics = evaluate_forecast(series([100.0, 101.0]), series([100.0, 101.0]), reference=100.0)
        json.dumps(metrics.to_dict())

    def test_direction_sets_are_disjoint_and_cover_the_bundle(self):
        assert not (LOWER_IS_BETTER & HIGHER_IS_BETTER)
        metrics = evaluate_forecast(
            series([100.0, 101.0]),
            series([100.0, 101.0]),
            reference=100.0,
            lower=series([90.0, 91.0]),
            upper=series([110.0, 111.0]),
        ).to_dict()
        for name in LOWER_IS_BETTER | HIGHER_IS_BETTER:
            assert name in metrics
