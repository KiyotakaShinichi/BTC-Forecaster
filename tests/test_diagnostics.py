"""Diagnostics, checked against series whose properties are known by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.diagnostics.multiple_testing import (
    benjamini_hochberg,
    bonferroni,
    expected_false_positives,
    family_wise_error_rate,
)
from btc_forecaster.diagnostics.suite import (
    DiagnosticReport,
    acf_values,
    adf_test,
    arch_lm_test,
    diagnose_prices,
    diagnose_residuals,
    diagnose_returns,
    durbin_watson_statistic,
    jarque_bera_test,
    kpss_test,
    ljung_box_test,
    pacf_values,
    significance_band,
)
from btc_forecaster.testing import ar1_return_prices, random_walk_prices, synthetic_market_frame


@pytest.fixture
def white_noise() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(scale=0.02, size=1000))


@pytest.fixture
def unit_root() -> pd.Series:
    return np.log(random_walk_prices(1000, seed=1))


class TestStationarity:
    def test_adf_rejects_a_unit_root_for_white_noise(self, white_noise):
        result = adf_test(white_noise)
        assert result.rejects_null
        assert "unit root" in result.null_hypothesis

    def test_adf_does_not_reject_for_a_random_walk(self, unit_root):
        assert not adf_test(unit_root).rejects_null

    def test_kpss_null_is_the_opposite_of_adf(self, white_noise):
        """The direction trap: p<0.05 means opposite things for the two tests."""
        adf = adf_test(white_noise)
        kpss = kpss_test(white_noise)

        assert "non-stationary" in adf.null_hypothesis
        assert kpss.null_hypothesis == "the series is stationary"
        assert adf.rejects_null and not kpss.rejects_null, "both should say 'stationary'"

    def test_both_tests_agree_that_a_random_walk_is_not_stationary(self, unit_root):
        assert not adf_test(unit_root).rejects_null
        assert kpss_test(unit_root).rejects_null

    def test_log_price_of_a_random_walk_looks_like_a_unit_root(self):
        frame = synthetic_market_frame(periods=1000, seed=2)
        report = diagnose_prices(frame["close"])
        assert not report.get("adf").rejects_null

    def test_conclusion_text_states_the_direction(self, white_noise):
        assert "evidence of stationarity" in adf_test(white_noise).conclusion


class TestAutocorrelation:
    def test_ljung_box_finds_no_structure_in_white_noise(self, white_noise):
        assert not ljung_box_test(white_noise).rejects_null

    def test_ljung_box_detects_an_ar1_process(self):
        returns = np.log(ar1_return_prices(1000, phi=0.5, seed=3)).diff().dropna()
        assert ljung_box_test(returns).rejects_null

    def test_durbin_watson_is_near_two_for_white_noise(self, white_noise):
        result = durbin_watson_statistic(white_noise)
        assert result.statistic == pytest.approx(2.0, abs=0.15)
        assert "no strong first-order" in result.detail["interpretation"]

    def test_durbin_watson_detects_positive_autocorrelation(self):
        returns = np.log(ar1_return_prices(1000, phi=0.7, seed=4)).diff().dropna()
        result = durbin_watson_statistic(returns)
        assert result.statistic < 1.5
        assert result.detail["interpretation"] == "positive autocorrelation"

    def test_durbin_watson_reports_no_p_value(self, white_noise):
        """It has no p-value, and must not be dressed up as if it did."""
        result = durbin_watson_statistic(white_noise)
        assert np.isnan(result.p_value)
        assert not result.rejects_null

    def test_acf_at_lag_zero_is_one(self, white_noise):
        assert acf_values(white_noise, nlags=10).iloc[0] == pytest.approx(1.0)

    def test_pacf_recovers_a_known_ar1_coefficient(self):
        returns = np.log(ar1_return_prices(2000, phi=0.5, seed=5)).diff().dropna()
        assert pacf_values(returns, nlags=10).iloc[1] == pytest.approx(0.5, abs=0.1)

    def test_significance_band_shrinks_with_sample_size(self):
        assert significance_band(100) > significance_band(10_000)
        assert significance_band(400) == pytest.approx(1.96 / 20.0, rel=1e-3)


class TestHeteroskedasticity:
    def test_arch_lm_finds_nothing_in_homoskedastic_noise(self, white_noise):
        assert not arch_lm_test(white_noise).rejects_null

    def test_arch_lm_detects_volatility_clustering(self):
        rng = np.random.default_rng(6)
        n = 1500
        sigma = np.empty(n)
        returns = np.empty(n)
        sigma[0], returns[0] = 0.02, 0.0
        for t in range(1, n):
            sigma[t] = np.sqrt(1e-5 + 0.15 * returns[t - 1] ** 2 + 0.8 * sigma[t - 1] ** 2)
            returns[t] = sigma[t] * rng.standard_normal()

        result = arch_lm_test(pd.Series(returns))
        assert result.rejects_null
        assert "conditional variance model is justified" in result.reject_means


class TestNormality:
    def test_jarque_bera_accepts_gaussian_data(self, white_noise):
        assert not jarque_bera_test(white_noise).rejects_null

    def test_jarque_bera_rejects_fat_tails_and_reports_excess_kurtosis(self):
        rng = np.random.default_rng(7)
        fat = pd.Series(rng.standard_t(df=3, size=2000))
        result = jarque_bera_test(fat)

        assert result.rejects_null
        assert result.detail["excess_kurtosis"] > 1.0


class TestReports:
    def test_return_report_runs_the_whole_battery(self):
        frame = synthetic_market_frame(periods=1000, seed=8)
        report = diagnose_returns(frame["close"].pct_change())

        assert report.n_tests == 6
        assert {"adf", "kpss", "ljung_box", "arch_lm", "jarque_bera", "durbin_watson"} == set(
            report.p_values()
        )

    def test_report_states_how_many_hypotheses_were_tested(self):
        frame = synthetic_market_frame(periods=800, seed=9)
        payload = diagnose_returns(frame["close"].pct_change()).to_dict()

        assert payload["n_tests"] == 6
        assert "uncorrected" in payload["multiple_testing_note"]
        assert "do not select models" in payload["multiple_testing_note"]

    def test_report_is_serialisable(self):
        import json

        frame = synthetic_market_frame(periods=600, seed=10)
        json.dumps(diagnose_returns(frame["close"].pct_change()).to_dict())

    def test_residual_report_finds_no_structure_in_clean_residuals(self, white_noise):
        report = diagnose_residuals(white_noise)
        assert not report.get("ljung_box").rejects_null

    def test_summary_is_human_readable(self):
        frame = synthetic_market_frame(periods=600, seed=11)
        text = diagnose_returns(frame["close"].pct_change()).summary()
        assert "returns" in text
        assert "adf" in text

    def test_missing_test_lookup_returns_none(self, white_noise):
        assert DiagnosticReport("x", (), 0).get("nope") is None


class TestMultipleTesting:
    def test_bonferroni_scales_by_the_number_of_tests(self):
        adjusted, rejected = bonferroni([0.01, 0.02, 0.03], alpha=0.05)
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.09])
        assert list(rejected) == [True, False, False]

    def test_bonferroni_clips_at_one(self):
        adjusted, _ = bonferroni([0.5, 0.6], alpha=0.05)
        assert (adjusted <= 1.0).all()

    def test_benjamini_hochberg_is_more_powerful_than_bonferroni(self):
        p_values = [0.001, 0.008, 0.02, 0.04, 0.3]
        _, bh_rejected = benjamini_hochberg(p_values, alpha=0.05)
        _, bonf_rejected = bonferroni(p_values, alpha=0.05)
        assert bh_rejected.sum() > bonf_rejected.sum()

    def test_benjamini_hochberg_adjusted_values_are_monotone(self):
        adjusted, _ = benjamini_hochberg([0.001, 0.01, 0.02, 0.04, 0.5])
        assert list(adjusted) == sorted(adjusted)

    def test_nan_p_values_do_not_shift_the_threshold(self):
        """A test that failed to converge must not affect the others."""
        with_nan, _ = benjamini_hochberg([0.01, np.nan, 0.02])
        without, _ = benjamini_hochberg([0.01, 0.02])
        np.testing.assert_allclose([with_nan[0], with_nan[2]], without)
        assert np.isnan(with_nan[1])

    def test_all_nan_input_rejects_nothing(self):
        adjusted, rejected = benjamini_hochberg([np.nan, np.nan])
        assert not rejected.any()
        assert np.isnan(adjusted).all()

    def test_expected_false_positives_quantifies_the_pacf_scan(self):
        """60 lags at a 5% band: about three hits on pure noise."""
        assert expected_false_positives(60, 0.05) == pytest.approx(3.0)

    def test_family_wise_error_rate_grows_fast(self):
        assert family_wise_error_rate(1, 0.05) == pytest.approx(0.05)
        assert family_wise_error_rate(11, 0.05) > 0.40
        assert family_wise_error_rate(48, 0.05) > 0.90

    def test_report_correction_flags_survivors(self):
        frame = synthetic_market_frame(periods=1000, seed=12)
        report = diagnose_returns(frame["close"].pct_change())
        corrected = report.adjusted("benjamini-hochberg")

        assert corrected["n_tests"] == 6
        assert len(corrected["results"]) == 6
        assert all("significant_after_correction" in row for row in corrected["results"])

    def test_unknown_correction_method_is_rejected(self):
        frame = synthetic_market_frame(periods=400, seed=13)
        with pytest.raises(ValueError, match="unknown correction"):
            diagnose_returns(frame["close"].pct_change()).adjusted("magic")
