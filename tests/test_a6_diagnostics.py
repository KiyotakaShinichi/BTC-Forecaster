"""Residual diagnostics, and the discipline of refusing to answer.

The statistical machinery is reused and already tested. What is tested here is
the restraint around it:

* a p-value from too few observations is INSUFFICIENT_N, not a small number;
* a failed assumption is reported as a failed assumption, not as a verdict;
* 200 p-values produce ten rejections from nothing, and the expected count is
  reported beside the observed one.

The last is the one that matters most in a forty-model zoo. A residual table
without it reads as a list of discoveries.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research.diagnostics import (
    INSUFFICIENT_N,
    MINIMUM_OBSERVATIONS,
    diagnose,
    diagnose_all,
)

RNG = np.random.default_rng(909)


class TestSmallSamplesAreRefused:
    @pytest.mark.parametrize("n", [0, 1, 10, MINIMUM_OBSERVATIONS - 1])
    def test_every_test_declines_below_the_minimum(self, n: int) -> None:
        """Ljung-Box at 10 lags on 30 residuals is arithmetic without evidence."""
        report = diagnose(RNG.normal(size=n))
        assert {t["status"] for t in report["tests"]} == {INSUFFICIENT_N}
        assert all(t["p_value"] is None for t in report["tests"])

    def test_the_refusal_explains_itself(self) -> None:
        report = diagnose(RNG.normal(size=10))
        interpretation = report["tests"][0]["interpretation"]
        assert str(MINIMUM_OBSERVATIONS) in interpretation
        assert "noise, not evidence" in interpretation

    def test_the_correlogram_declines_too(self) -> None:
        assert diagnose(RNG.normal(size=10))["correlogram"]["status"] == INSUFFICIENT_N

    def test_at_the_minimum_the_tests_run(self) -> None:
        report = diagnose(RNG.normal(size=MINIMUM_OBSERVATIONS))
        assert INSUFFICIENT_N not in {t["status"] for t in report["tests"]}


class TestTheDiagnosticsDetectWhatTheyClaim:
    def test_white_noise_passes_ljung_box(self) -> None:
        report = diagnose(RNG.normal(size=500))
        ljung = next(t for t in report["tests"] if t["name"] == "ljung_box")
        assert ljung["status"] == "NOT_REJECTED"

    def test_autocorrelated_residuals_fail_ljung_box(self) -> None:
        """The control: a test that never rejects is not a test."""
        values = np.zeros(500)
        innovations = RNG.normal(size=500)
        for t in range(1, 500):
            values[t] = 0.6 * values[t - 1] + innovations[t]
        ljung = next(t for t in diagnose(values)["tests"] if t["name"] == "ljung_box")
        assert ljung["status"] == "REJECT"

    def test_volatility_clustering_fails_arch_lm(self) -> None:
        values = np.zeros(600)
        variance = 1.0
        for t in range(600):
            variance = 0.05 + 0.15 * values[t - 1] ** 2 + 0.8 * variance if t else variance
            values[t] = RNG.normal(0.0, np.sqrt(variance))
        arch = next(t for t in diagnose(values)["tests"] if t["name"] == "arch_lm")
        assert arch["status"] == "REJECT"

    def test_a_heavy_tailed_series_fails_jarque_bera(self) -> None:
        jb = next(
            t for t in diagnose(RNG.standard_t(3, size=500))["tests"] if t["name"] == "jarque_bera"
        )
        assert jb["status"] == "REJECT"

    def test_failed_assumptions_are_listed(self) -> None:
        report = diagnose(RNG.standard_t(2, size=500))
        assert "jarque_bera" in report["assumptions_failed"]


class TestDiagnosticsAreNotVerdicts:
    def test_the_report_says_so(self) -> None:
        """Almost every conditional-mean model on this series fails ARCH-LM,
        and that says nothing about forecast quality."""
        note = diagnose(RNG.normal(size=200))["note"]
        assert "not a verdict" in note
        assert "ARCH-LM" in note

    def test_a_broken_test_is_recorded_rather_than_raised(self) -> None:
        """A diagnostic that errors must not end a forty-model report."""
        report = diagnose(np.full(200, 1.0))
        assert all(t["status"] in {"REJECT", "NOT_REJECTED", "ERROR"} for t in report["tests"])

    def test_a_constant_residual_series_is_reported_as_degenerate(self) -> None:
        """Autocorrelation of a constant is undefined. Saying so beats returning
        a column of NaN with a RuntimeWarning attached."""
        correlogram = diagnose(np.full(200, 1.0))["correlogram"]
        assert correlogram["status"] == "DEGENERATE"
        assert "undefined" in correlogram["reason"]

    def test_non_finite_values_are_dropped_not_propagated(self) -> None:
        values = RNG.normal(size=200)
        values[5] = np.nan
        values[9] = np.inf
        assert diagnose(values)["n"] == 198


class TestMultipleTestingAcrossTheZoo:
    def test_the_expected_false_positive_count_is_reported(self) -> None:
        """Forty models times five tests is 200 p-values; ten are expected to be
        significant from nothing at all. Reporting the expectation beside the
        observation is what stops the table reading as a list of discoveries."""
        residuals = {f"model_{i}": RNG.normal(size=300) for i in range(20)}
        report = diagnose_all(residuals)
        multiple = report["multiple_testing"]
        assert multiple["comparisons"] == 100
        assert multiple["expected_false_positives_at_alpha"] == pytest.approx(5.0)
        assert "raw_rejections" in multiple
        assert "rejections_after_bh" in multiple

    def test_correction_never_increases_the_rejection_count(self) -> None:
        residuals = {f"model_{i}": RNG.normal(size=300) for i in range(15)}
        multiple = diagnose_all(residuals)["multiple_testing"]
        assert multiple["rejections_after_bh"] <= multiple["raw_rejections"]

    def test_every_p_value_gets_a_q_value(self) -> None:
        residuals = {f"model_{i}": RNG.normal(size=200) for i in range(5)}
        multiple = diagnose_all(residuals)["multiple_testing"]
        assert len(multiple["results"]) == multiple["comparisons"]
        assert all("q_value" in r for r in multiple["results"])

    def test_q_values_are_at_least_their_p_values(self) -> None:
        residuals = {f"model_{i}": RNG.normal(size=200) for i in range(8)}
        for row in diagnose_all(residuals)["multiple_testing"]["results"]:
            assert row["q_value"] >= row["p_value"] - 1e-12

    def test_short_series_contribute_no_p_values(self) -> None:
        """A model with too few residuals must not enlarge the testing family."""
        residuals = {"long": RNG.normal(size=300), "short": RNG.normal(size=10)}
        multiple = diagnose_all(residuals)["multiple_testing"]
        assert multiple["comparisons"] == 5

    def test_the_report_warns_about_its_own_arithmetic(self) -> None:
        note = diagnose_all({"a": RNG.normal(size=200)})["note"]
        assert "arithmetic rather than evidence" in note
