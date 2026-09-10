"""Paired comparison, multiplicity, stability, and error diversity.

The results table ranks forty models. These four analyses are what stop that
ranking from being read as forty findings:

* Diebold-Mariano says whether a difference is distinguishable from noise;
* Benjamini-Hochberg says how many of those p-values are arithmetic;
* the block analysis says whether a mean skill describes anything;
* the diversity analysis says whether the forty models are really one.

Every one of them is a way of *reducing* what the table appears to show, which
is the correct direction for a study with this sample size.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research.comparison import (
    DEFAULT_BASELINE,
    INSUFFICIENT_FORWARD_EVIDENCE,
    MINIMUM_PAIRED_ORIGINS,
    compare_against_baseline,
    correct_for_multiplicity,
    error_diversity,
    stability_by_block,
)

RNG = np.random.default_rng(77)


@pytest.fixture()
def scenario():
    """A baseline, a genuinely better model, and a genuinely worse one."""
    n = 400
    actual = RNG.normal(0.0, 0.02, n)
    return {
        "actual": actual,
        "forecasts": {
            DEFAULT_BASELINE: np.zeros(n),
            "informative": 0.6 * actual + RNG.normal(0.0, 0.004, n),
            "noisy": RNG.normal(0.0, 0.05, n),
            "copy_of_baseline": np.zeros(n),
        },
    }


class TestDieboldMariano:
    def test_a_better_model_is_detected(self, scenario) -> None:
        comparisons = compare_against_baseline(scenario["actual"], scenario["forecasts"])
        informative = next(c for c in comparisons if c.model_id == "informative")
        assert informative.status == "OK"
        assert informative.p_value < 0.01
        # Negative mean loss difference: the candidate's loss is lower.
        assert informative.mean_loss_difference < 0

    def test_a_worse_model_is_detected_as_worse(self, scenario) -> None:
        """The direction matters. A significant DM result with a positive loss
        difference means significantly *worse*, and reporting only the p-value
        would let that read as a discovery."""
        comparisons = compare_against_baseline(scenario["actual"], scenario["forecasts"])
        noisy = next(c for c in comparisons if c.model_id == "noisy")
        assert noisy.p_value < 0.01
        assert noisy.mean_loss_difference > 0

    def test_an_identical_model_is_not_distinguishable(self, scenario) -> None:
        comparisons = compare_against_baseline(scenario["actual"], scenario["forecasts"])
        copy = next(c for c in comparisons if c.model_id == "copy_of_baseline")
        assert copy.mean_loss_difference == pytest.approx(0.0, abs=1e-15)

    def test_every_comparison_shares_the_same_window(self, scenario) -> None:
        """Comparing two models on different evaluation blocks is not a
        comparison, and is the easiest way to make one look better."""
        comparisons = compare_against_baseline(scenario["actual"], scenario["forecasts"])
        assert len({c.n_origins for c in comparisons}) == 1

    def test_too_few_origins_is_refused(self) -> None:
        n = MINIMUM_PAIRED_ORIGINS - 1
        actual = RNG.normal(size=n)
        comparisons = compare_against_baseline(
            actual, {DEFAULT_BASELINE: np.zeros(n), "other": RNG.normal(size=n)}
        )
        assert comparisons[0].status == INSUFFICIENT_FORWARD_EVIDENCE
        assert comparisons[0].p_value is None

    def test_a_missing_baseline_raises(self, scenario) -> None:
        with pytest.raises(KeyError, match="did not produce a forecast"):
            compare_against_baseline(
                scenario["actual"], {"a": np.zeros(len(scenario["actual"]))}
            )

    def test_the_baseline_is_the_naive_forecast(self) -> None:
        """A2's hypothesis, and the one nothing has beaten."""
        assert DEFAULT_BASELINE == "naive_last_value"


class TestMultiplicity:
    def test_the_family_size_counts_only_usable_p_values(self, scenario) -> None:
        """Three candidates, two comparisons.

        `copy_of_baseline` produces an identical loss series, so the DM
        statistic is 0/0 and there is no p-value to correct. It is excluded from
        the family rather than counted as a non-rejection -- which is the right
        direction: an undefined test must not make the correction look more
        conservative than it is.
        """
        comparisons = compare_against_baseline(scenario["actual"], scenario["forecasts"])
        assert len(comparisons) == 3
        corrected = correct_for_multiplicity(comparisons)
        assert corrected["family_size"] == 2

    def test_the_expected_false_positive_count_is_reported(self) -> None:
        """Forty comparisons at alpha=0.05 produce two significant results from
        nothing at all."""
        n = 300
        actual = RNG.normal(size=n)
        forecasts = {DEFAULT_BASELINE: np.zeros(n)}
        forecasts.update({f"m{i}": RNG.normal(size=n) * 1e-9 for i in range(40)})
        corrected = correct_for_multiplicity(
            compare_against_baseline(actual, forecasts)
        )
        assert corrected["family_size"] == 40
        assert corrected["expected_false_positives_at_alpha"] == pytest.approx(2.0)

    def test_correction_never_increases_significance(self, scenario) -> None:
        corrected = correct_for_multiplicity(
            compare_against_baseline(scenario["actual"], scenario["forecasts"])
        )
        assert corrected["significant_after_bh"] <= corrected["raw_significant"]

    def test_q_values_are_never_below_their_p_values(self, scenario) -> None:
        corrected = correct_for_multiplicity(
            compare_against_baseline(scenario["actual"], scenario["forecasts"])
        )
        for row in corrected["results"]:
            assert row["q_value"] >= row["p_value"] - 1e-12

    def test_the_note_points_at_the_q_value(self, scenario) -> None:
        corrected = correct_for_multiplicity(
            compare_against_baseline(scenario["actual"], scenario["forecasts"])
        )
        assert "q-value is what a claim should rest on" in corrected["note"]

    def test_an_empty_family_is_reported_as_insufficient(self) -> None:
        assert correct_for_multiplicity([])["status"] == INSUFFICIENT_FORWARD_EVIDENCE


class TestStabilityByBlock:
    def test_a_model_good_in_only_one_block_is_visible(self) -> None:
        """The whole point. A mean skill hides a model that worked once and
        reversed afterwards, and the mean is what a results table shows."""
        n = 300
        actual = RNG.normal(0.0, 0.02, n)
        streaky = np.zeros(n)
        streaky[: n // 3] = 0.9 * actual[: n // 3]  # excellent early
        streaky[n // 3 :] = -0.9 * actual[n // 3 :]  # inverted afterwards

        frame = stability_by_block(actual, {"streaky": streaky, "flat": np.zeros(n)})
        row = frame.set_index("model_id").loc["streaky"]
        assert row["block_1_skill"] > 0.5
        assert row["worst_block_skill"] < 0
        assert row["positive_in_every_block"] is np.False_ or not row["positive_in_every_block"]

    def test_a_consistent_model_is_positive_everywhere(self) -> None:
        n = 300
        actual = RNG.normal(0.0, 0.02, n)
        frame = stability_by_block(actual, {"good": 0.8 * actual})
        assert bool(frame.set_index("model_id").loc["good", "positive_in_every_block"])

    def test_the_blocks_partition_the_holdout(self) -> None:
        n = 300
        actual = RNG.normal(size=n)
        frame = stability_by_block(actual, {"m": np.zeros(n)}, blocks=3)
        assert {"block_1_skill", "block_2_skill", "block_3_skill"} <= set(frame.columns)

    def test_worst_block_is_never_above_the_mean(self) -> None:
        n = 300
        actual = RNG.normal(0.0, 0.02, n)
        frame = stability_by_block(actual, {"m": 0.3 * actual, "n": np.zeros(n)})
        assert bool((frame["worst_block_skill"] <= frame["mean_skill"] + 1e-12).all())


class TestErrorDiversity:
    def test_identical_models_are_reported_as_identical(self) -> None:
        """If forty models make the same errors, an ensemble of them is one
        model with extra steps."""
        n = 200
        actual = RNG.normal(0.0, 0.02, n)
        same = 0.5 * actual
        diversity = error_diversity(actual, {"a": same, "b": same.copy()})
        assert diversity["mean_pairwise_error_correlation"] == pytest.approx(1.0)
        assert diversity["mean_pairwise_sign_disagreement"] == pytest.approx(0.0)

    def test_genuinely_different_models_are_distinguishable(self) -> None:
        n = 400
        actual = RNG.normal(0.0, 0.02, n)
        diversity = error_diversity(
            actual,
            {
                "one": RNG.normal(0.0, 0.02, n),
                "two": RNG.normal(0.0, 0.02, n),
            },
        )
        assert diversity["mean_pairwise_error_correlation"] < 0.9
        assert diversity["mean_pairwise_sign_disagreement"] > 0.2

    def test_shared_large_errors_are_measured(self) -> None:
        n = 300
        actual = RNG.normal(0.0, 0.02, n)
        diversity = error_diversity(actual, {"a": np.zeros(n), "b": 0.5 * actual})
        assert 0.0 <= diversity["mean_shared_large_error_jaccard"] <= 1.0
        assert diversity["large_error_quantile"] == 0.95

    def test_it_refuses_a_single_model(self) -> None:
        assert "at least two" in error_diversity(np.zeros(10), {"a": np.zeros(10)})["status"]

    def test_it_says_a6_builds_no_ensemble(self) -> None:
        """Preparation for a possible future study, explicitly not one."""
        n = 120
        actual = RNG.normal(size=n)
        diversity = error_diversity(actual, {"a": np.zeros(n), "b": RNG.normal(size=n)})
        assert "A6 builds no ensemble" in diversity["interpretation"]
