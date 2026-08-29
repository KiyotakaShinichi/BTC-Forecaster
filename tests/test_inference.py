"""Dependence-aware inference: block bootstrap, Diebold-Mariano, corrections.

The tests that matter most here are the ones showing the naive procedure gives
the wrong answer on dependent data and these give a defensible one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.evaluation.inference import (
    INFERENCE_NOTES,
    KNOWN_NESTED_PAIRS,
    best_constant_accuracy,
    compare_models,
    diebold_mariano,
    directional_accuracy_ci,
    directional_base_rate,
    is_nested,
    loss_series,
    moving_block_indices,
    optimal_block_length,
    stationary_bootstrap,
    stationary_bootstrap_indices,
)


def ar1_series(n: int, phi: float, *, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(scale=scale, size=n)
    out = np.zeros(n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + innovations[t]
    return out


class TestResamplingMechanics:
    def test_resample_has_the_original_length(self):
        rng = np.random.default_rng(0)
        assert len(stationary_bootstrap_indices(50, expected_block_length=5, rng=rng)) == 50

    def test_indices_stay_in_range(self):
        rng = np.random.default_rng(1)
        idx = stationary_bootstrap_indices(30, expected_block_length=4, rng=rng)
        assert idx.min() >= 0 and idx.max() < 30

    def test_blocks_wrap_around_the_end(self):
        """Wrapping is what keeps every observation equally likely to appear."""
        rng = np.random.default_rng(3)
        seen = set()
        for _ in range(200):
            seen.update(stationary_bootstrap_indices(20, expected_block_length=5, rng=rng).tolist())
        assert seen == set(range(20))

    def test_a_long_block_preserves_consecutive_runs(self):
        rng = np.random.default_rng(2)
        idx = stationary_bootstrap_indices(200, expected_block_length=50, rng=rng)
        consecutive = np.mean(np.diff(idx) == 1)
        assert consecutive > 0.9

    def test_block_length_one_is_the_iid_bootstrap(self):
        rng = np.random.default_rng(4)
        idx = stationary_bootstrap_indices(200, expected_block_length=1, rng=rng)
        assert np.mean(np.diff(idx) == 1) < 0.15

    def test_moving_block_returns_the_right_length(self):
        rng = np.random.default_rng(5)
        assert len(moving_block_indices(37, block_length=5, rng=rng)) == 37

    def test_optimal_block_length_follows_the_cube_root_rule(self):
        assert optimal_block_length(np.zeros(1000)) == pytest.approx(10.0)
        assert optimal_block_length(np.zeros(8)) == 2.0

    @pytest.mark.parametrize("bad", [0, 0.5])
    def test_invalid_block_length_is_rejected(self, bad):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="expected_block_length"):
            stationary_bootstrap_indices(10, expected_block_length=bad, rng=rng)


class TestBootstrapDeterminism:
    def test_the_same_seed_gives_the_same_interval(self):
        sample = ar1_series(80, 0.5, seed=7)
        a = stationary_bootstrap(sample, seed=42)
        b = stationary_bootstrap(sample, seed=42)
        assert (a.statistic, a.lower, a.upper) == (b.statistic, b.lower, b.upper)

    def test_different_seeds_give_similar_but_distinct_intervals(self):
        sample = ar1_series(80, 0.5, seed=7)
        a = stationary_bootstrap(sample, seed=1)
        b = stationary_bootstrap(sample, seed=2)
        assert a.lower != b.lower
        assert abs(a.lower - b.lower) < abs(a.statistic) + 0.5

    def test_the_point_statistic_is_the_sample_value(self):
        sample = np.array([1.0, 2.0, 3.0, 4.0])
        assert stationary_bootstrap(sample, seed=0).statistic == pytest.approx(2.5)

    def test_the_interval_brackets_the_statistic(self):
        ci = stationary_bootstrap(ar1_series(100, 0.4, seed=3), seed=0)
        assert ci.lower <= ci.statistic <= ci.upper

    def test_assumptions_travel_with_the_result(self):
        ci = stationary_bootstrap(ar1_series(64, 0.3, seed=1), seed=0)
        assert ci.method.startswith("stationary bootstrap")
        assert ci.block_length == pytest.approx(4.0)
        assert ci.n_observations == 64

    def test_too_few_observations_is_an_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            stationary_bootstrap([1.0])

    def test_result_is_serialisable(self):
        import json

        json.dumps(stationary_bootstrap(ar1_series(40, 0.2, seed=1), seed=0).to_dict())


class TestDependenceWidensIntervals:
    """The whole point of the module, stated as a test."""

    def test_dependent_data_gives_a_wider_interval_than_the_iid_assumption(self):
        dependent = ar1_series(200, 0.8, seed=11)

        honest = stationary_bootstrap(dependent, seed=0, block_length=20)
        pretend_iid = stationary_bootstrap(dependent, seed=0, block_length=1)

        assert honest.width > pretend_iid.width * 1.5, (
            "ignoring serial dependence understates uncertainty"
        )

    def test_on_independent_data_the_two_agree(self):
        independent = np.random.default_rng(12).normal(size=300)
        blocked = stationary_bootstrap(independent, seed=0, block_length=10)
        iid = stationary_bootstrap(independent, seed=0, block_length=1)
        assert blocked.width == pytest.approx(iid.width, rel=0.35)


class TestDirectionalAccuracyCi:
    def test_a_fair_coin_interval_contains_one_half(self):
        hits = np.random.default_rng(0).random(200) < 0.5
        ci = directional_accuracy_ci(hits, seed=0)
        assert ci.lower <= 0.5 <= ci.upper
        assert not ci.excludes_null

    def test_a_strongly_biased_sample_excludes_one_half(self):
        hits = np.random.default_rng(0).random(200) < 0.75
        ci = directional_accuracy_ci(hits, seed=0)
        assert ci.excludes_null
        assert ci.lower > 0.5

    def test_the_null_is_recorded_as_one_half(self):
        assert directional_accuracy_ci([True, False, True, False], seed=0).null_value == 0.5

    def test_a_small_sample_gives_a_wide_interval(self):
        """8 origins cannot establish an edge, and the interval must say so."""
        hits = np.array([True] * 6 + [False] * 2)
        ci = directional_accuracy_ci(hits, seed=0)
        assert ci.n_observations == 8
        assert ci.width > 0.3

    def test_serially_dependent_hits_widen_the_interval(self):
        """A trending market makes consecutive hits agree; the CI must not
        treat those as independent evidence."""
        clustered = np.array(([True] * 10 + [False] * 10) * 5)
        blocked = directional_accuracy_ci(clustered, seed=0, block_length=10)
        naive = directional_accuracy_ci(clustered, seed=0, block_length=1)
        assert blocked.width > naive.width


class TestDirectionalNull:
    """A coin is the wrong null on a trending asset over a multi-week horizon."""

    def test_base_rate_counts_realised_up_moves(self):
        actual = np.array([110.0, 120.0, 90.0, 105.0])
        reference = np.full(4, 100.0)
        assert directional_base_rate(actual, reference) == pytest.approx(0.75)

    def test_the_best_constant_predictor_is_the_null(self):
        assert best_constant_accuracy(0.61) == pytest.approx(0.61)
        assert best_constant_accuracy(0.39) == pytest.approx(0.61)
        assert best_constant_accuracy(0.5) == pytest.approx(0.5)

    def test_an_always_up_forecast_scores_the_base_rate_and_has_no_edge(self):
        """The exact artefact found in the first 36-fold live run:
        random_walk_drift predicted up on 100% of origins and scored 0.611,
        which is the base rate. Against 0.5 that looks significant."""
        # Construct the base rate exactly rather than sampling it, so the test
        # is about the null and not about which draw the RNG produced.
        n, n_up = 300, 183  # 0.61
        actual_up = np.array([True] * n_up + [False] * (n - n_up))
        actual_up = np.random.default_rng(0).permutation(actual_up)
        hits = actual_up  # an always-up predictor is right exactly when it went up

        base_rate = float(actual_up.mean())
        assert base_rate == pytest.approx(0.61)
        against_coin = directional_accuracy_ci(hits, seed=0, null=0.5)
        against_base = directional_accuracy_ci(
            hits, seed=0, null=best_constant_accuracy(base_rate)
        )

        assert against_coin.exceeds_null, "a coin null calls the base rate significant"
        assert not against_base.exceeds_null, "the correct null finds no edge"

    def test_a_genuine_edge_survives_the_base_rate_null(self):
        # Right 85% of the time regardless of direction -- real skill.
        n, n_hit = 300, 255
        hits = np.array([True] * n_hit + [False] * (n - n_hit))
        hits = np.random.default_rng(1).permutation(hits)

        against_base = directional_accuracy_ci(hits, seed=0, null=best_constant_accuracy(0.61))
        assert against_base.exceeds_null
        assert against_base.lower > 0.61

    def test_excluding_the_null_is_two_sided_and_beating_it_is_not(self):
        """The bug this distinction exists to prevent: a flat random-walk
        forecast scores 0.333 against a 0.611 base rate. Its interval excludes
        the null while being significantly WORSE than it, and a two-sided flag
        reported that as 'beats the null'."""
        n, n_hit = 300, 100  # 0.333
        hits = np.random.default_rng(2).permutation(
            np.array([True] * n_hit + [False] * (n - n_hit))
        )
        ci = directional_accuracy_ci(hits, seed=0, null=0.6111)

        assert ci.excludes_null, "two-sided: the interval does exclude the null"
        assert not ci.exceeds_null, "but it lies below, so it does not beat it"
        assert ci.below_null
        assert ci.versus_null == "below"

    def test_a_genuinely_better_model_exceeds_the_null(self):
        n, n_hit = 300, 255
        hits = np.random.default_rng(3).permutation(
            np.array([True] * n_hit + [False] * (n - n_hit))
        )
        ci = directional_accuracy_ci(hits, seed=0, null=0.6111)
        assert ci.exceeds_null
        assert not ci.below_null
        assert ci.versus_null == "above"

    def test_an_indistinguishable_model_is_labelled_as_such(self):
        n, n_hit = 300, 184
        hits = np.random.default_rng(4).permutation(
            np.array([True] * n_hit + [False] * (n - n_hit))
        )
        ci = directional_accuracy_ci(hits, seed=0, null=0.6111)
        assert not ci.exceeds_null
        assert not ci.below_null
        assert ci.versus_null == "indistinguishable"

    def test_the_null_is_recorded_on_the_result(self):
        ci = directional_accuracy_ci([True, False, True], seed=0, null=0.61)
        assert ci.null_value == pytest.approx(0.61)
        assert ci.to_dict()["null_value"] == pytest.approx(0.61)

    def test_the_default_remains_a_coin_but_is_documented_as_usually_wrong(self):
        assert directional_accuracy_ci([True, False], seed=0).null_value == 0.5
        assert "base rate" in INFERENCE_NOTES["directional_null"]

    def test_an_empty_sample_gives_nan(self):
        assert np.isnan(directional_base_rate([], []))
        assert np.isnan(best_constant_accuracy(float("nan")))


class TestDieboldMariano:
    def test_identical_forecasts_are_not_distinguishable(self):
        loss = np.abs(np.random.default_rng(0).normal(size=100))
        result = diebold_mariano(loss, loss.copy(), model_a="x", model_b="y")
        assert not np.isfinite(result.statistic) or result.p_value > 0.5

    def test_a_clearly_better_model_is_detected(self):
        rng = np.random.default_rng(1)
        good = np.abs(rng.normal(scale=1.0, size=200))
        bad = np.abs(rng.normal(scale=3.0, size=200))
        result = diebold_mariano(good, bad, model_a="good", model_b="bad")
        assert result.p_value < 0.01
        assert result.favours == "good"
        assert result.mean_loss_difference < 0

    def test_favours_reports_lower_loss_without_claiming_significance(self):
        rng = np.random.default_rng(2)
        a = np.abs(rng.normal(size=30)) * 0.99
        b = np.abs(rng.normal(size=30))
        result = diebold_mariano(a, b, model_a="a", model_b="b")
        assert result.favours in {"a", "b"}

    def test_hac_lags_follow_the_horizon(self):
        loss_a = np.abs(ar1_series(150, 0.3, seed=1))
        loss_b = np.abs(ar1_series(150, 0.3, seed=2))
        assert diebold_mariano(loss_a, loss_b, horizon=1).hac_lags == 0
        assert diebold_mariano(loss_a, loss_b, horizon=10).hac_lags == 9

    def test_overlapping_horizons_widen_the_standard_error(self):
        """Ignoring overlap understates variance and over-rejects."""
        rng = np.random.default_rng(3)
        shared = ar1_series(300, 0.9, seed=4)
        loss_a = np.abs(shared + rng.normal(scale=0.1, size=300)) + 0.05
        loss_b = np.abs(shared + rng.normal(scale=0.1, size=300))

        naive = diebold_mariano(loss_a, loss_b, horizon=1)
        honest = diebold_mariano(loss_a, loss_b, horizon=20)
        assert abs(honest.statistic) < abs(naive.statistic)

    def test_small_sample_correction_shrinks_the_statistic(self):
        rng = np.random.default_rng(5)
        a = np.abs(rng.normal(size=25))
        b = np.abs(rng.normal(size=25)) + 0.3
        corrected = diebold_mariano(a, b, horizon=5, small_sample=True)
        raw = diebold_mariano(a, b, horizon=5, small_sample=False)
        assert abs(corrected.statistic) < abs(raw.statistic)
        assert corrected.small_sample_corrected

    def test_nested_pairs_are_flagged_and_marked_unusable(self):
        """DM is invalid for nested models; saying so beats reporting a number."""
        rng = np.random.default_rng(6)
        result = diebold_mariano(
            np.abs(rng.normal(size=100)),
            np.abs(rng.normal(size=100)),
            model_a="random_walk",
            model_b="arima",
        )
        assert result.nested
        assert not result.usable
        assert any("nested" in c.lower() for c in result.caveats)
        assert any("Clark-West" in c for c in result.caveats)

    def test_non_nested_pairs_are_usable(self):
        rng = np.random.default_rng(7)
        result = diebold_mariano(
            np.abs(rng.normal(size=100)),
            np.abs(rng.normal(size=100)),
            model_a="ets",
            model_b="prophet",
        )
        assert not result.nested
        assert result.usable

    def test_nesting_can_be_declared_explicitly(self):
        rng = np.random.default_rng(8)
        result = diebold_mariano(
            np.abs(rng.normal(size=50)), np.abs(rng.normal(size=50)),
            model_a="custom_a", model_b="custom_b", nested=True,
        )
        assert result.nested

    def test_a_tiny_sample_is_reported_not_computed(self):
        result = diebold_mariano([1.0, 2.0], [1.5, 2.5], model_a="a", model_b="b")
        assert np.isnan(result.statistic)
        assert any("Too few observations" in c for c in result.caveats)

    def test_a_short_sample_carries_a_power_warning(self):
        rng = np.random.default_rng(9)
        result = diebold_mariano(
            np.abs(rng.normal(size=8)), np.abs(rng.normal(size=8)), model_a="a", model_b="b"
        )
        assert any("very little power" in c for c in result.caveats)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="different lengths"):
            diebold_mariano([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_result_is_serialisable(self):
        import json

        rng = np.random.default_rng(10)
        json.dumps(diebold_mariano(np.abs(rng.normal(size=40)), np.abs(rng.normal(size=40))).to_dict())

    def test_known_nested_pairs_include_the_random_walk_family(self):
        assert is_nested("random_walk", "arima")
        assert is_nested("arima", "random_walk"), "nesting is symmetric for this check"
        assert not is_nested("ets", "prophet")
        assert ("random_walk", "random_walk_drift") in KNOWN_NESTED_PAIRS


class TestLossSeries:
    def test_absolute_and_squared_losses(self):
        np.testing.assert_allclose(loss_series([1.0, 2.0], [0.0, 0.0]), [1.0, 2.0])
        np.testing.assert_allclose(
            loss_series([1.0, 2.0], [0.0, 0.0], loss="squared"), [1.0, 4.0]
        )

    def test_unknown_loss_is_rejected(self):
        with pytest.raises(ValueError, match="unknown loss"):
            loss_series([1.0], [1.0], loss="huber")


class TestCompareModels:
    @pytest.fixture
    def records(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for origin in range(60):
            for step in (1, 2):
                actual = 100.0 + rng.normal(scale=5.0)
                rows.append({
                    "model": "random_walk", "origin": origin, "step": step,
                    "actual": actual, "predicted": 100.0, "origin_close": 100.0,
                })
                rows.append({
                    "model": "prophet_xgb_hybrid", "origin": origin, "step": step,
                    "actual": actual, "predicted": 100.0 + rng.normal(scale=8.0),
                    "origin_close": 100.0,
                })
                rows.append({
                    "model": "prophet", "origin": origin, "step": step,
                    "actual": actual, "predicted": 100.0 + rng.normal(scale=1.0),
                    "origin_close": 100.0,
                })
        return pd.DataFrame(rows)

    def test_every_challenger_is_compared_to_the_baseline(self, records):
        table = compare_models(records, baseline="random_walk", horizon=2)
        assert set(table.index) == {"prophet_xgb_hybrid", "prophet"}
        assert (table["model_b"] == "random_walk").all()

    def test_the_correction_is_applied_across_the_family(self, records):
        table = compare_models(records, baseline="random_walk", horizon=2)
        assert table["correction"].eq("benjamini-hochberg").all()

        # Only usable comparisons get an adjusted p-value; nested pairs are
        # excluded from the family and carry NaN by design.
        usable = table[table["usable"]]
        assert not usable.empty
        assert (usable["p_adjusted"] >= usable["p_value"] - 1e-12).all()
        assert table.loc[~table["usable"], "p_adjusted"].isna().all()

    def test_only_usable_comparisons_enter_the_correction(self, records):
        """Correcting a statistic that was never valid would launder it."""
        table = compare_models(records, baseline="random_walk", horizon=2)
        expected = int(table["usable"].sum())
        assert (table["n_hypotheses_corrected"] == expected).all()

    def test_nested_comparisons_are_excluded_from_correction(self, records):
        renamed = records.replace({"prophet_xgb_hybrid": "arima"})
        table = compare_models(renamed, baseline="random_walk", horizon=2)
        assert table.loc["arima", "nested"]
        assert not table.loc["arima", "usable"]
        assert np.isnan(table.loc["arima", "p_adjusted"])
        assert table.loc["prophet", "usable"]

    def test_step_filter_restricts_the_comparison(self, records):
        table = compare_models(records, baseline="random_walk", horizon=1, step=1)
        assert (table["n_observations"] == 60).all()

    def test_pooling_all_steps_uses_more_rows(self, records):
        table = compare_models(records, baseline="random_walk", horizon=2)
        assert (table["n_observations"] == 120).all()

    def test_a_missing_baseline_is_an_error(self, records):
        with pytest.raises(ValueError, match="not present"):
            compare_models(records, baseline="nonexistent")

    def test_missing_columns_are_reported(self):
        with pytest.raises(ValueError, match="missing column"):
            compare_models(pd.DataFrame({"model": ["a"]}), baseline="a")


class TestInferenceNotes:
    def test_the_binomial_replacement_is_documented(self):
        assert "binomial" in INFERENCE_NOTES["directional_uncertainty"]
        assert "Politis-Romano" in INFERENCE_NOTES["directional_uncertainty"]

    def test_nesting_and_hac_are_documented(self):
        assert "Nested" in INFERENCE_NOTES["forecast_comparison"]
        assert "HAC" in INFERENCE_NOTES["forecast_comparison"]

    def test_the_remaining_limitation_is_stated(self):
        assert "structural breaks" in INFERENCE_NOTES["residual_limitations"]

    def test_notes_are_serialisable(self):
        import json

        json.dumps(INFERENCE_NOTES)
