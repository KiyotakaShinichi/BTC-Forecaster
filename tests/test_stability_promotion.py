"""Stability, failure analysis, resource cost, and the promotion policy.

The promotion tests matter most: a policy that cannot reject is not a policy,
so several of these construct a model that clears the headline metric and check
that it is still refused.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.backtesting.engine import run_walk_forward
from btc_forecaster.backtesting.splits import WalkForwardSplitter
from btc_forecaster.evaluation.promotion import (
    Decision,
    PromotionPolicy,
    decide_all,
    evaluate_promotion,
)
from btc_forecaster.evaluation.regimes import label_fold_origins
from btc_forecaster.evaluation.stability import (
    failure_summary,
    largest_failures,
    residual_diagnostics,
    resource_costs,
    stability_by_period,
    stability_profile,
)
from btc_forecaster.models.baselines import HistoricalMeanReturn, RandomWalk, RandomWalkWithDrift
from btc_forecaster.testing import synthetic_market_frame


@pytest.fixture(scope="module")
def backtest():
    frame = synthetic_market_frame(periods=1400, seed=23)
    # 40 origins: enough step-1 residuals for the diagnostics to be computable
    # at all. With 14 folds there are 14, which residual_diagnostics correctly
    # refuses as too few for a Ljung-Box test.
    splitter = WalkForwardSplitter(horizon=10, n_folds=40, min_train_bars=400)
    result = run_walk_forward(
        frame, [RandomWalk(), RandomWalkWithDrift(), HistoricalMeanReturn()], splitter
    )
    return frame, result


class TestStabilityProfile:
    def test_one_row_per_model(self, backtest):
        _, result = backtest
        profile = stability_profile(result.to_frame())
        assert set(profile.index) == {
            "random_walk",
            "random_walk_drift",
            "historical_mean_return",
        }

    def test_it_reports_the_full_distribution_not_just_a_mean(self, backtest):
        _, result = backtest
        profile = stability_profile(result.to_frame())
        for column in ("mean", "median", "std", "best", "worst", "iqr"):
            assert column in profile.columns

    def test_best_and_worst_folds_are_identified(self, backtest):
        _, result = backtest
        table = result.to_frame()
        profile = stability_profile(table)

        for model in profile.index:
            rows = table[table["model"] == model]
            assert profile.loc[model, "best"] == pytest.approx(rows["mae"].min())
            assert profile.loc[model, "worst"] == pytest.approx(rows["mae"].max())
            worst_fold = int(profile.loc[model, "worst_fold"])
            assert rows[rows["fold"] == worst_fold]["mae"].iloc[0] == pytest.approx(
                profile.loc[model, "worst"]
            )

    def test_worst_to_median_flags_a_model_with_a_blow_up(self):
        """A model with the best mean and one disaster is not the better model."""
        table = pd.DataFrame(
            {
                "model": ["steady"] * 5 + ["spiky"] * 5,
                "fold": list(range(5)) * 2,
                "error": [None] * 10,
                "mae": [100, 100, 100, 100, 100] + [10, 10, 10, 10, 400],
            }
        )
        profile = stability_profile(table)

        # spiky wins on the mean (88 vs 100) and is far worse in the tail.
        assert profile.loc["spiky", "mean"] < profile.loc["steady", "mean"]
        assert profile.loc["spiky", "worst_to_median"] == pytest.approx(40.0)
        assert profile.loc["steady", "worst_to_median"] == pytest.approx(1.0)

    def test_coefficient_of_variation_is_comparable_across_scales(self):
        table = pd.DataFrame(
            {
                "model": ["small"] * 4 + ["large"] * 4,
                "fold": list(range(4)) * 2,
                "error": [None] * 8,
                "mae": [10, 12, 8, 10] + [1000, 1200, 800, 1000],
            }
        )
        profile = stability_profile(table)
        assert profile.loc["small", "coefficient_of_variation"] == pytest.approx(
            profile.loc["large", "coefficient_of_variation"]
        )

    def test_failed_folds_are_excluded(self):
        table = pd.DataFrame(
            {
                "model": ["m"] * 3,
                "fold": [0, 1, 2],
                "error": [None, "boom", None],
                "mae": [10.0, np.nan, 12.0],
            }
        )
        assert stability_profile(table).loc["m", "n_folds"] == 2

    def test_missing_columns_are_reported(self):
        with pytest.raises(ValueError, match="missing column"):
            stability_profile(pd.DataFrame({"model": ["a"]}))

    def test_it_is_serialisable(self, backtest):
        _, result = backtest
        json.dumps(stability_profile(result.to_frame()).reset_index().to_dict(orient="records"))


class TestStabilityByPeriod:
    def test_it_groups_by_calendar_year(self, backtest):
        _, result = backtest
        table = stability_by_period(result.to_frame(), freq="YE")
        assert table.index.names == ["model", "period"]
        assert (table["n_folds"] > 0).all()

    def test_it_needs_a_test_start_column(self):
        with pytest.raises(ValueError, match="test_start"):
            stability_by_period(pd.DataFrame({"model": ["a"], "mae": [1.0]}))


class TestFailureAnalysis:
    def test_it_returns_the_largest_errors(self, backtest):
        _, result = backtest
        failures = largest_failures(result.prediction_records(), model="random_walk", top_n=10)
        assert len(failures) == 10
        assert failures["abs_error"].is_monotonic_decreasing

    def test_each_failure_carries_the_context_to_diagnose_it(self, backtest):
        _, result = backtest
        failures = largest_failures(result.prediction_records(), model="random_walk", top_n=5)
        for column in ("origin", "date", "step", "actual_move", "predicted_move", "direction_hit"):
            assert column in failures.columns

    def test_regime_context_is_attached_when_supplied(self, backtest):
        frame, result = backtest
        origins = [fold.origin.last_observed_bar for fold in result.folds]
        regimes = label_fold_origins(frame, origins)

        failures = largest_failures(
            result.prediction_records(), model="random_walk", top_n=5, origin_regimes=regimes
        )
        assert "regime" in failures.columns

    def test_failure_summary_quantifies_error_concentration(self, backtest):
        _, result = backtest
        summary = failure_summary(result.prediction_records(), model="random_walk")

        assert 0.0 <= summary["share_of_total_error"] <= 1.0
        assert summary["tail_mean_error"] > summary["overall_mean_error"]
        assert summary["n_in_tail"] < summary["n_predictions"]

    def test_a_concentrated_error_profile_is_detected(self):
        """If the headline MAE is mostly a few disasters, the number means
        something different from what it appears to."""
        n = 200
        records = pd.DataFrame(
            {
                "model": ["m"] * n,
                "step": [1] * n,
                "origin_close": [100.0] * n,
                "actual": [100.0] * n,
                "predicted": [100.0] * (n - 10) + [1000.0] * 10,
            }
        )
        summary = failure_summary(records, model="m", quantile=0.95)
        assert summary["share_of_total_error"] > 0.9

    def test_missing_columns_are_reported(self):
        with pytest.raises(ValueError, match="missing column"):
            largest_failures(pd.DataFrame({"model": ["a"]}))

    def test_an_unknown_model_gives_an_empty_frame(self, backtest):
        _, result = backtest
        assert largest_failures(result.prediction_records(), model="nope").empty


class TestResidualDiagnostics:
    def test_it_reports_all_four_families(self, backtest):
        _, result = backtest
        report = residual_diagnostics(result.prediction_records(), model="random_walk")
        for key in ("ljung_box", "arch_lm", "jarque_bera", "mean_residual"):
            assert key in report

    def test_it_uses_step_one_residuals_only(self, backtest):
        """Pooling horizons mixes errors whose autocorrelation is mechanical."""
        _, result = backtest
        records = result.prediction_records()
        report = residual_diagnostics(records, model="random_walk")

        step_one = records[(records["model"] == "random_walk") & (records["step"] == 1)]
        assert report["n"] == len(step_one)
        assert report["n"] < (records["model"] == "random_walk").sum()

    def test_too_few_origins_is_refused_rather_than_computed(self):
        """14 folds gives 14 step-1 residuals, which cannot support a
        Ljung-Box test. Saying so beats returning a number."""
        frame = synthetic_market_frame(periods=1400, seed=23)
        sparse = run_walk_forward(
            frame,
            [RandomWalk()],
            WalkForwardSplitter(horizon=15, n_folds=14, min_train_bars=500),
        )
        report = residual_diagnostics(sparse.prediction_records(), model="random_walk")
        assert "too few" in report["note"]
        assert "ljung_box" not in report

    def test_the_interpretation_is_prose_not_a_verdict(self, backtest):
        """A2.18: interpret, do not dispatch on a single rejected test."""
        _, result = backtest
        report = residual_diagnostics(result.prediction_records(), model="random_walk")
        assert isinstance(report["interpretation"], str)
        assert len(report["interpretation"]) > 20

    def test_bias_is_detected_when_present(self):
        n = 300
        records = pd.DataFrame(
            {
                "model": ["m"] * n,
                "step": [1] * n,
                "origin": pd.date_range("2020-01-01", periods=n, freq="D"),
                "origin_close": [100.0] * n,
                "actual": [105.0] * n,
                "predicted": [100.0] * n,
            }
        )
        report = residual_diagnostics(records, model="m")
        assert report["biased"]
        assert "under-predicting" in report["interpretation"]

    def test_too_few_residuals_is_reported_not_computed(self):
        records = pd.DataFrame(
            {
                "model": ["m"] * 5,
                "step": [1] * 5,
                "origin_close": [100.0] * 5,
                "actual": [100.0] * 5,
                "predicted": [100.0] * 5,
            }
        )
        assert "too few" in residual_diagnostics(records, model="m")["note"]

    def test_it_is_serialisable(self, backtest):
        _, result = backtest
        json.dumps(residual_diagnostics(result.prediction_records(), model="random_walk"))


class TestResourceCosts:
    def test_fit_and_inference_are_reported_separately(self, backtest):
        _, result = backtest
        costs = resource_costs(result.to_frame())
        assert "fit_seconds_mean" in costs.columns
        assert "predict_seconds_mean" in costs.columns

    def test_the_cost_multiple_is_relative_to_the_cheapest_model(self, backtest):
        _, result = backtest
        costs = resource_costs(result.to_frame())
        assert costs["cost_multiple_vs_cheapest"].min() == pytest.approx(1.0)
        assert (costs["cost_multiple_vs_cheapest"] >= 1.0).all()

    def test_an_empty_table_gives_an_empty_frame(self):
        assert resource_costs(pd.DataFrame()).empty


class TestPromotionPolicy:
    """A policy that cannot reject is not a policy."""

    def test_the_policy_is_declared_as_pre_registered(self):
        assert "before outer-fold aggregates" in PromotionPolicy().to_dict()["declared"]

    def test_a_model_clearing_everything_is_promoted(self):
        verdict = evaluate_promotion(
            "good",
            mae_skill=0.15,
            n_folds=20,
            skill_ci=(0.05, 0.25),
            worst_to_median=1.4,
            coverage_error=0.03,
            cost_multiple=2.0,
        )
        assert verdict.decision is Decision.PROMOTE
        assert "A2 replaces no production model" in verdict.rationale

    def test_losing_to_the_baseline_is_rejected(self):
        """The legacy hybrid's case, at -1.64."""
        verdict = evaluate_promotion(
            "hybrid", mae_skill=-1.64, n_folds=20, skill_ci=(-2.0, -1.2),
            worst_to_median=1.5, coverage_error=0.05,
        )
        assert verdict.decision is Decision.REJECT
        assert "mae_skill" in verdict.failed

    def test_a_trivial_improvement_is_rejected(self):
        """A2.21: not promoted merely because MAE is 0.001 smaller."""
        verdict = evaluate_promotion(
            "marginal", mae_skill=0.001, n_folds=20, skill_ci=(0.0005, 0.0015),
            worst_to_median=1.1, coverage_error=0.01,
        )
        assert verdict.decision is Decision.REJECT

    def test_skill_whose_interval_includes_zero_is_inconclusive(self):
        verdict = evaluate_promotion(
            "noisy", mae_skill=0.06, n_folds=20, skill_ci=(-0.15, 0.19),
            worst_to_median=1.5, coverage_error=0.04,
        )
        assert verdict.decision is Decision.INCONCLUSIVE
        assert "skill_interval_includes_zero" in verdict.failed

    def test_an_unstable_model_is_not_promoted(self):
        verdict = evaluate_promotion(
            "spiky", mae_skill=0.20, n_folds=20, skill_ci=(0.05, 0.35),
            worst_to_median=9.0, coverage_error=0.02,
        )
        assert verdict.decision is Decision.INCONCLUSIVE
        assert "stability" in verdict.failed

    def test_a_miscalibrated_model_is_not_promoted(self):
        """A 95% band covering 23% is misleading, not merely imprecise."""
        verdict = evaluate_promotion(
            "miscalibrated", mae_skill=0.20, n_folds=20, skill_ci=(0.05, 0.35),
            worst_to_median=1.2, coverage_error=0.72,
        )
        assert verdict.decision is Decision.INCONCLUSIVE
        assert "interval_calibration" in verdict.failed

    def test_an_expensive_model_must_clear_a_higher_bar(self):
        modest = {
            "n_folds": 20,
            "skill_ci": (0.02, 0.08),
            "worst_to_median": 1.2,
            "coverage_error": 0.02,
        }
        cheap = evaluate_promotion("cheap", mae_skill=0.05, cost_multiple=1.0, **modest)
        expensive = evaluate_promotion("expensive", mae_skill=0.05, cost_multiple=50.0, **modest)

        assert cheap.decision is Decision.PROMOTE
        assert expensive.decision is Decision.REJECT
        assert "compute multiple" in expensive.rationale

    def test_too_few_folds_is_inconclusive_not_a_verdict_on_the_model(self):
        verdict = evaluate_promotion(
            "untested", mae_skill=0.5, n_folds=3, skill_ci=(0.3, 0.7),
            worst_to_median=1.1, coverage_error=0.01,
        )
        assert verdict.decision is Decision.INCONCLUSIVE
        assert "Not a statement about the model" in verdict.rationale

    def test_a_missing_interval_is_inconclusive_not_a_pass(self):
        verdict = evaluate_promotion(
            "unmeasured", mae_skill=0.20, n_folds=20, skill_ci=None,
            worst_to_median=1.2, coverage_error=0.02,
        )
        assert verdict.decision is Decision.INCONCLUSIVE
        assert "skill_interval_missing" in verdict.failed

    def test_reject_and_inconclusive_are_distinguished(self):
        """'We showed it does not work' is a different result from 'we could
        not show it works'."""
        rejected = evaluate_promotion("a", mae_skill=-0.5, n_folds=20)
        inconclusive = evaluate_promotion(
            "b", mae_skill=0.3, n_folds=20, skill_ci=(-0.1, 0.7),
            worst_to_median=1.1, coverage_error=0.01,
        )
        assert rejected.decision is Decision.REJECT
        assert inconclusive.decision is Decision.INCONCLUSIVE

    def test_verdicts_are_serialisable(self):
        json.dumps(evaluate_promotion("m", mae_skill=0.1, n_folds=20).to_dict())

    def test_thresholds_are_configurable_and_recorded(self):
        strict = PromotionPolicy(min_mae_skill=0.5)
        verdict = evaluate_promotion(
            "m", mae_skill=0.2, n_folds=20, skill_ci=(0.1, 0.3),
            worst_to_median=1.1, coverage_error=0.01, policy=strict,
        )
        assert verdict.decision is Decision.REJECT
        assert strict.to_dict()["min_mae_skill"] == 0.5


class TestDecideAll:
    def test_it_covers_every_model_except_the_baseline(self, backtest):
        _, result = backtest
        summary = result.summary()
        verdicts = decide_all(
            summary,
            result.skill_table("random_walk"),
            stability_profile(result.to_frame()),
            resource_costs(result.to_frame()),
            baseline="random_walk",
            n_folds=len(result.folds),
        )
        assert "random_walk" not in verdicts.index
        assert set(verdicts.index) == {"random_walk_drift", "historical_mean_return"}

    def test_no_model_is_promoted_on_a_random_walk(self, backtest):
        """The honest outcome on data with no signal."""
        _, result = backtest
        verdicts = decide_all(
            result.summary(),
            result.skill_table("random_walk"),
            stability_profile(result.to_frame()),
            resource_costs(result.to_frame()),
            baseline="random_walk",
            n_folds=len(result.folds),
        )
        assert (verdicts["decision"] != Decision.PROMOTE.value).all()

    def test_it_is_serialisable(self, backtest):
        _, result = backtest
        verdicts = decide_all(
            result.summary(),
            result.skill_table("random_walk"),
            stability_profile(result.to_frame()),
            resource_costs(result.to_frame()),
            baseline="random_walk",
            n_folds=len(result.folds),
        )
        json.dumps(verdicts.reset_index().to_dict(orient="records"))
