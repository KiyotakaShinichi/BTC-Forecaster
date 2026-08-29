"""The composed A2 benchmark: table, manifest integrity, immutability."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from btc_forecaster.backtesting.splits import WalkForwardSplitter
from btc_forecaster.benchmark import (
    BENCHMARK_MANIFEST,
    BENCHMARK_TABLE_CSV,
    benchmark_manifest,
    format_benchmark_report,
    run_benchmark,
    write_benchmark,
)
from btc_forecaster.data.snapshot import MarketSnapshot
from btc_forecaster.evaluation.promotion import Decision, PromotionPolicy
from btc_forecaster.models.baselines import (
    HistoricalMeanReturn,
    RandomWalk,
    RandomWalkWithDrift,
)
from btc_forecaster.testing import synthetic_market_frame


@pytest.fixture(scope="module")
def benchmark():
    frame = synthetic_market_frame(periods=1500, seed=31)
    snapshot = MarketSnapshot.build(
        frame, ticker="TEST-USD", provider="synthetic", normalise=False
    )
    result = run_benchmark(
        frame,
        [RandomWalk(), RandomWalkWithDrift(), HistoricalMeanReturn()],
        snapshot=snapshot,
        splitter=WalkForwardSplitter(horizon=10, n_folds=40, min_train_bars=400),
        run_id="test-run",
    )
    return frame, result


class TestBenchmarkTable:
    def test_one_row_per_model(self, benchmark):
        _, result = benchmark
        assert set(result.table.index) == {
            "random_walk",
            "random_walk_drift",
            "historical_mean_return",
        }

    def test_it_reports_every_required_dimension(self, benchmark):
        """A2.20: not rankable on one metric."""
        _, result = benchmark
        for column in (
            "mae",
            "mae_skill_vs_random_walk",
            "rmse",
            "dir_acc_step1",
            "dir_ci_lower",
            "dir_ci_upper",
            "interval_coverage",
            "mae_worst_to_median",
            "cost_multiple_vs_cheapest",
        ):
            assert column in result.table.columns, column

    def test_direction_carries_an_interval_not_just_a_point(self, benchmark):
        _, result = benchmark
        assert (result.table["dir_ci_lower"] < result.table["dir_acc_step1"]).all()
        assert (result.table["dir_acc_step1"] < result.table["dir_ci_upper"]).all()

    def test_no_model_beats_a_coin_on_a_random_walk(self, benchmark):
        """The honest outcome, expressed through the interval rather than a
        p-value: every directional CI contains 0.5."""
        _, result = benchmark
        assert not result.table["dir_beats_coin"].any()

    def test_cost_multiple_is_not_distorted_by_import_warm_up(self, benchmark):
        """RandomWalk is the cheapest model there is. Using the mean fold would
        charge it the one-time scipy import and rank it ~18x the drift model."""
        _, result = benchmark
        assert result.table.loc["random_walk", "cost_multiple_vs_cheapest"] < 3.0

    def test_per_step_metrics_cover_every_horizon(self, benchmark):
        _, result = benchmark
        assert sorted(result.per_step["step"].unique()) == list(range(1, 11))
        assert set(result.per_step["model"]) == set(result.table.index)


class TestPromotionInTheRun:
    def test_the_policy_is_recorded_not_chosen_after_the_fact(self, benchmark):
        _, result = benchmark
        assert "before outer-fold aggregates" in result.policy.to_dict()["declared"]

    def test_nothing_is_promoted_on_a_random_walk(self, benchmark):
        _, result = benchmark
        assert (result.promotion["decision"] != Decision.PROMOTE.value).all()

    def test_the_baseline_is_not_judged_against_itself(self, benchmark):
        _, result = benchmark
        assert "random_walk" not in result.promotion.index

    def test_every_verdict_carries_a_rationale(self, benchmark):
        _, result = benchmark
        assert result.promotion["rationale"].str.len().gt(30).all()


class TestPairwiseComparisons:
    def test_nested_pairs_are_flagged_rather_than_reported(self, benchmark):
        """Every baseline nests the random walk, so DM is invalid for all of
        them here. Saying so is the correct output."""
        _, result = benchmark
        assert not result.comparisons["usable"].any()
        assert result.comparisons["nested"].all()

    def test_the_correction_family_is_recorded(self, benchmark):
        _, result = benchmark
        assert (result.comparisons["correction"] == "benjamini-hochberg").all()


class TestRegimeStratification:
    def test_results_are_split_by_regime(self, benchmark):
        _, result = benchmark
        assert not result.by_regime.empty
        assert result.by_regime.index.names == ["model", "regime"]


class TestManifest:
    def test_it_records_everything_needed_to_reproduce_the_run(self, benchmark):
        _, result = benchmark
        manifest = benchmark_manifest(result)

        assert manifest["run_id"] == "test-run"
        assert len(manifest["dataset"]["sha256"]) == 64
        assert manifest["outer_folds"]["n_folds"] == 40
        assert len(manifest["outer_folds"]["folds"]) == 40
        assert manifest["baseline"] == "random_walk"
        assert manifest["environment"]["python"]

    def test_it_carries_the_target_and_inference_audits(self, benchmark):
        _, result = benchmark
        manifest = benchmark_manifest(result)
        assert manifest["target_audit"]["primary_task"] == "price_level"
        assert "binomial" in manifest["inference_notes"]["directional_uncertainty"]

    def test_it_re_verifies_the_preserved_evidence(self, benchmark):
        """A benchmark run confirms the negative results are still intact."""
        _, result = benchmark
        manifest = benchmark_manifest(result)
        verified = manifest["preserved_evidence"]["verified"]
        assert "LEGACY_INVALID_RESULT" in verified
        assert "LEAKAGE_CORRECTED_REFERENCE" in verified

    def test_it_records_the_promotion_policy_and_decisions(self, benchmark):
        _, result = benchmark
        manifest = benchmark_manifest(result)
        assert manifest["promotion_policy"]["min_mae_skill"] > 0
        assert len(manifest["promotion_decisions"]) == 2

    def test_it_is_serialisable(self, benchmark):
        _, result = benchmark
        json.dumps(benchmark_manifest(result), default=str)


class TestWriting:
    def test_a_run_writes_its_artifacts_and_the_manifest(self, benchmark, tmp_path):
        _, result = benchmark
        written = write_benchmark(result, tmp_path / "run")

        assert BENCHMARK_TABLE_CSV in written
        assert BENCHMARK_MANIFEST in written
        assert "per_fold.csv" in written
        assert "predictions.csv" in written
        assert "promotion.csv" in written

    def test_a_completed_run_is_never_overwritten(self, benchmark, tmp_path):
        """A2.22: research runs are immutable."""
        _, result = benchmark
        target = tmp_path / "run"
        write_benchmark(result, target)

        with pytest.raises(FileExistsError, match="immutable"):
            write_benchmark(result, target)

    def test_the_manifest_is_written_last(self, benchmark, tmp_path):
        """Its presence must mean the run completed, not that it started."""
        _, result = benchmark
        target = tmp_path / "run"
        write_benchmark(result, target)

        manifest_mtime = (target / BENCHMARK_MANIFEST).stat().st_mtime_ns
        others = [p for p in target.glob("*") if p.name != BENCHMARK_MANIFEST]
        assert others
        assert manifest_mtime >= max(p.stat().st_mtime_ns for p in others)

    def test_the_written_table_round_trips(self, benchmark, tmp_path):
        _, result = benchmark
        target = tmp_path / "run"
        write_benchmark(result, target)

        table = pd.read_csv(target / BENCHMARK_TABLE_CSV, index_col=0)
        assert set(table.index) == set(result.table.index)


class TestReport:
    def test_the_report_states_the_data_and_the_folds(self, benchmark):
        _, result = benchmark
        report = format_benchmark_report(result)
        assert "A2 BENCHMARK" in report
        assert "sha256=" in report
        assert "40 expanding" in report

    def test_it_states_the_promotion_decisions(self, benchmark):
        _, result = benchmark
        assert "PROMOTION DECISIONS" in format_benchmark_report(result)

    def test_it_says_when_every_comparison_is_nested(self, benchmark):
        _, result = benchmark
        assert "nested" in format_benchmark_report(result)


class TestPolicyIsConfigurable:
    def test_a_stricter_policy_changes_the_verdicts(self, benchmark):
        frame, _ = benchmark
        snapshot = MarketSnapshot.build(
            frame, ticker="TEST-USD", provider="synthetic", normalise=False
        )
        strict = run_benchmark(
            frame,
            [RandomWalk(), RandomWalkWithDrift()],
            snapshot=snapshot,
            splitter=WalkForwardSplitter(horizon=10, n_folds=40, min_train_bars=400),
            policy=PromotionPolicy(min_mae_skill=0.90),
            run_id="strict",
        )
        assert (strict.promotion["decision"] == Decision.REJECT.value).all()
