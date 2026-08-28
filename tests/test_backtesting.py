"""Walk-forward fold construction and the evaluation engine."""

from __future__ import annotations

import pandas as pd
import pytest

from btc_forecaster.backtesting.engine import (
    has_useful_skill,
    rank_models,
    run_walk_forward,
)
from btc_forecaster.backtesting.splits import (
    WalkForwardSplitter,
    assert_folds_are_disjoint,
)
from btc_forecaster.models.base import TrainingWindow
from btc_forecaster.models.baselines import (
    HistoricalMeanReturn,
    RandomWalk,
    RandomWalkWithDrift,
)
from btc_forecaster.testing import constant_growth_frame, synthetic_market_frame
from btc_forecaster.timebase import BAR_DURATION


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_market_frame(periods=1200, seed=13)


@pytest.fixture
def splitter() -> WalkForwardSplitter:
    return WalkForwardSplitter(horizon=30, n_folds=4, min_train_bars=400)


class TestFoldConstruction:
    def test_it_builds_the_requested_number_of_folds(self, frame, splitter):
        assert len(splitter.split(frame.index)) == 4

    def test_training_never_overlaps_its_own_test_window(self, frame, splitter):
        folds = splitter.split(frame.index)
        assert_folds_are_disjoint(folds)
        for fold in folds:
            assert fold.train_end < fold.test_start

    def test_test_windows_are_exactly_the_horizon(self, frame, splitter):
        for fold in splitter.split(frame.index):
            assert fold.horizon == 30

    def test_origins_advance_monotonically(self, frame, splitter):
        folds = splitter.split(frame.index)
        origins = [fold.origin.last_observed_bar for fold in folds]
        assert origins == sorted(origins)
        assert len(set(origins)) == len(origins)

    def test_expanding_mode_grows_the_training_set(self, frame, splitter):
        folds = splitter.split(frame.index)
        sizes = [len(fold.train_slice(frame)) for fold in folds]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]
        assert all(fold.train_start == frame.index[0] for fold in folds)

    def test_rolling_mode_keeps_the_training_set_fixed(self, frame):
        splitter = WalkForwardSplitter(
            horizon=30, n_folds=4, min_train_bars=400, mode="rolling", window_bars=300
        )
        folds = splitter.split(frame.index)
        assert {len(fold.train_slice(frame)) for fold in folds} == {300}
        assert folds[-1].train_start > folds[0].train_start

    def test_rolling_mode_requires_a_window(self):
        with pytest.raises(ValueError, match="window_bars"):
            WalkForwardSplitter(mode="rolling")

    def test_minimum_training_size_is_respected(self, frame):
        splitter = WalkForwardSplitter(horizon=30, n_folds=5, min_train_bars=800)
        for fold in splitter.split(frame.index):
            assert len(fold.train_slice(frame)) >= 800

    def test_insufficient_data_is_a_clear_error(self, frame):
        splitter = WalkForwardSplitter(horizon=30, n_folds=3, min_train_bars=5000)
        with pytest.raises(ValueError, match="need at least"):
            splitter.split(frame.index)

    def test_a_single_fold_lands_at_the_end_of_the_data(self, frame):
        folds = WalkForwardSplitter(horizon=30, n_folds=1, min_train_bars=400).split(frame.index)
        assert len(folds) == 1
        assert folds[0].test_end == frame.index[-1]

    def test_step_bars_controls_origin_spacing(self, frame):
        splitter = WalkForwardSplitter(
            horizon=30, n_folds=4, min_train_bars=400, step_bars=100
        )
        folds = splitter.split(frame.index)
        gaps = {
            int((b.train_end - a.train_end) / BAR_DURATION)
            for a, b in zip(folds, folds[1:], strict=False)
        }
        assert gaps == {100}

    def test_splitter_configuration_is_serialisable(self, splitter):
        import json

        json.dumps(splitter.describe())

    @pytest.mark.parametrize("bad", [{"horizon": 0}, {"n_folds": 0}, {"embargo_bars": -1}])
    def test_invalid_configuration_is_rejected(self, bad):
        with pytest.raises(ValueError):
            WalkForwardSplitter(**bad)


class TestEmbargo:
    def test_embargo_inserts_a_gap_between_train_and_test(self, frame):
        splitter = WalkForwardSplitter(
            horizon=30, n_folds=3, min_train_bars=400, embargo_bars=10
        )
        for fold in splitter.split(frame.index):
            gap = int((fold.test_start - fold.train_end) / BAR_DURATION)
            assert gap == 11, "embargo of 10 leaves 10 skipped bars plus the step to the next bar"

    def test_zero_embargo_puts_test_immediately_after_train(self, frame, splitter):
        for fold in splitter.split(frame.index):
            assert fold.test_start == fold.train_end + BAR_DURATION

    def test_forecast_steps_cover_the_embargo_gap(self, frame):
        splitter = WalkForwardSplitter(
            horizon=30, n_folds=2, min_train_bars=400, embargo_bars=10
        )
        for fold in splitter.split(frame.index):
            assert fold.steps_to_test_end == 40, "10 embargoed + 30 scored"
            assert fold.horizon == 30

    def test_embargoed_folds_still_pass_the_disjointness_check(self, frame):
        splitter = WalkForwardSplitter(
            horizon=30, n_folds=3, min_train_bars=400, embargo_bars=20
        )
        assert_folds_are_disjoint(splitter.split(frame.index))


class TestEngineFairness:
    """The property that makes a model comparison mean anything."""

    def test_every_model_is_scored_on_identical_folds(self, frame, splitter):
        models = [RandomWalk(), RandomWalkWithDrift(), HistoricalMeanReturn()]
        result = run_walk_forward(frame, models, splitter)
        table = result.to_frame()

        per_model = {
            name: group[["fold", "train_start", "train_end", "test_start", "test_end"]]
            .reset_index(drop=True)
            for name, group in table.groupby("model")
        }
        reference = per_model["random_walk"]
        for name, folds in per_model.items():
            pd.testing.assert_frame_equal(folds, reference, obj=name)

    def test_models_never_see_their_own_test_window(self, frame, splitter):
        """Asserted by recording what each fit actually received."""
        seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        class Spy(RandomWalk):
            def _fit(self, window: TrainingWindow) -> None:
                seen.append((window.index.min(), window.index.max()))
                super()._fit(window)

        folds = splitter.split(frame.index)
        run_walk_forward(frame, [Spy(name="spy")], splitter)

        assert len(seen) == len(folds)
        for (start, end), fold in zip(seen, folds, strict=False):
            assert start == fold.train_start
            assert end == fold.train_end
            assert end < fold.test_start

    def test_each_fold_refits_from_scratch(self, frame, splitter):
        fits: list[int] = []

        class Counting(RandomWalk):
            def _fit(self, window: TrainingWindow) -> None:
                fits.append(len(window))
                super()._fit(window)

        run_walk_forward(frame, [Counting(name="counting")], splitter)
        assert len(fits) == 4
        assert fits == sorted(fits), "expanding window should grow on each refit"

    def test_results_are_reproducible(self, frame, splitter):
        # fit_seconds is a wall-clock measurement and is excluded by design;
        # every scored quantity must be bit-identical between runs.
        a = run_walk_forward(frame, [RandomWalk()], splitter).to_frame().drop(columns=["fit_seconds"])
        b = run_walk_forward(frame, [RandomWalk()], splitter).to_frame().drop(columns=["fit_seconds"])
        pd.testing.assert_frame_equal(a, b)


class TestEngineResults:
    def test_summary_has_one_row_per_model(self, frame, splitter):
        models = [RandomWalk(), RandomWalkWithDrift()]
        summary = run_walk_forward(frame, models, splitter).summary()
        assert set(summary.index) == {"random_walk", "random_walk_drift"}
        assert (summary["n_folds_ok"] == 4).all()

    def test_summary_reports_dispersion_across_folds(self, frame, splitter):
        summary = run_walk_forward(frame, [RandomWalk()], splitter).summary()
        assert "mae_std" in summary.columns
        assert summary.loc["random_walk", "mae_std"] > 0

    def test_a_failing_model_is_recorded_not_swallowed(self, frame, splitter):
        class Broken(RandomWalk):
            def _fit(self, window):
                raise RuntimeError("deliberate failure")

        result = run_walk_forward(frame, [RandomWalk(), Broken(name="broken")], splitter)

        failures = result.failures()
        assert len(failures) == 4
        assert all("deliberate failure" in f.error for f in failures)

        summary = result.summary()
        assert summary.loc["broken", "n_folds_failed"] == 4
        assert summary.loc["random_walk", "n_folds_ok"] == 4, "one failure must not abort the run"

    def test_skill_table_measures_against_the_baseline_on_the_same_folds(self, frame, splitter):
        models = [RandomWalk(), RandomWalkWithDrift()]
        result = run_walk_forward(frame, models, splitter, baseline="random_walk")
        table = result.skill_table()

        assert table.loc["random_walk", "mae_skill_vs_random_walk"] == pytest.approx(0.0)
        assert "mae_skill_vs_random_walk" in table.columns

    def test_drift_shows_positive_skill_on_a_genuinely_trending_series(self):
        """The engine must be able to detect skill when skill exists."""
        frame = constant_growth_frame(periods=900, daily_growth=0.002)
        splitter = WalkForwardSplitter(horizon=30, n_folds=4, min_train_bars=400)
        result = run_walk_forward(
            frame, [RandomWalk(), RandomWalkWithDrift()], splitter, baseline="random_walk"
        )
        assert has_useful_skill(result, "random_walk_drift")

    def test_no_model_shows_skill_on_a_pure_random_walk(self, frame, splitter):
        """The negative result the platform exists to be able to report.

        On data with no predictable structure, nothing should beat the naive
        forecast. A framework that finds skill here is measuring its own bugs.
        """
        result = run_walk_forward(
            frame,
            [RandomWalk(), RandomWalkWithDrift(), HistoricalMeanReturn()],
            splitter,
            baseline="random_walk",
        )
        for model in ("random_walk_drift", "historical_mean_return"):
            assert not has_useful_skill(result, model), f"{model} found skill in noise"

    def test_directional_accuracy_on_noise_is_near_a_coin_flip(self, frame):
        splitter = WalkForwardSplitter(horizon=60, n_folds=6, min_train_bars=400)
        result = run_walk_forward(frame, [RandomWalkWithDrift()], splitter)
        accuracy = result.summary().loc["random_walk_drift", "directional_accuracy"]
        assert 0.3 < accuracy < 0.7

    def test_ranking_orders_by_the_chosen_metric(self, frame, splitter):
        models = [RandomWalk(), RandomWalkWithDrift(), HistoricalMeanReturn()]
        ranked = rank_models(run_walk_forward(frame, models, splitter), metric="mae")
        assert list(ranked["mae"]) == sorted(ranked["mae"])

    def test_pooled_direction_test_carries_its_caveat(self, frame, splitter):
        result = run_walk_forward(frame, [RandomWalk()], splitter)
        payload = result.direction_test("random_walk")
        assert payload["n_total"] == 4 * 30
        assert "independent Bernoulli trials" in payload["caveat"]

    def test_manifest_records_the_folds_and_configuration(self, frame, splitter):
        import json

        result = run_walk_forward(frame, [RandomWalk()], splitter)
        manifest = result.to_manifest()

        assert manifest["n_folds"] == 4
        assert manifest["splitter"]["mode"] == "expanding"
        assert len(manifest["folds"]) == 4
        assert manifest["baseline"] == "random_walk"
        json.dumps(manifest)

    def test_interval_coverage_is_measured_per_fold(self, frame, splitter):
        result = run_walk_forward(frame, [RandomWalk()], splitter)
        table = result.to_frame()
        assert "interval_coverage" in table.columns
        assert table["interval_coverage"].between(0.0, 1.0).all()

    def test_coverage_is_summarised_even_though_it_is_not_rankable(self, frame, splitter):
        """Coverage is a calibration metric: 0.60 and 1.00 are both wrong.

        It must still appear in the summary, so it is carried by SUMMARY_METRICS
        rather than by the higher/lower-is-better sets used for ranking.
        """
        summary = run_walk_forward(frame, [RandomWalk()], splitter).summary()
        assert "interval_coverage" in summary.columns
        assert "coverage_error" in summary.columns

    def test_coverage_error_measures_distance_from_the_nominal_level(self, frame, splitter):
        table = run_walk_forward(frame, [RandomWalk()], splitter).to_frame()
        expected = (table["interval_coverage"] - table["interval_level"]).abs()
        pd.testing.assert_series_equal(
            table["coverage_error"], expected, check_names=False
        )

    def test_random_walk_bands_are_roughly_calibrated_on_a_random_walk(self, frame):
        """A 95% band on data that really is a random walk should cover ~95%."""
        splitter = WalkForwardSplitter(horizon=30, n_folds=10, min_train_bars=400)
        result = run_walk_forward(frame, [RandomWalk()], splitter)
        coverage = result.summary().loc["random_walk", "interval_coverage"]
        assert 0.80 < coverage < 1.0

    def test_exog_must_align_with_the_market_frame(self, frame, splitter):
        exog = pd.DataFrame({"x": [0.0]}, index=frame.index[:1])
        with pytest.raises(ValueError, match="indexed identically"):
            run_walk_forward(frame, [RandomWalk()], splitter, exog=exog)

    def test_on_fold_callback_fires_once_per_model_fold(self, frame, splitter):
        seen = []
        run_walk_forward(
            frame, [RandomWalk(), RandomWalkWithDrift()], splitter, on_fold=seen.append
        )
        assert len(seen) == 8
