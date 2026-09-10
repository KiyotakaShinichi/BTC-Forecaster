"""The fold evaluator and the walk-forward scorer.

What a job must do, pinned: score every origin exactly once against the return
that actually followed, reach each horizon by the strategy the model declares,
record every failure instead of dropping it, and be re-runnable to the same
numbers. And what the scorer must do: compare a model to the naive forecast only
where both were scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.contracts import (
    RESOURCE_BUDGET_SECONDS,
    EvaluationContext,
    Family,
    ModelStatus,
    ResourceClass,
    TrainingSet,
    ZooModel,
)
from btc_forecaster.research.registry import ZooRegistration
from btc_forecaster.research.walk_forward.config import BASELINE, WalkForwardConfig, WindowSpec
from btc_forecaster.research.walk_forward.evaluator import (
    DIRECT,
    ITERATED,
    RECORD_COLUMNS,
    InformationCache,
    JobSpec,
    run_job,
    schedule_for,
    strategy_for,
)
from btc_forecaster.research.walk_forward.scoring import NOT_DECLARED, score_configs
from btc_forecaster.research.walk_forward.targets import cumulative_log_return
from btc_forecaster.testing import synthetic_market_frame


def available(*model_ids: str) -> list[str]:
    return [m for m in model_ids if registry.get(m).is_available()]


FRAME = synthetic_market_frame(periods=700, kind="ar1", seed=21)
ROLL, EXP = WindowSpec("rolling", 150), WindowSpec("expanding")
HORIZONS = (1, 5)
MODELS = available("naive_last_value", "random_walk_drift", "ar_p", "arima", "theta", "ridge", "mlp")
CONFIG = WalkForwardConfig(
    models=tuple(MODELS),
    horizons=HORIZONS,
    windows=(ROLL, EXP),
    n_refits=3,
    bootstrap_resamples=100,
    minimum_paired_origins=30,
)
SCHEDULE = schedule_for(FRAME, CONFIG)


@pytest.fixture(scope="module")
def cache():
    return InformationCache(FRAME, CONFIG)


@pytest.fixture(scope="module")
def jobs(cache):
    return {
        (m, h, w.label): run_job(JobSpec(m, h, w), SCHEDULE, CONFIG, cache)
        for m in MODELS
        for h in HORIZONS
        for w in (ROLL, EXP)
    }


@pytest.fixture(scope="module")
def records(jobs):
    return pd.concat([j.records for j in jobs.values()], ignore_index=True)


def register_probe(monkeypatch, model_id, factory, *, resource=ResourceClass.TRIVIAL, requires=()):
    registry.get("naive_last_value")  # load the adapters before patching the registry
    monkeypatch.setitem(
        registry._REGISTRY,
        model_id,
        ZooRegistration(
            model_id=model_id, factory=factory, family=Family.BASELINE,
            resource_class=resource, description="test probe", requires=requires,
        ),
    )


class TestTheSchedule:
    def test_the_first_origin_leaves_room_for_the_largest_window(self) -> None:
        assert FRAME.index.get_loc(SCHEDULE.origins[0]) == 60 + 150 + 5 - 1
        assert FRAME.index.get_loc(SCHEDULE.origins[-1]) == len(FRAME) - 1 - 5
        assert len(SCHEDULE.folds) == 3


class TestStrategy:
    def test_series_models_iterate(self) -> None:
        assert strategy_for(registry.build("ar_p")) == ITERATED
        assert strategy_for(registry.build("naive_last_value")) == ITERATED

    def test_tabular_and_sequence_models_train_directly(self) -> None:
        for model_id in available("ridge", "mlp"):
            assert strategy_for(registry.build(model_id)) == DIRECT


class TestEveryOriginIsScoredOnce:
    def test_every_job_ran_every_fold_cleanly(self, jobs) -> None:
        for key, job in jobs.items():
            assert job.clean, (key, job.statuses)

    def test_one_row_per_origin_in_canonical_columns(self, jobs) -> None:
        for job in jobs.values():
            assert tuple(job.records.columns) == RECORD_COLUMNS
            assert pd.DatetimeIndex(job.records["origin"]).equals(SCHEDULE.origins)

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_the_actual_is_the_return_h_bars_later(self, jobs, horizon) -> None:
        rows = jobs[("ar_p", horizon, ROLL.label)].records
        origin = FRAME.index.get_indexer(pd.DatetimeIndex(rows["origin"]))
        target = FRAME.index.get_indexer(pd.DatetimeIndex(rows["target_bar"]))
        assert (target - origin == horizon).all()
        expected = cumulative_log_return(FRAME["close"], horizon).reindex(pd.DatetimeIndex(rows["target_bar"]))
        assert np.allclose(rows["actual"].to_numpy(), expected.to_numpy())


class TestTheBaselinesAreWhatTheyClaim:
    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_naive_forecasts_no_change(self, jobs, horizon) -> None:
        assert (jobs[(BASELINE, horizon, ROLL.label)].records["predicted"] == 0.0).all()

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_drift_is_h_times_the_folds_own_mean(self, jobs, cache, horizon) -> None:
        rows = jobs[("random_walk_drift", horizon, ROLL.label)].records
        for fold in SCHEDULE.folds:
            mean = float(cache.get(fold, ROLL).training_set(1).y.mean())
            fold_rows = rows[rows["fold"] == fold.index]
            assert np.allclose(fold_rows["predicted"].to_numpy(), horizon * mean)

    def test_the_constant_direction_is_chosen_on_training_targets(self, jobs, cache) -> None:
        rows = jobs[("ar_p", 5, ROLL.label)].records
        for fold in SCHEDULE.folds:
            majority_up = bool((cache.get(fold, ROLL).training_set(5).y > 0).mean() >= 0.5)
            assert set(rows.loc[rows["fold"] == fold.index, "train_direction_up"]) == {majority_up}


class TestWindowsAndEarlyStopping:
    def test_a_rolling_job_trains_on_exactly_its_rows(self, jobs) -> None:
        for model_id in available("ridge", "mlp", "ar_p"):
            assert {f.train_rows for f in jobs[(model_id, 5, ROLL.label)].folds} == {150}

    def test_an_expanding_job_grows(self, jobs) -> None:
        sizes = [f.train_rows for f in jobs[("ar_p", 1, EXP.label)].folds]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


class TestDeterminism:
    def test_a_rerun_gives_identical_records(self, jobs) -> None:
        for model_id in available("arima", "ridge", "mlp"):
            again = run_job(JobSpec(model_id, 5, ROLL), SCHEDULE, CONFIG, InformationCache(FRAME, CONFIG))
            pd.testing.assert_frame_equal(again.records, jobs[(model_id, 5, ROLL.label)].records)


class TestFailuresAreRecordedNotDropped:
    def test_a_raising_model_fails_every_fold_and_says_why(self, monkeypatch, cache) -> None:
        class Boom(ZooModel):
            model_id = "boom_walk_forward_probe"
            family = Family.BASELINE

            def _fit(self, train: TrainingSet) -> None:
                raise ValueError("deliberate failure")

            def _predict_point(self, context: EvaluationContext) -> np.ndarray:
                return np.zeros(len(context))

        register_probe(monkeypatch, "boom_walk_forward_probe", Boom)
        job = run_job(JobSpec("boom_walk_forward_probe", 1, ROLL), SCHEDULE, CONFIG, cache)
        assert job.statuses == [ModelStatus.FAILED.value] * 3
        assert job.folds[0].failure["exception"] == "ValueError"
        assert job.records.empty and not job.clean

    def test_a_fold_over_budget_is_reported_outside_the_result(self, monkeypatch, cache) -> None:
        """Whether a fold overran its budget depends on machine load. The first
        canonical BTC run showed it: fifteen LSTM folds over budget under four
        workers that take a third of the time alone. So an overrun is reported in
        the timings and never changes the canonical outcome -- the same job with
        a starved budget must produce the same records and fold records."""
        normal = run_job(JobSpec("ar_p", 1, ROLL), SCHEDULE, CONFIG, cache)
        monkeypatch.setitem(RESOURCE_BUDGET_SECONDS, ResourceClass.TRIVIAL, 0.0)
        starved = run_job(JobSpec("ar_p", 1, ROLL), SCHEDULE, CONFIG, cache)
        assert all(f.over_budget for f in starved.folds)
        assert not any(f.over_budget for f in normal.folds)
        assert [f.canonical() for f in starved.folds] == [f.canonical() for f in normal.folds]
        pd.testing.assert_frame_equal(starved.records, normal.records)
        assert starved.clean and normal.clean
        assert starved.folds[0].timing()["over_budget"] is True
        assert "over_budget" not in starved.folds[0].canonical()

    def test_a_missing_dependency_is_skipped_for_every_fold(self, monkeypatch, cache) -> None:
        register_probe(
            monkeypatch, "absent_walk_forward_probe", lambda: None, requires=("a7_absent_package",)
        )
        job = run_job(JobSpec("absent_walk_forward_probe", 1, ROLL), SCHEDULE, CONFIG, cache)
        assert job.statuses == [ModelStatus.SKIPPED_DEPENDENCY.value] * 3
        assert job.strategy is None and job.records.empty


class TestScoring:
    def test_the_naive_forecast_scores_itself_as_no_skill(self, records) -> None:
        naive = [s for s in score_configs(records, SCHEDULE, baseline=BASELINE) if s.model_id == BASELINE]
        for score in naive:
            assert score.skill_vs_naive == 0.0 and score.mase == 1.0
            # A forecast of exactly zero is a wrong call, as in A6.
            assert score.directional_accuracy == 0.0

    def test_drift_skill_matches_a_hand_computation(self, records) -> None:
        score = next(
            s for s in score_configs(records, SCHEDULE, baseline=BASELINE)
            if (s.model_id, s.horizon, s.window) == ("random_walk_drift", 5, ROLL.label)
        )
        rows = records[(records["model_id"] == "random_walk_drift") & (records["horizon"] == 5) & (records["window"] == ROLL.label)]
        expected = 1.0 - np.abs(rows["actual"] - rows["predicted"]).sum() / np.abs(rows["actual"]).sum()
        assert score.skill_vs_naive == pytest.approx(expected)

    def test_skill_is_reported_per_fold_and_per_block(self, records) -> None:
        for score in score_configs(records, SCHEDULE, baseline=BASELINE):
            assert set(score.fold_skill) == {0, 1, 2}
            assert set(score.block_skill) == {"early", "middle", "late"}
            assert 0.0 <= score.positive_fold_fraction <= 1.0

    def test_the_constant_direction_null_is_scored(self, records) -> None:
        score = next(s for s in score_configs(records, SCHEDULE, baseline=BASELINE) if s.model_id == "ar_p")
        rows = records[(records["model_id"] == "ar_p") & (records["horizon"] == score.horizon) & (records["window"] == score.window)]
        expected = float(np.mean((rows["actual"] > 0) == rows["train_direction_up"]))
        assert score.constant_direction_accuracy == pytest.approx(expected)

    def test_undeclared_scores_are_named_not_invented(self, records) -> None:
        payload = score_configs(records, SCHEDULE, baseline=BASELINE)[0].as_dict()
        assert payload["not_declared"] == NOT_DECLARED

    def test_only_origins_both_sides_covered_are_scored(self, records) -> None:
        """A model that lost a fold is not compared on a different stretch."""
        mask = (records["model_id"] == "ar_p") & (records["fold"] == 0)
        thinned = records[~mask]
        scores = {(s.model_id, s.horizon, s.window): s for s in score_configs(thinned, SCHEDULE, baseline=BASELINE)}
        assert scores[("ar_p", 1, ROLL.label)].n_origins == len(SCHEDULE.origins) - len(SCHEDULE.folds[0].origins)
        assert scores[(BASELINE, 1, ROLL.label)].n_origins == len(SCHEDULE.origins)

    def test_an_origin_outside_the_schedule_is_refused(self, records) -> None:
        bad = records.copy()
        bad.loc[bad.index[0], "origin"] = FRAME.index[0]
        with pytest.raises(ValueError, match="outside the schedule"):
            score_configs(bad, SCHEDULE, baseline=BASELINE)
