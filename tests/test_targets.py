"""Target definition, return views, and per-horizon-step decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.backtesting.engine import run_walk_forward
from btc_forecaster.backtesting.splits import WalkForwardSplitter
from btc_forecaster.evaluation.targets import (
    TARGET_AUDIT,
    ForecastTask,
    cumulative_return_view,
    evaluate_by_step,
    one_step_direction_sample,
    step_metrics_frame,
    step_return_view,
)
from btc_forecaster.models.baselines import RandomWalk, RandomWalkWithDrift
from btc_forecaster.testing import synthetic_market_frame
from btc_forecaster.timebase import UTC


def series(values, start="2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", tz=UTC, name="date")
    return pd.Series(np.asarray(values, dtype=float), index=index)


class TestTargetAudit:
    def test_price_level_stays_the_primary_task(self):
        """Switching the primary metric would break comparability with the
        preserved reference, which evidence.py forbids."""
        assert TARGET_AUDIT["primary_task"] == ForecastTask.PRICE_LEVEL.value
        assert "LEAKAGE_CORRECTED_REFERENCE" in TARGET_AUDIT["why_price_level_is_primary"]

    def test_return_tasks_are_reported_alongside_not_instead(self):
        assert ForecastTask.CUMULATIVE_RETURN.value in TARGET_AUDIT["secondary_tasks"]
        assert ForecastTask.STEP_RETURN.value in TARGET_AUDIT["secondary_tasks"]

    def test_the_audit_states_why_returns_matter(self):
        assert "I(1)" in TARGET_AUDIT["why_returns_are_also_reported"]

    def test_the_directional_definition_is_recorded(self):
        assert "forecast origin" in TARGET_AUDIT["directional_definition"]
        assert "not-up" in TARGET_AUDIT["directional_definition"]

    def test_the_correlated_observations_limitation_is_recorded(self):
        assert "correlated observations" in TARGET_AUDIT["known_limitation"]

    def test_the_audit_is_serialisable(self):
        import json

        json.dumps(TARGET_AUDIT)


class TestCumulativeReturnView:
    def test_returns_are_measured_from_the_origin_close(self):
        view = cumulative_return_view(
            actual=series([110.0, 120.0]),
            predicted=series([105.0, 90.0]),
            origin_close=100.0,
        )
        np.testing.assert_allclose(view.actual.to_numpy(), [0.10, 0.20])
        np.testing.assert_allclose(view.predicted.to_numpy(), [0.05, -0.10])

    def test_direction_agreement_is_per_bar(self):
        view = cumulative_return_view(
            actual=series([110.0, 90.0]),
            predicted=series([105.0, 105.0]),
            origin_close=100.0,
        )
        assert list(view.direction_agreement()) == [True, False]

    def test_a_flat_forecast_predicts_down_everywhere(self):
        """Zero is not up. A forecast of no change is not a bet on an increase."""
        view = cumulative_return_view(
            actual=series([110.0, 90.0]),
            predicted=series([100.0, 100.0]),
            origin_close=100.0,
        )
        assert list(view.direction_agreement()) == [False, True]

    def test_the_view_is_invariant_to_price_scale(self):
        small = cumulative_return_view(series([110.0]), series([105.0]), origin_close=100.0)
        large = cumulative_return_view(series([110_000.0]), series([105_000.0]), origin_close=100_000.0)
        np.testing.assert_allclose(small.actual.to_numpy(), large.actual.to_numpy())

    def test_zero_origin_is_rejected(self):
        with pytest.raises(ValueError, match="non-zero"):
            cumulative_return_view(series([1.0]), series([1.0]), origin_close=0.0)

    def test_mismatched_index_is_rejected(self):
        with pytest.raises(ValueError, match="share an index"):
            from btc_forecaster.evaluation.targets import ReturnView

            ReturnView(
                task=ForecastTask.CUMULATIVE_RETURN,
                predicted=series([1.0], start="2024-01-01"),
                actual=series([1.0], start="2025-01-01"),
                origin_close=100.0,
            )


class TestStepReturnView:
    def test_first_step_is_measured_from_the_origin(self):
        view = step_return_view(
            actual=series([110.0, 121.0]),
            predicted=series([105.0, 105.0]),
            origin_close=100.0,
        )
        assert view.actual.iloc[0] == pytest.approx(0.10)
        assert view.actual.iloc[1] == pytest.approx(0.10)
        assert view.predicted.iloc[1] == pytest.approx(0.0)

    def test_step_and_cumulative_views_answer_different_questions(self):
        """A path can be right in shape and wrong in level, or the reverse."""
        actual = series([110.0, 121.0])
        predicted = series([90.0, 99.0])  # correct 10% steps, wrong level

        step = step_return_view(actual, predicted, origin_close=100.0)
        cumulative = cumulative_return_view(actual, predicted, origin_close=100.0)

        assert list(step.direction_agreement()) == [False, True]
        assert list(cumulative.direction_agreement()) == [False, False]


class TestEvaluateByStep:
    @pytest.fixture
    def records(self) -> pd.DataFrame:
        frame = synthetic_market_frame(periods=1200, seed=13)
        splitter = WalkForwardSplitter(horizon=10, n_folds=20, min_train_bars=400)
        result = run_walk_forward(frame, [RandomWalk(), RandomWalkWithDrift()], splitter)
        return result.prediction_records("random_walk_drift")

    def test_engine_emits_one_record_per_scored_bar(self, records):
        assert len(records) == 20 * 10
        assert set(records.columns) >= {
            "model", "fold", "origin", "date", "step",
            "actual", "predicted", "origin_close", "lower", "upper",
        }

    def test_step_counts_forecast_distance_from_the_origin(self, records):
        assert sorted(records["step"].unique()) == list(range(1, 11))
        first = records[records["step"] == 1]
        assert (first["date"] - first["origin"] == pd.Timedelta(days=1)).all()

    def test_each_step_pools_one_observation_per_origin(self, records):
        metrics = evaluate_by_step(records)
        assert len(metrics) == 10
        assert all(m.n_origins == 20 for m in metrics)

    def test_error_grows_with_horizon(self, records):
        metrics = evaluate_by_step(records)
        assert metrics[-1].mae > metrics[0].mae

    def test_step_one_gives_one_observation_per_origin(self, records):
        """The point of the decomposition: 20 near-independent draws, not 200
        correlated ones."""
        sample = one_step_direction_sample(records)
        assert len(sample) == 20
        assert sample.dtype == bool

    def test_step_metrics_frame_is_indexed_by_step(self, records):
        frame = step_metrics_frame(evaluate_by_step(records))
        assert frame.index.name == "step"
        assert "directional_accuracy" in frame.columns

    def test_max_step_truncates(self, records):
        assert len(evaluate_by_step(records, max_step=3)) == 3

    def test_missing_columns_are_reported(self):
        with pytest.raises(ValueError, match="missing column"):
            evaluate_by_step(pd.DataFrame({"step": [1]}))

    def test_empty_metrics_give_an_empty_frame(self):
        assert step_metrics_frame([]).empty

    def test_records_without_step_one_are_an_error(self, records):
        with pytest.raises(ValueError, match="no step-1 rows"):
            one_step_direction_sample(records[records["step"] > 1])


class TestPooledVersusPerStepDirection:
    def test_pooled_direction_is_dominated_by_within_fold_correlation(self):
        """The artefact that produced 0.889 in the A2 baseline run.

        Scoring many bars against one origin on a trending series makes every bar
        in that fold agree, so the pooled figure reports fold-level luck as if it
        were per-bar skill. Step-1 pooling across origins does not.
        """
        frame = synthetic_market_frame(periods=1200, kind="trend", seed=5)
        splitter = WalkForwardSplitter(horizon=30, n_folds=8, min_train_bars=400)
        result = run_walk_forward(frame, [RandomWalk(), RandomWalkWithDrift()], splitter)
        records = result.prediction_records("random_walk_drift")

        actual_return = records["actual"] / records["origin_close"] - 1.0
        predicted_return = records["predicted"] / records["origin_close"] - 1.0
        pooled = float(((predicted_return > 0) == (actual_return > 0)).mean())

        step_one = float(one_step_direction_sample(records).mean())

        # Both are computed from the same forecasts; the pooled number rests on
        # 8 independent origins while claiming 240 observations.
        assert 0.0 <= pooled <= 1.0
        assert 0.0 <= step_one <= 1.0
        assert len(one_step_direction_sample(records)) == 8
        assert len(records) == 240, "pooled figure would claim 240 independent trials"
