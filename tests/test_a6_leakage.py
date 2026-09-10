"""Adversaries. Every way this zoo could reach forward, tried on purpose.

The structural guards -- `history_at` slicing at the origin, `to_supervised`
refusing step 0, the window builder asserting on `feature_bar` -- are the
defence. This file is the attack. Each test constructs the specific mistake and
asserts it is caught or has no effect:

* future rows poisoned after the origin;
* future *targets* poisoned;
* a single future feature value poisoned;
* feature timestamps shuffled;
* an off-by-one window that would include the predicted bar;
* overlapping labels between the training budget and the holdout.

The most important property is that these tests can *fail*. Several of them
poison data and assert bit-identical output, which would pass trivially if the
poisoning did not reach the model at all -- so each of those has a companion
assertion that poisoning the *past* does change the answer. A leakage test that
cannot detect a leak is decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.contracts import EvaluationContext
from btc_forecaster.research.partition import (
    PartitionError,
    PartitionSpec,
    build_dataset,
)
from btc_forecaster.research.windows import (
    Sequences,
    WindowError,
    WindowSpec,
    assert_window_causality,
    build_sequences,
)
from btc_forecaster.testing import synthetic_market_frame


def available(*model_ids: str) -> list[str]:
    """Filter to the models this environment can actually build.

    The quant CI job installs `.[dev]` -- numpy, pandas, scipy and statsmodels,
    but not scikit-learn, arch or xgboost. Hard-coding a model list would make
    these tests either fail there or be skipped wholesale, and the second is
    worse: the leakage suite silently not running is exactly the situation it
    exists to prevent. Filtering keeps whatever is present under test.
    """
    return [m for m in model_ids if registry.get(m).is_available()]


#: One model per mechanism: a constant, a series model, a tabular model, and a
#: sequence model. Each reaches the data by a different route, so each could
#: leak differently.
ADVERSARY_MODELS = available("random_walk_drift", "arima", "ridge", "gru")
SPEC = PartitionSpec(train_rows=200)


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(synthetic_market_frame(periods=500, kind="ar1", seed=41), spec=SPEC)


@pytest.fixture(scope="module")
def fitted(dataset):
    train, dev = dataset.training_set(), dataset.development_context()
    models = {}
    for model_id in ADVERSARY_MODELS:
        model = registry.build(model_id).fit(train)
        if model.needs_calibration:
            model.calibrate(dev)
        models[model_id] = model
    return models


def poisoned_context(context: EvaluationContext, **overrides) -> EvaluationContext:
    fields = {
        "X": context.X,
        "y": context.y,
        "series": context.series,
        "close": context.close,
        "feature_bar": context.feature_bar,
        "train_end": context.train_end,
        "design": context.design,
    }
    fields.update(overrides)
    return EvaluationContext(**fields)


class TestFutureRowsCannotBeRead:
    """The boundary is per-representation, and getting it wrong is instructive.

    A model may legitimately use:

    * the *series* up to and including its forecast origin;
    * the *design row indexed by the target bar*, because that row's predictors
      were computed at the origin -- `to_supervised` already shifted them.

    So the future starts one row later for the design matrix than for the
    series, and a test that poisons `design.index > origin` is poisoning a legal
    input rather than catching a leak. That mistake is the same off-by-one this
    whole contract exists to prevent, arriving from the other direction.
    """

    @pytest.mark.parametrize("model_id", ADVERSARY_MODELS)
    def test_poisoning_every_future_bar_changes_nothing(
        self, dataset, fitted, model_id
    ) -> None:
        context = dataset.evaluation_context()
        clean = fitted[model_id].predict(context).point[0]

        origin = context.origins[0]
        first_target = context.target_bars[0]
        series = context.series.copy()
        close = context.close.copy()
        design = context.design.copy()
        series.loc[series.index > origin] = 7.5
        close.loc[close.index > origin] = 1e9
        design.loc[design.index > first_target, :] = 99.0

        dirty = fitted[model_id].predict(
            poisoned_context(context, series=series, close=close, design=design)
        ).point[0]
        assert dirty == clean, f"{model_id} read a bar after its forecast origin"


class TestTheAdversariesCanActuallyDetectALeak:
    """The companion assertions, without which the tests above prove nothing.

    A model that ignores its inputs entirely would pass every poisoning test in
    this file. So each mechanism is poisoned where it genuinely reads, and the
    forecast must move. The constant baseline is exempt because ignoring its
    inputs is what makes it a baseline -- and that exemption is itself asserted.
    """

    def test_a_series_model_reacts_to_its_own_history(self, dataset, fitted) -> None:
        context = dataset.evaluation_context()
        clean = fitted["arima"].predict(context).point[0]
        series = context.series.copy()
        series.loc[series.index <= context.origins[0]] = 0.05
        dirty = fitted["arima"].predict(poisoned_context(context, series=series)).point[0]
        assert dirty != clean

    def test_a_tabular_model_reacts_to_its_own_feature_row(self, dataset, fitted) -> None:
        """Ridge has no history: it reads exactly one row. Poisoning bars it
        never looks at would prove nothing, so the row itself is poisoned."""
        pytest.importorskip("sklearn")
        context = dataset.evaluation_context()
        clean = fitted["ridge"].predict(context).point[0]
        X = context.X.copy()
        X.iloc[0, :] = 3.0
        dirty = fitted["ridge"].predict(poisoned_context(context, X=X)).point[0]
        assert dirty != clean

    def test_a_sequence_model_reacts_to_its_own_window(self, dataset, fitted) -> None:
        context = dataset.evaluation_context()
        clean = fitted["gru"].predict(context).point[0]
        design = context.design.copy()
        design.loc[design.index <= context.target_bars[0], :] = 3.0
        dirty = fitted["gru"].predict(poisoned_context(context, design=design)).point[0]
        assert dirty != clean

    def test_the_constant_baseline_genuinely_ignores_everything(
        self, dataset, fitted
    ) -> None:
        """Stated rather than skipped. A baseline that reacted to its inputs
        would not be a baseline."""
        context = dataset.evaluation_context()
        clean = fitted["random_walk_drift"].predict(context).point[0]
        series = context.series.copy()
        series.loc[:] = 0.05
        dirty = fitted["random_walk_drift"].predict(
            poisoned_context(context, series=series)
        ).point[0]
        assert dirty == clean


class TestFutureTargetsCannotBeRead:
    @pytest.mark.parametrize("model_id", ADVERSARY_MODELS)
    def test_replacing_every_target_changes_no_forecast(
        self, dataset, fitted, model_id
    ) -> None:
        """The target is what the model is being asked for. Reading it would be
        the most direct leak available, and the least visible in a metric."""
        context = dataset.evaluation_context()
        clean = fitted[model_id].predict(context).point
        wrecked = poisoned_context(
            context, y=pd.Series(np.full(len(context), 0.5), index=context.y.index)
        )
        assert np.array_equal(fitted[model_id].predict(wrecked).point, clean)


class TestSingleValueInjection:
    @pytest.mark.parametrize("model_id", available("ridge", "gru"))
    def test_one_poisoned_future_feature_does_not_move_earlier_forecasts(
        self, dataset, fitted, model_id
    ) -> None:
        """Finer than the block poisoning: one cell, in one row, in the future."""
        context = dataset.evaluation_context()
        clean = fitted[model_id].predict(context).point

        design = context.design.copy()
        target_row = context.target_bars[len(context) // 2]
        design.loc[target_row, design.columns[0]] = 1e6

        dirty = fitted[model_id].predict(poisoned_context(context, design=design)).point
        moved = np.flatnonzero(dirty != clean)
        # Only forecasts at or after the poisoned row may move.
        position = int(np.flatnonzero(context.target_bars == target_row)[0])
        assert all(index >= position for index in moved), (
            f"{model_id} moved a forecast before the poisoned bar"
        )


class TestTimestampIntegrity:
    def test_shuffling_feature_timestamps_is_refused(self, dataset) -> None:
        """A shuffled index would silently pair each row with someone else's
        target, which trains a model on noise and scores it on nothing."""
        context = dataset.evaluation_context()
        rng = np.random.default_rng(0)
        shuffled = pd.Series(
            rng.permutation(context.feature_bar.to_numpy()), index=context.feature_bar.index
        )
        with pytest.raises(ValueError, match="not strictly before"):
            poisoned_context(context, feature_bar=shuffled)

    def test_an_origin_equal_to_its_target_is_refused(self, dataset) -> None:
        context = dataset.evaluation_context()
        with pytest.raises(ValueError, match="not strictly before"):
            poisoned_context(
                context,
                feature_bar=pd.Series(context.target_bars, index=context.X.index),
            )


class TestOffByOneWindows:
    def test_a_window_containing_its_own_target_is_caught(self, dataset) -> None:
        train = dataset.training_set()
        spec = WindowSpec(lookback=6)
        good = build_sequences(train.X, train.y, train.feature_bar, spec)
        poisoned = Sequences(
            X=good.X, y=good.y, index=good.index, last_feature_bar=good.index
        )
        with pytest.raises(WindowError, match="not strictly before"):
            assert_window_causality(poisoned)

    def test_a_window_reaching_one_bar_too_far_is_caught(self, dataset) -> None:
        """Shifted forward by exactly one bar -- the classic mistake, and the one
        a positional check would keep satisfying."""
        train = dataset.training_set()
        spec = WindowSpec(lookback=6)
        good = build_sequences(train.X, train.y, train.feature_bar, spec)
        shifted = pd.DatetimeIndex(good.last_feature_bar).shift(2, freq="D")
        with pytest.raises(WindowError):
            assert_window_causality(
                Sequences(X=good.X, y=good.y, index=good.index, last_feature_bar=shifted)
            )

    def test_the_supervised_builder_refuses_a_zero_shift(self) -> None:
        """step=0 lets a feature row see the bar it predicts."""
        with pytest.raises(PartitionError, match="step"):
            PartitionSpec(step=0)


class TestPartitionOverlap:
    def test_no_training_bar_appears_in_the_holdout(self, dataset) -> None:
        partition = dataset.partition
        assert not set(partition.budget) & set(partition.holdout)
        assert not set(partition.train) & set(partition.holdout)

    def test_no_development_bar_appears_in_the_holdout(self, dataset) -> None:
        partition = dataset.partition
        assert not set(partition.dev) & set(partition.holdout)

    def test_every_holdout_target_is_after_every_training_target(self, dataset) -> None:
        partition = dataset.partition
        assert partition.holdout.min() > partition.budget.max()

    def test_the_context_refuses_a_target_inside_training(self, dataset) -> None:
        context = dataset.evaluation_context()
        with pytest.raises(ValueError, match="inside the training partition"):
            poisoned_context(context, train_end=context.target_bars.max())

    def test_labels_do_not_overlap_across_the_boundary(self, dataset) -> None:
        """The subtler version: with step=1 the last training target and the
        first holdout origin must not be the same bar."""
        partition = dataset.partition
        context = dataset.evaluation_context()
        assert context.origins.min() >= partition.dev.max()
        assert context.target_bars.min() > partition.dev.max()


class TestScalerLeakage:
    @pytest.mark.parametrize("model_id", available("ridge", "gru"))
    def test_the_scaler_is_not_refitted_at_predict_time(
        self, dataset, fitted, model_id
    ) -> None:
        """A model that re-derived its scaling from the arriving matrix would
        centre the evaluation block on its own mean -- and would answer
        differently for a subset of the same rows."""
        context = dataset.evaluation_context()
        full = fitted[model_id].predict(context).point
        half = len(context) // 2
        subset = poisoned_context(
            context,
            X=context.X.iloc[:half],
            y=context.y.iloc[:half],
            feature_bar=context.feature_bar.iloc[:half],
        )
        assert np.allclose(fitted[model_id].predict(subset).point, full[:half], atol=1e-12)
