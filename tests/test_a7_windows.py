"""Horizon targets and the information set of one fold.

Two properties are pinned here, both at the level of data rather than models:

**Nothing a model trains on completes after its origin.** For a 7-bar target the
last six feature rows before the origin have targets that end after it; they are
excluded, and a test counts them.

**Nothing outside the declared extent is read.** Poisoning a raw bar before a
rolling window's extent leaves everything a model at that fold can see
bit-identical -- and a control shows the same poison *does* leak into the
window under A6's full-history features, which is why the extent exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.features.pipeline import build_feature_frame
from btc_forecaster.features.spec import default_specs, log_returns
from btc_forecaster.research.walk_forward.config import WindowSpec
from btc_forecaster.research.walk_forward.origins import build_schedule, first_origin_position
from btc_forecaster.research.walk_forward.targets import (
    TARGET_DEFINITION,
    assert_targets_realised_by,
    cumulative_log_return,
    horizon_supervised,
)
from btc_forecaster.research.walk_forward.windows import (
    InformationSetError,
    extent_start_position,
    information_fingerprint,
    information_set,
)
from btc_forecaster.testing import synthetic_market_frame

FRAME = synthetic_market_frame(periods=700, kind="ar1", seed=5)
WARMUP, ROWS, MAX_H = 60, 200, 7
ROLLING, EXPANDING = WindowSpec("rolling", ROWS), WindowSpec("expanding")
SCHEDULE = build_schedule(
    FRAME.index,
    first_position=first_origin_position(warmup_bars=WARMUP, training_rows=ROWS, max_horizon=MAX_H),
    max_horizon=MAX_H,
    n_refits=3,
    n_blocks=3,
)


def info(window=ROLLING, fold=0, frame=FRAME):
    return information_set(
        frame, SCHEDULE.folds[fold], window, max_horizon=MAX_H, warmup_bars=WARMUP
    )


def poison(frame: pd.DataFrame, position: int) -> pd.DataFrame:
    out = frame.copy()
    out.iloc[position, out.columns.get_loc("close")] *= 5.0
    return out


class TestTheTarget:
    def test_it_is_the_log_ratio_h_bars_apart(self) -> None:
        close = FRAME["close"]
        y = cumulative_log_return(close, 7)
        assert y.iloc[100] == pytest.approx(np.log(close.iloc[100] / close.iloc[93]))

    def test_it_counts_bars_not_days(self) -> None:
        """A missing calendar day shifts nothing: three bars back is three rows back."""
        gapped = FRAME.drop(FRAME.index[50])
        y = cumulative_log_return(gapped["close"], 3)
        assert y.iloc[52] == pytest.approx(np.log(gapped["close"].iloc[52] / gapped["close"].iloc[49]))

    def test_at_h1_it_is_the_a6_target(self) -> None:
        assert np.allclose(
            cumulative_log_return(FRAME["close"], 1).to_numpy()[1:],
            log_returns(FRAME).to_numpy()[1:],
        )

    def test_a_zero_horizon_is_refused(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            cumulative_log_return(FRAME["close"], 0)

    def test_features_come_from_h_bars_before_the_target_bar(self) -> None:
        data = horizon_supervised(build_feature_frame(FRAME, list(default_specs())), FRAME["close"], 7)
        positions = FRAME.index.get_indexer(data.X.index)
        origins = FRAME.index.get_indexer(pd.DatetimeIndex(data.feature_bar))
        assert (positions - origins == 7).all()

    def test_an_unrealised_target_is_caught(self) -> None:
        data = horizon_supervised(build_feature_frame(FRAME, list(default_specs())), FRAME["close"], 7)
        with pytest.raises(AssertionError, match="complete after the origin"):
            assert_targets_realised_by(data, data.X.index[-10])

    def test_the_definition_is_on_record(self) -> None:
        assert TARGET_DEFINITION["equals_a6_target_at_h1"] is True
        assert TARGET_DEFINITION["comparable_to_a2"] is False
        assert "bars" in TARGET_DEFINITION["horizon_unit"]


class TestTheExtent:
    def test_the_rolling_formula(self) -> None:
        assert extent_start_position(1000, ROLLING, max_horizon=7, warmup_bars=60) == 1000 - 7 - 200 + 1 - 60

    def test_an_expanding_window_reads_from_the_start(self) -> None:
        assert extent_start_position(1000, EXPANDING, max_horizon=7, warmup_bars=60) == 0
        assert info(EXPANDING).extent_start == FRAME.index[0]

    def test_a_window_that_does_not_fit_is_refused(self) -> None:
        with pytest.raises(InformationSetError, match="before the start"):
            extent_start_position(100, ROLLING, max_horizon=7, warmup_bars=60)

    def test_the_slice_ends_where_the_last_target_does(self) -> None:
        i = info()
        last_origin = SCHEDULE.folds[0].origins[-1]
        assert i.frame.index[-1] == FRAME.index[FRAME.index.get_loc(last_origin) + MAX_H]

    def test_a_warmup_shorter_than_the_features_is_refused(self) -> None:
        with pytest.raises(InformationSetError, match="warmup"):
            information_set(FRAME, SCHEDULE.folds[0], ROLLING, max_horizon=MAX_H, warmup_bars=5)


class TestTrainingRows:
    @pytest.mark.parametrize("horizon", [1, 3, 7])
    def test_a_rolling_window_holds_exactly_its_rows(self, horizon) -> None:
        for fold in range(3):
            train = info(fold=fold).training_set(horizon)
            assert len(train) == ROWS

    @pytest.mark.parametrize("horizon", [1, 3, 7])
    def test_every_target_is_realised_by_the_refit_origin(self, horizon) -> None:
        i = info()
        train = i.training_set(horizon)
        assert train.X.index.max() == i.refit_origin
        # ...so the newest feature row sits h bars before it: the last h-1 are embargoed.
        newest = FRAME.index.get_loc(pd.Timestamp(train.feature_bar.iloc[-1]))
        assert newest == FRAME.index.get_loc(i.refit_origin) - horizon

    def test_no_training_row_is_still_warming_up(self) -> None:
        i = info()
        assert pd.DatetimeIndex(i.training_set(7).feature_bar).min() >= i.first_settled_bar

    def test_an_expanding_window_grows_across_folds(self) -> None:
        sizes = [len(info(EXPANDING, fold).training_set(1)) for fold in range(3)]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]

    def test_the_series_is_the_target_and_close_is_aligned(self) -> None:
        train = info().training_set(3)
        assert np.array_equal(train.series.to_numpy(), train.y.to_numpy())
        assert train.close.index.equals(train.X.index)


class TestTheEvaluationBlock:
    @pytest.mark.parametrize("horizon", [1, 3, 7])
    def test_one_row_per_origin_predicting_h_bars_ahead(self, horizon) -> None:
        i = info()
        train = i.training_set(horizon)
        context = i.evaluation_context(horizon, train_end=train.X.index.max())
        assert pd.DatetimeIndex(context.feature_bar).equals(SCHEDULE.folds[0].origins)
        gaps = FRAME.index.get_indexer(context.X.index) - FRAME.index.get_indexer(context.origins)
        assert (gaps == horizon).all()
        assert (context.X.index > train.X.index.max()).all()

    def test_series_models_see_one_bar_returns(self) -> None:
        context = info().evaluation_context(7, train_end=info().training_set(7).X.index.max())
        assert np.allclose(
            context.series.dropna().to_numpy(),
            log_returns(info().frame).dropna().to_numpy(),
        )


class TestEarlyStopping:
    def test_the_dev_block_is_the_tail_and_the_overlap_is_purged(self) -> None:
        i = info()
        train = i.training_set(7)
        fit, dev = i.early_stopping_split(train, horizon=7, dev_fraction=0.2)
        assert len(dev) == 40
        assert len(fit) == ROWS - 40 - 6  # six purged rows for a 7-bar target
        assert dev.X.index[-1] == train.X.index[-1]
        assert (dev.X.index > fit.X.index.max()).all()

    def test_one_bar_targets_need_no_purge(self) -> None:
        i = info()
        fit, dev = i.early_stopping_split(i.training_set(1), horizon=1, dev_fraction=0.2)
        assert len(fit) + len(dev) == ROWS

    def test_a_window_too_small_to_split_is_refused(self) -> None:
        i = info()
        train = i.training_set(7)
        tiny = type(train)(
            X=train.X.iloc[:5], y=train.y.iloc[:5], series=train.series.iloc[:5],
            close=train.close.iloc[:5], feature_bar=train.feature_bar.iloc[:5],
        )
        with pytest.raises(InformationSetError, match="purge"):
            i.early_stopping_split(tiny, horizon=7, dev_fraction=0.2)


class TestTheExtentIsABoundary:
    #: The last fold: its extent starts well inside the data, so there are bars
    #: before it to poison. At fold 0 the largest window starts at bar 0, and
    #: "the bar before" would be position -1 -- which `iloc` reads as the *last*
    #: bar, silently poisoning the future instead. That is how this test was
    #: first written, and why it now asserts the position it poisons.
    FOLD = 2

    def test_a_bar_before_the_extent_changes_nothing_a_model_can_see(self) -> None:
        clean = info(fold=self.FOLD)
        before = FRAME.index.get_loc(clean.extent_start) - 1
        assert before >= 1, "the poisoned bar must be inside the data, before the extent"
        dirty = info(fold=self.FOLD, frame=poison(FRAME, before))
        for horizon in (1, 3, 7):
            assert information_fingerprint(dirty, horizon) == information_fingerprint(clean, horizon)

    def test_a_bar_inside_the_window_does_change_it(self) -> None:
        """The sensitivity half: a detector that reports 'no change' for every
        input detects nothing."""
        clean = info()
        inside = FRAME.index.get_loc(clean.refit_origin) - 20
        dirty = info(frame=poison(FRAME, inside))
        assert information_fingerprint(dirty, 1) != information_fingerprint(clean, 1)

    def test_a_bar_after_the_refit_origin_changes_no_training_row(self) -> None:
        clean = info()
        after = FRAME.index.get_loc(clean.refit_origin) + 3
        dirty = info(frame=poison(FRAME, after))
        assert information_fingerprint(dirty, 7) == information_fingerprint(clean, 7)

    def test_without_the_extent_the_same_poison_would_leak(self) -> None:
        """Why the extent exists. With features over the whole history -- A6's
        convention -- the EMA carries the poisoned bar into the window."""
        clean = info(fold=self.FOLD)
        before = FRAME.index.get_loc(clean.extent_start) - 1
        window_rows = clean.training_set(1).feature_bar
        full_clean = build_feature_frame(FRAME, list(default_specs()))
        full_dirty = build_feature_frame(poison(FRAME, before), list(default_specs()))
        first = pd.Timestamp(window_rows.iloc[0])
        assert full_clean.loc[first, "ema_7"] != full_dirty.loc[first, "ema_7"]
