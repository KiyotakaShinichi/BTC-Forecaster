"""The origin schedule: where every model is scored, and how it is cut.

A comparison is only a comparison if both sides were scored at the same
origins. These tests pin that the schedule is shared, contiguous, deterministic
and never reaches past the end of the data for any horizon.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research.walk_forward.origins import (
    OriginSchedule,
    RefitFold,
    block_names,
    build_schedule,
    first_origin_position,
)
from btc_forecaster.testing import daily_index

BARS = daily_index(900, "2020-01-01")


def schedule(**overrides) -> OriginSchedule:
    kwargs = {"first_position": 400, "max_horizon": 30, "n_refits": 6, "n_blocks": 3}
    kwargs.update(overrides)
    return build_schedule(BARS, **kwargs)


class TestTheFirstOrigin:
    def test_the_arithmetic_is_the_documented_one(self) -> None:
        """Warm-up, then the window's rows, then the horizon those rows need to
        have been realised -- minus the one bar the origin itself contributes."""
        assert first_origin_position(warmup_bars=60, training_rows=2000, max_horizon=30) == 2089

    def test_a_window_actually_fits_at_the_first_origin(self) -> None:
        """Checked by counting rather than by re-deriving the formula."""
        warmup, rows, horizon = 10, 50, 7
        p = first_origin_position(warmup_bars=warmup, training_rows=rows, max_horizon=horizon)
        feature_bars = np.arange(p + 1)
        realised = feature_bars[(feature_bars + horizon <= p) & (feature_bars >= warmup)]
        assert len(realised) == rows

    def test_nonpositive_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError):
            first_origin_position(warmup_bars=0, training_rows=100, max_horizon=1)


class TestTheSchedule:
    def test_origins_are_consecutive_bars(self) -> None:
        s = schedule()
        assert s.origins.equals(BARS[400 : len(BARS) - 30])

    def test_every_origin_has_the_longest_horizon_inside_the_data(self) -> None:
        s = schedule()
        last_bar = BARS[-1]
        positions = BARS.get_indexer(s.origins)
        assert (positions + s.max_horizon <= len(BARS) - 1).all()
        assert BARS[positions[-1] + s.max_horizon] == last_bar

    def test_folds_partition_the_origins_in_order(self) -> None:
        s = schedule()
        joined = s.folds[0].origins.append([f.origins for f in s.folds[1:]])
        assert joined.equals(s.origins)
        assert joined.tz is not None  # the timezone survives the partition
        assert [f.index for f in s.folds] == list(range(6))

    def test_each_fold_starts_at_its_refit_origin(self) -> None:
        for fold in schedule().folds:
            assert fold.origins[0] == fold.refit_origin

    def test_fold_sizes_differ_by_at_most_one(self) -> None:
        sizes = [len(f.origins) for f in schedule().folds]
        assert max(sizes) - min(sizes) <= 1

    def test_blocks_are_early_middle_late(self) -> None:
        s = schedule()
        assert s.block_names == ("early", "middle", "late")
        assert s.blocks[0][-1] < s.blocks[1][0] and s.blocks[1][-1] < s.blocks[2][0]

    def test_more_blocks_get_neutral_names(self) -> None:
        assert block_names(4) == ("block-1", "block-2", "block-3", "block-4")

    def test_labels_cover_every_origin(self) -> None:
        s = schedule()
        assert s.fold_of().index.equals(s.origins)
        assert s.block_of().index.equals(s.origins)
        assert set(s.block_of()) == {"early", "middle", "late"}


class TestDeterminism:
    def test_two_builds_are_identical(self) -> None:
        assert schedule().fingerprint() == schedule().fingerprint()

    def test_the_fingerprint_moves_with_the_design(self) -> None:
        assert schedule(n_refits=7).fingerprint() != schedule().fingerprint()
        assert schedule(max_horizon=7).fingerprint() != schedule().fingerprint()


class TestBoundaries:
    def test_too_little_data_is_refused_with_the_arithmetic(self) -> None:
        with pytest.raises(ValueError, match="need at least"):
            build_schedule(BARS[:420], first_position=400, max_horizon=30, n_refits=6, n_blocks=3)

    def test_more_folds_than_origins_is_refused(self) -> None:
        with pytest.raises(ValueError, match="need at least"):
            build_schedule(BARS, first_position=860, max_horizon=30, n_refits=20, n_blocks=3)

    def test_an_unsorted_index_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            build_schedule(BARS[::-1], first_position=400, max_horizon=30, n_refits=6, n_blocks=3)

    def test_duplicate_bars_are_refused(self) -> None:
        doubled = BARS.append(BARS[-1:]).sort_values()
        with pytest.raises(ValueError, match="duplicates"):
            build_schedule(doubled, first_position=400, max_horizon=30, n_refits=6, n_blocks=3)

    def test_a_fold_that_does_not_start_at_its_refit_origin_is_refused(self) -> None:
        with pytest.raises(ValueError, match="refit origin"):
            RefitFold(index=0, refit_origin=BARS[5], origins=BARS[6:9])

    def test_folds_that_skip_an_origin_are_refused(self) -> None:
        s = schedule()
        broken = (s.folds[0], s.folds[2])
        with pytest.raises(ValueError, match="partition"):
            OriginSchedule(origins=s.origins, folds=broken, blocks=s.blocks, max_horizon=30)
