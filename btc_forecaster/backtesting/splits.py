"""Walk-forward fold construction.

One splitter produces one list of folds, and every model in a comparison is
scored on that same list. This is the property that makes a model comparison
mean anything: if two models see different origins, different training lengths
or different test windows, their scores are not comparable and the difference
between them is partly an artefact of the split.

The original pipeline had two separate, inconsistent evaluation paths -- a
single 90-day tail holdout and a four-fold walk-forward built with
``np.linspace`` -- and no model other than the hybrid was ever run through
either.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ..timebase import BAR_DURATION, ForecastOrigin, to_utc_index

SplitMode = Literal["expanding", "rolling"]


@dataclass(frozen=True)
class Fold:
    """One train/test split, anchored to a forecast origin."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_bars: int

    @property
    def origin(self) -> ForecastOrigin:
        """The forecast is issued at the close of the last training bar."""
        return ForecastOrigin(self.train_end)

    @property
    def steps_to_test_end(self) -> int:
        """Forecast steps needed to reach ``test_end`` from the origin.

        With an embargo this exceeds the scored horizon: the model forecasts
        across the embargo gap and only the tail is scored.
        """
        return int((self.test_end - self.train_end) / BAR_DURATION)

    @property
    def horizon(self) -> int:
        """Scored bars in this fold."""
        return int((self.test_end - self.test_start) / BAR_DURATION) + 1

    def train_slice(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[(frame.index >= self.train_start) & (frame.index <= self.train_end)]

    def test_slice(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[(frame.index >= self.test_start) & (frame.index <= self.test_end)]

    def to_dict(self) -> dict:
        return {
            "fold": self.index,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "train_bars": None,
            "embargo_bars": self.embargo_bars,
            "horizon": self.horizon,
        }


@dataclass(frozen=True)
class WalkForwardSplitter:
    """Builds evenly spaced walk-forward folds over a daily index.

    ``mode="expanding"``
        Training starts at the beginning of the data every time and grows. Uses
        all available history, and is what most published backtests do.

    ``mode="rolling"``
        Training is a fixed-length window that slides. Costs data but adapts to
        regime change, which for crypto is not a minor consideration.

    ``embargo_bars`` inserts a gap between the last training bar and the first
    scored bar. It matters whenever features or targets are built from
    overlapping windows: a 30-day rolling feature computed just before the split
    shares observations with the first days after it, so training and test are
    not independent even though no bar appears in both. The gap costs data and
    buys a cleaner claim.
    """

    horizon: int = 30
    n_folds: int = 5
    min_train_bars: int = 730
    mode: SplitMode = "expanding"
    window_bars: int | None = None
    embargo_bars: int = 0
    step_bars: int | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if self.embargo_bars < 0:
            raise ValueError("embargo_bars must be >= 0")
        if self.mode not in ("expanding", "rolling"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "rolling" and not self.window_bars:
            raise ValueError("rolling mode requires window_bars")

    def split(self, index: pd.DatetimeIndex) -> list[Fold]:
        """Build the folds. Raises if the data cannot support the request."""
        idx = to_utc_index(index)
        if not idx.is_monotonic_increasing:
            raise ValueError("index must be sorted ascending")

        n = len(idx)
        needed = self.min_train_bars + self.embargo_bars + self.horizon
        if n < needed:
            raise ValueError(
                f"need at least {needed} bars for {self.n_folds} fold(s) "
                f"(min_train_bars={self.min_train_bars} + embargo={self.embargo_bars} "
                f"+ horizon={self.horizon}), got {n}"
            )

        first_train_end = self.min_train_bars - 1
        last_train_end = n - 1 - self.embargo_bars - self.horizon

        if last_train_end < first_train_end:
            raise ValueError("not enough data after min_train_bars for even one fold")

        positions = self._origin_positions(first_train_end, last_train_end)

        folds: list[Fold] = []
        for fold_number, train_end_pos in enumerate(positions):
            test_start_pos = train_end_pos + self.embargo_bars + 1
            test_end_pos = test_start_pos + self.horizon - 1
            if test_end_pos > n - 1:
                continue

            if self.mode == "expanding":
                train_start_pos = 0
            else:
                if self.window_bars is None:  # unreachable: __post_init__ enforces it
                    raise ValueError("rolling mode requires window_bars")
                train_start_pos = max(0, train_end_pos - self.window_bars + 1)

            folds.append(
                Fold(
                    index=fold_number,
                    train_start=idx[train_start_pos],
                    train_end=idx[train_end_pos],
                    test_start=idx[test_start_pos],
                    test_end=idx[test_end_pos],
                    embargo_bars=self.embargo_bars,
                )
            )

        if not folds:
            raise ValueError("no valid folds could be constructed")
        return folds

    def _origin_positions(self, first: int, last: int) -> list[int]:
        if self.step_bars:
            positions = list(range(last, first - 1, -self.step_bars))[: self.n_folds]
            return sorted(positions)

        if self.n_folds == 1:
            return [last]

        span = last - first
        stride = span / (self.n_folds - 1)
        positions = sorted({int(round(first + i * stride)) for i in range(self.n_folds)})
        return positions

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "horizon": self.horizon,
            "n_folds_requested": self.n_folds,
            "min_train_bars": self.min_train_bars,
            "window_bars": self.window_bars,
            "embargo_bars": self.embargo_bars,
            "step_bars": self.step_bars,
        }

    def __iter__(self) -> Iterator:  # pragma: no cover - convenience only
        raise TypeError("call split(index) to build folds")


def assert_folds_are_disjoint(folds: list[Fold]) -> None:
    """No training bar may appear in that fold's own test window."""
    for fold in folds:
        if fold.train_end >= fold.test_start:
            raise AssertionError(
                f"fold {fold.index}: training ends {fold.train_end.date()} but testing "
                f"starts {fold.test_start.date()}"
            )
        expected_gap = fold.embargo_bars + 1
        actual_gap = int((fold.test_start - fold.train_end) / BAR_DURATION)
        if actual_gap != expected_gap:
            raise AssertionError(
                f"fold {fold.index}: gap between train and test is {actual_gap} bars, "
                f"expected {expected_gap} (embargo {fold.embargo_bars} + 1)"
            )


__all__ = [
    "Fold",
    "SplitMode",
    "WalkForwardSplitter",
    "assert_folds_are_disjoint",
]
