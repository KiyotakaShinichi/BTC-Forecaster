"""The information set of one fold: which raw bars a model may read.

Every (refit fold, training window) pair declares an **extent** -- the first raw
bar anything in that fold may read -- and the features are rebuilt from the raw
bars inside it. Nothing older is touched.

That is stricter than A6, deliberately. A6 computes features over the whole
history, which is causal but not bounded: the EMA is seeded at the first bar of
the data and carries a geometrically fading memory of every bar since. For a
single partition that is harmless. For a rolling window it means a "250-row"
model is in fact reading, however faintly, bars from years earlier -- and an
adversary that poisons a bar outside the window would move its forecast. Here it
cannot, and the adversary suite checks that it cannot.

For a rolling window of ``w`` rows at refit origin ``p`` (a bar position):

    extent_start = p - max_horizon - w + 1 - warmup_bars

so the window's earliest training row, at the longest horizon, still has
``warmup_bars`` bars before it to settle its features. Shorter horizons get more
warm-up, never less. An expanding window's extent is the start of the data.

Training rows are the rows whose target is **realised by the refit origin** --
for ``h > 1`` the last ``h - 1`` feature rows before the origin are excluded,
because their targets end after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...features.pipeline import SupervisedData, build_feature_frame
from ...features.spec import FeatureSpec, default_specs, log_returns
from ..contracts import EvaluationContext, TrainingSet
from .config import WindowSpec
from .origins import RefitFold
from .targets import assert_targets_realised_by, horizon_supervised


class InformationSetError(ValueError):
    """The data cannot supply the declared window at this origin."""


def extent_start_position(
    refit_position: int, window: WindowSpec, *, max_horizon: int, warmup_bars: int
) -> int:
    """The first raw bar position this (fold, window) may read."""
    if window.rows is None:
        return 0
    start = refit_position - max_horizon - window.rows + 1 - warmup_bars
    if start < 0:
        raise InformationSetError(
            f"{window.label} at bar {refit_position} needs {-start} bar(s) before the "
            "start of the data"
        )
    return start


@dataclass(frozen=True)
class InformationSet:
    """The raw bars, features and returns one (fold, window) may use."""

    fold: RefitFold
    window: WindowSpec
    extent_start: pd.Timestamp
    #: Raw bars from the extent start to the last bar any target in the fold needs.
    frame: pd.DataFrame
    features: pd.DataFrame
    returns: pd.Series
    warmup_bars: int

    @property
    def refit_origin(self) -> pd.Timestamp:
        return self.fold.refit_origin

    @property
    def first_settled_bar(self) -> pd.Timestamp:
        """Features computed before this bar are still warming up and unused."""
        return self.frame.index[self.warmup_bars]

    def supervised(self, horizon: int) -> SupervisedData:
        """Every settled row in the extent, at this horizon."""
        data = horizon_supervised(self.features, self.frame["close"], horizon)
        settled = pd.DatetimeIndex(data.feature_bar) >= self.first_settled_bar
        return SupervisedData(
            X=data.X[settled], y=data.y[settled], feature_bar=data.feature_bar[settled], step=horizon
        )

    def training_set(self, horizon: int) -> TrainingSet:
        """The window's rows at this horizon, targets realised by the refit origin."""
        data = self.supervised(horizon).before(self.refit_origin)
        if self.window.rows is not None:
            if len(data) < self.window.rows:
                raise InformationSetError(
                    f"{self.window.label} at {self.refit_origin.date()} h={horizon}: only "
                    f"{len(data)} realised row(s) available"
                )
            data = SupervisedData(
                X=data.X.iloc[-self.window.rows :],
                y=data.y.iloc[-self.window.rows :],
                feature_bar=data.feature_bar.iloc[-self.window.rows :],
                step=horizon,
            )
        assert_targets_realised_by(data, self.refit_origin)
        return _training_set(data, self.frame["close"])

    def evaluation_context(self, horizon: int, *, train_end: pd.Timestamp) -> EvaluationContext:
        """One row per daily origin in the fold, predicting ``horizon`` bars ahead.

        ``series`` is the 1-bar return history of the extent. Models reach it
        only through ``history_at``, which stops at each row's origin.
        """
        data = self.supervised(horizon)
        rows = pd.DatetimeIndex(data.feature_bar).isin(self.fold.origins)
        if int(rows.sum()) != len(self.fold.origins):
            raise InformationSetError(
                f"fold {self.fold.index} h={horizon}: {int(rows.sum())} evaluation row(s) "
                f"for {len(self.fold.origins)} origin(s)"
            )
        return EvaluationContext(
            X=data.X[rows],
            y=data.y[rows],
            series=self.returns,
            close=self.frame["close"],
            feature_bar=data.feature_bar[rows],
            train_end=train_end,
            design=data.X,
        )

    def early_stopping_split(
        self, train: TrainingSet, *, horizon: int, dev_fraction: float
    ) -> tuple[TrainingSet, EvaluationContext]:
        """Hold out the tail of the window to decide when a network stops.

        Taken from inside the window, never from after the origin. For ``h > 1``
        the ``h - 1`` rows between the fitted rows and the held-out rows are
        purged: their targets overlap the first held-out targets, and leaving
        them in would let the stopping rule reward memorising the answer.
        """
        n = len(train)
        n_dev = max(1, int(round(n * dev_fraction)))
        purge = horizon - 1
        n_fit = n - n_dev - purge
        if n_fit < 1:
            raise InformationSetError(
                f"{n} row(s) cannot hold a {n_dev}-row DEV block and a {purge}-row purge"
            )
        fit = SupervisedData(
            X=train.X.iloc[:n_fit], y=train.y.iloc[:n_fit], feature_bar=train.feature_bar.iloc[:n_fit], step=horizon
        )
        dev_slice = slice(n - n_dev, n)
        dev = EvaluationContext(
            X=train.X.iloc[dev_slice],
            y=train.y.iloc[dev_slice],
            series=self.returns.loc[self.returns.index <= self.refit_origin],
            close=self.frame["close"].loc[self.frame.index <= self.refit_origin],
            feature_bar=train.feature_bar.iloc[dev_slice],
            train_end=fit.X.index.max(),
            design=train.X,
        )
        return _training_set(fit, self.frame["close"]), dev

    def declared_extent(self) -> dict:
        return {
            "fold": self.fold.index,
            "window": self.window.label,
            "extent_start": self.extent_start.isoformat(),
            "first_settled_bar": self.first_settled_bar.isoformat(),
            "refit_origin": self.refit_origin.isoformat(),
            "last_bar": self.frame.index[-1].isoformat(),
        }


def _training_set(data: SupervisedData, close: pd.Series) -> TrainingSet:
    return TrainingSet(
        X=data.X,
        y=data.y,
        series=data.y.copy(),
        close=close.loc[data.X.index],
        feature_bar=data.feature_bar,
    )


def information_set(
    frame: pd.DataFrame,
    fold: RefitFold,
    window: WindowSpec,
    *,
    max_horizon: int,
    warmup_bars: int,
    specs: list[FeatureSpec] | None = None,
) -> InformationSet:
    """Slice the raw bars this (fold, window) may read, and build features from them.

    The slice runs to the last bar any target in the fold needs. Bars after an
    origin are present so that targets can be scored; models reach them only
    through the contracts' origin-bounded accessors, and the adversary suite
    poisons them to prove it.
    """
    specs = list(specs) if specs is not None else list(default_specs())
    minimum = max(spec.min_history for spec in specs)
    if warmup_bars < minimum:
        raise InformationSetError(
            f"warmup_bars={warmup_bars} is shorter than the longest feature history ({minimum})"
        )

    index = pd.DatetimeIndex(frame.index)
    refit_position = int(index.get_loc(fold.refit_origin))
    last_needed = int(index.get_loc(fold.origins[-1])) + max_horizon
    if last_needed > len(index) - 1:
        raise InformationSetError(f"fold {fold.index} needs bars beyond the end of the data")

    start = extent_start_position(
        refit_position, window, max_horizon=max_horizon, warmup_bars=warmup_bars
    )
    extent = frame.iloc[start : last_needed + 1]
    return InformationSet(
        fold=fold,
        window=window,
        extent_start=extent.index[0],
        frame=extent,
        features=build_feature_frame(extent, specs),
        returns=log_returns(extent),
        warmup_bars=warmup_bars,
    )


def information_fingerprint(info: InformationSet, horizon: int) -> str:
    """A hash of everything a model at this (fold, window, horizon) can see.

    Used by the adversaries: poisoning a bar outside the declared extent must
    leave it unchanged.
    """
    import hashlib

    train = info.training_set(horizon)
    parts = [
        train.X.to_numpy(dtype=float).tobytes(),
        train.y.to_numpy(dtype=float).tobytes(),
        np.asarray(info.features.loc[info.features.index <= info.refit_origin].to_numpy(dtype=float)).tobytes(),
    ]
    return hashlib.sha256(b"".join(parts)).hexdigest()


__all__ = [
    "InformationSet",
    "InformationSetError",
    "extent_start_position",
    "information_fingerprint",
    "information_set",
]
