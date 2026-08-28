"""Assembling features into a supervised learning problem, without leakage.

Two steps, deliberately separated:

1. :func:`build_feature_frame` computes each spec over the market frame. Every
   column is causal -- row ``D`` uses bars ``<= D`` -- but row ``D`` *does*
   include bar ``D``'s own close.

2. :func:`to_supervised` shifts the feature frame forward by the forecast step
   so that the row used to predict bar ``T`` is the feature row from bar
   ``T - step``. This is what stops the model from seeing the close it is being
   asked to predict.

Skipping step 2 is exactly the defect in the pre-Track-A pipeline: it fed
``roll_mean_ret_7[D]``, ``ema_7[D]`` and ``sma_14[D]`` -- all of which contain
``close[D]`` -- into a model whose target was ``close[D]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..timebase import ForecastOrigin, to_utc_index
from .spec import FeatureSpec, log_price, simple_returns


@dataclass(frozen=True)
class SupervisedData:
    """An aligned design matrix and target, plus the origin of each row.

    ``feature_bar[T]`` records which bar the predictors in row ``T`` came from.
    Carrying it explicitly means the alignment can be audited after the fact
    instead of being an unstated assumption about a shift.
    """

    X: pd.DataFrame
    y: pd.Series
    feature_bar: pd.Series
    step: int

    def __post_init__(self) -> None:
        if not self.X.index.equals(self.y.index):
            raise ValueError("X and y must share an index")
        if len(self.X) == 0:
            raise ValueError("supervised data is empty; check min_history and window sizes")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def before(self, bar: pd.Timestamp) -> "SupervisedData":
        """Rows whose *target* bar is at or before ``bar``."""
        mask = self.X.index <= bar
        return SupervisedData(
            X=self.X[mask], y=self.y[mask], feature_bar=self.feature_bar[mask], step=self.step
        )

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> "SupervisedData":
        mask = (self.X.index >= start) & (self.X.index <= end)
        return SupervisedData(
            X=self.X[mask], y=self.y[mask], feature_bar=self.feature_bar[mask], step=self.step
        )


def build_feature_frame(
    frame: pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    dropna: bool = False,
) -> pd.DataFrame:
    """Compute every spec over ``frame``. Row ``D`` uses bars ``<= D``.

    Features are always computed over the longest history available rather than
    over a training slice, because recursive features (EMA) are seeded at the
    first observation and would otherwise differ between fit and predict. This
    is safe: computing over more *past* data does not create look-ahead.
    """
    if not specs:
        raise ValueError("no feature specs given")

    names = [spec.name for spec in specs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate feature names: {sorted(duplicates)}")

    columns = {spec.name: spec.compute(frame) for spec in specs}
    out = pd.DataFrame(columns, index=frame.index)
    out.index.name = "date"
    return out.dropna() if dropna else out


def target_log_price(frame: pd.DataFrame) -> pd.Series:
    """Default modelling target: log close.

    Log space is what the existing Prophet + residual decomposition operates in,
    and it makes multiplicative price dynamics additive.
    """
    return log_price(frame)


def target_simple_return(frame: pd.DataFrame) -> pd.Series:
    return simple_returns(frame)


def to_supervised(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    step: int = 1,
    dropna: bool = True,
) -> SupervisedData:
    """Align features to a target ``step`` bars ahead.

    The row used to predict bar ``T`` is the feature row computed at bar
    ``T - step``. With ``step=1`` -- a one-day-ahead forecast -- the predictors
    for Tuesday are everything knowable at Monday's close.

    ``step`` must be at least 1. A zero shift would let the model see the bar it
    is predicting, which is the leak this function exists to prevent.
    """
    if step < 1:
        raise ValueError(
            f"step must be >= 1, got {step}; a zero shift lets features see their own target bar"
        )

    features = features.copy()
    features.index = to_utc_index(features.index)
    target = target.copy()
    target.index = to_utc_index(target.index)

    shifted = features.shift(step)
    feature_bar = pd.Series(features.index, index=features.index).shift(step)

    aligned_index = shifted.index.intersection(target.index)
    X = shifted.loc[aligned_index]
    y = target.loc[aligned_index]
    bars = feature_bar.loc[aligned_index]

    if dropna:
        keep = X.notna().all(axis=1) & y.notna() & bars.notna()
        X, y, bars = X[keep], y[keep], bars[keep]

    return SupervisedData(X=X, y=y, feature_bar=bars.rename("feature_bar"), step=step)


def future_feature_row(
    history: pd.DataFrame,
    specs: list[FeatureSpec],
    origin: ForecastOrigin,
) -> pd.DataFrame:
    """The single predictor row available at ``origin`` for forecasting ahead.

    Returns the feature values computed at the origin bar -- the most recent
    information legitimately available when the forecast is made.
    """
    observable = history.loc[history.index <= origin.last_observed_bar]
    if observable.empty:
        raise ValueError(f"no observations at or before {origin.last_observed_bar}")

    computed = build_feature_frame(observable, specs)
    row = computed.iloc[[-1]]
    if row.isna().any(axis=None):
        missing = [c for c in row.columns if bool(row[c].isna().iloc[0])]
        raise ValueError(
            f"insufficient history at {origin.last_observed_bar.date()} for features: {missing}"
        )
    return row


def required_history(specs: list[FeatureSpec], step: int = 1) -> int:
    """Bars of history needed before the first usable supervised row exists."""
    return max(spec.min_history for spec in specs) + step


def summarise_features(features: pd.DataFrame) -> pd.DataFrame:
    """Coverage and dispersion per column. A constant column is a dead feature."""
    return pd.DataFrame(
        {
            "non_null": features.notna().sum(),
            "coverage": features.notna().mean(),
            "mean": features.mean(numeric_only=True),
            "std": features.std(numeric_only=True),
            "is_constant": features.std(numeric_only=True).fillna(0.0).eq(0.0),
        }
    )


def assert_causal(
    frame: pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    cut: int = -30,
    rng_seed: int = 0,
) -> None:
    """Empirically verify that no feature depends on data after its own bar.

    Perturbs every bar after ``cut`` and asserts that feature values at or before
    ``cut`` are unchanged. This catches accidental use of centred windows,
    backward fills, or a negative shift, none of which are visible by reading a
    column name.
    """
    rng = np.random.default_rng(rng_seed)
    cut_bar = frame.index[cut]

    baseline = build_feature_frame(frame, specs).loc[:cut_bar]

    perturbed = frame.copy()
    tail = perturbed.index > cut_bar
    perturbed.loc[tail, "close"] *= 1.0 + rng.normal(scale=0.25, size=int(tail.sum()))
    perturbed.loc[tail, "close"] = perturbed.loc[tail, "close"].abs().clip(lower=1e-6)
    if "volume" in perturbed.columns:
        perturbed.loc[tail, "volume"] *= 5.0

    after = build_feature_frame(perturbed, specs).loc[:cut_bar]

    for column in baseline.columns:
        if not baseline[column].equals(after[column]):
            first = (baseline[column] != after[column]).idxmax()
            raise AssertionError(
                f"feature {column!r} is not causal: its value at {first} changed when "
                f"data after {cut_bar} was modified"
            )


__all__ = [
    "SupervisedData",
    "assert_causal",
    "build_feature_frame",
    "future_feature_row",
    "required_history",
    "summarise_features",
    "target_log_price",
    "target_simple_return",
    "to_supervised",
]
