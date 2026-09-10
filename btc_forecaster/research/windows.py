"""One sequence contract, shared by every deep model.

A recurrent or convolutional model needs a *window* of the past, not a row, and
building that window is where sequence models leak. The three classic ways:

* an off-by-one that includes the bar being predicted;
* a window assembled after shuffling, so rows from the future sit inside it;
* a target aligned to the wrong end of the window.

None of them raises. All of them produce a model that looks excellent.

So there is exactly one builder, every deep adapter uses it, and it asserts its
own contract on every call: for a target at bar ``T``, the window holds the
``lookback`` causal feature rows ending at the row whose predictors come from
``T - step``. Every timestamp inside the window is strictly before the bar being
predicted, and :func:`assert_window_causality` proves it by comparing the
recorded ``feature_bar`` of each window row against the target bar.

The design matrix is already shifted by :func:`to_supervised` -- row ``T`` holds
predictors computed at ``T - step``. Stacking the last ``L`` such rows therefore
cannot reach forward, and the assertion is a guard against a future edit rather
than against the current code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class WindowError(ValueError):
    """The requested sequence cannot be built causally from the data supplied."""


@dataclass(frozen=True)
class WindowSpec:
    """How much past a sequence model sees.

    ``lookback`` is deliberately small. At n=1,000 a 64-step window costs 6% of
    the training rows to the burn-in and gives every sequence 64x the input
    dimension of a tabular row -- which at this sample size buys variance, not
    signal. 24 is roughly a month of daily bars.
    """

    lookback: int = 24

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise WindowError("lookback must be at least 1")

    def as_dict(self) -> dict:
        return {"lookback": self.lookback}


@dataclass(frozen=True)
class Sequences:
    """Stacked windows, their targets, and the bar each window predicts."""

    #: (n_windows, lookback, n_features)
    X: np.ndarray
    #: (n_windows,)
    y: np.ndarray
    #: the target bar of each window
    index: pd.DatetimeIndex
    #: the latest feature bar inside each window; must be strictly before index
    last_feature_bar: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2])

    @property
    def lookback(self) -> int:
        return int(self.X.shape[1])


def build_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    feature_bar: pd.Series,
    spec: WindowSpec,
) -> Sequences:
    """Stack ``lookback`` causal feature rows per target.

    The first ``lookback - 1`` targets are dropped: there is not enough past to
    build their window, and padding them with zeros would train the model on
    windows that never occur at prediction time.
    """
    if not X.index.equals(y.index):
        raise WindowError("X and y must share an index")
    values = X.to_numpy(dtype=float)
    n_rows, n_features = values.shape
    if n_rows < spec.lookback:
        raise WindowError(
            f"need at least {spec.lookback} rows to build one window, got {n_rows}"
        )

    n_windows = n_rows - spec.lookback + 1
    windows = np.empty((n_windows, spec.lookback, n_features), dtype=float)
    for i in range(n_windows):
        windows[i] = values[i : i + spec.lookback]

    index = pd.DatetimeIndex(X.index[spec.lookback - 1 :])
    targets = y.to_numpy(dtype=float)[spec.lookback - 1 :]
    last_bar = pd.DatetimeIndex(pd.Series(feature_bar).to_numpy()[spec.lookback - 1 :])

    sequences = Sequences(X=windows, y=targets, index=index, last_feature_bar=last_bar)
    assert_window_causality(sequences)
    return sequences


def window_at(X: pd.DataFrame, position: int, spec: WindowSpec) -> np.ndarray:
    """The single window ending at row ``position``, for one-step prediction.

    Used by the adapters at evaluation time, one origin at a time. Slicing
    forward is impossible: the window is ``[position - lookback + 1, position]``
    and a position with insufficient history raises rather than padding.
    """
    if position < spec.lookback - 1:
        raise WindowError(
            f"row {position} has only {position + 1} rows of history; "
            f"a window needs {spec.lookback}"
        )
    values = X.to_numpy(dtype=float)
    return values[position - spec.lookback + 1 : position + 1]


def assert_window_causality(sequences: Sequences) -> None:
    """Every timestamp inside a window is strictly before the bar it predicts.

    Checked on the recorded ``feature_bar`` rather than on the window's position
    in an array, because a positional argument is exactly what an off-by-one
    edit would keep satisfying.
    """
    if len(sequences) == 0:
        return
    offending = sequences.last_feature_bar >= sequences.index
    if bool(np.any(offending)):
        first = int(np.argmax(offending))
        raise WindowError(
            f"window {first} contains feature bar {sequences.last_feature_bar[first]} "
            f"which is not strictly before its target bar {sequences.index[first]}"
        )


def evaluation_windows(
    X_full: pd.DataFrame,
    origins: pd.DatetimeIndex,
    spec: WindowSpec,
) -> np.ndarray:
    """One window per evaluation origin, drawn from the full design matrix.

    ``X_full`` spans training and evaluation, which is what lets the first
    evaluation window reach back into training history rather than starting from
    nothing. That is legitimate -- those rows are the realised past -- and the
    causality assertion still holds because every row in the window is a causal
    feature row whose own bar precedes its target.
    """
    positions = X_full.index.get_indexer(origins)
    if bool((positions < 0).any()):
        raise WindowError("an evaluation origin is not present in the design matrix")
    if int(positions.min()) < spec.lookback - 1:
        raise WindowError(
            "the first evaluation origin has insufficient history for a full window; "
            "the design matrix must include the training rows before it"
        )
    return np.stack([window_at(X_full, int(p), spec) for p in positions])


__all__ = [
    "Sequences",
    "WindowError",
    "WindowSpec",
    "assert_window_causality",
    "build_sequences",
    "evaluation_windows",
    "window_at",
]
