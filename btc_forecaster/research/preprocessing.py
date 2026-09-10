"""Scaling, fitted on the training rows and nowhere else.

The leak this module exists to prevent is the quietest one in applied machine
learning. Fit a `StandardScaler` on train and evaluation together and every
prediction still respects its timestamps, every fold is still disjoint, and the
mean and variance of the evaluation block have nonetheless been whispered into
the training data. Nothing looks wrong. Every metric improves slightly.

So a scaler here is fitted once, on the budget rows, and afterwards only
transforms. :func:`fit_scaler` is the only constructor and it takes a training
matrix; there is no path that reaches an evaluation row while fitting.

Implemented on numpy rather than scikit-learn on purpose: statistical models
need scaling too, and `statsmodels` is a core dependency while `sklearn` is an
extra. A repository where standardisation requires an optional package is a
repository where the statistical family silently gets a different contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import Preprocessing


class ScalerNotFittedError(RuntimeError):
    """Raised when a scaler is asked to transform before it has seen training data."""


@dataclass(frozen=True)
class Scaler:
    """Centre and scale, with the statistics that produced them recorded.

    ``kind`` is the declared :class:`Preprocessing` mode. ``NONE`` and
    ``MODEL_NATIVE`` produce an identity transform, so a model that handles its
    own scaling still goes through the same call path and the same recorded
    fingerprint.
    """

    kind: Preprocessing
    centre: np.ndarray
    spread: np.ndarray
    columns: tuple[str, ...]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if list(X.columns) != list(self.columns):
            raise ValueError(
                "column mismatch: the scaler was fitted on "
                f"{list(self.columns)} and asked to transform {list(X.columns)}"
            )
        if self.kind in (Preprocessing.NONE, Preprocessing.MODEL_NATIVE):
            return X.copy()
        values = (X.to_numpy(dtype=float) - self.centre) / self.spread
        return pd.DataFrame(values, index=X.index, columns=X.columns)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "columns": list(self.columns),
            "centre": [float(v) for v in self.centre],
            "spread": [float(v) for v in self.spread],
        }


def fit_scaler(X: pd.DataFrame, kind: Preprocessing) -> Scaler:
    """Fit on ``X`` -- which must be the training rows, and only those.

    ``STANDARDIZED`` uses mean and standard deviation. ``ROBUST`` uses median
    and inter-quartile range, which matters here: daily crypto returns carry
    genuine outliers that a standard deviation absorbs into the scale, shrinking
    every ordinary day toward zero.

    A zero or non-finite spread is replaced by 1.0. A constant feature carries
    no information, and dividing by its zero spread would turn it into infinity
    and take the rest of the matrix with it.
    """
    values = X.to_numpy(dtype=float)
    columns = tuple(X.columns)

    if kind in (Preprocessing.NONE, Preprocessing.MODEL_NATIVE):
        return Scaler(
            kind=kind,
            centre=np.zeros(values.shape[1]),
            spread=np.ones(values.shape[1]),
            columns=columns,
        )

    if kind is Preprocessing.STANDARDIZED:
        centre = values.mean(axis=0)
        spread = values.std(axis=0, ddof=0)
    elif kind is Preprocessing.ROBUST:
        centre = np.median(values, axis=0)
        q75, q25 = np.percentile(values, [75, 25], axis=0)
        spread = q75 - q25
    else:  # pragma: no cover -- the enum is exhaustive
        raise ValueError(f"unknown preprocessing mode {kind!r}")

    spread = np.where(np.isfinite(spread) & (spread > 1e-12), spread, 1.0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    return Scaler(kind=kind, centre=centre, spread=spread, columns=columns)


def scaled_matrices(
    train_X: pd.DataFrame,
    other_X: pd.DataFrame,
    kind: Preprocessing,
) -> tuple[np.ndarray, np.ndarray, Scaler]:
    """The common path: fit on train, transform both, hand back the scaler.

    Returning the scaler is not a convenience -- it goes into the model's
    serialized state, so a reloaded model transforms new rows with the
    statistics it was fitted under rather than re-deriving them.
    """
    scaler = fit_scaler(train_X, kind)
    return (
        scaler.transform(train_X).to_numpy(dtype=float),
        scaler.transform(other_X).to_numpy(dtype=float),
        scaler,
    )


__all__ = [
    "Scaler",
    "ScalerNotFittedError",
    "fit_scaler",
    "scaled_matrices",
]
