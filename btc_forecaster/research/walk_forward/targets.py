"""What is forecast at each horizon, and in what units.

One definition for every horizon, every model and every baseline:

    y_h(t) = ln( close[t + h] / close[t] )

the natural-log return over ``h`` **bars** from the forecast origin ``t`` -- the
last bar whose close is observable when the forecast is made. It is indexed by
the bar that completes it, ``t + h``, which is the convention the A6 contracts
use: a row's index is the bar it predicts, and ``feature_bar`` is the origin.

At ``h = 1`` this is exactly A6's target. At longer horizons it is a different
question, and its errors are larger by construction: a 30-bar return has
roughly thirty times the variance of a 1-bar return, so MAE is not comparable
across horizons and nothing here compares it that way. Skill against the naive
forecast, computed within one horizon, is the comparable quantity.

It is not A2's target. A2 forecast a 31-step price *path* and scored it in USD;
this is a single scale-free return. The two are not re-expressions of each
other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...features.pipeline import SupervisedData, to_supervised

TARGET_NAME = "cumulative_log_return"

#: Written into every manifest, so the unit of every number is on record.
TARGET_DEFINITION = {
    "name": TARGET_NAME,
    "formula": "ln(close[t + h] / close[t])",
    "units": "natural-log return over h bars (dimensionless)",
    "origin": "t, the last bar whose close is observable when the forecast is made",
    "indexed_by": "the target bar t + h; feature_bar records the origin t",
    "horizon_unit": "bars of the daily series, not calendar days",
    "equals_a6_target_at_h1": True,
    "comparable_to_a2": False,
    "why_not_comparable_to_a2": (
        "A2 scored a 31-step price path in USD at step 31. This is one "
        "scale-free return per origin."
    ),
    "cross_horizon_note": (
        "MAE grows with h by construction; compare skill against the naive "
        "forecast within a horizon, never MAE across horizons."
    ),
}


def cumulative_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """``ln(close[T] / close[T - h])``, indexed by the end bar ``T``.

    Positional, so a missing calendar day shifts nothing: ``h`` bars back is
    ``h`` rows back.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return np.log(close.astype(float)).diff(horizon).rename(f"{TARGET_NAME}_h{horizon}")


def horizon_supervised(features: pd.DataFrame, close: pd.Series, horizon: int) -> SupervisedData:
    """Features at origin ``t`` aligned to the return realised at ``t + h``.

    Built on the existing :func:`to_supervised`, which shifts the features by
    ``step`` rows and refuses a zero shift. A row therefore carries features
    computed ``h`` bars before the bar its target ends on, and nothing later.
    """
    return to_supervised(features, cumulative_log_return(close, horizon), step=horizon)


def assert_targets_realised_by(supervised: SupervisedData, origin: pd.Timestamp) -> None:
    """No row may have a target that completes after ``origin``.

    For ``h > 1`` this is the embargo that matters: the last ``h - 1`` feature
    rows before an origin have targets that end *after* it, and training on
    them would hand the model returns from the future it is about to forecast.
    """
    if len(supervised) and supervised.X.index.max() > origin:
        late = supervised.X.index[supervised.X.index > origin]
        raise AssertionError(
            f"{len(late)} training target(s) complete after the origin {origin}; "
            f"first at {late[0]}"
        )


__all__ = [
    "TARGET_DEFINITION",
    "TARGET_NAME",
    "assert_targets_realised_by",
    "cumulative_log_return",
    "horizon_supervised",
]
