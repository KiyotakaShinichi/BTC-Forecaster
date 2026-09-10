"""Walk-forward leakage adversaries: poison what a model must not read.

The A6 suite proved its one partition could not see the future. Walk-forward
multiplies the ways to get it wrong -- a refit origin one bar late, a window
that reaches past its extent, an h-bar target whose last rows complete after
the origin -- so each fold is attacked directly, through the same
:func:`~.evaluator.forecast_fold` the benchmark itself uses.

Five adversaries, each with the outcome it must produce:

``future_row``        every raw bar after a cut origin corrupted  -> forecasts at
                      and before the cut unchanged
``future_target``     every realised target replaced              -> no forecast
                      changes
``future_feature``    feature cells after the cut corrupted       -> forecasts at
                      and before the cut unchanged
``out_of_window``     the bar just before the fold's extent       -> no forecast
                      changes (rolling windows only)
``valid_past``        a bar inside the training window corrupted  -> forecasts
                      *do* change

The last is the one that makes the other four mean something. A detector that
reports "nothing moved" for every input detects nothing; if corrupting data the
model is entitled to changes nothing, the unchanged results above are passing
for the wrong reason. Models that ignore their data by construction -- the
naive forecast -- are recorded as insensitive by design, not silently passed.

A model passes only if every applicable adversary produced its expected
outcome. Comparisons are exact: bit-identical, not close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import registry
from ..contracts import EvaluationContext
from .config import BASELINE, WalkForwardConfig, WindowSpec
from .evaluator import JobSpec, forecast_fold, strategy_for
from .origins import OriginSchedule
from .windows import information_set

FUTURE_ROW = "future_row"
FUTURE_TARGET = "future_target"
FUTURE_FEATURE = "future_feature"
OUT_OF_WINDOW = "out_of_window"
VALID_PAST = "valid_past"
ADVERSARIES: tuple[str, ...] = (FUTURE_ROW, FUTURE_TARGET, FUTURE_FEATURE, OUT_OF_WINDOW, VALID_PAST)

#: Models whose forecast does not depend on data at all.
CONSTANT_BY_DESIGN: frozenset[str] = frozenset({BASELINE})

#: How far inside the training window the valid-past adversary strikes.
VALID_PAST_OFFSET = 10


@dataclass(frozen=True)
class Check:
    name: str
    #: "unchanged" or "changes".
    expectation: str
    changed: bool
    applicable: bool = True
    note: str | None = None

    @property
    def passed(self) -> bool:
        if not self.applicable:
            return True
        return self.changed == (self.expectation == "changes")

    def as_dict(self) -> dict:
        return {
            "adversary": self.name,
            "expectation": self.expectation,
            "changed": self.changed,
            "applicable": self.applicable,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True)
class AdversaryReport:
    model_id: str
    horizon: int
    window: str
    fold: int
    cut_origin: pd.Timestamp
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "horizon": self.horizon,
            "window": self.window,
            "fold": self.fold,
            "cut_origin": self.cut_origin.isoformat(),
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }


def _poison_bars(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out.loc[mask, "close"] = out.loc[mask, "close"] * 5.0
    if "volume" in out.columns:
        out.loc[mask, "volume"] = out.loc[mask, "volume"] * 7.0
    return out


def run_adversaries(
    model_id: str,
    frame: pd.DataFrame,
    schedule: OriginSchedule,
    config: WalkForwardConfig,
    *,
    horizon: int,
    window: WindowSpec,
    fold_index: int,
) -> AdversaryReport:
    """Attack one fold of one (model, horizon, window) five ways."""
    fold = schedule.folds[fold_index]
    spec = JobSpec(model_id, horizon, window)
    strategy = strategy_for(registry.build(model_id))

    def forecast(source: pd.DataFrame, hook=None) -> tuple[pd.DatetimeIndex, np.ndarray]:
        info = information_set(
            source, fold, window, max_horizon=config.max_horizon, warmup_bars=config.warmup_bars
        )
        result = forecast_fold(spec, info, config, strategy=strategy, context_hook=hook)
        return result.origins, result.predicted

    origins, clean = forecast(frame)
    cut = fold.origins[len(fold.origins) // 2]
    upto = np.asarray(origins <= cut)
    index = pd.DatetimeIndex(frame.index)
    checks: list[Check] = []

    # 1. Every raw bar after the cut.
    _, dirty = forecast(_poison_bars(frame, np.asarray(index > cut)))
    checks.append(Check(FUTURE_ROW, "unchanged", not np.array_equal(clean[upto], dirty[upto])))

    # 2. Every realised target the model is handed.
    def poison_targets(context: EvaluationContext) -> EvaluationContext:
        return EvaluationContext(
            X=context.X, y=context.y * 0.0 + 99.0, series=context.series, close=context.close,
            feature_bar=context.feature_bar, train_end=context.train_end, design=context.design,
        )

    _, dirty = forecast(frame, poison_targets)
    checks.append(Check(FUTURE_TARGET, "unchanged", not np.array_equal(clean, dirty)))

    # 3. Feature cells computed after the cut, in both the rows and the design.
    #    Design rows are indexed by target bar, so "computed after the cut" is
    #    every row after the cut row's own target bar -- found by position, not
    #    by adding a calendar offset, which a missing bar would shift.
    def poison_features(context: EvaluationContext) -> EvaluationContext:
        X = context.X.copy()
        late = np.asarray(pd.DatetimeIndex(context.feature_bar) > cut)
        X.iloc[late, 0] = 1e6
        design = None
        if context.design is not None:
            cut_target = context.X.index[context.origins.get_loc(cut)]
            design = context.design.copy()
            design.loc[design.index > cut_target, design.columns[0]] = 1e6
        return EvaluationContext(
            X=X, y=context.y, series=context.series, close=context.close,
            feature_bar=context.feature_bar, train_end=context.train_end, design=design,
        )

    _, dirty = forecast(frame, poison_features)
    checks.append(Check(FUTURE_FEATURE, "unchanged", not np.array_equal(clean[upto], dirty[upto])))

    # 4. The bar just before the fold's declared extent.
    info = information_set(frame, fold, window, max_horizon=config.max_horizon, warmup_bars=config.warmup_bars)
    extent = int(index.get_loc(info.extent_start))
    if extent >= 1:
        _, dirty = forecast(_poison_bars(frame, np.asarray(index == index[extent - 1])))
        checks.append(Check(OUT_OF_WINDOW, "unchanged", not np.array_equal(clean, dirty)))
    else:
        checks.append(
            Check(
                OUT_OF_WINDOW, "unchanged", False, applicable=False,
                note="the extent starts at the first bar; nothing precedes it",
            )
        )

    # 5. The valid past: the last bars up to and including the refit origin,
    #    shifted and left shifted. A persistent shift, not a one-bar spike: a
    #    spike raises one return and lowers the next by the same amount, and a
    #    drift estimate -- a mean return, i.e. a telescoping sum -- is exactly
    #    blind to it. That is how this adversary first reported the drift
    #    baseline as insensitive. A shift that persists to the origin moves every
    #    estimator the model could be using, so "nothing changed" means something.
    refit = int(index.get_loc(fold.refit_origin))
    shifted = (np.arange(len(index)) >= refit - VALID_PAST_OFFSET) & (np.arange(len(index)) <= refit)
    _, dirty = forecast(_poison_bars(frame, shifted))
    if model_id in CONSTANT_BY_DESIGN:
        checks.append(
            Check(
                VALID_PAST, "changes", not np.array_equal(clean, dirty), applicable=False,
                note="forecasts a constant by construction; insensitive by design",
            )
        )
    else:
        checks.append(Check(VALID_PAST, "changes", not np.array_equal(clean, dirty)))

    return AdversaryReport(
        model_id=model_id,
        horizon=horizon,
        window=window.label,
        fold=fold.index,
        cut_origin=cut,
        checks=tuple(checks),
    )


__all__ = [
    "ADVERSARIES",
    "CONSTANT_BY_DESIGN",
    "FUTURE_FEATURE",
    "FUTURE_ROW",
    "FUTURE_TARGET",
    "OUT_OF_WINDOW",
    "VALID_PAST",
    "AdversaryReport",
    "Check",
    "run_adversaries",
]
