"""The walk-forward engine every model is scored through."""

from .engine import (
    BacktestResult,
    FoldOutcome,
    has_useful_skill,
    rank_models,
    run_walk_forward,
)
from .splits import Fold, WalkForwardSplitter, assert_folds_are_disjoint

__all__ = [
    "BacktestResult",
    "Fold",
    "FoldOutcome",
    "WalkForwardSplitter",
    "assert_folds_are_disjoint",
    "has_useful_skill",
    "rank_models",
    "run_walk_forward",
]
