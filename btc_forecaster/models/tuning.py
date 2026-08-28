"""Nested hyperparameter selection, inside the training history only.

The rule (A2.3)::

    OUTER TRAIN HISTORY
        |-- inner train / validation splits -> pick config
        |-- freeze config
        v
    refit on the whole outer training history
        v
    OUTER TEST FOLD   (never seen during selection)

Because :meth:`ForecastModel.fit` receives a :class:`TrainingWindow` and nothing
else, a model that tunes inside ``_fit`` *cannot* see the outer test window --
it was never handed it. That is a structural guarantee rather than a discipline,
and it is why tuning lives here instead of in the backtest loop.

Two things are still easy to get wrong inside the training history, and both are
handled explicitly:

* **Inner validation must be later than inner training.** A random k-fold split
  would train on 2025 to validate on 2021. Splits are expanding and forward.
* **Inner validation must be embargoed.** Features built from rolling windows
  straddle the inner boundary just as they do the outer one, so the same gap is
  applied.

The search budget is deliberately small (A2.24). This is a laptop research
system, and the honest trade is many folds against few configurations, not the
reverse: a large search evaluated on few folds mostly discovers which
configuration best fits the noise in the validation window.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InnerSplit:
    """One train/validation split strictly inside an outer training history."""

    index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    embargo: int

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start + 1

    @property
    def n_val(self) -> int:
        return self.val_end - self.val_start + 1

    def to_dict(self) -> dict:
        return {
            "split": self.index,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "embargo": self.embargo,
        }


def inner_validation_splits(
    n_rows: int,
    *,
    n_splits: int = 3,
    val_bars: int = 60,
    embargo: int = 0,
    min_train_bars: int = 250,
) -> list[InnerSplit]:
    """Expanding forward splits over ``n_rows`` of training data.

    The validation windows are placed at the *end* of the history and walk
    backwards, so the most recent -- and most representative -- data is always
    used for selection. Raises if the history cannot support even one split,
    rather than silently returning fewer than asked for.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if val_bars < 1:
        raise ValueError("val_bars must be >= 1")

    needed = min_train_bars + embargo + val_bars
    if n_rows < needed:
        raise ValueError(
            f"need at least {needed} rows for one inner split "
            f"(min_train={min_train_bars} + embargo={embargo} + val={val_bars}), got {n_rows}"
        )

    splits: list[InnerSplit] = []
    for i in range(n_splits):
        val_end = n_rows - 1 - i * val_bars
        val_start = val_end - val_bars + 1
        train_end = val_start - embargo - 1
        if train_end + 1 < min_train_bars or val_start < 0:
            break
        splits.append(
            InnerSplit(
                index=len(splits),
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                embargo=embargo,
            )
        )

    if not splits:
        raise ValueError("no inner validation split could be constructed")
    return list(reversed(splits))


@dataclass(frozen=True)
class CandidateScore:
    """One configuration's inner-validation performance."""

    params: dict[str, Any]
    mean_score: float
    per_split: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "params": dict(self.params),
            "mean_score": self.mean_score,
            "per_split": list(self.per_split),
        }


@dataclass(frozen=True)
class SelectionRecord:
    """What was chosen, on what evidence, and over what window.

    Persisted per outer fold so an audit can confirm selection never saw the
    outer test window, and so a later reader can see how much of the ranking was
    real and how much was noise.
    """

    chosen: dict[str, Any]
    chosen_score: float
    n_candidates: int
    n_splits: int
    splits: tuple[dict, ...]
    ranking: tuple[dict, ...]
    scoring: str
    train_rows: int
    features: tuple[str, ...] = field(default=())
    notes: tuple[str, ...] = field(default=())

    @property
    def score_spread(self) -> float:
        """Gap between the best and worst candidate.

        A small spread means the search barely mattered -- worth reporting,
        because a tuned model whose configurations are indistinguishable has not
        earned the compute it took to find one.
        """
        if len(self.ranking) < 2:
            return 0.0
        return float(self.ranking[-1]["mean_score"] - self.ranking[0]["mean_score"])

    def to_dict(self) -> dict:
        return {
            "chosen": dict(self.chosen),
            "chosen_score": self.chosen_score,
            "n_candidates": self.n_candidates,
            "n_splits": self.n_splits,
            "splits": [dict(s) for s in self.splits],
            "top_5": [dict(r) for r in self.ranking[:5]],
            "score_spread": self.score_spread,
            "scoring": self.scoring,
            "train_rows": self.train_rows,
            "features": list(self.features),
            "notes": list(self.notes),
        }


def sample_parameter_grid(
    space: Mapping[str, Sequence[Any]],
    *,
    n_candidates: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Draw ``n_candidates`` distinct configurations from a discrete space.

    Random sampling rather than an exhaustive grid: with a fixed budget, random
    search covers each individual dimension far better than a coarse grid does,
    because a grid spends its budget on combinations of dimensions that mostly
    do not matter (Bergstra & Bengio 2012).

    Deterministic given ``seed``, so a fold's search is reproducible.
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")

    total = 1
    for values in space.values():
        total *= len(values)

    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []

    attempts = 0
    while len(out) < min(n_candidates, total) and attempts < n_candidates * 50:
        attempts += 1
        candidate = {key: values[int(rng.integers(len(values)))] for key, values in space.items()}
        key = tuple(sorted(candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def select_by_inner_validation(
    data_len: int,
    candidates: list[dict[str, Any]],
    fit_score: Callable[[dict[str, Any], InnerSplit], float],
    *,
    n_splits: int = 3,
    val_bars: int = 60,
    embargo: int = 0,
    min_train_bars: int = 250,
    scoring: str = "inner-validation MAE (lower is better)",
    features: tuple[str, ...] = (),
) -> SelectionRecord:
    """Score every candidate on every inner split and keep the best mean.

    ``fit_score`` receives a configuration and a split and returns a loss --
    lower is better. It must fit on ``split.train_*`` and evaluate on
    ``split.val_*`` and nothing else; :mod:`tests.test_tuning` verifies the
    indices it is handed never reach past the training history.

    A candidate that fails on any split is dropped with a note rather than
    aborting the search: a configuration that does not converge is evidence
    about that configuration.
    """
    if not candidates:
        raise ValueError("no candidate configurations given")

    splits = inner_validation_splits(
        data_len,
        n_splits=n_splits,
        val_bars=val_bars,
        embargo=embargo,
        min_train_bars=min_train_bars,
    )

    scored: list[CandidateScore] = []
    notes: list[str] = []

    for params in candidates:
        losses: list[float] = []
        failed = False
        for split in splits:
            try:
                losses.append(float(fit_score(params, split)))
            except Exception as exc:
                notes.append(f"dropped {params}: {type(exc).__name__}: {exc}")
                failed = True
                break
        if failed or not losses or not np.all(np.isfinite(losses)):
            continue
        scored.append(
            CandidateScore(
                params=params,
                mean_score=float(np.mean(losses)),
                per_split=tuple(float(x) for x in losses),
            )
        )

    if not scored:
        raise RuntimeError(
            f"no candidate configuration produced a finite score. Notes: {notes[:3]}"
        )

    scored.sort(key=lambda c: c.mean_score)
    best = scored[0]

    if len(scored) > 1 and np.isclose(scored[0].mean_score, scored[-1].mean_score, rtol=1e-3):
        notes.append(
            "Candidate scores are within 0.1% of each other; the search did not "
            "meaningfully discriminate between configurations."
        )

    return SelectionRecord(
        chosen=dict(best.params),
        chosen_score=best.mean_score,
        n_candidates=len(candidates),
        n_splits=len(splits),
        splits=tuple(s.to_dict() for s in splits),
        ranking=tuple(c.to_dict() for c in scored),
        scoring=scoring,
        train_rows=data_len,
        features=features,
        notes=tuple(notes),
    )


def assert_selection_precedes(record: SelectionRecord, outer_test_start: pd.Timestamp) -> None:
    """Sanity check for an audit: selection used fewer rows than the training history.

    Weak by design -- the real guarantee is structural (a model never receives
    the outer test window). This catches an accounting mistake in a bespoke
    tuning loop that bypasses :func:`select_by_inner_validation`.
    """
    for split in record.splits:
        if split["n_train"] + split["n_val"] > record.train_rows:
            raise AssertionError(
                f"inner split spans {split['n_train'] + split['n_val']} rows but the "
                f"training history has only {record.train_rows}; selection reached "
                f"past {outer_test_start}"
            )


__all__ = [
    "CandidateScore",
    "InnerSplit",
    "SelectionRecord",
    "assert_selection_precedes",
    "inner_validation_splits",
    "sample_parameter_grid",
    "select_by_inner_validation",
]
