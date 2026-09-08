"""Three partitions, one budget, and a rule about which one may be looked at.

The zoo has forty models and a thousand training rows. In that regime the
easiest way to produce an impressive-looking result is not to cheat on a
timestamp -- it is to choose an architecture, a learning rate or an early-stop
epoch by watching the number you are about to report. Nothing about that is
detectable in the output; the model is causal, the folds are clean, and the
score is inflated anyway.

So the data is cut three ways and the rules are structural:

``TRAIN``     parameters are estimated here, and nowhere else.
``DEV``       every choice a human or a loop makes is made here: early stopping,
              the at-most-three-configuration selection Phase 16 permits.
``HOLDOUT``   scored exactly once, at the end. Never read while choosing
              anything.

The **budget** is the last ``train_rows`` rows of ``TRAIN``. Deterministic,
contiguous, and the most recent data available before ``DEV`` begins -- which is
what anyone with a thousand-row compute limit would actually use. It is a slice,
not a sample: there is no seed, no shuffle, and re-running produces bit-identical
rows.

Contiguity matters more than it looks. Randomly sampling 1,000 rows from twelve
years would hand a sequence model a shuffled series and let a tabular model
interpolate between neighbours it should have had to extrapolate past.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.pipeline import build_feature_frame, to_supervised
from ..features.spec import FeatureSpec, default_specs, log_returns
from .contracts import EvaluationContext, TrainingSet


class PartitionError(ValueError):
    """The requested partition cannot be honoured by the data supplied."""


@dataclass(frozen=True)
class PartitionSpec:
    """How the series is cut, and how many rows the budget takes.

    Fractions are of the *supervised* rows -- after feature construction has
    consumed its leading history -- so the split does not silently move when a
    feature with a longer window is added.
    """

    train_fraction: float = 0.65
    dev_fraction: float = 0.15
    train_rows: int = 1000
    #: One-step-ahead. Every model in the zoo answers the same question, so that
    #: the comparison is between models rather than between horizons.
    step: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise PartitionError("train_fraction must be in (0, 1)")
        if not 0.0 < self.dev_fraction < 1.0:
            raise PartitionError("dev_fraction must be in (0, 1)")
        if self.train_fraction + self.dev_fraction >= 1.0:
            raise PartitionError("train + dev must leave a non-empty holdout")
        if self.train_rows < 1:
            raise PartitionError("train_rows must be positive")
        if self.step < 1:
            raise PartitionError("step must be >= 1; step 0 lets a row see its own target")

    def as_dict(self) -> dict:
        return {
            "train_fraction": self.train_fraction,
            "dev_fraction": self.dev_fraction,
            "train_rows": self.train_rows,
            "step": self.step,
        }


@dataclass(frozen=True)
class Partition:
    """The three index blocks and the budget slice inside TRAIN."""

    train: pd.DatetimeIndex
    dev: pd.DatetimeIndex
    holdout: pd.DatetimeIndex
    budget: pd.DatetimeIndex
    spec: PartitionSpec
    budget_truncated: bool

    def __post_init__(self) -> None:
        # Disjoint and ordered. Checked rather than assumed, because an
        # off-by-one here would put a holdout bar in the training set and
        # nothing downstream would notice.
        if len(self.train) and len(self.dev) and self.train.max() >= self.dev.min():
            raise PartitionError("train overlaps dev")
        if len(self.dev) and len(self.holdout) and self.dev.max() >= self.holdout.min():
            raise PartitionError("dev overlaps holdout")
        if not self.budget.isin(self.train).all():
            raise PartitionError("the training budget contains rows outside TRAIN")
        if len(self.holdout) == 0:
            raise PartitionError("holdout is empty")

    @property
    def train_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.train.max())

    def as_dict(self) -> dict:
        return {
            "spec": self.spec.as_dict(),
            "train": _block(self.train),
            "dev": _block(self.dev),
            "holdout": _block(self.holdout),
            "budget": _block(self.budget),
            "budget_is_contiguous_tail_of_train": True,
            "budget_truncated_to_available_rows": self.budget_truncated,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()
        ).hexdigest()


def _block(index: pd.DatetimeIndex) -> dict:
    if len(index) == 0:
        return {"rows": 0, "start": None, "end": None}
    return {
        "rows": int(len(index)),
        "start": pd.Timestamp(index.min()).isoformat(),
        "end": pd.Timestamp(index.max()).isoformat(),
    }


def partition_index(index: pd.DatetimeIndex, spec: PartitionSpec) -> Partition:
    """Cut a supervised index into TRAIN / DEV / HOLDOUT and take the budget.

    The budget is the **tail** of TRAIN. If TRAIN holds fewer rows than the
    budget asks for, the budget is the whole of TRAIN and the shortfall is
    recorded -- not quietly topped up from DEV.
    """
    total = len(index)
    if total < 3:
        raise PartitionError(f"need at least 3 supervised rows to partition, got {total}")

    n_train = int(total * spec.train_fraction)
    n_dev = int(total * spec.dev_fraction)
    if n_train < 1 or n_dev < 1 or n_train + n_dev >= total:
        raise PartitionError(
            f"{total} rows cannot be split {spec.train_fraction}/{spec.dev_fraction} "
            "into three non-empty blocks"
        )

    train = index[:n_train]
    dev = index[n_train : n_train + n_dev]
    holdout = index[n_train + n_dev :]

    truncated = len(train) < spec.train_rows
    budget = train[-spec.train_rows :] if not truncated else train

    return Partition(
        train=pd.DatetimeIndex(train),
        dev=pd.DatetimeIndex(dev),
        holdout=pd.DatetimeIndex(holdout),
        budget=pd.DatetimeIndex(budget),
        spec=spec,
        budget_truncated=truncated,
    )


@dataclass(frozen=True)
class ZooDataset:
    """The supervised data, its partition, and the three views models get."""

    X: pd.DataFrame
    y: pd.Series
    series: pd.Series
    close: pd.Series
    feature_bar: pd.Series
    partition: Partition
    feature_specs: tuple[FeatureSpec, ...]

    def _subset(self, index: pd.DatetimeIndex) -> TrainingSet:
        return TrainingSet(
            X=self.X.loc[index],
            y=self.y.loc[index],
            # The series is handed whole to series models, which slice it
            # themselves at their own training end; the TrainingSet's own index
            # is what bounds estimation.
            series=self.series.loc[self.series.index <= index.max()],
            close=self.close.loc[self.close.index <= index.max()],
            feature_bar=self.feature_bar.loc[index],
        )

    def training_set(self) -> TrainingSet:
        """The 1,000-row budget. What every model estimates parameters on."""
        return self._subset(self.partition.budget)

    def development_set(self) -> TrainingSet:
        """Where early stopping and the <=3-configuration choice happen."""
        return self._subset(self.partition.dev)

    def development_context(self) -> EvaluationContext:
        """DEV as a scoreable block, for early stopping only."""
        dev = self.partition.dev
        return EvaluationContext(
            X=self.X.loc[dev],
            y=self.y.loc[dev],
            series=self.series.loc[self.series.index <= dev.max()],
            close=self.close.loc[self.close.index <= dev.max()],
            feature_bar=self.feature_bar.loc[dev],
            train_end=self.partition.budget.max(),
        )

    def evaluation_context(self) -> EvaluationContext:
        """HOLDOUT. Scored once, at the end, after every choice is frozen."""
        holdout = self.partition.holdout
        return EvaluationContext(
            X=self.X.loc[holdout],
            y=self.y.loc[holdout],
            series=self.series,
            close=self.close,
            feature_bar=self.feature_bar.loc[holdout],
            train_end=self.partition.dev.max(),
        )

    def as_dict(self) -> dict:
        return {
            "supervised_rows": int(len(self.X)),
            "features": list(self.X.columns),
            "n_features": int(self.X.shape[1]),
            "target": "next_bar_log_return",
            "partition": self.partition.as_dict(),
        }


def build_dataset(
    frame: pd.DataFrame,
    *,
    spec: PartitionSpec | None = None,
    feature_specs: list[FeatureSpec] | None = None,
) -> ZooDataset:
    """Build the causal design matrix, the target, and the partition.

    The target is the **next bar's log return**, not the next price. Forecasting
    a price makes a random walk look excellent and every model look similar,
    because the level dominates; forecasting the return is the question that
    actually distinguishes them, and it is the same target A2's causal
    challenger used.
    """
    spec = spec or PartitionSpec()
    specs = list(feature_specs) if feature_specs is not None else list(default_specs())

    features = build_feature_frame(frame, specs)
    returns = log_returns(frame)
    supervised = to_supervised(features, returns, step=spec.step)

    dataset = ZooDataset(
        X=supervised.X,
        y=supervised.y,
        series=returns,
        close=frame["close"],
        feature_bar=supervised.feature_bar,
        partition=partition_index(pd.DatetimeIndex(supervised.X.index), spec),
        feature_specs=tuple(specs),
    )
    return dataset


def budget_rows_are_deterministic(frame: pd.DataFrame, spec: PartitionSpec) -> bool:
    """Two builds of the same frame select bit-identical budget rows.

    Used by the tests and by the runner's own self-check. A budget chosen by
    sampling would pass this only with a fixed seed; a budget chosen by slicing
    passes it because there is nothing to seed.
    """
    first = build_dataset(frame, spec=spec).training_set()
    second = build_dataset(frame, spec=spec).training_set()
    return (
        first.X.index.equals(second.X.index)
        and bool(np.array_equal(first.X.to_numpy(), second.X.to_numpy()))
        and first.fingerprint() == second.fingerprint()
    )


__all__ = [
    "Partition",
    "PartitionError",
    "PartitionSpec",
    "ZooDataset",
    "budget_rows_are_deterministic",
    "build_dataset",
    "partition_index",
]
