"""Forecast origins, refit folds and chronological blocks.

One schedule per benchmark. Every model, every horizon and every training
window is scored at exactly the same daily origins, because a comparison across
different origins is partly a comparison of the origins -- and the easiest way
to make one model look better than another is to let it be scored somewhere
kinder.

Three nested partitions of the same origins:

``origins``
    Consecutive bars, from the first bar at which the largest fixed window can
    be filled with fully realised targets, to the last bar that still has the
    longest horizon's target inside the data. Common to every horizon, so a
    30-day result and a 1-day result describe the same stretch of history.

``folds``
    Contiguous runs of origins. A model is re-estimated at the first origin of
    each fold and its parameters are frozen for the rest of it; the forecast
    origin still advances one bar at a time, so state and features see each new
    bar, but estimation does not. The fold is also the unit of the stability
    gate.

``blocks``
    Contiguous thirds (at minimum) of the same origins -- early, middle, late --
    for asking whether a result belongs to one period.

Horizons count **bars**, not calendar days. The daily series has occasional
missing bars, and "the close three bars later" is well defined where "the close
three days later" is sometimes not there.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RefitFold:
    """A contiguous run of origins sharing one set of estimated parameters."""

    index: int
    #: Parameters are estimated with data through this bar and no later.
    refit_origin: pd.Timestamp
    #: The daily origins scored with those frozen parameters. The first is the
    #: refit origin itself.
    origins: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if len(self.origins) == 0:
            raise ValueError(f"fold {self.index} has no origins")
        if self.origins[0] != self.refit_origin:
            raise ValueError(f"fold {self.index} does not start at its refit origin")

    def as_dict(self) -> dict:
        return {
            "fold": self.index,
            "refit_origin": self.refit_origin.isoformat(),
            "last_origin": self.origins[-1].isoformat(),
            "n_origins": len(self.origins),
        }


def _joined(parts: list[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Concatenate indexes without dropping their timezone."""
    if not parts:
        return pd.DatetimeIndex([])
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def block_names(n_blocks: int) -> tuple[str, ...]:
    if n_blocks == 3:
        return ("early", "middle", "late")
    return tuple(f"block-{i + 1}" for i in range(n_blocks))


@dataclass(frozen=True)
class OriginSchedule:
    """The origins of one benchmark and the two partitions of them."""

    origins: pd.DatetimeIndex
    folds: tuple[RefitFold, ...]
    blocks: tuple[pd.DatetimeIndex, ...]
    max_horizon: int

    def __post_init__(self) -> None:
        if not self.origins.is_monotonic_increasing or not self.origins.is_unique:
            raise ValueError("origins must be strictly increasing")
        # `append`, not `np.concatenate(... .values)`: `.values` on a tz-aware
        # index is naive UTC, and a naive index never equals an aware one -- so
        # the check would reject every schedule, including correct ones.
        if not _joined([f.origins for f in self.folds]).equals(self.origins):
            raise ValueError("folds must partition the origins contiguously and in order")
        if not _joined(list(self.blocks)).equals(self.origins):
            raise ValueError("blocks must partition the origins contiguously and in order")

    @property
    def block_names(self) -> tuple[str, ...]:
        return block_names(len(self.blocks))

    def fold_of(self) -> pd.Series:
        """Origin -> fold index."""
        return pd.Series(
            np.concatenate([np.full(len(f.origins), f.index) for f in self.folds]),
            index=self.origins,
            name="fold",
        )

    def block_of(self) -> pd.Series:
        """Origin -> block name."""
        names = self.block_names
        return pd.Series(
            np.concatenate([np.full(len(b), names[i], dtype=object) for i, b in enumerate(self.blocks)]),
            index=self.origins,
            name="block",
        )

    def as_dict(self) -> dict:
        return {
            "first_origin": self.origins[0].isoformat(),
            "last_origin": self.origins[-1].isoformat(),
            "n_origins": len(self.origins),
            "max_horizon": self.max_horizon,
            "folds": [f.as_dict() for f in self.folds],
            "blocks": [
                {
                    "block": name,
                    "first_origin": block[0].isoformat(),
                    "last_origin": block[-1].isoformat(),
                    "n_origins": len(block),
                }
                for name, block in zip(self.block_names, self.blocks, strict=True)
            ],
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_origin_position(
    *, warmup_bars: int, training_rows: int, max_horizon: int
) -> int:
    """The earliest bar position at which ``training_rows`` rows fit.

    A training row for horizon ``h`` has a feature bar ``f`` and a target bar
    ``f + h``, and at origin ``p`` only rows with ``f + h <= p`` are realised. A
    window of ``w`` such rows therefore starts at feature bar ``p - h - w + 1``,
    and that bar must itself sit at least ``warmup_bars`` into the data so its
    features are settled. Solving for ``p`` at the longest horizon:

        p >= warmup_bars + w + h - 1
    """
    if min(warmup_bars, training_rows, max_horizon) < 1:
        raise ValueError("warmup_bars, training_rows and max_horizon must all be positive")
    return warmup_bars + training_rows + max_horizon - 1


def build_schedule(
    bars: pd.DatetimeIndex,
    *,
    first_position: int,
    max_horizon: int,
    n_refits: int,
    n_blocks: int,
) -> OriginSchedule:
    """Every origin from ``first_position`` to the last with a full horizon ahead.

    Deterministic: no seed, no sampling. Folds and blocks are contiguous
    near-equal splits of the same origins, sizes differing by at most one.
    """
    bars = pd.DatetimeIndex(bars)
    if not bars.is_monotonic_increasing or not bars.is_unique:
        raise ValueError("bars must be sorted ascending without duplicates")
    if max_horizon < 1:
        raise ValueError("max_horizon must be positive")

    last_position = len(bars) - 1 - max_horizon
    n_origins = last_position - first_position + 1
    needed = max(n_refits, n_blocks)
    if first_position < 0 or n_origins < needed:
        raise ValueError(
            f"need at least {first_position + needed + max_horizon} bars for "
            f"{n_refits} refit folds and {n_blocks} blocks (first origin at bar "
            f"{first_position}, horizon {max_horizon}); got {len(bars)}"
        )

    origins = bars[first_position : last_position + 1]
    positions = np.arange(len(origins))

    folds = tuple(
        RefitFold(index=i, refit_origin=origins[chunk[0]], origins=origins[chunk])
        for i, chunk in enumerate(np.array_split(positions, n_refits))
    )
    blocks = tuple(origins[chunk] for chunk in np.array_split(positions, n_blocks))
    return OriginSchedule(origins=origins, folds=folds, blocks=blocks, max_horizon=max_horizon)


__all__ = [
    "OriginSchedule",
    "RefitFold",
    "block_names",
    "build_schedule",
    "first_origin_position",
]
