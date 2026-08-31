"""B4.2 / B4.3 — research-only BTC outcome dataset and its provenance.

Targets live in their own file, joined to features only by `forecast_origin`,
and are never computed inside the intelligence feature matrix. That separation
is not tidiness: the moment a target can be read from the same structure that
produces features, someone will eventually derive a feature from it, and no
test will notice because the values will look perfectly ordinary.

The one invariant every target here maintains:

    every price used in a target became available *strictly after* the origin.

`forward_return_h` at origin O is the return from the last close available at O
to the last close available at O+h. The base price is the only quantity in a
target that is known at the origin, and it is the same price a feature would
legitimately see — so it cannot leak anything a feature does not already have.

Track A/A2 targets are untouched. This dataset exists alongside them and shares
no code with them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from .contracts import B4DataError, EvidenceTier, LeakageError, MarketSeries, require_utc
from .market_data import close_as_of, last_available_index

#: Horizons are declared up front and all of them stay visible in results
#: (B4.12) -- reporting only the horizon that looked best is the single easiest
#: way to turn noise into a finding.
DEFAULT_HORIZONS_HOURLY: tuple[str, ...] = ("1h", "6h", "24h", "72h")
DEFAULT_HORIZONS_DAILY: tuple[str, ...] = ("1d", "3d", "7d", "30d")

HORIZON_SECONDS: dict[str, int] = {
    "1h": 3_600,
    "6h": 21_600,
    "24h": 86_400,
    "72h": 259_200,
    "1d": 86_400,
    "3d": 259_200,
    "7d": 604_800,
    "30d": 2_592_000,
}

TARGET_CONTRACT_VERSION = "b4-targets-v1"


def horizon_delta(horizon: str) -> timedelta:
    if horizon not in HORIZON_SECONDS:
        raise B4DataError(f"unknown horizon {horizon!r}")
    return timedelta(seconds=HORIZON_SECONDS[horizon])


@dataclass(frozen=True)
class TargetRow:
    """Outcomes for one origin. `None` where the horizon runs off the data."""

    forecast_origin: datetime
    base_close: float
    base_available_at: datetime
    forward_return: dict[str, float | None]
    forward_abs_return: dict[str, float | None]
    forward_direction: dict[str, int | None]
    realized_volatility: dict[str, float | None]

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "forecast_origin": self.forecast_origin.isoformat(),
            "base_close": self.base_close,
            "base_available_at": self.base_available_at.isoformat(),
        }
        for horizon, value in self.forward_return.items():
            record[f"forward_return_{horizon}"] = value
        for horizon, value in self.forward_abs_return.items():
            record[f"forward_abs_return_{horizon}"] = value
        for horizon, direction in self.forward_direction.items():
            record[f"forward_direction_{horizon}"] = direction
        for horizon, value in self.realized_volatility.items():
            record[f"realized_volatility_{horizon}"] = value
        return record


class TargetManifest(BaseModel):
    """B4.3. Everything needed to rebuild and audit the outcome dataset."""

    model_config = ConfigDict(frozen=True)

    target_contract_version: str = TARGET_CONTRACT_VERSION
    series_id: str
    ticker: str
    provider: str
    timestamp_convention: str
    evidence_tier: EvidenceTier
    source_timezone: str
    bar_period_seconds: int
    series_start: datetime
    series_end: datetime
    series_bar_count: int
    series_fingerprint: str
    origin_start: datetime
    origin_end: datetime
    origin_count: int
    horizons: tuple[str, ...]
    target_definitions: dict[str, str]
    row_count: int
    result_hash: str
    git_sha: str
    created_at: datetime


TARGET_DEFINITIONS: dict[str, str] = {
    "base_close": "last close available at the origin (available_at <= origin)",
    "forward_return_H": "close available at origin+H divided by base close, minus one",
    "forward_abs_return_H": "absolute value of forward_return_H",
    "forward_direction_H": "1 when forward_return_H > 0, 0 when < 0, omitted when exactly 0",
    "realized_volatility_H": (
        "population standard deviation of log returns between consecutive bar closes "
        "that became available in (origin, origin+H]"
    ),
}


def build_targets(
    series: MarketSeries,
    origins: Sequence[datetime],
    horizons: Sequence[str] = DEFAULT_HORIZONS_HOURLY,
) -> list[TargetRow]:
    """Compute forward outcomes for each origin.

    Origins with no prior bar are dropped rather than defaulted: an origin
    before the series begins has no base price, and inventing one would put a
    fabricated observation at the very start of every study.
    """
    if not series.bars:
        raise B4DataError(f"{series.series_id}: cannot build targets from an empty series")
    for horizon in horizons:
        horizon_delta(horizon)

    rows: list[TargetRow] = []
    for raw_origin in origins:
        origin = require_utc(raw_origin, "forecast_origin")
        base_index = last_available_index(series, origin)
        if base_index < 0:
            continue
        base_bar = series.bars[base_index]

        forward_return: dict[str, float | None] = {}
        forward_abs: dict[str, float | None] = {}
        direction: dict[str, int | None] = {}
        volatility: dict[str, float | None] = {}

        for horizon in horizons:
            end = origin + horizon_delta(horizon)
            end_index = last_available_index(series, end)
            if end_index <= base_index or end > (series.end or end):
                forward_return[horizon] = None
                forward_abs[horizon] = None
                direction[horizon] = None
                volatility[horizon] = None
                continue

            end_bar = series.bars[end_index]
            if end_bar.available_at <= origin:
                # Unreachable by construction; asserted because a silent
                # violation here would corrupt every downstream result.
                raise LeakageError(
                    f"{series.series_id}: target bar for {horizon} at origin {origin.isoformat()} "
                    "was already available at the origin"
                )

            value = end_bar.close / base_bar.close - 1.0
            forward_return[horizon] = value
            forward_abs[horizon] = abs(value)
            direction[horizon] = 1 if value > 0.0 else (0 if value < 0.0 else None)
            volatility[horizon] = _realized_volatility(series, base_index, end_index)

        rows.append(
            TargetRow(
                forecast_origin=origin,
                base_close=base_bar.close,
                base_available_at=base_bar.available_at,
                forward_return=forward_return,
                forward_abs_return=forward_abs,
                forward_direction=direction,
                realized_volatility=volatility,
            )
        )
    return rows


def _realized_volatility(series: MarketSeries, base_index: int, end_index: int) -> float | None:
    """Population sd of log returns strictly inside the forward window."""
    closes = [bar.close for bar in series.bars[base_index : end_index + 1]]
    if len(closes) < 3:
        return None
    log_returns = [math.log(later / earlier) for earlier, later in zip(closes, closes[1:], strict=False)]
    mean = sum(log_returns) / len(log_returns)
    variance = sum((value - mean) ** 2 for value in log_returns) / len(log_returns)
    return math.sqrt(variance)


def target_result_hash(rows: Sequence[TargetRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row.as_record(), sort_keys=True).encode())
    return digest.hexdigest()


def build_target_manifest(
    series: MarketSeries,
    rows: Sequence[TargetRow],
    horizons: Sequence[str],
    *,
    git_sha: str,
    created_at: datetime,
) -> TargetManifest:
    if not rows:
        raise B4DataError("cannot manifest an empty target dataset")
    start = series.start
    end = series.end
    if start is None or end is None:
        raise B4DataError("series has no time span")
    return TargetManifest(
        series_id=series.series_id,
        ticker=series.ticker,
        provider=series.provider,
        timestamp_convention=series.timestamp_convention.value,
        evidence_tier=series.evidence_tier,
        source_timezone=series.source_timezone,
        bar_period_seconds=series.period_seconds,
        series_start=start,
        series_end=end,
        series_bar_count=len(series),
        series_fingerprint=series.fingerprint(),
        origin_start=rows[0].forecast_origin,
        origin_end=rows[-1].forecast_origin,
        origin_count=len(rows),
        horizons=tuple(horizons),
        target_definitions=dict(TARGET_DEFINITIONS),
        row_count=len(rows),
        result_hash=target_result_hash(rows),
        git_sha=git_sha,
        created_at=require_utc(created_at, "created_at"),
    )


def join_targets_by_origin(
    feature_rows: Sequence[dict[str, Any]],
    target_rows: Sequence[TargetRow],
) -> list[dict[str, Any]]:
    """B4.39. Join the intelligence matrix to outcomes on `forecast_origin` only.

    An inner join: an origin without both sides is dropped rather than filled.
    Filling would put a fabricated feature or a fabricated outcome into the
    study, and either one is worse than a smaller sample.
    """
    by_origin = {row.forecast_origin: row for row in target_rows}
    joined: list[dict[str, Any]] = []
    for feature_row in feature_rows:
        raw = feature_row.get("forecast_origin")
        if raw is None:
            raise B4DataError("feature row has no forecast_origin")
        origin = require_utc(raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw)), "forecast_origin")
        target = by_origin.get(origin)
        if target is None:
            continue
        merged = dict(feature_row)
        merged.update(target.as_record())
        merged["forecast_origin"] = origin
        joined.append(merged)
    return joined


def assert_targets_are_future_only(series: MarketSeries, rows: Sequence[TargetRow], horizons: Sequence[str]) -> None:
    """B4.40. Re-derive the contract independently of how targets were built.

    Recomputing from the series rather than trusting the stored row is the
    whole value of the check: a bug in `build_targets` that also wrote the
    stored `base_available_at` would pass any assertion that only read the row.
    """
    for row in rows:
        if row.base_available_at > row.forecast_origin:
            raise LeakageError(
                f"base close at {row.forecast_origin.isoformat()} was not available at the origin"
            )
        for horizon in horizons:
            if row.forward_return.get(horizon) is None:
                continue
            end = row.forecast_origin + horizon_delta(horizon)
            index = last_available_index(series, end)
            if index < 0:
                raise LeakageError(f"no bar available at {end.isoformat()}")
            if series.bars[index].available_at <= row.forecast_origin:
                raise LeakageError(
                    f"{horizon} outcome at {row.forecast_origin.isoformat()} uses a bar "
                    "that was already available at the origin"
                )
            recomputed = series.bars[index].close / row.base_close - 1.0
            stored = row.forward_return[horizon]
            assert stored is not None
            if not math.isclose(recomputed, stored, rel_tol=0.0, abs_tol=0.0):
                raise LeakageError(
                    f"{horizon} outcome at {row.forecast_origin.isoformat()} does not reproduce "
                    "from the series it claims to come from"
                )


def base_close_matches_feature_view(series: MarketSeries, origin: datetime, base_close: float) -> bool:
    """True when a target's base price is exactly what a feature would see."""
    return close_as_of(series, origin) == base_close


__all__ = [
    "DEFAULT_HORIZONS_DAILY",
    "DEFAULT_HORIZONS_HOURLY",
    "HORIZON_SECONDS",
    "TARGET_CONTRACT_VERSION",
    "TARGET_DEFINITIONS",
    "TargetManifest",
    "TargetRow",
    "assert_targets_are_future_only",
    "base_close_matches_feature_view",
    "build_target_manifest",
    "build_targets",
    "horizon_delta",
    "join_targets_by_origin",
    "target_result_hash",
]
