"""Small, frozen set of deterministic C0 entry/exit policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import Side, TakeProfitLevel


class EntryPolicy(str, Enum):
    MARKET_AT_ORIGIN_V1 = "MARKET_AT_ORIGIN_V1"
    LIMIT_PULLBACK_V1 = "LIMIT_PULLBACK_V1"
    VOLATILITY_BAND_V1 = "VOLATILITY_BAND_V1"


class StopPolicy(str, Enum):
    FIXED_RISK_V1 = "FIXED_RISK_V1"
    VOLATILITY_BASED_V1 = "VOLATILITY_BASED_V1"
    FORECAST_ADVERSE_QUANTILE_V1 = "FORECAST_ADVERSE_QUANTILE_V1"


class TargetPolicy(str, Enum):
    FORECAST_MEDIAN_V1 = "FORECAST_MEDIAN_V1"
    FORECAST_UPPER_QUANTILE_V1 = "FORECAST_UPPER_QUANTILE_V1"
    R_MULTIPLE_V1 = "R_MULTIPLE_V1"


@dataclass(frozen=True)
class Entry:
    price: float
    zone_low: float
    zone_high: float


@dataclass(frozen=True)
class Stop:
    price: float
    distance_fraction: float
    reason: str


def entry(policy: EntryPolicy, origin_price: float, *, volatility: float = 0.0) -> Entry:
    if origin_price <= 0:
        raise ValueError("origin price must be positive")
    if policy is EntryPolicy.MARKET_AT_ORIGIN_V1:
        return Entry(origin_price, origin_price, origin_price)
    offset = max(volatility, 0.001) * (0.25 if policy is EntryPolicy.LIMIT_PULLBACK_V1 else 1.0)
    return Entry(origin_price * (1 - offset), origin_price * (1 - offset), origin_price)


def stop(
    policy: StopPolicy,
    side: Side,
    entry_price: float,
    *,
    fixed_fraction: float = 0.01,
    volatility: float = 0.0,
    adverse_return: float | None = None,
) -> Stop:
    if policy is StopPolicy.FIXED_RISK_V1:
        distance = fixed_fraction
    elif policy is StopPolicy.VOLATILITY_BASED_V1:
        distance = 1.5 * volatility
    else:
        if adverse_return is None:
            raise ValueError("adverse quantile required")
        distance = abs(adverse_return)
    if distance <= 0 or distance >= 1:
        raise ValueError("stop distance must be in (0, 1)")
    sign = -1 if side is Side.LONG_CANDIDATE else 1
    return Stop(entry_price * (1 + sign * distance), distance, policy.value)


def target(
    policy: TargetPolicy,
    side: Side,
    entry_price: float,
    *,
    forecast_return: float,
    stop_distance: float,
) -> tuple[TakeProfitLevel, ...]:
    distance = (
        abs(forecast_return) if policy is not TargetPolicy.R_MULTIPLE_V1 else 2 * stop_distance
    )
    if distance <= 0:
        raise ValueError("positive target distance required")
    sign = 1 if side is Side.LONG_CANDIDATE else -1
    return (TakeProfitLevel(entry_price * (1 + sign * distance), 1.0, policy.value),)
