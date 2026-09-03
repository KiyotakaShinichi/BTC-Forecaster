"""Immutable contracts for the C0 paper-trading research seam."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType


class Side(str, Enum):
    LONG_CANDIDATE = "LONG_CANDIDATE"
    SHORT_CANDIDATE = "SHORT_CANDIDATE"
    NO_TRADE = "NO_TRADE"


class ConfidenceBand(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class PromotionStatus(str, Enum):
    PROMOTED = "PROMOTED"
    UNPROMOTED = "UNPROMOTED"


@dataclass(frozen=True)
class ForecastDistribution:
    forecast_id: str
    forecast_origin: datetime
    instrument: str
    horizon: timedelta
    model_id: str
    model_version: str
    expected_return: float
    return_quantiles: Mapping[float, float]
    direction_probability_raw: float
    direction_probability_calibrated: float | None
    calibration_version: str | None
    promotion_status: PromotionStatus
    stability_status: str
    data_freshness: timedelta
    feature_snapshot_id: str
    regime_support: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.direction_probability_raw <= 1:
            raise ValueError("raw probability must be in [0, 1]")
        if self.direction_probability_calibrated is not None and not (
            0 <= self.direction_probability_calibrated <= 1
        ):
            raise ValueError("calibrated probability must be in [0, 1]")
        if self.horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        quantiles = dict(self.return_quantiles)
        if any(not 0 < q < 1 for q in quantiles):
            raise ValueError("quantile keys must be in (0, 1)")
        if list(quantiles) != sorted(quantiles):
            raise ValueError("quantiles must be ordered")
        if any(
            a > b for a, b in zip(quantiles.values(), list(quantiles.values())[1:], strict=False)
        ):
            raise ValueError("quantile returns must be monotonic")
        object.__setattr__(self, "return_quantiles", MappingProxyType(quantiles))


@dataclass(frozen=True)
class ExternalSignalBundle:
    signal_contract_version: str
    signal_ids: tuple[str, ...]
    aggregate_confidence: float | None
    availability_state: str


@dataclass(frozen=True)
class TakeProfitLevel:
    price: float
    fraction: float
    policy: str


@dataclass(frozen=True)
class TradeDecision:
    decision_id: str
    forecast_origin: datetime
    instrument: str
    side: Side
    forecast_horizon: timedelta
    entry_policy: str
    entry_price: float | None
    entry_zone_low: float | None
    entry_zone_high: float | None
    expected_return: float
    expected_return_interval: tuple[float, float]
    raw_probability: float
    calibrated_probability: float | None
    confidence_band: ConfidenceBand
    stop_price: float | None
    stop_distance_fraction: float | None
    stop_policy: str
    take_profit_levels: tuple[TakeProfitLevel, ...]
    time_stop: datetime
    expected_value_after_costs: float
    expected_R: float | None
    risk_fraction: float
    risk_amount: float
    position_notional: float
    required_leverage: float
    allowed_leverage: float
    promotion_gate: bool
    risk_gate: bool
    data_gate: bool
    reason_codes: tuple[str, ...]
    invalidation_reasons: tuple[str, ...]
    forecast_id: str
    model_id: str
    feature_snapshot_id: str
    cost_model_version: str
    research_only: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.side is Side.NO_TRADE and self.allowed_leverage != 0:
            raise ValueError("NO_TRADE must have zero allowed leverage")
        if not self.research_only:
            raise ValueError("C0 decisions are research-only")
