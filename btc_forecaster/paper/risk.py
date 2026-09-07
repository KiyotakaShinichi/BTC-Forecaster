"""Portfolio risk states, limits, and risk-based sizing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskState(str, Enum):
    ACTIVE = "ACTIVE"
    REDUCED = "REDUCED"
    HALTED = "HALTED"


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade: float = 0.01
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_simultaneous_positions: int = 1
    max_daily_loss: float = 0.03
    max_rolling_drawdown: float = 0.10
    loss_streak_cooldown: int = 3
    paper_leverage_cap: float = 1.0
    liquidity_cap: float = 1_000_000.0


@dataclass(frozen=True)
class RiskSnapshot:
    equity: float
    allocated_capital: float
    daily_loss_fraction: float = 0.0
    rolling_drawdown: float = 0.0
    consecutive_losses: int = 0
    open_positions: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    model_integrity: bool = True
    data_fresh: bool = True
    calibration_valid: bool = True


def risk_state(snapshot: RiskSnapshot, limits: RiskLimits) -> RiskState:
    if (
        not snapshot.model_integrity
        or not snapshot.data_fresh
        or not snapshot.calibration_valid
        or snapshot.daily_loss_fraction >= limits.max_daily_loss
        or snapshot.rolling_drawdown >= limits.max_rolling_drawdown
        or snapshot.consecutive_losses >= limits.loss_streak_cooldown
        or snapshot.open_positions >= limits.max_simultaneous_positions
        or snapshot.gross_exposure >= limits.max_gross_exposure
        or abs(snapshot.net_exposure) >= limits.max_net_exposure
    ):
        return RiskState.HALTED
    if (
        snapshot.daily_loss_fraction >= limits.max_daily_loss / 2
        or snapshot.rolling_drawdown >= limits.max_rolling_drawdown / 2
    ):
        return RiskState.REDUCED
    return RiskState.ACTIVE


@dataclass(frozen=True)
class PositionSize:
    risk_fraction: float
    risk_amount: float
    position_notional: float
    required_leverage: float
    allowed_leverage: float


def size_position(
    snapshot: RiskSnapshot,
    limits: RiskLimits,
    *,
    requested_risk_fraction: float,
    stop_distance_fraction: float,
    live_eligible: bool = False,
) -> PositionSize:
    if snapshot.equity <= 0 or snapshot.allocated_capital <= 0 or stop_distance_fraction <= 0:
        raise ValueError("positive equity, capital, and stop distance required")
    fraction = min(max(requested_risk_fraction, 0.0), limits.max_risk_per_trade)
    if risk_state(snapshot, limits) is RiskState.REDUCED:
        fraction /= 2
    amount = snapshot.equity * fraction
    uncapped = amount / stop_distance_fraction
    exposure_cap = max(0.0, limits.max_gross_exposure - snapshot.gross_exposure) * snapshot.equity
    notional = min(uncapped, exposure_cap, limits.liquidity_cap)
    required = notional / snapshot.allocated_capital
    allowed = min(required, limits.paper_leverage_cap) if live_eligible else 0.0
    return PositionSize(fraction, amount, notional, required, allowed)
