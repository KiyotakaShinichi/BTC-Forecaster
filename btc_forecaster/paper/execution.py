"""Deterministic OHLC paper execution; never connects to an exchange."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .contracts import Side, TradeDecision
from .costs import CostModel


class TradeState(str, Enum):
    PROPOSED = "PROPOSED"
    REJECTED = "REJECTED"
    OPEN = "OPEN"
    PARTIALLY_EXITED = "PARTIALLY_EXITED"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_TIME = "CLOSED_TIME"
    CLOSED_MANUAL = "CLOSED_MANUAL"
    INVALIDATED = "INVALIDATED"


TRANSITIONS = {
    TradeState.PROPOSED: {TradeState.REJECTED, TradeState.OPEN, TradeState.INVALIDATED},
    TradeState.OPEN: {
        TradeState.PARTIALLY_EXITED,
        TradeState.CLOSED_TP,
        TradeState.CLOSED_SL,
        TradeState.CLOSED_TIME,
        TradeState.CLOSED_MANUAL,
        TradeState.INVALIDATED,
    },
    TradeState.PARTIALLY_EXITED: {
        TradeState.CLOSED_TP,
        TradeState.CLOSED_SL,
        TradeState.CLOSED_TIME,
        TradeState.CLOSED_MANUAL,
        TradeState.INVALIDATED,
    },
}


def transition(current: TradeState, target: TradeState) -> TradeState:
    if target not in TRANSITIONS.get(current, set()):
        raise ValueError(f"impossible trade transition: {current.value} -> {target.value}")
    return target


@dataclass(frozen=True)
class MarketBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class PaperResult:
    state: TradeState
    entry_price: float | None
    exit_price: float | None
    fees: float
    funding: float
    pnl: float
    return_fraction: float
    r_multiple: float | None
    mfe: float
    mae: float
    holding_seconds: float
    exit_reason: str


def execute(decision: TradeDecision, bars: list[MarketBar], costs: CostModel) -> PaperResult:
    if decision.side is Side.NO_TRADE:
        return PaperResult(TradeState.REJECTED, None, None, 0, 0, 0, 0, None, 0, 0, 0, "NO_TRADE")
    if not bars or decision.entry_price is None or decision.stop_price is None:
        raise ValueError("candidate requires bars, entry, and stop")
    entry_price = decision.entry_price
    target_price = decision.take_profit_levels[0].price
    sign = 1 if decision.side is Side.LONG_CANDIDATE else -1
    mfe = mae = 0.0
    exit_price, state, reason, exit_time = (
        bars[-1].close,
        TradeState.CLOSED_TIME,
        "TIME_STOP",
        bars[-1].timestamp,
    )
    for bar in bars:
        favorable = (
            sign * (bar.high - entry_price) / entry_price
            if sign > 0
            else sign * (bar.low - entry_price) / entry_price
        )
        adverse = (
            sign * (bar.low - entry_price) / entry_price
            if sign > 0
            else sign * (bar.high - entry_price) / entry_price
        )
        mfe, mae = max(mfe, favorable), min(mae, adverse)
        tp_hit = bar.high >= target_price if sign > 0 else bar.low <= target_price
        sl_hit = bar.low <= decision.stop_price if sign > 0 else bar.high >= decision.stop_price
        if tp_hit and sl_hit:
            # Unknown ordering: resolve against the strategy, explicitly and conservatively.
            exit_price, state, reason, exit_time = (
                decision.stop_price,
                TradeState.CLOSED_SL,
                "AMBIGUOUS_STOP_FIRST",
                bar.timestamp,
            )
            break
        if sl_hit:
            exit_price, state, reason, exit_time = (
                decision.stop_price,
                TradeState.CLOSED_SL,
                "STOP_LOSS",
                bar.timestamp,
            )
            break
        if tp_hit:
            exit_price, state, reason, exit_time = (
                target_price,
                TradeState.CLOSED_TP,
                "TAKE_PROFIT",
                bar.timestamp,
            )
            break
        if bar.timestamp >= decision.time_stop:
            exit_price, state, reason, exit_time = (
                bar.close,
                TradeState.CLOSED_TIME,
                "TIME_STOP",
                bar.timestamp,
            )
            break
    gross = sign * (exit_price - entry_price) / entry_price
    cost_fraction = costs.round_trip_fraction()
    pnl = decision.position_notional * (gross - cost_fraction)
    fees = decision.position_notional * (cost_fraction - costs.funding - costs.borrow_cost)
    risk = decision.risk_amount
    return PaperResult(
        state,
        entry_price,
        exit_price,
        fees,
        decision.position_notional * costs.funding,
        pnl,
        gross - cost_fraction,
        pnl / risk if risk else None,
        mfe,
        mae,
        (exit_time - bars[0].timestamp).total_seconds(),
        reason,
    )
