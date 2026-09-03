"""Performance, no-trade, dependence-aware bootstrap, and promotion research."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .contracts import Side, TradeDecision
from .execution import PaperResult


@dataclass(frozen=True)
class PerformanceMetrics:
    trade_count: int
    win_rate: float
    loss_rate: float
    expectancy: float
    median_r: float
    profit_factor: float
    max_drawdown: float
    turnover: float
    fee_drag: float
    slippage_drag: float


def performance(results: list[PaperResult], notionals: list[float]) -> PerformanceMetrics:
    pnl = np.asarray([r.pnl for r in results], dtype=float)
    r_values = np.asarray([r.r_multiple for r in results if r.r_multiple is not None], dtype=float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    curve = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(np.r_[0.0, curve]) - np.r_[0.0, curve]
    total_notional = sum(notionals)
    return PerformanceMetrics(
        len(results),
        float(np.mean(pnl > 0)) if len(pnl) else 0.0,
        float(np.mean(pnl < 0)) if len(pnl) else 0.0,
        float(pnl.mean()) if len(pnl) else 0.0,
        float(np.median(r_values)) if len(r_values) else 0.0,
        float(wins / losses) if losses else float("inf"),
        float(drawdown.max()),
        total_notional,
        sum(r.fees for r in results),
        0.0,
    )


def no_trade_analysis(decisions: list[TradeDecision]) -> dict[str, object]:
    rejected = [decision for decision in decisions if decision.side is Side.NO_TRADE]
    return {
        "candidate_forecasts_rejected": len(rejected),
        "reason_distribution": dict(
            Counter(code for decision in rejected for code in decision.reason_codes)
        ),
    }


def block_bootstrap(
    values: list[float], *, block_size: int = 3, samples: int = 1000, seed: int = 0
) -> np.ndarray:
    """Stationary chronology-preserving block bootstrap of mean expectancy."""
    if not values or block_size <= 0:
        raise ValueError("values and positive block size required")
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    starts = np.arange(max(1, len(data) - block_size + 1))
    output = np.empty(samples)
    for index in range(samples):
        chunks = [
            data[start : start + block_size]
            for start in rng.choice(starts, size=int(np.ceil(len(data) / block_size)), replace=True)
        ]
        output[index] = np.concatenate(chunks)[: len(data)].mean()
    return output


def ruin_proxy(
    returns: list[float],
    *,
    drawdown_limit: float,
    block_size: int = 3,
    samples: int = 1000,
    seed: int = 0,
) -> float:
    if not returns:
        raise ValueError("returns required")
    rng = np.random.default_rng(seed)
    data = np.asarray(returns)
    ruined = 0
    for _ in range(samples):
        starts = rng.integers(
            0, max(1, len(data) - block_size + 1), int(np.ceil(len(data) / block_size))
        )
        path = np.concatenate([data[s : s + block_size] for s in starts])[: len(data)]
        equity = np.cumprod(1 + path)
        dd = 1 - equity / np.maximum.accumulate(np.r_[1.0, equity])[: len(equity)]
        ruined += bool(np.max(dd) >= drawdown_limit)
    return ruined / samples


@dataclass(frozen=True)
class ExecutionAblation:
    name: str
    stop_enabled: bool
    target_enabled: bool
    time_stop_enabled: bool = True


PREDEFINED_ABLATIONS = (
    ExecutionAblation("TIME_ONLY", False, False),
    ExecutionAblation("SL_TIME", True, False),
    ExecutionAblation("SL_TP_TIME", True, True),
)


def validate_temporal_isolation(tuning_end: object, validation_start: object) -> None:
    if tuning_end >= validation_start:  # type: ignore[operator]
        raise ValueError("execution tuning overlaps final validation")
