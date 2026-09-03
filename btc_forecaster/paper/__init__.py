"""C0 paper trade opportunity, risk, and execution research engine."""

from .a2 import current_live_permission
from .contracts import ForecastDistribution, Side, TradeDecision
from .decision import LIVE_TRADING_ELIGIBLE, decide

__all__ = [
    "ForecastDistribution",
    "LIVE_TRADING_ELIGIBLE",
    "Side",
    "TradeDecision",
    "current_live_permission",
    "decide",
]
