"""Track B4 — historical external-signal validation.

Research-only. Nothing here produces a forecast, a fused signal, a position or a
trading decision, and no module in this package imports a Track A/A2 forecasting
model. The output is evidence about whether external signals have any
reproducible, point-in-time-safe historical association with future BTC
outcomes — including, and especially, the finding that they do not.
"""

from __future__ import annotations

B4_RESEARCH_NAMESPACE = "research/market_intelligence/b4"

__all__ = ["B4_RESEARCH_NAMESPACE"]
