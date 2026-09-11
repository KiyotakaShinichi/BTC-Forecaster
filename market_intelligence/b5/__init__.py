"""Track B5 — point-in-time market-intelligence event study.

Research-only. B5 asks whether independently timestamped external information is
associated with BTC market behaviour beyond what the price-only studies (A2, A6,
A7) could see. Its first gate is whether a point-in-time corpus exists to study;
if it does not, the track records that and stops. Nothing here produces a
forecast, a fused signal, a position or a trading decision, and no module in
this package imports a Track A forecasting model or the paper-trading engine.
"""

from __future__ import annotations

B5_RESEARCH_NAMESPACE = "research/market_intelligence/b5"

__all__ = ["B5_RESEARCH_NAMESPACE"]
