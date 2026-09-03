"""Pure human-readable presentation: it invents no decision values."""

from .contracts import TradeDecision


def format_opportunity(decision: TradeDecision) -> str:
    targets = ", ".join(f"{item.price:.2f}" for item in decision.take_profit_levels) or "N/A"
    return "\n".join(
        (
            "BTC PAPER OPPORTUNITY",
            "PAPER RESEARCH — NOT LIVE EXECUTION",
            "",
            f"Decision: {decision.side.value}",
            f"Horizon: {decision.forecast_horizon}",
            f"Entry: {decision.entry_price if decision.entry_price is not None else 'N/A'}",
            f"Expected return: {decision.expected_return:.4%}",
            f"Calibrated probability: {decision.calibrated_probability if decision.calibrated_probability is not None else 'N/A'}",
            f"Confidence: {decision.confidence_band.value}",
            f"Stop: {decision.stop_price if decision.stop_price is not None else 'N/A'}",
            f"TP: {targets}",
            f"Time stop: {decision.time_stop.isoformat()}",
            f"Risk: {decision.risk_amount:.2f}",
            f"Required leverage: {decision.required_leverage:.2f}x",
            f"Allowed leverage: {decision.allowed_leverage:.2f}x",
            f"Why: {', '.join(decision.reason_codes)}",
            f"Invalidation: {', '.join(decision.invalidation_reasons)}",
            f"Model promotion: {decision.promotion_gate}",
        )
    )
