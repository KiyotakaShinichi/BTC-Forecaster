from __future__ import annotations

from datetime import datetime, timedelta

from .models import EventSignal, EventType, TransferContext


class FeatureAggregator:
    """Produces deterministic, point-in-time-safe lagged signal features."""

    def aggregate(self, signals: list[EventSignal], forecast_origin: datetime) -> dict[str, float]:
        eligible = [s for s in signals if s.available_time <= forecast_origin]
        recent24 = [s for s in eligible if s.available_time > forecast_origin - timedelta(hours=24)]
        recent72 = [s for s in eligible if s.available_time > forecast_origin - timedelta(hours=72)]
        weight = sum(s.btc_relevance * s.novelty * s.confidence for s in recent24)
        weighted = sum(s.sentiment * s.btc_relevance * s.novelty * s.confidence for s in recent24)
        regulatory = [s for s in recent72 if s.event_type == EventType.REGULATION]
        inflows = [s for s in recent72 if s.event_type == EventType.WHALE_TRANSFER and s.transfer_context == TransferContext.EXCHANGE_INFLOW]
        macro = [s for s in recent72 if s.event_type in {EventType.MACRO_SHOCK, EventType.MONETARY_POLICY, EventType.LIQUIDITY_EVENT}]
        return {
            "sentiment_mean_24h": sum(s.sentiment for s in recent24) / len(recent24) if recent24 else 0.0,
            "sentiment_weighted_24h": weighted / weight if weight else 0.0,
            "event_count_24h": float(len(recent24)),
            "high_relevance_event_count_24h": float(sum(s.btc_relevance >= 0.75 for s in recent24)),
            "regulatory_signal_72h": self._weighted_mean(regulatory),
            "whale_exchange_inflow_signal_72h": self._weighted_mean(inflows),
            "macro_news_signal_72h": self._weighted_mean(macro),
        }

    @staticmethod
    def _weighted_mean(signals: list[EventSignal]) -> float:
        weights = [s.btc_relevance * s.novelty * s.confidence for s in signals]
        total = sum(weights)
        return sum(s.sentiment * w for s, w in zip(signals, weights)) / total if total else 0.0
