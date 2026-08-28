from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FeatureDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    dtype: str = "float64"
    window: str
    definition: str
    missing_value_semantics: str
    version: str


FEATURE_CONTRACT_VERSION = "market-intelligence-features-v1"
FEATURE_DEFINITIONS = (
    FeatureDefinition(
        name="sentiment_mean_24h",
        window="24h",
        definition="unweighted mean event sentiment",
        missing_value_semantics="0 only when coverage is available and no weighted observations exist",
        version="1.0.0",
    ),
    FeatureDefinition(
        name="sentiment_weighted_24h",
        window="24h",
        definition="sentiment weighted by relevance, novelty and confidence",
        missing_value_semantics="0 only when coverage is available and total evidence weight is zero",
        version="1.0.0",
    ),
    FeatureDefinition(
        name="event_count_24h",
        window="24h",
        definition="point-in-time eligible event count",
        missing_value_semantics="0 means no observed event; provider health is separate",
        version="1.0.0",
    ),
    FeatureDefinition(
        name="high_relevance_event_count_24h",
        window="24h",
        definition="event count with BTC relevance >= 0.75",
        missing_value_semantics="0 means no qualifying observed event",
        version="1.0.0",
    ),
    FeatureDefinition(
        name="regulatory_signal_72h",
        window="72h",
        definition="weighted regulatory sentiment",
        missing_value_semantics="0 means no weighted regulatory evidence",
        version="1.0.0",
    ),
    FeatureDefinition(
        name="whale_exchange_inflow_signal_72h",
        window="72h",
        definition="weighted sentiment for contextualized exchange inflows",
        missing_value_semantics="0 means no contextualized inflow evidence",
        version="1.0.0",
    ),
    FeatureDefinition(
        name="macro_news_signal_72h",
        window="72h",
        definition="weighted macro, monetary policy and liquidity sentiment",
        missing_value_semantics="0 means no weighted macro evidence",
        version="1.0.0",
    ),
)
