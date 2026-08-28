"""Provider-agnostic, point-in-time external intelligence for BTC forecasts."""

from .aggregation import FeatureAggregator
from .models import (
    Direction,
    Document,
    EventSignal,
    EventType,
    SignalCategory,
    TransferContext,
)
from .cycle import ReplayService, run_intelligence_cycle
from .features import FEATURE_CONTRACT_VERSION, FEATURE_DEFINITIONS

__all__ = [
    "Direction",
    "Document",
    "EventSignal",
    "EventType",
    "FeatureAggregator",
    "SignalCategory",
    "TransferContext",
    "ReplayService",
    "run_intelligence_cycle",
    "FEATURE_CONTRACT_VERSION",
    "FEATURE_DEFINITIONS",
]
