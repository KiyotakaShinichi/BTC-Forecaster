"""The ForecastModel contract, naive baselines, and model implementations.

Heavy third-party models (Prophet, XGBoost, arch, statsmodels state-space) are
NOT imported here. Import them from their own modules, or build them through
btc_forecaster.models.registry, which reports what is unavailable rather than
failing at import time.
"""

from .base import (
    ForecastModel,
    ForecastResult,
    MissingDependencyError,
    NotFittedError,
    TrainingWindow,
    random_walk_bands,
)
from .baselines import (
    HistoricalMeanReturn,
    RandomWalk,
    RandomWalkWithDrift,
    default_baselines,
)

__all__ = [
    "ForecastModel",
    "ForecastResult",
    "HistoricalMeanReturn",
    "MissingDependencyError",
    "NotFittedError",
    "RandomWalk",
    "RandomWalkWithDrift",
    "TrainingWindow",
    "default_baselines",
    "random_walk_bands",
]
