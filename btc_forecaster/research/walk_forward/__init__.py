"""Track A7: walk-forward robustness around the A6 model zoo.

A6 fitted forty models once, on 1,000 rows, and scored them on one holdout
block. Nothing beat the random walk. A7 asks whether that conclusion survives
being asked again from many origins, at several horizons, with several amounts
of history, across time -- and it is built to make a spurious edge harder to
manufacture, not easier to find.

Nothing here promotes a model. The strongest outcome A7 can reach is
``ROBUST_RESEARCH_CANDIDATE``: a reason to preregister a separate study.
"""

from .config import (
    BASELINE,
    CURATED_MODELS,
    DRIFT_BASELINE,
    HORIZONS,
    PREPROCESSING_VERSION,
    GateThresholds,
    WalkForwardConfig,
    WindowSpec,
)

__all__ = [
    "BASELINE",
    "CURATED_MODELS",
    "DRIFT_BASELINE",
    "HORIZONS",
    "PREPROCESSING_VERSION",
    "GateThresholds",
    "WalkForwardConfig",
    "WindowSpec",
]
