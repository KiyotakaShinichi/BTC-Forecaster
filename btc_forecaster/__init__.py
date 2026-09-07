"""BTC-Forecaster: a point-in-time correct forecasting research platform.

The package is layered so that each layer can be tested without the one above:

    timebase      the time contract -- UTC, event_time, available_time, origins
    config        frozen run configuration
    data          providers, schema contract, hash-verified snapshots
    features      point-in-time feature construction and train-only selection
    models        the ForecastModel contract, baselines, statistical models
    evaluation    metrics for point and interval forecasts
    backtesting   the walk-forward engine every model is scored through
    diagnostics   stationarity, autocorrelation, heteroskedasticity tests
    artifacts     run output and its manifest

Nothing here imports Prophet, XGBoost, arch, matplotlib or yfinance at module
scope, so the core is importable -- and unit-testable -- without them.
"""

from .timebase import (
    UTC,
    ForecastOrigin,
    HorizonSpec,
    PointInTimeViolation,
    assert_available,
    available_time,
    bar_event_time,
    to_utc_index,
    to_utc_timestamp,
)

__version__ = "0.2.0"

__all__ = [
    "UTC",
    "ForecastOrigin",
    "HorizonSpec",
    "PointInTimeViolation",
    "__version__",
    "assert_available",
    "available_time",
    "bar_event_time",
    "to_utc_index",
    "to_utc_timestamp",
]
