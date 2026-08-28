"""Run configuration as a frozen, serialisable object.

The original pipeline read ``os.getenv`` at eleven points scattered through a
700-line module executing at import time, so the configuration of a run existed
only as whatever the environment happened to be at that moment and was never
recorded alongside the output. A result you cannot reconstruct the settings for
is not reproducible.

Configuration is now one immutable object, built once, written into every run
manifest, and passed explicitly. The environment variable names are unchanged so
existing Docker images and shell scripts keep working.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

DEFAULT_MODELS: tuple[str, ...] = (
    "random_walk",
    "random_walk_drift",
    "historical_mean_return",
    "arima",
    "ets",
    "prophet",
    "prophet_xgb_hybrid",
)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class WalkForwardConfig:
    """How the backtest folds are built."""

    n_folds: int = 6
    horizon: int = 30
    min_train_bars: int = 730
    mode: str = "expanding"
    window_bars: int | None = None
    embargo_bars: int = 0

    @classmethod
    def from_env(cls) -> WalkForwardConfig:
        window = _env_int("WF_WINDOW_BARS", 0)
        return cls(
            n_folds=_env_int("WF_FOLDS", 6),
            horizon=_env_int("WF_HORIZON", 30),
            min_train_bars=_env_int("WF_MIN_TRAIN_BARS", 730),
            mode=_env_str("WF_MODE", "expanding"),
            window_bars=window or None,
            embargo_bars=_env_int("WF_EMBARGO_BARS", 0),
        )


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines what a run does."""

    ticker: str = "BTC-USD"
    start: str = "2017-01-01"
    horizon_days: int = 365
    interval_level: float = 0.95
    random_state: int = 42
    monte_carlo_runs: int = 1000
    max_lag: int = 60

    models: tuple[str, ...] = DEFAULT_MODELS
    primary_model: str = "prophet_xgb_hybrid"
    baseline_model: str = "random_walk"

    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    output_dir: Path = Path("out")
    snapshot_dir: Path = Path("data/snapshots")
    refresh_data: bool = False
    make_plots: bool = True
    show_plots: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "snapshot_dir", Path(self.snapshot_dir))
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if not 0.0 < self.interval_level < 1.0:
            raise ValueError("interval_level must be in (0, 1)")
        if self.baseline_model not in self.models:
            raise ValueError(
                f"baseline_model {self.baseline_model!r} must be one of the models being run; "
                "skill cannot be measured against a model that was never scored"
            )

    @classmethod
    def from_env(cls, **overrides) -> RunConfig:
        """Build from environment variables, honouring the original names."""
        base = cls(
            ticker=_env_str("TICKER", "BTC-USD"),
            start=_env_str("DATA_START", "2017-01-01"),
            horizon_days=_env_int("HORIZON_DAYS", 365),
            interval_level=_env_float("INTERVAL_LEVEL", 0.95),
            random_state=_env_int("RANDOM_STATE", 42),
            monte_carlo_runs=_env_int("MONTE_CARLO_RUNS", 1000),
            max_lag=_env_int("MAX_LAG", 60),
            models=_env_tuple("MODELS", DEFAULT_MODELS),
            primary_model=_env_str("PRIMARY_MODEL", "prophet_xgb_hybrid"),
            baseline_model=_env_str("BASELINE_MODEL", "random_walk"),
            walk_forward=WalkForwardConfig.from_env(),
            output_dir=Path(_env_str("OUTPUT_DIR", "out")),
            snapshot_dir=Path(_env_str("SNAPSHOT_DIR", "data/snapshots")),
            refresh_data=_env_bool("REFRESH_DATA", False),
            make_plots=_env_bool("MAKE_PLOTS", True),
            show_plots=_env_bool("PLOT_SHOW", False),
        )
        return replace(base, **overrides) if overrides else base

    @property
    def snapshot_path(self) -> Path:
        """Where this ticker's snapshot lives. One directory per ticker."""
        return self.snapshot_dir / self.ticker.replace("/", "_")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["snapshot_dir"] = str(self.snapshot_dir)
        payload["models"] = list(self.models)
        return payload


__all__ = ["DEFAULT_MODELS", "RunConfig", "WalkForwardConfig"]
