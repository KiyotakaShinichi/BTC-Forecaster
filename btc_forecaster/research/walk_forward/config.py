"""What an A7 benchmark is, fixed before it runs.

Every choice that could be tuned after seeing a result lives here: the models,
the horizons, the training windows, the refit schedule, the information-set
warm-up, and every threshold the robustness gate applies. The configuration has
a canonical serialisation and a digest, and the digest is part of the result
digest -- so a run whose thresholds were moved afterwards is a different run,
visibly, rather than the same run with a better headline.

The gate thresholds are deliberately not permissive. The failure this track
guards against is not a missed signal; it is an apparent edge assembled from a
favourable horizon, a favourable window and a favourable block, reported as if
it were one finding. Each threshold closes one of those doors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

WindowKind = Literal["rolling", "expanding"]

#: Changes whenever feature construction, target definition or the
#: information-set rule changes. Part of the configuration digest, so evidence
#: produced under one preprocessing contract cannot be mistaken for another's.
PREPROCESSING_VERSION = "a7-walk-forward-1"

#: The random-walk hypothesis A2 and A6 both failed to reject. Every comparison
#: is against it.
BASELINE = "naive_last_value"

#: Required alongside the naive forecast. A model that beats a zero-return
#: forecast only by capturing average drift has found the drift, not a signal.
DRIFT_BASELINE = "random_walk_drift"

#: The curated A6 subset. Enough to ask the robustness questions -- two
#: baselines, the autoregressive and state-space families, one tree ensemble,
#: one linear learner, and a feed-forward and a recurrent network -- without
#: re-running a forty-model zoo at every origin, horizon and window.
CURATED_MODELS: tuple[str, ...] = (
    "naive_last_value",
    "random_walk_drift",
    "ar_p",
    "arima",
    "theta",
    "local_level",
    "local_linear_trend",
    "xgboost",
    "ridge",
    "mlp",
    "lstm",
)

#: Days ahead. h=1 is A6's target; the others are new questions, not
#: re-expressions of A6's answer.
HORIZONS: tuple[int, ...] = (1, 3, 7, 30)

#: Below this a rolling window cannot hold a DEV block for early stopping and a
#: 24-bar sequence lookback and still leave anything to train on.
MIN_ROLLING_ROWS = 100


@dataclass(frozen=True)
class WindowSpec:
    """How much history a model may train on at each refit origin.

    ``rolling`` keeps the most recent ``rows`` supervised rows whose targets are
    fully realised by the origin. ``expanding`` keeps every such row since the
    start of the data. Rows, not calendar days, because a row is what a model
    actually estimates on.
    """

    kind: WindowKind
    rows: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("rolling", "expanding"):
            raise ValueError(f"unknown window kind {self.kind!r}")
        if self.kind == "rolling":
            if self.rows is None or self.rows < MIN_ROLLING_ROWS:
                raise ValueError(
                    f"a rolling window needs at least {MIN_ROLLING_ROWS} rows, got {self.rows}"
                )
        elif self.rows is not None:
            raise ValueError("an expanding window has no fixed row count")

    @property
    def label(self) -> str:
        return f"rolling-{self.rows}" if self.kind == "rolling" else "expanding"

    @classmethod
    def parse(cls, label: str) -> WindowSpec:
        if label == "expanding":
            return cls("expanding")
        kind, _, rows = label.partition("-")
        if kind != "rolling" or not rows.isdigit():
            raise ValueError(f"cannot parse window label {label!r}")
        return cls("rolling", int(rows))


DEFAULT_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec("rolling", 250),
    WindowSpec("rolling", 500),
    WindowSpec("rolling", 1000),
    WindowSpec("rolling", 2000),
    WindowSpec("expanding"),
)


@dataclass(frozen=True)
class GateThresholds:
    """The robustness gate, declared before any result exists.

    A configuration is a ``ROBUST_RESEARCH_CANDIDATE`` only if it clears every
    one of these, and a signal that clears some but not all is recorded as
    fragile rather than rounded up.
    """

    #: Benjamini-Hochberg level over the primary comparison family.
    alpha: float = 0.05
    #: Aggregate MAE skill against the naive forecast must be strictly above this.
    min_aggregate_skill: float = 0.0
    #: The last chronological block must be strictly above this. A model that
    #: worked once and has since stopped working is a history lesson.
    min_late_block_skill: float = 0.0
    #: Share of refit folds with positive skill. Three in four, not "most".
    min_positive_fold_fraction: float = 0.75
    #: Practically useful: a one-percent MAE improvement, not a rounding error.
    practical_min_skill: float = 0.01
    #: What counts as a raw signal when deciding FRAGILE versus uninteresting:
    #: one-sided, unadjusted. Deliberately loose, because its only effect is to
    #: route a result into the fragile bin, never to promote it.
    raw_signal_alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5), got {self.alpha}")
        if not 0.0 < self.raw_signal_alpha < 0.5:
            raise ValueError(f"raw_signal_alpha must be in (0, 0.5), got {self.raw_signal_alpha}")
        if not 0.5 < self.min_positive_fold_fraction <= 1.0:
            raise ValueError(
                "min_positive_fold_fraction must exceed one half -- a gate a coin "
                f"flip could pass is not a gate; got {self.min_positive_fold_fraction}"
            )
        if self.min_aggregate_skill < 0.0 or self.min_late_block_skill < 0.0:
            raise ValueError("skill floors below zero would admit models worse than naive")
        if self.practical_min_skill < self.min_aggregate_skill:
            raise ValueError("the practical floor cannot sit below the aggregate floor")

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "min_aggregate_skill": self.min_aggregate_skill,
            "min_late_block_skill": self.min_late_block_skill,
            "min_positive_fold_fraction": self.min_positive_fold_fraction,
            "practical_min_skill": self.practical_min_skill,
            "raw_signal_alpha": self.raw_signal_alpha,
        }


@dataclass(frozen=True)
class WalkForwardConfig:
    """One benchmark, completely specified."""

    models: tuple[str, ...] = CURATED_MODELS
    horizons: tuple[int, ...] = HORIZONS
    windows: tuple[WindowSpec, ...] = DEFAULT_WINDOWS
    #: Refit origins, evenly spaced over the evaluation span. Between refits the
    #: parameters are frozen and the forecast origin advances one bar at a time.
    n_refits: int = 12
    #: Chronological blocks for the regime view: early, middle, late at minimum.
    n_blocks: int = 3
    #: Raw bars before a fold's first training row that may be read, and only to
    #: settle features such as the EMA. Anything older is out of window.
    warmup_bars: int = 60
    #: The tail of each training window a deep model holds out to decide when to
    #: stop. Taken from the window, never from after the origin.
    dev_fraction: float = 0.2
    baseline: str = BASELINE
    loss: Literal["absolute"] = "absolute"
    bootstrap_resamples: int = 1000
    seed: int = 20260909
    #: Below this many paired origins a Diebold-Mariano test is not reported.
    minimum_paired_origins: int = 100
    gates: GateThresholds = field(default_factory=GateThresholds)
    preprocessing_version: str = PREPROCESSING_VERSION

    def __post_init__(self) -> None:
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be a non-empty list without duplicates")
        for required in (BASELINE, DRIFT_BASELINE):
            if required not in self.models:
                raise ValueError(
                    f"{required!r} is required: every benchmark carries the naive and "
                    "drift baselines, scored on the same target as everything else"
                )
        if self.baseline != BASELINE:
            raise ValueError(f"the comparison baseline is {BASELINE!r}; got {self.baseline!r}")
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("horizons must be positive")
        if tuple(sorted(set(self.horizons))) != tuple(self.horizons):
            raise ValueError("horizons must be unique and ascending")
        labels = [w.label for w in self.windows]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("windows must be non-empty and unique")
        if self.n_refits < 2:
            raise ValueError("at least two refit folds are needed to say anything about stability")
        if self.n_blocks < 3:
            raise ValueError("at least three chronological blocks: early, middle and late")
        if self.warmup_bars < 1:
            raise ValueError("warmup_bars must be positive")
        if not 0.0 < self.dev_fraction < 0.5:
            raise ValueError("dev_fraction must be in (0, 0.5)")
        if self.bootstrap_resamples < 100:
            raise ValueError("fewer than 100 bootstrap resamples is not an interval")
        if self.minimum_paired_origins < 30:
            raise ValueError("minimum_paired_origins below 30 would report noise as a test")

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def largest_fixed_window(self) -> int:
        fixed = [w.rows for w in self.windows if w.rows is not None]
        return max(fixed) if fixed else 0

    def as_dict(self) -> dict:
        return {
            "models": list(self.models),
            "horizons": list(self.horizons),
            "windows": [w.label for w in self.windows],
            "n_refits": self.n_refits,
            "n_blocks": self.n_blocks,
            "warmup_bars": self.warmup_bars,
            "dev_fraction": self.dev_fraction,
            "baseline": self.baseline,
            "loss": self.loss,
            "bootstrap_resamples": self.bootstrap_resamples,
            "seed": self.seed,
            "minimum_paired_origins": self.minimum_paired_origins,
            "gates": self.gates.as_dict(),
            "preprocessing_version": self.preprocessing_version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> WalkForwardConfig:
        data = dict(payload)
        data["models"] = tuple(data["models"])
        data["horizons"] = tuple(data["horizons"])
        data["windows"] = tuple(WindowSpec.parse(label) for label in data["windows"])
        data["gates"] = GateThresholds(**data["gates"])
        return cls(**data)

    def canonical_json(self) -> str:
        """Sorted keys, no whitespace: the bytes the digest is taken over."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


__all__ = [
    "BASELINE",
    "CURATED_MODELS",
    "DEFAULT_WINDOWS",
    "DRIFT_BASELINE",
    "HORIZONS",
    "MIN_ROLLING_ROWS",
    "PREPROCESSING_VERSION",
    "GateThresholds",
    "WalkForwardConfig",
    "WindowKind",
    "WindowSpec",
]
