"""One contract every model in the zoo implements, and one that admits ignorance.

Forty models from six families cannot be compared unless they answer the same
question in the same units. But they genuinely differ in *what they can say*: a
Ridge regression produces a number, a quantile gradient booster produces a
distribution, and an ARIMA produces both. The usual way to paper over that is to
give every model a `predict_proba` and fabricate whatever it cannot compute.

This contract does the opposite. Capabilities are **declared**, and asking a
model for something it did not declare raises rather than guessing. A
deterministic point forecaster has no probability to report, and a benchmark
table with a blank cell is more honest than one filled by a Gaussian assumption
nobody chose.

The evaluation protocol is one-step-ahead, and it is the same for every family:

* parameters are estimated **once**, on the training partition only;
* at each evaluation origin ``t`` the model predicts the log return of bar
  ``t+1``, conditioning only on realised data up to and including ``t``;
* series models roll their state forward with realised values without
  re-estimating; tabular models consume the causal feature row at ``t``.

The structural guard is :meth:`EvaluationContext.history_at`, which is the only
way to reach the series and cannot return a bar after its origin. A model that
wants to cheat has to be written to cheat -- and the leakage suite injects
future rows and asserts nothing moves.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Family(str, Enum):
    """The six families the zoo reports a best-in-class for.

    Reported per family, never as a global winner: "the best of six deep models
    at n=1,000" is a statement about six models, and collapsing forty into one
    ranking invites exactly the multiple-comparison error Phase 21 exists for.
    """

    BASELINE = "BASELINE"
    STATISTICAL = "STATISTICAL"
    VOLATILITY = "VOLATILITY"
    LINEAR_ML = "LINEAR_ML"
    TREE_ENSEMBLE = "TREE_ENSEMBLE"
    KERNEL_LOCAL = "KERNEL_LOCAL"
    PROBABILISTIC = "PROBABILISTIC"
    DEEP = "DEEP"


class ResourceClass(str, Enum):
    """What a model costs, declared before it runs so a budget can be enforced."""

    TRIVIAL = "TRIVIAL"
    LIGHT = "LIGHT"
    MODERATE = "MODERATE"
    HEAVY = "HEAVY"


#: Wall-clock ceiling per model, by declared class. A model that exceeds its own
#: declaration is recorded as RESOURCE_LIMIT rather than left running: an
#: experiment with no deadline is an experiment that never reports.
RESOURCE_BUDGET_SECONDS: dict[ResourceClass, float] = {
    ResourceClass.TRIVIAL: 5.0,
    ResourceClass.LIGHT: 30.0,
    ResourceClass.MODERATE: 120.0,
    ResourceClass.HEAVY: 600.0,
}


class Preprocessing(str, Enum):
    """How a model wants its inputs, declared rather than inferred.

    Scaler statistics are fitted on the training rows only. Fitting a scaler on
    train+eval is the quietest leak in applied machine learning: it changes no
    prediction visibly and improves every metric slightly.
    """

    NONE = "NONE"
    STANDARDIZED = "STANDARDIZED"
    ROBUST = "ROBUST"
    MODEL_NATIVE = "MODEL_NATIVE"


class Capability(str, Enum):
    """What a model can be asked for. Undeclared means unavailable, not zero."""

    POINT = "POINT"
    QUANTILES = "QUANTILES"
    DIRECTION_PROBABILITY = "DIRECTION_PROBABILITY"
    VARIANCE = "VARIANCE"
    SERIALIZE = "SERIALIZE"


class ModelStatus(str, Enum):
    """Where a model stands in this lab. Never silently omitted."""

    ACTIVE = "ACTIVE"
    OPTIONAL = "OPTIONAL"
    UNSUITABLE_FOR_CONSTRAINED_LAB = "UNSUITABLE_FOR_CONSTRAINED_LAB"
    FAILED = "FAILED"
    SKIPPED_DEPENDENCY = "SKIPPED_DEPENDENCY"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


#: The one scientific status any A6 model may hold. There is no other value, and
#: the runner asserts it -- a promotion decision needs a preregistered study with
#: a sample size this lab does not have.
EXPLORATORY = "EXPLORATORY"


class CapabilityNotSupported(NotImplementedError):
    """Raised when a model is asked for something it did not declare.

    Deliberately loud. The alternative -- returning a fabricated Gaussian
    interval or a 0.5 probability -- produces a number that scores, tabulates
    and misleads.
    """


class ModelFitError(RuntimeError):
    """A model failed to fit. Recorded against the model, never swallowed."""


@dataclass(frozen=True)
class TrainingSet:
    """Everything a model may see while estimating parameters, and nothing else.

    Both representations of the same rows, aligned:

    * ``X`` / ``y`` -- the causal design matrix, where row ``t`` holds features
      computed at bar ``t`` and ``y[t]`` is the log return of bar ``t+1``;
    * ``series`` -- the log-return series itself, for models that estimate on a
      series rather than a design matrix.

    ``feature_bar`` records which bar each predictor row came from, so the
    alignment is auditable rather than an unstated assumption about a shift.
    """

    X: pd.DataFrame
    y: pd.Series
    series: pd.Series
    close: pd.Series
    feature_bar: pd.Series

    def __post_init__(self) -> None:
        if not self.X.index.equals(self.y.index):
            raise ValueError("X and y must share an index")
        if len(self.X) == 0:
            raise ValueError("training set is empty")
        if self.X.isna().to_numpy().any():
            raise ValueError("training features contain NaN; the builder should have dropped them")
        # The whole point-in-time claim in one assertion: the bar a row's
        # predictors came from is strictly before the bar it predicts.
        if bool((pd.DatetimeIndex(self.feature_bar) >= self.X.index).any()):
            raise ValueError("a feature row is not strictly before the bar it predicts")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def fingerprint(self) -> str:
        """Content hash of the exact rows a model was fitted on."""
        payload = np.concatenate(
            [
                self.X.to_numpy(dtype=float).ravel(),
                self.y.to_numpy(dtype=float).ravel(),
            ]
        )
        digest = hashlib.sha256(payload.tobytes())
        digest.update(json.dumps(self.feature_names, sort_keys=True).encode())
        digest.update(str(self.X.index[0]).encode())
        digest.update(str(self.X.index[-1]).encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationContext:
    """The evaluation block, exposed so that reaching the future is impossible.

    ``X`` holds one causal feature row per evaluation origin and ``y`` the
    realised log return of the bar after it. ``series`` is the *full* realised
    log-return history, but a model reaches it only through
    :meth:`history_at`, which slices at the origin.

    That is the structural guard: a model cannot accidentally index past its
    origin, because the accessor will not hand it those rows. It can still be
    written to cheat -- `test_a6_leakage.py` injects poisoned future bars and
    asserts every prediction is bit-identical.
    """

    X: pd.DataFrame
    y: pd.Series
    series: pd.Series
    close: pd.Series
    train_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.X.index.equals(self.y.index):
            raise ValueError("X and y must share an index")
        if len(self.X) == 0:
            raise ValueError("evaluation context is empty")
        if bool((self.X.index <= self.train_end).any()):
            raise ValueError("an evaluation origin is inside the training partition")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def origins(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.X.index)

    def history_at(self, i: int) -> pd.Series:
        """Realised log returns up to and including evaluation origin ``i``.

        The only route to the series. It cannot return a bar after the origin,
        so a model that uses this accessor is causal by construction.
        """
        if not 0 <= i < len(self.X):
            raise IndexError(f"evaluation origin {i} out of range (0..{len(self.X) - 1})")
        origin = self.X.index[i]
        return self.series.loc[self.series.index <= origin]

    def close_at(self, i: int) -> pd.Series:
        """Realised closes up to and including evaluation origin ``i``."""
        if not 0 <= i < len(self.X):
            raise IndexError(f"evaluation origin {i} out of range (0..{len(self.X) - 1})")
        origin = self.X.index[i]
        return self.close.loc[self.close.index <= origin]


@dataclass(frozen=True)
class ZooForecast:
    """One model's one-step-ahead forecasts over the evaluation block.

    ``point`` is a log return per origin. Optional fields are ``None`` when the
    model did not declare the capability -- never a filled-in default.
    """

    model_id: str
    point: np.ndarray
    quantiles: dict[float, np.ndarray] | None = None
    direction_probability: np.ndarray | None = None
    variance: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.point)
        for name, value in (
            ("direction_probability", self.direction_probability),
            ("variance", self.variance),
        ):
            if value is not None and len(value) != n:
                raise ValueError(f"{name} must have one value per origin")
        if self.quantiles is not None:
            for q, values in self.quantiles.items():
                if not 0.0 < q < 1.0:
                    raise ValueError(f"quantile {q} is not in (0, 1)")
                if len(values) != n:
                    raise ValueError(f"quantile {q} must have one value per origin")
            levels = sorted(self.quantiles)
            stacked = np.vstack([self.quantiles[q] for q in levels])
            if bool((np.diff(stacked, axis=0) < -1e-9).any()):
                raise ValueError(f"{self.model_id} produced crossing quantiles")
        if self.direction_probability is not None:
            p = self.direction_probability
            if bool(((p < 0.0) | (p > 1.0)).any()):
                raise ValueError(f"{self.model_id} produced a probability outside [0, 1]")


class ZooModel(ABC):
    """The single interface. Capabilities are declared, never assumed.

    Subclasses implement :meth:`_fit` and :meth:`_predict_point`. Anything
    richer -- quantiles, direction probabilities, conditional variance -- is
    opt-in by declaring the capability and overriding the matching hook.
    """

    #: Stable identity. The registry key, and what every table and card uses.
    model_id: str = "unnamed"
    family: Family = Family.BASELINE
    version: str = "1"
    resource_class: ResourceClass = ResourceClass.LIGHT
    preprocessing: Preprocessing = Preprocessing.NONE
    capabilities: frozenset[Capability] = frozenset({Capability.POINT})
    #: Third-party modules required. Absent means SKIPPED_DEPENDENCY, not a crash.
    requires: tuple[str, ...] = ()
    #: Set on a model that genuinely cannot run in this lab, with the reason.
    unsuitable_reason: str | None = None

    def __init__(self) -> None:
        self._fitted = False
        self._train_fingerprint: str | None = None

    # -- capability ------------------------------------------------------

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def _require(self, capability: Capability) -> None:
        if not self.supports(capability):
            raise CapabilityNotSupported(
                f"{self.model_id} does not support {capability.value}; "
                f"it declares {sorted(c.value for c in self.capabilities)}"
            )

    @property
    def requires_scaling(self) -> bool:
        return self.preprocessing in (Preprocessing.STANDARDIZED, Preprocessing.ROBUST)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # -- lifecycle -------------------------------------------------------

    def fit(self, train: TrainingSet) -> ZooModel:
        self._train_fingerprint = train.fingerprint()
        self._fit(train)
        self._fitted = True
        return self

    def predict(self, context: EvaluationContext) -> ZooForecast:
        if not self._fitted:
            raise ModelFitError(f"{self.model_id} has not been fitted")
        point = np.asarray(self._predict_point(context), dtype=float)
        if len(point) != len(context):
            raise ValueError(
                f"{self.model_id} returned {len(point)} predictions for {len(context)} origins"
            )
        if not np.isfinite(point).all():
            raise ModelFitError(f"{self.model_id} produced a non-finite point forecast")

        quantiles = (
            self._predict_quantiles(context) if self.supports(Capability.QUANTILES) else None
        )
        direction = (
            self._predict_direction_probability(context)
            if self.supports(Capability.DIRECTION_PROBABILITY)
            else None
        )
        variance = self._predict_variance(context) if self.supports(Capability.VARIANCE) else None
        return ZooForecast(
            model_id=self.model_id,
            point=point,
            quantiles=quantiles,
            direction_probability=direction,
            variance=variance,
            metadata=self.describe(),
        )

    # -- the four optional hooks ----------------------------------------

    def predict_distribution(self, context: EvaluationContext) -> dict[float, np.ndarray]:
        self._require(Capability.QUANTILES)
        return self._predict_quantiles(context)

    def predict_direction_probability(self, context: EvaluationContext) -> np.ndarray:
        self._require(Capability.DIRECTION_PROBABILITY)
        return self._predict_direction_probability(context)

    def predict_variance(self, context: EvaluationContext) -> np.ndarray:
        self._require(Capability.VARIANCE)
        return self._predict_variance(context)

    # -- description -----------------------------------------------------

    def describe(self) -> dict:
        """What a run manifest records about this model."""
        return {
            "model_id": self.model_id,
            "family": self.family.value,
            "version": self.version,
            "resource_class": self.resource_class.value,
            "preprocessing": self.preprocessing.value,
            "capabilities": sorted(c.value for c in self.capabilities),
            "requires": list(self.requires),
            "hyperparameters": self.hyperparameters(),
            "train_fingerprint": self._train_fingerprint,
            "scientific_status": EXPLORATORY,
        }

    def hyperparameters(self) -> dict:
        """The frozen, resource-bounded configuration this model was given."""
        return {}

    def parameter_count(self) -> int | None:
        """Learned parameters, where the notion is well defined. None otherwise."""
        return None

    # -- subclass surface ------------------------------------------------

    @abstractmethod
    def _fit(self, train: TrainingSet) -> None: ...

    @abstractmethod
    def _predict_point(self, context: EvaluationContext) -> np.ndarray: ...

    def _predict_quantiles(self, context: EvaluationContext) -> dict[float, np.ndarray]:
        raise CapabilityNotSupported(f"{self.model_id} declared QUANTILES but did not implement it")

    def _predict_direction_probability(self, context: EvaluationContext) -> np.ndarray:
        raise CapabilityNotSupported(
            f"{self.model_id} declared DIRECTION_PROBABILITY but did not implement it"
        )

    def _predict_variance(self, context: EvaluationContext) -> np.ndarray:
        raise CapabilityNotSupported(f"{self.model_id} declared VARIANCE but did not implement it")


__all__ = [
    "EXPLORATORY",
    "RESOURCE_BUDGET_SECONDS",
    "Capability",
    "CapabilityNotSupported",
    "EvaluationContext",
    "Family",
    "ModelFitError",
    "ModelStatus",
    "Preprocessing",
    "ResourceClass",
    "TrainingSet",
    "ZooForecast",
    "ZooModel",
]
