"""Shared plumbing for every model that consumes the causal design matrix.

The linear, tree, kernel and quantile families differ in what they estimate and
agree completely on how they see the data: one causal feature row per forecast
origin, scaled by statistics taken from the training rows alone, predicting the
next bar's log return.

Putting that in one place is not tidiness. Every one of these adapters could
independently get the scaler wrong -- fit it on the evaluation block, or forget
to store it and re-derive it at predict time -- and each mistake would improve
the metrics and raise nothing. There is one implementation, and it is tested
once.

Determinism is enforced the same way: every estimator that accepts a
``random_state`` gets :data:`SEED`, and a test asserts that two fits of the same
model on the same rows produce bit-identical forecasts. A benchmark that cannot
be re-run to the same number is not evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import (
    Capability,
    EvaluationContext,
    ModelFitError,
    Preprocessing,
    TrainingSet,
    ZooModel,
)
from ..preprocessing import Scaler, fit_scaler

#: One seed for the whole zoo. Recorded in every manifest.
SEED = 20260909


class TabularModel(ZooModel):
    """A model that maps a causal feature row to a next-bar log return.

    Subclasses supply :meth:`_estimator`; everything else -- scaling, storing
    the scaler, refusing to predict on mismatched columns -- happens here.
    """

    preprocessing = Preprocessing.STANDARDIZED
    capabilities = frozenset({Capability.POINT, Capability.SERIALIZE})
    requires = ("sklearn",)

    def __init__(self) -> None:
        super().__init__()
        self._estimator_instance: Any = None
        self._scaler: Scaler | None = None
        self._feature_names: tuple[str, ...] = ()

    def _estimator(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fit(self, train: TrainingSet) -> None:
        # The scaler is fitted here, on the training rows, and kept. Re-deriving
        # it at predict time from whatever matrix arrives is the mistake this
        # base class exists to make impossible.
        self._scaler = fit_scaler(train.X, self.preprocessing)
        self._feature_names = tuple(train.X.columns)
        X = self._scaler.transform(train.X).to_numpy(dtype=float)
        y = train.y.to_numpy(dtype=float)
        # Constructed outside the guard on purpose. A subclass that never
        # supplied an estimator is a programming error, and wrapping it as a
        # model failure would record "adaboost did not converge" against a
        # missing method -- the benchmark would carry on and the bug would ship.
        estimator = self._estimator()
        try:
            self._estimator_instance = estimator.fit(X, y)
        except Exception as exc:  # noqa: BLE001 -- a genuine fit failure, recorded
            raise ModelFitError(f"{self.model_id} failed to fit: {exc}") from exc

    def _design(self, context: EvaluationContext) -> np.ndarray:
        if self._scaler is None:
            raise ModelFitError(f"{self.model_id} has no fitted scaler")
        if tuple(context.X.columns) != self._feature_names:
            raise ModelFitError(
                f"{self.model_id} was fitted on {list(self._feature_names)} "
                f"and asked to predict on {list(context.X.columns)}"
            )
        return self._scaler.transform(context.X).to_numpy(dtype=float)

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        return np.asarray(
            self._estimator_instance.predict(self._design(context)), dtype=float
        )

    def parameter_count(self) -> int | None:
        """Learned parameters where the notion is well defined.

        A linear model has coefficients plus an intercept. A forest does not
        have "parameters" in any comparable sense, so its subclass reports node
        counts under its own key and this returns None rather than inventing a
        number that would sit in the same column as a coefficient count.
        """
        estimator = self._estimator_instance
        if estimator is None:
            return None
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return None
        intercept = getattr(estimator, "intercept_", None)
        extra = 0 if intercept is None else int(np.size(intercept))
        return int(np.size(coefficients)) + extra

    def hyperparameters(self) -> dict:
        estimator = self._estimator_instance
        configured = self._configuration()
        if estimator is None:
            return configured
        return {**configured, "seed": SEED, "n_features": len(self._feature_names)}

    def _configuration(self) -> dict:
        """The frozen, resource-bounded settings this model was given."""
        return {}


def seeded(estimator_class: Any, **kwargs: Any) -> Any:
    """Construct an estimator with the zoo seed, if it accepts one.

    Checked by signature rather than by a hand-maintained list of which
    estimators are stochastic -- that list would go stale the first time
    scikit-learn changed one.
    """
    import inspect

    if "random_state" in inspect.signature(estimator_class).parameters:
        kwargs.setdefault("random_state", SEED)
    return estimator_class(**kwargs)


__all__ = ["SEED", "TabularModel", "seeded"]
