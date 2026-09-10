"""The linear and kernel-regression family, on causal lag and rolling features.

Eight estimators that all fit the same design matrix and differ in what they
penalise or how they measure error. At n=1,000 with 11 features that is a
regime where regularisation choice matters more than model class, which is
precisely why the family is worth having in one comparison rather than being
represented by whichever variant somebody reached for first.

Every one of them is standardised, and that is not optional: a penalty term is
a statement about coefficient magnitude, so an unscaled matrix makes the penalty
mean something different for each column. The scaler is fitted on the training
rows and stored, never re-derived at predict time.

Regularisation strengths are **frozen**, not searched. Phase 16 permits at most
three development-only configurations; where a choice was made, the alternatives
considered are recorded on the model so the manifest shows what was compared
rather than only what won.
"""

from __future__ import annotations

from typing import Any

from ..contracts import Family, Preprocessing, ResourceClass
from ..registry import ZooRegistration, register
from ._tabular import TabularModel, seeded


class Ridge(TabularModel):
    """L2 penalty. The default answer when predictors are correlated, which
    lag and rolling features of one series inevitably are."""

    model_id = "ridge"
    family = Family.LINEAR_ML
    resource_class = ResourceClass.TRIVIAL
    alpha = 1.0
    considered = (0.1, 1.0, 10.0)

    def _estimator(self) -> Any:
        from sklearn.linear_model import Ridge as SkRidge

        return seeded(SkRidge, alpha=self.alpha)

    def _configuration(self) -> dict:
        return {"alpha": self.alpha, "considered_on_dev": list(self.considered)}


class Lasso(TabularModel):
    """L1 penalty: coefficients go to exactly zero, so it selects as it fits.

    Worth watching rather than scoring at this sample size -- which features
    survive is more interesting than the MAE.
    """

    model_id = "lasso"
    family = Family.LINEAR_ML
    resource_class = ResourceClass.TRIVIAL
    alpha = 1e-4
    considered = (1e-5, 1e-4, 1e-3)

    def _estimator(self) -> Any:
        from sklearn.linear_model import Lasso as SkLasso

        return seeded(SkLasso, alpha=self.alpha, max_iter=10_000)

    def _configuration(self) -> dict:
        return {"alpha": self.alpha, "considered_on_dev": list(self.considered)}

    def hyperparameters(self) -> dict:
        base = super().hyperparameters()
        estimator = self._estimator_instance
        if estimator is not None:
            nonzero = int((estimator.coef_ != 0).sum())
            base["nonzero_coefficients"] = nonzero
            base["selected_features"] = [
                name
                for name, coefficient in zip(self._feature_names, estimator.coef_, strict=False)
                if coefficient != 0
            ]
        return base


class ElasticNet(TabularModel):
    """L1 and L2 together. Between Ridge and Lasso, and registered because the
    mix ratio is the thing being tested, not the penalty type."""

    model_id = "elastic_net"
    family = Family.LINEAR_ML
    resource_class = ResourceClass.TRIVIAL
    alpha = 1e-4
    l1_ratio = 0.5

    def _estimator(self) -> Any:
        from sklearn.linear_model import ElasticNet as SkElasticNet

        return seeded(SkElasticNet, alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=10_000)

    def _configuration(self) -> dict:
        return {"alpha": self.alpha, "l1_ratio": self.l1_ratio}


class Huber(TabularModel):
    """Squared error inside a threshold, absolute outside it.

    The reason to include it is specific to this series: daily crypto returns
    carry genuine outliers, and least squares spends its coefficients on them.
    Huber is the direct test of whether that matters here.
    """

    model_id = "huber"
    family = Family.LINEAR_ML
    resource_class = ResourceClass.TRIVIAL
    epsilon = 1.35

    def _estimator(self) -> Any:
        from sklearn.linear_model import HuberRegressor

        return seeded(HuberRegressor, epsilon=self.epsilon, max_iter=1000)

    def _configuration(self) -> dict:
        return {"epsilon": self.epsilon}


class BayesianRidge(TabularModel):
    """Ridge with the penalty inferred rather than chosen.

    Its value at n=1,000 is that it does not require the regularisation
    strength to be guessed -- the evidence approximation picks it -- so it is a
    control on how much the frozen alphas above are costing.
    """

    model_id = "bayesian_ridge"
    family = Family.LINEAR_ML
    resource_class = ResourceClass.TRIVIAL

    def _estimator(self) -> Any:
        from sklearn.linear_model import BayesianRidge as SkBayesianRidge

        return seeded(SkBayesianRidge)

    def _configuration(self) -> dict:
        return {"penalty": "inferred by evidence approximation"}


class LinearSvr(TabularModel):
    """Epsilon-insensitive linear regression: errors inside epsilon cost nothing.

    Distinct from Huber rather than a variant of it -- Huber down-weights large
    errors, this one ignores small ones.
    """

    model_id = "linear_svr"
    family = Family.LINEAR_ML
    resource_class = ResourceClass.LIGHT
    epsilon = 0.0
    C = 1.0

    def _estimator(self) -> Any:
        from sklearn.svm import LinearSVR

        return seeded(LinearSVR, epsilon=self.epsilon, C=self.C, max_iter=100_000, tol=1e-6)

    def _configuration(self) -> dict:
        return {"epsilon": self.epsilon, "C": self.C}


class RbfSvr(TabularModel):
    """Kernel SVR. The only genuinely non-linear member of this family.

    Its cost is quadratic in the sample, which is affordable at n=1,000 and is
    exactly the kind of thing that would not be affordable at n=100,000 -- one
    of the few places the constrained budget makes a model *more* usable rather
    than less.
    """

    model_id = "rbf_svr"
    family = Family.KERNEL_LOCAL
    resource_class = ResourceClass.LIGHT
    #: Chosen on DEV from three candidates, holdout untouched. C=1.0 lets the
    #: kernel fit the noise -- it reached -0.537 DEV skill against -0.149 here,
    #: with a prediction spread twice as wide as anything the series supports.
    #: An epsilon of 0.01 was also rejected: it exceeds half the target's own
    #: standard deviation, so the tube swallows the signal along with the noise.
    C = 0.1
    gamma = "scale"
    epsilon = 0.0
    considered = ((0.01, 1.0), (0.005, 1.0), (0.0, 0.1))

    def _estimator(self) -> Any:
        from sklearn.svm import SVR

        return seeded(SVR, kernel="rbf", C=self.C, gamma=self.gamma, epsilon=self.epsilon)

    def _configuration(self) -> dict:
        return {
            "kernel": "rbf",
            "C": self.C,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "considered_on_dev": [list(c) for c in self.considered],
            "dev_skill_of_considered": [-0.537, -0.539, -0.149],
        }


class PartialLeastSquares(TabularModel):
    """PLS: components chosen to covary with the target, not merely to explain X.

    Justified here by the same collinearity that motivates Ridge -- eleven
    features derived from one price series do not carry eleven directions of
    information. Two components, because at n=1,000 more is fitting noise.
    """

    model_id = "pls"
    family = Family.LINEAR_ML
    resource_class = ResourceClass.TRIVIAL
    n_components = 2

    def _estimator(self) -> Any:
        from sklearn.cross_decomposition import PLSRegression

        return seeded(PLSRegression, n_components=self.n_components)

    def _configuration(self) -> dict:
        return {"n_components": self.n_components}

    def _predict_point(self, context):  # type: ignore[no-untyped-def]
        # PLSRegression returns a column vector; everything else in the zoo
        # returns a flat array, and a silent (n, 1) would broadcast against the
        # actuals into an (n, n) matrix of nonsense.
        import numpy as np

        return np.asarray(
            self._estimator_instance.predict(self._design(context)), dtype=float
        ).ravel()


class RobustRidge(TabularModel):
    """Ridge on robustly scaled features.

    Registered as its own entry rather than as a flag on `ridge` because it
    tests a different hypothesis: not how much to penalise, but whether the
    outliers should be allowed to set the scale at all.
    """

    model_id = "ridge_robust_scaled"
    family = Family.LINEAR_ML
    resource_class = ResourceClass.TRIVIAL
    preprocessing = Preprocessing.ROBUST
    alpha = 1.0

    def _estimator(self) -> Any:
        from sklearn.linear_model import Ridge as SkRidge

        return seeded(SkRidge, alpha=self.alpha)

    def _configuration(self) -> dict:
        return {"alpha": self.alpha, "scaling": "median / IQR"}


for _cls, _description, _notes in (
    (Ridge, "L2-penalised least squares on standardised causal features.", ()),
    (
        Lasso,
        "L1-penalised least squares; reports which features survived.",
        ("At n=1,000 which coefficients survive is more informative than the MAE.",),
    ),
    (ElasticNet, "L1+L2 penalty at a 0.5 mix ratio.", ()),
    (
        Huber,
        "Huber loss: squared inside epsilon, absolute outside.",
        ("Included specifically because daily crypto returns carry real outliers.",),
    ),
    (
        BayesianRidge,
        "Ridge with the penalty inferred by evidence approximation.",
        ("A control on how much the frozen alphas cost the other linear models.",),
    ),
    (LinearSvr, "Epsilon-insensitive linear SVR.", ()),
    (
        RbfSvr,
        "RBF-kernel SVR; the only non-linear member of the linear/kernel group.",
        ("Quadratic in the sample size, which is affordable only because n=1,000.",),
    ),
    (
        PartialLeastSquares,
        "PLS regression with 2 components.",
        ("Eleven features from one price series do not carry eleven directions.",),
    ),
    (
        RobustRidge,
        "Ridge on median/IQR-scaled features.",
        ("Tests whether outliers should set the feature scale, not how much to penalise.",),
    ),
):
    register(
        ZooRegistration(
            model_id=_cls.model_id,
            factory=_cls,
            family=_cls.family,
            resource_class=_cls.resource_class,
            description=_description,
            requires=("sklearn",),
            notes=_notes,
        )
    )


__all__ = [
    "BayesianRidge",
    "ElasticNet",
    "Huber",
    "Lasso",
    "LinearSvr",
    "PartialLeastSquares",
    "RbfSvr",
    "Ridge",
    "RobustRidge",
]
