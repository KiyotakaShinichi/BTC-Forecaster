"""Trees and ensembles, deliberately kept small.

Every model here is capable of fitting 1,000 rows of daily returns perfectly and
learning nothing. Depth, leaf size and estimator count are therefore frozen at
values chosen for the sample size rather than for accuracy on a development
split: max_depth 3-5, min_samples_leaf 20, and a few hundred shallow trees.

That is a real constraint, not a nod to one. An unbounded random forest on this
matrix reaches a training R^2 near 1 and an evaluation skill near zero, and the
gap is the entire lesson. Bounding the capacity in advance means the benchmark
measures the family rather than measuring how long somebody was willing to tune.

Trees are also the one family here that needs no scaling: they split on
thresholds, and a monotone rescale of a feature moves the threshold rather than
the split. They declare `NONE` and go through the same code path anyway, so the
recorded preprocessing is a fact about each model rather than an assumption
about its family.

XGBoost is registered as OPTIONAL rather than ACTIVE: it is a separate wheel
outside the core install, and a benchmark that cannot run without it would make
the whole zoo depend on an extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import Family, Preprocessing, ResourceClass
from ..registry import ZooRegistration, register
from ._tabular import SEED, TabularModel, seeded

#: Shared capacity ceiling. One number, so "small" means the same thing across
#: the family and a change to it is a change to all of them at once.
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 20
N_ESTIMATORS = 300


class TreeModel(TabularModel):
    """Trees split on thresholds, so scaling is a no-op for them."""

    family = Family.TREE_ENSEMBLE
    preprocessing = Preprocessing.NONE

    def parameter_count(self) -> int | None:
        """Node counts, not coefficients.

        Returned as None from the shared column and reported under its own key,
        because a forest's 12,000 nodes and a Ridge's 12 coefficients are not
        the same quantity and must not share an axis.
        """
        return None

    def _node_count(self) -> int | None:
        estimator = self._estimator_instance
        if estimator is None:
            return None
        if hasattr(estimator, "tree_"):
            return int(estimator.tree_.node_count)
        sub = getattr(estimator, "estimators_", None)
        if sub is None:
            return None
        flat = np.ravel(np.asarray(sub, dtype=object))
        counts = [int(e.tree_.node_count) for e in flat if hasattr(e, "tree_")]
        return sum(counts) if counts else None

    def hyperparameters(self) -> dict:
        base = super().hyperparameters()
        nodes = self._node_count()
        if nodes is not None:
            base["total_nodes"] = nodes
        estimator = self._estimator_instance
        importances = getattr(estimator, "feature_importances_", None)
        if importances is not None and len(self._feature_names) == len(importances):
            ranked = sorted(
                zip(self._feature_names, [float(v) for v in importances], strict=False),
                key=lambda item: -item[1],
            )
            base["top_features"] = ranked[:3]
        return base


class DecisionTree(TreeModel):
    """One tree. The interpretable control the ensembles are measured against."""

    model_id = "decision_tree"
    resource_class = ResourceClass.TRIVIAL

    def _estimator(self) -> Any:
        from sklearn.tree import DecisionTreeRegressor

        return seeded(
            DecisionTreeRegressor, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF
        )

    def _configuration(self) -> dict:
        return {"max_depth": MAX_DEPTH, "min_samples_leaf": MIN_SAMPLES_LEAF}


class RandomForest(TreeModel):
    """Bagging plus feature subsampling. Variance reduction, not bias reduction."""

    model_id = "random_forest"
    resource_class = ResourceClass.LIGHT

    def _estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return seeded(
            RandomForestRegressor,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            n_jobs=1,
        )

    def _configuration(self) -> dict:
        return {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
        }


class ExtraTrees(TreeModel):
    """Random split thresholds rather than optimal ones.

    More variance reduction than a random forest and more bias. Registered
    beside it because on a low-signal series the extra randomisation is a
    genuine hypothesis, not a footnote.
    """

    model_id = "extra_trees"
    resource_class = ResourceClass.LIGHT

    def _estimator(self) -> Any:
        from sklearn.ensemble import ExtraTreesRegressor

        return seeded(
            ExtraTreesRegressor,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            n_jobs=1,
        )

    def _configuration(self) -> dict:
        return {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
        }


class AdaBoost(TreeModel):
    """Sequential reweighting toward hard examples.

    On a series where the hard examples are largely noise, that is a hypothesis
    worth testing explicitly rather than assuming.
    """

    model_id = "adaboost"
    resource_class = ResourceClass.LIGHT
    n_estimators = 200
    learning_rate = 0.05

    def _estimator(self) -> Any:
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor

        return seeded(
            AdaBoostRegressor,
            estimator=DecisionTreeRegressor(max_depth=3, random_state=SEED),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
        )

    def _configuration(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "base_max_depth": 3,
        }


class GradientBoosting(TreeModel):
    """Stagewise fitting of the residual. Shallow trees, small steps."""

    model_id = "gradient_boosting"
    resource_class = ResourceClass.LIGHT
    n_estimators = 300
    learning_rate = 0.03

    def _estimator(self) -> Any:
        from sklearn.ensemble import GradientBoostingRegressor

        return seeded(
            GradientBoostingRegressor,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=3,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            subsample=0.8,
        )

    def _configuration(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": 3,
            "subsample": 0.8,
        }


class HistGradientBoosting(TreeModel):
    """Histogram-binned boosting.

    Its advantage is speed at large n, which this lab does not have -- so it is
    here as a check that the binning approximation costs nothing at n=1,000,
    rather than as a faster way to get the same answer.
    """

    model_id = "hist_gradient_boosting"
    resource_class = ResourceClass.LIGHT
    max_iter = 300
    learning_rate = 0.03

    def _estimator(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return seeded(
            HistGradientBoostingRegressor,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            early_stopping=False,
        )

    def _configuration(self) -> dict:
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": MAX_DEPTH,
            "early_stopping": False,
        }

    def _node_count(self) -> int | None:
        return None


class XgboostCausal(TreeModel):
    """XGBoost on the same causal matrix, at the same capacity ceiling.

    Registered OPTIONAL: it is a separate wheel, and a zoo whose default
    benchmark cannot run without an extra is a zoo with a hidden dependency.

    Not the same model as A2's `xgboost_causal_retuned`, which selected its
    hyperparameters by nested inner validation inside every one of 36 folds.
    This one is frozen at the family ceiling and fitted once on 1,000 rows. The
    two are not comparable and the model card says so.
    """

    model_id = "xgboost"
    resource_class = ResourceClass.LIGHT
    requires = ("xgboost",)
    n_estimators = 300
    learning_rate = 0.03

    def _estimator(self) -> Any:
        from xgboost import XGBRegressor

        return seeded(
            XGBRegressor,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=MAX_DEPTH,
            min_child_weight=MIN_SAMPLES_LEAF,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=1,
            verbosity=0,
        )

    def _configuration(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": MAX_DEPTH,
            "min_child_weight": MIN_SAMPLES_LEAF,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    def _node_count(self) -> int | None:
        return None


for _cls, _description, _requires, _notes in (
    (
        DecisionTree,
        f"Single regression tree, max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}.",
        ("sklearn",),
        ("The interpretable control the ensembles are measured against.",),
    ),
    (
        RandomForest,
        f"{N_ESTIMATORS} bagged trees at the shared capacity ceiling.",
        ("sklearn",),
        ("Capacity frozen for n=1,000; an unbounded forest reaches training R^2 ~ 1 "
         "and evaluation skill ~ 0, and that gap is the lesson.",),
    ),
    (
        ExtraTrees,
        f"{N_ESTIMATORS} extremely randomised trees at the shared ceiling.",
        ("sklearn",),
        ("Random split thresholds: more variance reduction, more bias.",),
    ),
    (
        AdaBoost,
        "AdaBoost.R2 over depth-3 stumps, 200 rounds at lr=0.05.",
        ("sklearn",),
        ("Reweights toward hard examples, which on this series are largely noise -- "
         "a hypothesis worth testing rather than assuming.",),
    ),
    (
        GradientBoosting,
        "Stagewise gradient boosting, 300 rounds at lr=0.03, subsample 0.8.",
        ("sklearn",),
        (),
    ),
    (
        HistGradientBoosting,
        "Histogram-binned gradient boosting, 300 iterations, early stopping off.",
        ("sklearn",),
        ("Its advantage is speed at large n, which this lab does not have; it is "
         "here to check the binning approximation costs nothing at n=1,000.",),
    ),
    (
        XgboostCausal,
        "XGBoost on the causal matrix at the shared capacity ceiling.",
        ("xgboost",),
        ("NOT comparable to A2's xgboost_causal_retuned, which selected "
         "hyperparameters by nested inner validation inside all 36 folds. This "
         "one is frozen and fitted once on 1,000 rows.",),
    ),
):
    register(
        ZooRegistration(
            model_id=_cls.model_id,
            factory=_cls,
            family=Family.TREE_ENSEMBLE,
            resource_class=_cls.resource_class,
            description=_description,
            requires=_requires,
            notes=_notes,
        )
    )


__all__ = [
    "MAX_DEPTH",
    "MIN_SAMPLES_LEAF",
    "N_ESTIMATORS",
    "AdaBoost",
    "DecisionTree",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "RandomForest",
    "TreeModel",
    "XgboostCausal",
]
