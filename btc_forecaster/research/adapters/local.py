"""K-nearest-neighbour regression, and an honest account of what it means here.

Phase 7 asks for KNN "only if the temporal feature representation makes the
experiment meaningful", and the honest answer is: **partly**.

What makes it meaningful. The design matrix is not raw prices -- it is lagged
returns, rolling means, rolling volatilities and price-over-SMA ratios, all of
which are approximately stationary. Euclidean distance between two such rows is
a defensible statement that two days looked alike, which is exactly the
analogue-forecasting question KNN asks: *when the market last looked like this,
what happened next?*

What limits it, stated rather than buried:

* **The curse.** Eleven dimensions and 1,000 points is sparse. The nearest
  neighbour of a query is not close in any absolute sense, and "nearest" starts
  to mean "least far".
* **Non-stationary scale.** A 2% daily move in 2018 and in 2023 are the same
  number and different events. Standardising on training statistics helps and
  does not fix it.
* **Temporal correlation.** Adjacent rows overlap in their rolling windows, so
  a query's neighbours are often its own calendar neighbours. That inflates
  apparent similarity without adding independent evidence.

It is registered because the analogue question is genuinely interesting on this
series and the answer is informative either way -- not to increase the model
count. The limitations above are on the model card, and a distance diagnostic is
reported alongside the forecast so a reader can see how far "nearest" actually
was.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import Family, ResourceClass
from ..registry import ZooRegistration, register
from ._tabular import TabularModel, seeded


class KnnRegression(TabularModel):
    """Distance-weighted KNN on standardised causal features."""

    model_id = "knn"
    family = Family.KERNEL_LOCAL
    resource_class = ResourceClass.TRIVIAL
    n_neighbors = 25
    weights = "distance"
    considered = (10, 25, 50)

    def _estimator(self) -> Any:
        from sklearn.neighbors import KNeighborsRegressor

        return seeded(
            KNeighborsRegressor, n_neighbors=self.n_neighbors, weights=self.weights, n_jobs=1
        )

    def _configuration(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "metric": "euclidean",
            "considered_on_dev": list(self.considered),
        }

    def neighbour_distances(self, context: Any) -> dict:
        """How far the "nearest" neighbours actually are.

        Reported so the curse of dimensionality is visible as a number rather
        than as a caveat. If the mean distance to the nearest neighbour is
        comparable to the mean distance to a random training row, the model is
        averaging the training set with extra steps.
        """
        design = self._design(context)
        distances, _ = self._estimator_instance.kneighbors(design)
        return {
            "mean_nearest": float(distances[:, 0].mean()),
            "mean_kth": float(distances[:, -1].mean()),
            "k": int(self.n_neighbors),
        }

    def hyperparameters(self) -> dict:
        base = super().hyperparameters()
        estimator = self._estimator_instance
        if estimator is not None:
            # Distance to the k-th neighbour within the training set itself:
            # the scale against which any query distance should be read.
            distances, _ = estimator.kneighbors()
            base["train_mean_kth_neighbour_distance"] = float(distances[:, -1].mean())
            base["n_features"] = len(self._feature_names)
        return base

    def parameter_count(self) -> int | None:
        """None, and deliberately so.

        KNN stores the training set; it fits nothing. Reporting 1,000 x 11 as a
        "parameter count" would put memorised data in the same column as a
        Ridge's coefficients.
        """
        return None


register(
    ZooRegistration(
        model_id="knn",
        factory=KnnRegression,
        family=Family.KERNEL_LOCAL,
        resource_class=ResourceClass.TRIVIAL,
        description="Distance-weighted 25-NN on standardised causal features.",
        requires=("sklearn",),
        notes=(
            "Meaningful because the features are approximately stationary -- "
            "lagged returns and rolling statistics, not raw prices -- so "
            "Euclidean distance is a defensible statement that two days looked "
            "alike.",
            "Limited by dimensionality: 11 features and 1,000 points is sparse, "
            "and 'nearest' begins to mean 'least far'. The distance to the k-th "
            "training neighbour is reported so that is visible as a number.",
            "Limited by temporal correlation: adjacent rows share rolling "
            "windows, so a query's neighbours are often its calendar neighbours, "
            "which inflates similarity without adding independent evidence.",
            "Fits nothing and stores everything, so it reports no parameter "
            "count rather than reporting the size of the training set.",
        ),
    )
)


__all__ = ["KnnRegression"]
