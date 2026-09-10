"""Linear, tree and local models: the shared design matrix and its hazards.

Seventeen estimators reach the data through one base class, and the reason that
base class exists is that each of them could independently get the scaler wrong
-- fit it on the evaluation block, or forget to store it and re-derive it at
predict time -- and every one of those mistakes improves the metrics and raises
nothing.

So the tests here are mostly about the shared path: the scaler is fitted once
and kept, the column set is checked, the seed makes the result reproducible, and
no model claims a capability it does not have. The per-model assertions are
about capacity ceilings and honest reporting, not accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.adapters._tabular import SEED, TabularModel, seeded
from btc_forecaster.research.adapters.trees import MAX_DEPTH, MIN_SAMPLES_LEAF
from btc_forecaster.research.contracts import (
    Capability,
    CapabilityNotSupported,
    Family,
    ModelFitError,
    ModelStatus,
    Preprocessing,
)
from btc_forecaster.research.partition import PartitionSpec, build_dataset
from btc_forecaster.testing import synthetic_market_frame

pytest.importorskip("sklearn", reason="the tabular families are OPTIONAL adapters")

LINEAR_IDS = registry.model_ids(family=Family.LINEAR_ML)
TREE_IDS = [m for m in registry.model_ids(family=Family.TREE_ENSEMBLE) if m != "xgboost"]
LOCAL_IDS = registry.model_ids(family=Family.KERNEL_LOCAL)
TABULAR_IDS = LINEAR_IDS + TREE_IDS + LOCAL_IDS


@pytest.fixture(scope="module")
def dataset():
    frame = synthetic_market_frame(periods=800, kind="ar1", seed=13)
    return build_dataset(frame, spec=PartitionSpec(train_rows=300))


@pytest.fixture(scope="module")
def fitted(dataset):
    train = dataset.training_set()
    return {mid: registry.build(mid).fit(train) for mid in TABULAR_IDS}


class TestTheSharedPathIsSafe:
    @pytest.mark.parametrize("model_id", TABULAR_IDS)
    def test_it_fits_and_predicts_one_value_per_origin(self, dataset, fitted, model_id) -> None:
        context = dataset.evaluation_context()
        forecast = fitted[model_id].predict(context)
        assert forecast.point.shape == (len(context),)
        assert np.isfinite(forecast.point).all()

    @pytest.mark.parametrize("model_id", TABULAR_IDS)
    def test_two_fits_give_bit_identical_forecasts(self, dataset, model_id) -> None:
        """A benchmark that cannot be re-run to the same number is not evidence."""
        train, context = dataset.training_set(), dataset.evaluation_context()
        first = registry.build(model_id).fit(train).predict(context).point
        second = registry.build(model_id).fit(train).predict(context).point
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("model_id", TABULAR_IDS)
    def test_the_scaler_is_stored_not_re_derived(self, dataset, fitted, model_id) -> None:
        """The leak this base class exists to prevent.

        A model that re-derived its scaling from whatever matrix arrived would
        centre the evaluation block on its own mean -- and would give a
        different answer when handed a subset of the same rows. This one must
        not.
        """
        context = dataset.evaluation_context()
        full = fitted[model_id].predict(context).point
        half = len(context) // 2

        from btc_forecaster.research.contracts import EvaluationContext

        subset = EvaluationContext(
            X=context.X.iloc[:half],
            y=context.y.iloc[:half],
            series=context.series,
            close=context.close,
            feature_bar=context.feature_bar.iloc[:half],
            train_end=context.train_end,
        )
        assert np.allclose(fitted[model_id].predict(subset).point, full[:half], atol=1e-12)

    @pytest.mark.parametrize("model_id", TABULAR_IDS)
    def test_a_different_feature_set_is_refused(self, dataset, fitted, model_id) -> None:
        from btc_forecaster.research.contracts import EvaluationContext

        context = dataset.evaluation_context()
        narrowed = EvaluationContext(
            X=context.X.iloc[:, :3],
            y=context.y,
            series=context.series,
            close=context.close,
            feature_bar=context.feature_bar,
            train_end=context.train_end,
        )
        with pytest.raises(ModelFitError, match="was fitted on"):
            fitted[model_id].predict(narrowed)

    @pytest.mark.parametrize("model_id", TABULAR_IDS)
    def test_it_claims_only_point_and_serialize(self, fitted, model_id) -> None:
        """No manufactured distributions. These are point regressors."""
        model = fitted[model_id]
        assert model.capabilities == frozenset({Capability.POINT, Capability.SERIALIZE})
        with pytest.raises(CapabilityNotSupported):
            model.predict_distribution(None)  # type: ignore[arg-type]

    def test_the_seed_is_applied_only_where_accepted(self) -> None:
        """Checked by signature rather than by a hand-maintained list of which
        estimators are stochastic -- that list goes stale the first time
        scikit-learn changes one."""
        from sklearn.linear_model import LinearRegression, Ridge

        assert seeded(Ridge).random_state == SEED
        assert not hasattr(seeded(LinearRegression), "random_state")


class TestScalingIsDeclaredPerModel:
    @pytest.mark.parametrize("model_id", LINEAR_IDS)
    def test_linear_models_are_scaled(self, fitted, model_id) -> None:
        """A penalty is a statement about coefficient magnitude, so an unscaled
        matrix makes it mean something different for every column."""
        assert fitted[model_id].requires_scaling

    @pytest.mark.parametrize("model_id", TREE_IDS)
    def test_trees_declare_no_scaling(self, fitted, model_id) -> None:
        """Trees split on thresholds; a monotone rescale moves the threshold,
        not the split. Declared as a fact per model, not assumed per family."""
        assert fitted[model_id].preprocessing is Preprocessing.NONE
        assert not fitted[model_id].requires_scaling

    def test_the_robust_variant_really_uses_a_different_scaler(self, fitted, dataset) -> None:
        """Otherwise it would be `ridge` registered twice."""
        assert fitted["ridge_robust_scaled"].preprocessing is Preprocessing.ROBUST
        context = dataset.evaluation_context()
        assert not np.allclose(
            fitted["ridge"].predict(context).point,
            fitted["ridge_robust_scaled"].predict(context).point,
        )


class TestCapacityIsBoundedInAdvance:
    @pytest.mark.parametrize("model_id", ["decision_tree", "random_forest", "extra_trees"])
    def test_the_depth_ceiling_is_actually_applied(self, fitted, model_id) -> None:
        """An unbounded forest reaches training R^2 near 1 and evaluation skill
        near 0. Bounding capacity in advance is what makes the benchmark measure
        the family rather than measuring how long somebody tuned."""
        configuration = fitted[model_id].hyperparameters()
        assert configuration["max_depth"] == MAX_DEPTH
        assert configuration["min_samples_leaf"] == MIN_SAMPLES_LEAF

    @pytest.mark.parametrize("model_id", TREE_IDS)
    def test_a_tree_reports_no_parameter_count(self, fitted, model_id) -> None:
        """A forest's 12,000 nodes and a Ridge's 12 coefficients are not the
        same quantity and must not share a column."""
        assert fitted[model_id].parameter_count() is None

    @pytest.mark.parametrize("model_id", LINEAR_IDS)
    def test_a_linear_model_reports_a_real_parameter_count(self, fitted, model_id) -> None:
        count = fitted[model_id].parameter_count()
        assert isinstance(count, int) and 0 < count <= 20

    def test_node_counts_are_reported_under_their_own_key(self, fitted) -> None:
        assert fitted["random_forest"].hyperparameters()["total_nodes"] > 0
        assert fitted["decision_tree"].hyperparameters()["total_nodes"] > 0


class TestModelsReportWhatTheyLearned:
    def test_lasso_reports_which_features_survived(self, fitted) -> None:
        """At n=1,000 which coefficients survive is more informative than the MAE."""
        configuration = fitted["lasso"].hyperparameters()
        assert "nonzero_coefficients" in configuration
        assert len(configuration["selected_features"]) == configuration["nonzero_coefficients"]

    def test_trees_report_their_top_features(self, fitted) -> None:
        top = fitted["random_forest"].hyperparameters()["top_features"]
        assert len(top) == 3
        assert all(isinstance(name, str) for name, _ in top)

    def test_knn_reports_how_far_nearest_actually_is(self, fitted, dataset) -> None:
        """The curse of dimensionality as a number rather than a caveat. If the
        nearest neighbour is as far as a random row, the model is averaging the
        training set with extra steps."""
        model = fitted["knn"]
        distances = model.neighbour_distances(dataset.evaluation_context())
        assert distances["mean_nearest"] <= distances["mean_kth"]
        assert model.hyperparameters()["train_mean_kth_neighbour_distance"] > 0

    def test_knn_reports_no_parameter_count(self, fitted) -> None:
        """It stores the training set and fits nothing. Reporting 300x11 as a
        parameter count would put memorised data beside a Ridge's coefficients."""
        assert fitted["knn"].parameter_count() is None

    def test_the_svr_records_what_it_rejected(self, fitted) -> None:
        """Phase 16 permits three development-only configurations; the manifest
        has to show what was compared, not only what won."""
        configuration = fitted["rbf_svr"].hyperparameters()
        assert len(configuration["considered_on_dev"]) == 3
        assert len(configuration["dev_skill_of_considered"]) == 3


class TestOptionalDependenciesStayOptional:
    def test_xgboost_is_registered_as_optional_not_required(self) -> None:
        """A zoo whose default benchmark cannot run without an extra is a zoo
        with a hidden dependency."""
        registration = registry.get("xgboost")
        assert registration.requires == ("xgboost",)
        assert registration.effective_status() in (
            ModelStatus.ACTIVE,
            ModelStatus.SKIPPED_DEPENDENCY,
        )

    def test_it_is_not_confused_with_a2s_retuned_model(self) -> None:
        """A2's challenger selected hyperparameters by nested inner validation
        inside all 36 folds. This one is frozen and fitted once on 1,000 rows.
        The card must say so, because the names are one word apart."""
        notes = " ".join(registry.get("xgboost").notes)
        assert "NOT comparable" in notes
        assert "xgboost_causal_retuned" in notes

    def test_every_tabular_model_declares_its_dependency(self) -> None:
        for model_id in TABULAR_IDS:
            assert registry.get(model_id).requires, model_id


class TestTheBaseClassIsAbstract:
    def test_a_subclass_must_supply_an_estimator(self, dataset) -> None:
        class Incomplete(TabularModel):
            model_id = "incomplete"

        with pytest.raises(NotImplementedError):
            Incomplete().fit(dataset.training_set())
