"""The zoo's foundations: capability honesty, partition discipline, causality.

Everything the model zoo reports rests on four things being true, and none of
them is visible in a benchmark table:

* a model that cannot produce a probability says so instead of producing one;
* the 1,000 training rows are a deterministic slice, and no evaluation row can
  reach them;
* a scaler sees training data only;
* a sequence window cannot contain the bar it predicts.

These are the tests for those four. They use synthetic frames rather than the
market snapshot, because a foundation test that needs a 3,500-row download is a
foundation test that stops running.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.contracts import (
    EXPLORATORY,
    Capability,
    CapabilityNotSupported,
    EvaluationContext,
    Family,
    ModelFitError,
    ModelStatus,
    Preprocessing,
    ResourceClass,
    TrainingSet,
    ZooForecast,
    ZooModel,
)
from btc_forecaster.research.partition import (
    PartitionError,
    PartitionSpec,
    budget_rows_are_deterministic,
    build_dataset,
    partition_index,
)
from btc_forecaster.research.preprocessing import fit_scaler, scaled_matrices
from btc_forecaster.research.windows import (
    Sequences,
    WindowError,
    WindowSpec,
    assert_window_causality,
    build_sequences,
    evaluation_windows,
    window_at,
)
from btc_forecaster.testing import synthetic_market_frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_market_frame(periods=900, kind="ar1", seed=11)


@pytest.fixture(scope="module")
def dataset(frame: pd.DataFrame):
    return build_dataset(frame, spec=PartitionSpec(train_rows=200))


class TestCapabilitiesAreDeclaredNotAssumed:
    """The rule that keeps a benchmark table honest.

    A blank cell is a fact. A cell filled by assuming a Gaussian is a number
    that scores, tabulates and misleads.
    """

    class PointOnly(ZooModel):
        model_id = "point_only"
        capabilities = frozenset({Capability.POINT})

        def _fit(self, train: TrainingSet) -> None:
            return None

        def _predict_point(self, context: EvaluationContext) -> np.ndarray:
            return np.zeros(len(context))

    def test_asking_for_an_undeclared_capability_raises(self, dataset) -> None:
        model = self.PointOnly().fit(dataset.training_set())
        context = dataset.evaluation_context()
        with pytest.raises(CapabilityNotSupported, match="QUANTILES"):
            model.predict_distribution(context)
        with pytest.raises(CapabilityNotSupported, match="DIRECTION_PROBABILITY"):
            model.predict_direction_probability(context)
        with pytest.raises(CapabilityNotSupported, match="VARIANCE"):
            model.predict_variance(context)

    def test_the_forecast_leaves_undeclared_fields_empty(self, dataset) -> None:
        model = self.PointOnly().fit(dataset.training_set())
        forecast = model.predict(dataset.evaluation_context())
        assert forecast.quantiles is None
        assert forecast.direction_probability is None
        assert forecast.variance is None

    def test_declaring_without_implementing_is_caught(self, dataset) -> None:
        """A declaration is a promise; the base class refuses to fake it."""

        class Liar(self.PointOnly):
            model_id = "liar"
            capabilities = frozenset({Capability.POINT, Capability.QUANTILES})

        model = Liar().fit(dataset.training_set())
        with pytest.raises(CapabilityNotSupported, match="did not implement"):
            model.predict(dataset.evaluation_context())

    def test_predicting_before_fitting_raises(self, dataset) -> None:
        with pytest.raises(ModelFitError, match="has not been fitted"):
            self.PointOnly().predict(dataset.evaluation_context())

    def test_every_model_is_exploratory(self, dataset) -> None:
        """There is one scientific status in A6 and this is it."""
        model = self.PointOnly().fit(dataset.training_set())
        assert model.describe()["scientific_status"] == EXPLORATORY


class TestForecastsAreValidatedNotTrusted:
    def test_crossing_quantiles_are_rejected(self) -> None:
        """A 90th percentile below the 10th is not a wide interval; it is a bug."""
        with pytest.raises(ValueError, match="crossing quantiles"):
            ZooForecast(
                model_id="m",
                point=np.zeros(3),
                quantiles={0.1: np.ones(3), 0.9: np.zeros(3)},
            )

    def test_a_probability_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="probability outside"):
            ZooForecast(
                model_id="m", point=np.zeros(3), direction_probability=np.array([0.5, 1.4, 0.2])
            )

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="one value per origin"):
            ZooForecast(model_id="m", point=np.zeros(3), variance=np.ones(2))

    def test_a_model_returning_the_wrong_count_is_caught(self, dataset) -> None:
        class Short(ZooModel):
            model_id = "short"

            def _fit(self, train: TrainingSet) -> None:
                return None

            def _predict_point(self, context: EvaluationContext) -> np.ndarray:
                return np.zeros(len(context) - 1)

        model = Short().fit(dataset.training_set())
        with pytest.raises(ValueError, match="predictions for"):
            model.predict(dataset.evaluation_context())

    def test_a_non_finite_forecast_is_caught(self, dataset) -> None:
        class Broken(ZooModel):
            model_id = "broken"

            def _fit(self, train: TrainingSet) -> None:
                return None

            def _predict_point(self, context: EvaluationContext) -> np.ndarray:
                out = np.zeros(len(context))
                out[0] = np.nan
                return out

        model = Broken().fit(dataset.training_set())
        with pytest.raises(ModelFitError, match="non-finite"):
            model.predict(dataset.evaluation_context())


class TestThePartitionIsDisciplined:
    def test_the_budget_is_a_deterministic_slice(self, frame: pd.DataFrame) -> None:
        """No seed, no shuffle: re-running selects bit-identical rows."""
        assert budget_rows_are_deterministic(frame, PartitionSpec(train_rows=200))

    def test_the_budget_is_the_tail_of_train(self, dataset) -> None:
        partition = dataset.partition
        assert partition.budget.equals(partition.train[-len(partition.budget) :])
        assert partition.budget.isin(partition.train).all()

    def test_no_budget_row_is_in_dev_or_holdout(self, dataset) -> None:
        """The rule Phase 15 exists for: never sample from the final holdout."""
        partition = dataset.partition
        assert not partition.budget.isin(partition.dev).any()
        assert not partition.budget.isin(partition.holdout).any()

    def test_the_blocks_are_ordered_and_disjoint(self, dataset) -> None:
        partition = dataset.partition
        assert partition.train.max() < partition.dev.min()
        assert partition.dev.max() < partition.holdout.min()

    def test_an_evaluation_origin_inside_training_is_refused(self, dataset) -> None:
        with pytest.raises(ValueError, match="inside the training partition"):
            EvaluationContext(
                X=dataset.X,
                y=dataset.y,
                series=dataset.series,
                close=dataset.close,
                train_end=dataset.X.index[-1],
            )

    def test_a_short_budget_is_recorded_not_topped_up(self, frame: pd.DataFrame) -> None:
        """Asking for more rows than TRAIN holds must not borrow from DEV."""
        dataset = build_dataset(frame, spec=PartitionSpec(train_rows=10_000))
        assert dataset.partition.budget_truncated is True
        assert dataset.partition.budget.equals(dataset.partition.train)
        assert not dataset.partition.budget.isin(dataset.partition.dev).any()

    def test_an_impossible_split_raises(self) -> None:
        index = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=4, tz="UTC"))
        with pytest.raises(PartitionError):
            partition_index(index, PartitionSpec(train_fraction=0.99, dev_fraction=0.005))

    def test_step_zero_is_refused(self) -> None:
        """A zero shift lets a feature row see the bar it predicts."""
        with pytest.raises(PartitionError, match="step"):
            PartitionSpec(step=0)

    def test_the_training_set_is_strictly_causal(self, dataset) -> None:
        train = dataset.training_set()
        assert bool((pd.DatetimeIndex(train.feature_bar) < train.X.index).all())

    def test_the_fingerprint_changes_with_the_rows(self, frame: pd.DataFrame) -> None:
        a = build_dataset(frame, spec=PartitionSpec(train_rows=200)).training_set()
        b = build_dataset(frame, spec=PartitionSpec(train_rows=180)).training_set()
        assert a.fingerprint() != b.fingerprint()


class TestHistoryCannotReachForward:
    def test_history_at_stops_at_the_origin(self, dataset) -> None:
        context = dataset.evaluation_context()
        for i in (0, len(context) // 2, len(context) - 1):
            history = context.history_at(i)
            assert history.index.max() <= context.origins[i]

    def test_close_at_stops_at_the_origin(self, dataset) -> None:
        context = dataset.evaluation_context()
        assert context.close_at(3).index.max() <= context.origins[3]

    def test_an_out_of_range_origin_raises(self, dataset) -> None:
        context = dataset.evaluation_context()
        with pytest.raises(IndexError):
            context.history_at(len(context))


class TestScalersSeeTrainingOnly:
    def test_statistics_come_from_the_training_matrix(self, dataset) -> None:
        train = dataset.training_set()
        context = dataset.evaluation_context()
        _, _, scaler = scaled_matrices(train.X, context.X, Preprocessing.STANDARDIZED)
        expected = train.X.to_numpy(dtype=float).mean(axis=0)
        assert np.allclose(scaler.centre, expected)

    def test_transforming_evaluation_does_not_recentre_it(self, dataset) -> None:
        """The tell-tale of a leaked scaler: evaluation ends up exactly centred."""
        train = dataset.training_set()
        context = dataset.evaluation_context()
        _, evaluation, _ = scaled_matrices(train.X, context.X, Preprocessing.STANDARDIZED)
        assert not np.allclose(evaluation.mean(axis=0), 0.0, atol=1e-8)

    def test_robust_scaling_uses_median_and_iqr(self, dataset) -> None:
        train = dataset.training_set()
        scaler = fit_scaler(train.X, Preprocessing.ROBUST)
        assert np.allclose(scaler.centre, np.median(train.X.to_numpy(dtype=float), axis=0))

    def test_a_constant_column_does_not_divide_by_zero(self) -> None:
        X = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
        scaler = fit_scaler(X, Preprocessing.STANDARDIZED)
        assert np.isfinite(scaler.transform(X).to_numpy()).all()
        assert scaler.spread[0] == 1.0

    def test_none_and_native_are_the_identity(self, dataset) -> None:
        train = dataset.training_set()
        for kind in (Preprocessing.NONE, Preprocessing.MODEL_NATIVE):
            scaler = fit_scaler(train.X, kind)
            assert np.allclose(scaler.transform(train.X).to_numpy(), train.X.to_numpy())

    def test_column_mismatch_is_refused(self, dataset) -> None:
        train = dataset.training_set()
        scaler = fit_scaler(train.X, Preprocessing.STANDARDIZED)
        with pytest.raises(ValueError, match="column mismatch"):
            scaler.transform(train.X.iloc[:, :2])


class TestWindowsCannotContainTheirTarget:
    SPEC = WindowSpec(lookback=8)

    def test_every_window_ends_before_its_target(self, dataset) -> None:
        train = dataset.training_set()
        sequences = build_sequences(train.X, train.y, train.feature_bar, self.SPEC)
        assert bool((sequences.last_feature_bar < sequences.index).all())

    def test_the_shapes_are_what_they_claim(self, dataset) -> None:
        train = dataset.training_set()
        sequences = build_sequences(train.X, train.y, train.feature_bar, self.SPEC)
        assert sequences.X.shape == (len(train) - self.SPEC.lookback + 1, 8, train.n_features)
        assert len(sequences) == sequences.X.shape[0]

    def test_an_off_by_one_window_is_caught(self, dataset) -> None:
        """The assertion is against a future edit, so it must actually fire."""
        train = dataset.training_set()
        sequences = build_sequences(train.X, train.y, train.feature_bar, self.SPEC)
        poisoned = Sequences(
            X=sequences.X,
            y=sequences.y,
            index=sequences.index,
            # Shift the recorded feature bar forward past its target.
            last_feature_bar=sequences.index,
        )
        with pytest.raises(WindowError, match="not strictly before"):
            assert_window_causality(poisoned)

    def test_insufficient_history_raises_rather_than_pads(self, dataset) -> None:
        """Zero-padding would train on windows that never occur at prediction."""
        with pytest.raises(WindowError, match="needs"):
            window_at(dataset.X, 3, WindowSpec(lookback=16))

    def test_a_window_shorter_than_the_lookback_raises(self, dataset) -> None:
        train = dataset.training_set()
        with pytest.raises(WindowError, match="at least"):
            build_sequences(
                train.X.iloc[:4], train.y.iloc[:4], train.feature_bar.iloc[:4], self.SPEC
            )

    def test_evaluation_windows_reach_back_into_training(self, dataset) -> None:
        """Legitimately: those rows are the realised past, not the future."""
        context = dataset.evaluation_context()
        windows = evaluation_windows(dataset.X, context.origins, self.SPEC)
        assert windows.shape == (len(context), 8, dataset.X.shape[1])

    def test_an_unknown_origin_is_refused(self, dataset) -> None:
        stranger = pd.DatetimeIndex([pd.Timestamp("1999-01-01", tz="UTC")])
        with pytest.raises(WindowError, match="not present"):
            evaluation_windows(dataset.X, stranger, self.SPEC)


class TestTheRegistryIsTheOnlySourceOfTruth:
    def test_every_registration_builds_a_model_with_a_matching_id(self) -> None:
        """Identity is stated once. A key that disagrees with the model's own
        id is how a table row and a model card come to describe different things."""
        for registration in registry.all_registrations():
            if registration.effective_status() in (
                ModelStatus.SKIPPED_DEPENDENCY,
                ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB,
            ):
                continue
            assert registry.build(registration.model_id).model_id == registration.model_id

    def test_registering_a_duplicate_id_raises(self) -> None:
        from btc_forecaster.research.registry import ZooRegistration, register

        existing = registry.all_registrations()[0]
        with pytest.raises(ValueError, match="already registered"):
            register(
                ZooRegistration(
                    model_id=existing.model_id,
                    factory=existing.factory,
                    family=existing.family,
                    resource_class=existing.resource_class,
                    description="duplicate",
                )
            )

    def test_an_unsuitable_model_must_say_why(self) -> None:
        """An unexplained exclusion is not a scientific statement."""
        from btc_forecaster.research.registry import ZooRegistration

        with pytest.raises(ValueError, match="without a reason"):
            ZooRegistration(
                model_id="nameless",
                factory=lambda: None,  # type: ignore[arg-type,return-value]
                family=Family.DEEP,
                resource_class=ResourceClass.HEAVY,
                description="x",
                status=ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB,
            )

    def test_the_capability_matrix_covers_every_registration(self) -> None:
        matrix = registry.capability_matrix()
        assert {row["model_id"] for row in matrix} == set(registry.model_ids())

    def test_the_summary_counts_add_up(self) -> None:
        summary = registry.summary()
        assert summary["total_registered"] == len(registry.all_registrations())
        assert sum(summary["by_status"].values()) == summary["total_registered"]
        assert sum(summary["by_family"].values()) == summary["total_registered"]

    def test_an_unknown_model_raises_with_the_list(self) -> None:
        with pytest.raises(KeyError, match="registered:"):
            registry.get("no_such_model")


class TestTheBaselinesAreRealBaselines:
    def test_naive_last_value_forecasts_exactly_zero(self, dataset) -> None:
        model = registry.build("naive_last_value").fit(dataset.training_set())
        forecast = model.predict(dataset.evaluation_context())
        assert np.array_equal(forecast.point, np.zeros(len(forecast.point)))
        assert model.parameter_count() == 0

    def test_drift_is_the_mean_training_log_return(self, dataset) -> None:
        train = dataset.training_set()
        model = registry.build("random_walk_drift").fit(train)
        assert model.hyperparameters()["constant_log_return"] == pytest.approx(
            float(train.y.mean())
        )

    def test_the_two_means_are_jensen_distinct(self, dataset) -> None:
        """Not a duplicate registration: the arithmetic mean of simple returns
        exceeds the mean log return by roughly half the variance."""
        train = dataset.training_set()
        drift = registry.build("random_walk_drift").fit(train)
        mean_simple = registry.build("historical_mean_return").fit(train)
        a = drift.hyperparameters()["constant_log_return"]
        b = mean_simple.hyperparameters()["constant_log_return"]
        assert b > a
        assert b - a == pytest.approx(0.5 * float(train.y.var()), rel=0.35)

    def test_a_constant_forecast_is_constant(self, dataset) -> None:
        model = registry.build("random_walk_drift").fit(dataset.training_set())
        point = model.predict(dataset.evaluation_context()).point
        assert len(np.unique(point)) == 1
