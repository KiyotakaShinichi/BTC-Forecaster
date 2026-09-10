"""The seven deep challengers: causality, capacity, determinism, distinctness.

A deep model is the easiest thing in this zoo to get quietly wrong. It will
train, report a falling loss, produce finite forecasts and score plausibly
whether or not its convolution is causal, its attention mask is off by one, or
its window contains the bar it predicts. None of those raise.

So these tests attack the mechanism rather than the metric:

* a causal convolution's output at step t must not move when step t+1 changes;
* the attention mask must make the same true for self-attention;
* poisoning every future bar must leave every prediction bit-identical;
* the same seed must give the same forecast, twice;
* the seven must actually disagree, or some of them are one model.

`test_a6_autodiff.py` covers the gradients underneath. This file covers what is
built on them.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.adapters.deep import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOOKBACK,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    DeepModel,
)
from btc_forecaster.research.autodiff import Tensor
from btc_forecaster.research.contracts import (
    Capability,
    EvaluationContext,
    Family,
    ModelFitError,
)
from btc_forecaster.research.nn import CausalConv1d, MultiHeadAttention
from btc_forecaster.research.partition import PartitionSpec, build_dataset
from btc_forecaster.testing import synthetic_market_frame

DEEP_IDS = registry.model_ids(family=Family.DEEP)


@pytest.fixture(scope="module")
def dataset():
    # Small on purpose: these train for real, and the suite has to stay runnable.
    frame = synthetic_market_frame(periods=500, kind="ar1", seed=23)
    return build_dataset(frame, spec=PartitionSpec(train_rows=200))


@pytest.fixture(scope="module")
def fitted(dataset):
    train, dev = dataset.training_set(), dataset.development_context()
    models = {}
    for model_id in DEEP_IDS:
        model = registry.build(model_id).fit(train)
        model.calibrate(dev)
        models[model_id] = model
    return models


class TestCausalityIsStructural:
    def test_a_causal_convolution_cannot_see_the_next_step(self) -> None:
        """The property a mask gets wrong by one. Checked by perturbing a later
        timestep and asserting every earlier output is unchanged."""
        rng = np.random.default_rng(0)
        conv = CausalConv1d(rng, in_channels=2, out_channels=3, kernel_size=3)
        x = rng.standard_normal((1, 8, 2))

        baseline = conv(Tensor(x)).data.copy()
        poisoned = x.copy()
        poisoned[0, 5:, :] = 99.0
        after = conv(Tensor(poisoned)).data

        assert np.allclose(baseline[0, :5], after[0, :5])
        assert not np.allclose(baseline[0, 5:], after[0, 5:])

    def test_a_dilated_convolution_is_causal_too(self) -> None:
        rng = np.random.default_rng(1)
        conv = CausalConv1d(rng, in_channels=2, out_channels=2, kernel_size=3, dilation=4)
        x = rng.standard_normal((1, 16, 2))
        baseline = conv(Tensor(x)).data.copy()
        poisoned = x.copy()
        poisoned[0, 10:, :] = -50.0
        assert np.allclose(baseline[0, :10], conv(Tensor(poisoned)).data[0, :10])

    def test_attention_cannot_read_forward(self) -> None:
        """An additive -inf mask before the softmax. A multiplicative mask
        applied afterwards would renormalise the surviving weights and let a
        future position influence the total."""
        rng = np.random.default_rng(2)
        attention = MultiHeadAttention(rng, d_model=8, heads=2)
        x = rng.standard_normal((1, 6, 8))
        baseline = attention(Tensor(x)).data.copy()
        poisoned = x.copy()
        poisoned[0, 4:, :] = 30.0
        assert np.allclose(baseline[0, :4], attention(Tensor(poisoned)).data[0, :4], atol=1e-9)

    @pytest.mark.parametrize("model_id", DEEP_IDS)
    def test_poisoning_the_future_leaves_predictions_identical(
        self, dataset, fitted, model_id
    ) -> None:
        """The end-to-end version. Every bar after the first origin is replaced
        with nonsense; the first prediction must not move by one bit."""
        context = dataset.evaluation_context()
        clean = fitted[model_id].predict(context).point

        poisoned_design = context.design.copy()
        cut = context.target_bars[0]
        poisoned_design.loc[poisoned_design.index > cut, :] = 42.0
        poisoned = EvaluationContext(
            X=context.X,
            y=context.y,
            series=context.series,
            close=context.close,
            feature_bar=context.feature_bar,
            train_end=context.train_end,
            design=poisoned_design,
        )
        assert fitted[model_id].predict(poisoned).point[0] == clean[0]


class TestTheTrainingBudgetIsShared:
    @pytest.mark.parametrize("model_id", DEEP_IDS)
    def test_every_model_uses_the_same_budget(self, fitted, model_id) -> None:
        """A per-model tuning loop would compare how long each was tuned for
        rather than comparing architectures."""
        configuration = fitted[model_id].hyperparameters()
        assert configuration["lookback"] == LOOKBACK
        assert configuration["max_epochs"] == MAX_EPOCHS
        assert configuration["batch_size"] == BATCH_SIZE
        assert configuration["learning_rate"] == LEARNING_RATE
        assert configuration["patience"] == PATIENCE
        assert configuration["seed"] == SEED

    @pytest.mark.parametrize("model_id", DEEP_IDS)
    def test_early_stopping_watches_dev(self, fitted, model_id) -> None:
        """An early-stop epoch chosen on HOLDOUT is a hyperparameter fitted to
        the number being reported."""
        configuration = fitted[model_id].hyperparameters()
        assert configuration["early_stopping_block"] == "DEV"
        training = configuration["training"]
        assert 1 <= training["best_epoch"] <= training["epochs_run"] <= MAX_EPOCHS

    @pytest.mark.parametrize("model_id", DEEP_IDS)
    def test_the_epoch_ceiling_is_respected(self, fitted, model_id) -> None:
        assert fitted[model_id].hyperparameters()["training"]["epochs_run"] <= MAX_EPOCHS

    def test_a_model_must_be_trained_before_it_predicts(self, dataset) -> None:
        """Deep models train during calibration, so DEV decides when to stop.
        Predicting from an untrained network would return the initialisation."""
        model = registry.build("gru").fit(dataset.training_set())
        with pytest.raises(ModelFitError, match="never trained"):
            model.predict(dataset.evaluation_context())


class TestCapacityIsReportedNotEstimated:
    @pytest.mark.parametrize("model_id", DEEP_IDS)
    def test_it_reports_a_real_parameter_count(self, fitted, model_id) -> None:
        count = fitted[model_id].parameter_count()
        assert isinstance(count, int) and count > 0

    @pytest.mark.parametrize("model_id", [m for m in DEEP_IDS if m != "nbeats"])
    def test_six_of_the_seven_stay_small(self, fitted, model_id) -> None:
        """977 sequences at the full budget. More parameters than sequences is
        memorising, and these six are held well under it."""
        assert fitted[model_id].parameter_count() < 10_000

    def test_nbeats_is_the_declared_outlier(self, fitted) -> None:
        """Its backcast head projects back to the full flattened window, so it
        is structurally parameter-heavy. Left that way, and said so, because the
        parameters-per-sequence ratio is the axis Phase 24 asks about."""
        assert fitted["nbeats"].parameter_count() > 10_000
        notes = " ".join(registry.get("nbeats").notes)
        assert "parameter-heavy outlier" in notes
        assert "the ratio is the finding" in notes


class TestReproducibilityAndDistinctness:
    @pytest.mark.parametrize("model_id", ["mlp", "gru", "nbeats"])
    def test_the_same_seed_gives_the_same_forecast(self, dataset, model_id) -> None:
        """A benchmark that cannot be re-run to the same number is not evidence."""
        train, dev = dataset.training_set(), dataset.development_context()
        context = dataset.evaluation_context()

        first = registry.build(model_id).fit(train)
        first.calibrate(dev)
        second = registry.build(model_id).fit(train)
        second.calibrate(dev)
        assert np.array_equal(first.predict(context).point, second.predict(context).point)

    def test_the_seven_are_not_one_model(self, dataset, fitted) -> None:
        """Seven architectures that agreed to twelve decimal places would be one
        architecture registered seven times."""
        context = dataset.evaluation_context()
        predictions = {mid: fitted[mid].predict(context).point for mid in DEEP_IDS}
        for i, a in enumerate(DEEP_IDS):
            for b in DEEP_IDS[i + 1 :]:
                assert not np.allclose(predictions[a], predictions[b]), f"{a} == {b}"

    def test_the_tcn_reaches_further_back_than_the_cnn(self) -> None:
        """The whole reason both are registered. Dilations 1, 2, 4 at kernel 3
        reach 15 bars; two undilated kernel-3 layers reach 5."""
        tcn_notes = " ".join(registry.get("tcn").notes)
        assert "15 bars" in tcn_notes
        assert "cnn_1d" in tcn_notes


class TestTheFamilyIsHonestlyDeclared:
    @pytest.mark.parametrize("model_id", DEEP_IDS)
    def test_it_claims_only_point_and_serialize(self, fitted, model_id) -> None:
        assert fitted[model_id].capabilities == frozenset(
            {Capability.POINT, Capability.SERIALIZE}
        )

    @pytest.mark.parametrize("model_id", DEEP_IDS)
    def test_it_needs_no_third_party_framework(self, model_id) -> None:
        """The engine is in this repository. torch is not a dependency, and the
        card says why rather than leaving it to be inferred."""
        assert registry.get(model_id).requires == ()
        notes = " ".join(registry.get(model_id).notes)
        assert "torch is not a dependency" in notes
        assert "finite differences" in notes

    def test_the_control_is_labelled_as_one(self) -> None:
        """If nothing beats a model that cannot tell the order of its own
        inputs, the finding is about the series rather than about architectures."""
        notes = " ".join(registry.get("mlp").notes)
        assert "control" in notes.lower()

    def test_the_base_class_refuses_to_be_used_directly(self, dataset) -> None:
        with pytest.raises(NotImplementedError):
            DeepModel().fit(dataset.training_set())
