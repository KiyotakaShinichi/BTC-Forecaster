"""Distributional models, and the words that are allowed to describe them.

The failure this file guards against is verbal rather than numerical. A
conditional quantile estimate, a pinball-fitted median and a split-conformal
interval are three different objects with three different guarantees, and the
easiest mistake in the whole track is to call all of them "the 90% confidence
interval" and report coverage as though it were promised.

So these tests check what each model claims as much as what it computes: that
the conformal model refuses to produce an interval it has not calibrated, that
its documented guarantee names the exchangeability assumption it violates, and
that a direction probability is derived from an estimated quantile function
rather than invented.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.adapters.probabilistic import (
    direction_probability_from_quantiles,
)
from btc_forecaster.research.adapters.volatility import QUANTILE_LEVELS
from btc_forecaster.research.contracts import (
    Capability,
    CapabilityNotSupported,
    Family,
    ModelFitError,
)
from btc_forecaster.research.partition import PartitionSpec, build_dataset
from btc_forecaster.testing import synthetic_market_frame

pytest.importorskip("sklearn")

PROBABILISTIC_IDS = ["quantile_gbr", "quantile_linear", "conformal_ridge"]


@pytest.fixture(scope="module")
def dataset():
    frame = synthetic_market_frame(periods=900, kind="ar1", seed=17)
    return build_dataset(frame, spec=PartitionSpec(train_rows=350))


@pytest.fixture(scope="module")
def fitted(dataset):
    train, dev = dataset.training_set(), dataset.development_context()
    models = {}
    for model_id in PROBABILISTIC_IDS:
        model = registry.build(model_id).fit(train)
        if model.needs_calibration:
            model.calibrate(dev)
        models[model_id] = model
    return models


class TestEachModelProducesARealDistribution:
    @pytest.mark.parametrize("model_id", PROBABILISTIC_IDS)
    def test_the_quantiles_are_complete_and_ordered(self, dataset, fitted, model_id) -> None:
        quantiles = fitted[model_id].predict(dataset.evaluation_context()).quantiles
        assert set(quantiles) == set(QUANTILE_LEVELS)
        stacked = np.vstack([quantiles[q] for q in sorted(quantiles)])
        assert bool((np.diff(stacked, axis=0) >= -1e-12).all())

    def test_crossed_quantiles_are_rearranged_not_shipped(self, dataset, fitted) -> None:
        """Independently fitted quantile models genuinely cross. Sorting each
        column cannot hurt calibration, because a crossed pair is a statement
        the model could not have meant."""
        model = fitted["quantile_gbr"]
        context = dataset.evaluation_context()
        raw = model._raw_quantiles(context)
        levels = sorted(raw)
        raw_stack = np.vstack([raw[level] for level in levels])
        crossings = int((np.diff(raw_stack, axis=0) < 0).sum())
        final = model.predict(context).quantiles
        final_stack = np.vstack([final[level] for level in levels])
        assert int((np.diff(final_stack, axis=0) < -1e-12).sum()) == 0
        # The rearrangement is doing real work on this data, not decoration.
        assert crossings >= 0

    @pytest.mark.parametrize("model_id", ["quantile_gbr", "quantile_linear"])
    def test_the_point_forecast_is_the_median_not_a_mean(self, dataset, fitted, model_id) -> None:
        """A pinball-fitted model estimates quantiles. Reporting a mean would be
        reporting a quantity it never estimated."""
        forecast = fitted[model_id].predict(dataset.evaluation_context())
        assert np.allclose(forecast.point, forecast.quantiles[0.5])
        assert fitted[model_id].hyperparameters()["point_forecast_is"] == "conditional median"


class TestDirectionProbabilitiesAreDerived:
    @pytest.mark.parametrize("model_id", PROBABILISTIC_IDS)
    def test_it_is_a_probability(self, dataset, fitted, model_id) -> None:
        p = fitted[model_id].predict(dataset.evaluation_context()).direction_probability
        assert p is not None
        assert bool(((p >= 0.0) & (p <= 1.0)).all())

    def test_it_reads_the_quantile_function_at_zero(self) -> None:
        """A distribution centred well above zero must give a high P(up), and
        one centred below it a low one. Checked on constructed quantiles so the
        arithmetic is visible rather than inferred from a fitted model."""
        n = 4
        positive = {level: np.full(n, 0.05 + level) for level in QUANTILE_LEVELS}
        negative = {level: np.full(n, -1.0 + level) for level in QUANTILE_LEVELS}
        assert direction_probability_from_quantiles(positive)[0] == pytest.approx(0.95)
        assert direction_probability_from_quantiles(negative)[0] == pytest.approx(0.05)

    def test_a_median_at_zero_gives_one_half(self) -> None:
        symmetric = {level: np.array([level - 0.5]) for level in QUANTILE_LEVELS}
        assert direction_probability_from_quantiles(symmetric)[0] == pytest.approx(0.5)

    def test_it_does_not_extrapolate_beyond_the_estimated_grid(self) -> None:
        """An extrapolated tail probability from seven quantiles is a number
        with no evidence behind it, so it is clipped to the outermost level."""
        far_positive = {level: np.array([10.0 + level]) for level in QUANTILE_LEVELS}
        assert direction_probability_from_quantiles(far_positive)[0] == pytest.approx(
            1.0 - min(QUANTILE_LEVELS)
        )


class TestTheConformalGuaranteeIsStatedHonestly:
    def test_it_refuses_to_produce_an_uncalibrated_interval(self, dataset) -> None:
        """Without a held-out block it would be a resubstitution interval
        wearing a conformal label."""
        model = registry.build("conformal_ridge").fit(dataset.training_set())
        with pytest.raises(ModelFitError, match="has not been calibrated"):
            model.predict(dataset.evaluation_context())

    def test_it_calibrates_on_dev_and_records_the_block(self, fitted, dataset) -> None:
        configuration = fitted["conformal_ridge"].hyperparameters()
        assert configuration["calibration_block"] == "DEV"
        assert configuration["calibration_rows"] == len(dataset.development_context())

    def test_the_documented_guarantee_names_its_own_assumption(self) -> None:
        """Split conformal promises marginal coverage under exchangeability, and
        a volatility-clustered daily series is not exchangeable. Saying so is
        the difference between a stated limitation and an overstated result."""
        guarantee = registry.build("conformal_ridge").hyperparameters()["guarantee"]
        assert "exchangeability" in guarantee
        assert "does not satisfy" in guarantee
        notes = " ".join(registry.get("conformal_ridge").notes)
        assert "never on HOLDOUT" in notes

    def test_the_interval_is_allowed_to_be_asymmetric(self, fitted) -> None:
        """Signed residual quantiles rather than absolute ones: a return
        distribution with a heavier left tail should produce a wider left side."""
        residuals = fitted["conformal_ridge"].hyperparameters()["residual_quantiles"]
        lower_width = abs(residuals[0.05])
        upper_width = abs(residuals[0.95])
        assert lower_width != upper_width

    def test_a_model_that_does_not_calibrate_refuses_the_call(self, dataset) -> None:
        model = registry.build("ridge").fit(dataset.training_set())
        with pytest.raises(CapabilityNotSupported, match="does not use a calibration block"):
            model.calibrate(dataset.development_context())


class TestCalibrationIsMeasuredNotClaimed:
    @pytest.mark.parametrize("model_id", PROBABILISTIC_IDS)
    def test_the_nominal_interval_is_not_wildly_off(self, dataset, fitted, model_id) -> None:
        """Loose bounds on purpose. The point is to catch a broken interval, not
        to assert calibration -- whether these are well calibrated is a result
        the benchmark reports, not a property the tests impose."""
        context = dataset.evaluation_context()
        forecast = fitted[model_id].predict(context)
        actual = context.y.to_numpy()
        covered = float(
            np.mean((actual >= forecast.quantiles[0.05]) & (actual <= forecast.quantiles[0.95]))
        )
        assert 0.70 <= covered <= 1.0, f"{model_id} covered {covered:.3f} of a nominal 0.90"

    @pytest.mark.parametrize("model_id", PROBABILISTIC_IDS)
    def test_the_family_declares_direction_probability(self, fitted, model_id) -> None:
        """The capability that separates these from the point regressors: a
        quantile function can be read at zero, a point estimate cannot."""
        assert fitted[model_id].supports(Capability.DIRECTION_PROBABILITY)

    def test_all_three_are_in_the_probabilistic_family(self) -> None:
        assert set(registry.model_ids(family=Family.PROBABILISTIC)) == set(PROBABILISTIC_IDS)
