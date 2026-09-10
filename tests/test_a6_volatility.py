"""The volatility family, and the claim it is not allowed to make.

The failure mode these tests exist for is not a wrong number. It is a category
error: scoring a conditional variance and presenting the result as forecasting
skill. So the assertions are mostly about the *contract* -- what these models
declare, what they refuse to declare, and that their point output is the
constant mean it says it is rather than something dressed up as direction.

The one accuracy-shaped test is calibration, because a distributional model that
cannot cover its own nominal interval is not merely inaccurate, it is
misdescribed.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.adapters.volatility import QUANTILE_LEVELS, RETURN_SCALE
from btc_forecaster.research.contracts import (
    Capability,
    CapabilityNotSupported,
    Family,
    ModelStatus,
)
from btc_forecaster.research.partition import PartitionSpec, build_dataset
from btc_forecaster.testing import synthetic_market_frame

VOLATILITY_IDS = ["garch_11", "egarch_11", "gjr_garch_11"]

pytest.importorskip("arch", reason="the volatility family is an OPTIONAL adapter")


@pytest.fixture(scope="module")
def dataset():
    # A t-distributed AR(1) return series gives the variance models something
    # with genuine tails to model.
    frame = synthetic_market_frame(periods=900, kind="ar1", seed=7)
    return build_dataset(frame, spec=PartitionSpec(train_rows=400))


@pytest.fixture(scope="module")
def fitted(dataset):
    train = dataset.training_set()
    return {mid: registry.build(mid).fit(train) for mid in VOLATILITY_IDS}


class TestTheOutputContractIsExplicit:
    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_it_declares_variance_and_quantiles(self, fitted, model_id) -> None:
        model = fitted[model_id]
        assert model.supports(Capability.VARIANCE)
        assert model.supports(Capability.QUANTILES)

    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_it_does_not_claim_a_direction_probability(self, fitted, dataset, model_id) -> None:
        """The category error this family invites. A GARCH model makes no claim
        about tomorrow's sign, and asking it for one must raise rather than
        return the 0.5 that a symmetric distribution would imply."""
        model = fitted[model_id]
        assert not model.supports(Capability.DIRECTION_PROBABILITY)
        with pytest.raises(CapabilityNotSupported):
            model.predict_direction_probability(dataset.evaluation_context())

    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_the_point_forecast_is_the_constant_mean(self, fitted, dataset, model_id) -> None:
        """Not a hidden direction signal: one fitted number, repeated."""
        forecast = fitted[model_id].predict(dataset.evaluation_context())
        assert len(np.unique(forecast.point)) == 1
        expected = fitted[model_id].hyperparameters()["estimated"]["mu"] / RETURN_SCALE
        assert forecast.point[0] == pytest.approx(expected)

    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_the_registration_says_so_too(self, model_id) -> None:
        """The caveat has to reach the model card, not only the source."""
        notes = " ".join(registry.get(model_id).notes).lower()
        assert "not direction" in notes


class TestTheVarianceForecastIsUsable:
    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_variance_is_positive_and_finite(self, fitted, dataset, model_id) -> None:
        variance = fitted[model_id].predict(dataset.evaluation_context()).variance
        assert variance is not None
        assert np.isfinite(variance).all()
        assert bool((variance > 0).all())

    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_variance_actually_varies(self, fitted, dataset, model_id) -> None:
        """A conditional variance that never moves is an unconditional one, and
        the whole point of the family is the conditioning."""
        variance = fitted[model_id].predict(dataset.evaluation_context()).variance
        assert float(np.std(variance)) > 0.0

    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_variance_is_in_return_units_not_percent(self, fitted, dataset, model_id) -> None:
        """Estimation runs on percent returns; a forgotten rescale would make
        the variance 10,000x too large and every interval useless."""
        variance = fitted[model_id].predict(dataset.evaluation_context()).variance
        assert 1e-6 < float(np.sqrt(variance).mean()) < 0.5


class TestTheDistributionIsTheModelsOwn:
    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_quantiles_are_ordered_and_complete(self, fitted, dataset, model_id) -> None:
        quantiles = fitted[model_id].predict(dataset.evaluation_context()).quantiles
        assert quantiles is not None
        assert set(quantiles) == set(QUANTILE_LEVELS)
        stacked = np.vstack([quantiles[q] for q in sorted(quantiles)])
        assert bool((np.diff(stacked, axis=0) >= -1e-12).all())

    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_the_median_is_the_point_forecast(self, fitted, dataset, model_id) -> None:
        """For a symmetric conditional distribution these must coincide; if they
        do not, the mean and the quantiles came from different models."""
        forecast = fitted[model_id].predict(dataset.evaluation_context())
        assert np.allclose(forecast.quantiles[0.5], forecast.point, atol=1e-12)

    @pytest.mark.parametrize("model_id", VOLATILITY_IDS)
    def test_the_nominal_interval_is_approximately_covered(
        self, fitted, dataset, model_id
    ) -> None:
        """The one accuracy-shaped assertion. A model whose 90% interval covers
        60% of outcomes is not inaccurate, it is misdescribed -- and its pinball
        loss and Winkler score would both be meaningless."""
        context = dataset.evaluation_context()
        forecast = fitted[model_id].predict(context)
        actual = context.y.to_numpy()
        covered = float(
            np.mean((actual >= forecast.quantiles[0.05]) & (actual <= forecast.quantiles[0.95]))
        )
        assert 0.80 <= covered <= 0.99, f"{model_id} covered {covered:.3f} of a nominal 0.90"


class TestTheFamilyIsRegisteredHonestly:
    def test_all_three_are_in_the_volatility_family(self) -> None:
        assert set(registry.model_ids(family=Family.VOLATILITY)) == set(VOLATILITY_IDS)

    def test_they_require_arch_and_say_so(self) -> None:
        for model_id in VOLATILITY_IDS:
            assert registry.get(model_id).requires == ("arch",)

    def test_without_arch_they_would_be_skipped_not_missing(self) -> None:
        """The status the registry reports when a dependency is absent. It is a
        recorded outcome, not a silently shorter table."""
        registration = registry.get("garch_11")
        assert registration.effective_status() in (
            ModelStatus.ACTIVE,
            ModelStatus.SKIPPED_DEPENDENCY,
        )

    def test_the_two_asymmetric_models_are_not_duplicates(self, fitted, dataset) -> None:
        """EGARCH and GJR impose leverage differently -- logs versus levels --
        so they must disagree, or one of them is redundant."""
        context = dataset.evaluation_context()
        egarch = fitted["egarch_11"].predict(context).variance
        gjr = fitted["gjr_garch_11"].predict(context).variance
        assert not np.allclose(egarch, gjr)
