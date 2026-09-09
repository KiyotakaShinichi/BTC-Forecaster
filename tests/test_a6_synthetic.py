"""Behaviour on worlds where the answer is known.

On real BTC returns nothing works, so the benchmark cannot tell a correctly
wired model from a broken one -- both score approximately zero. These worlds
can.

The load-bearing test is NOISE_ONLY. It is unpredictable by construction, so a
model that beats the mean on it has found something that is not there, and the
finding would be a leak rather than a result. Every other assertion in this
repository about a model "working" is only worth as much as that control.

Rankings are never asserted. Phase 28 says these verify behaviour, not order,
and demanding that an LSTM beat a Ridge on a synthetic AR(1) would encode an
expectation rather than test one.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.partition import PartitionSpec, build_dataset
from btc_forecaster.research.synthetic import WORLD_NAMES, build_world

#: One cheap model per family. The full zoo on seven worlds would be an hour.
PROBES = ["naive_last_value", "random_walk_drift", "ar_p", "ridge", "decision_tree"]
SPEC = PartitionSpec(train_rows=200)


def skill_of(world_name: str, model_id: str) -> float:
    """MAE skill against the zero-return naive forecast, on this world's holdout."""
    dataset = build_dataset(build_world(world_name).frame, spec=SPEC)
    train, context = dataset.training_set(), dataset.evaluation_context()
    model = registry.build(model_id).fit(train)
    if model.needs_calibration:
        model.calibrate(dataset.development_context())
    actual = context.y.to_numpy()
    naive = float(np.mean(np.abs(actual)))
    mae = float(np.mean(np.abs(model.predict(context).point - actual)))
    return 1.0 - mae / naive


class TestTheNoiseControl:
    """The most important test in the file.

    NOISE_ONLY has no structure. A model that beats the mean on it has a leak,
    an alignment error, or a scaler that saw the future -- and every claim the
    other six worlds make depends on this one failing to find anything.
    """

    @pytest.mark.parametrize("model_id", PROBES)
    def test_nothing_beats_the_mean_on_pure_noise(self, model_id) -> None:
        skill = skill_of("NOISE_ONLY", model_id)
        assert skill < 0.05, (
            f"{model_id} scored {skill:+.4f} skill on pure noise; that is not a "
            "result, it is a leak"
        )

    @pytest.mark.parametrize("model_id", PROBES)
    def test_nothing_is_absurdly_bad_either(self, model_id) -> None:
        """The other direction. A model producing 10x the naive error on noise
        is mis-scaled, not conservative."""
        assert skill_of("NOISE_ONLY", model_id) > -2.0

    def test_the_control_is_documented_as_such(self) -> None:
        world = build_world("NOISE_ONLY")
        assert world.predictable_mean is False
        assert "leak" in world.expectation


class TestWorldsWithKnownStructure:
    def test_a_drift_is_recoverable(self) -> None:
        """The simplest wiring check: a constant-drift model must beat a
        zero-return forecast on a series that genuinely drifts."""
        assert skill_of("TREND", "random_walk_drift") > 0.05

    def test_the_naive_forecast_is_wrong_on_a_trend(self) -> None:
        """The control for the test above."""
        assert skill_of("TREND", "naive_last_value") == pytest.approx(0.0, abs=1e-12)

    def test_an_autoregressive_model_gains_on_an_ar_process(self) -> None:
        """AR(1) with phi=0.35. If an AR model cannot beat the mean here, its
        lag alignment is wrong."""
        assert skill_of("AR_PROCESS", "ar_p") > 0.02

    def test_an_autoregressive_model_recovers_the_coefficient(self) -> None:
        """Stronger than a skill check: the estimated phi must be near the true
        one, which a lag-shifted implementation would miss."""
        dataset = build_dataset(build_world("AR_PROCESS").frame, spec=SPEC)
        model = registry.build("ar_p").fit(dataset.training_set())
        first_lag = model.hyperparameters()["coefficients"][0]
        assert 0.2 < first_lag < 0.5, f"recovered phi={first_lag:.3f}, expected ~0.35"

    def test_mean_reversion_is_exploitable(self) -> None:
        assert skill_of("MEAN_REVERTING", "ar_p") > 0.02

    def test_nothing_gains_on_a_stationary_series(self) -> None:
        """i.i.d. around zero: the mean is the best available forecast."""
        for model_id in ("ar_p", "ridge"):
            assert skill_of("STATIONARY", model_id) < 0.05, model_id


class TestVolatilityAndDirectionAreSeparate:
    """The world that keeps the two claims apart."""

    def test_direction_is_not_predictable(self) -> None:
        for model_id in ("ar_p", "ridge"):
            assert skill_of("VOLATILITY_CLUSTERING", model_id) < 0.05, model_id

    def test_variance_is_predictable(self) -> None:
        """A GARCH fit must beat a constant-variance forecast on QLIKE, or the
        volatility family is not measuring what it claims."""
        pytest.importorskip("arch")
        dataset = build_dataset(build_world("VOLATILITY_CLUSTERING").frame, spec=SPEC)
        context = dataset.evaluation_context()
        model = registry.build("garch_11").fit(dataset.training_set())
        forecast = model.predict(context)

        actual = context.y.to_numpy()
        realised = actual**2

        def qlike(variance: np.ndarray) -> float:
            ratio = np.maximum(realised, 1e-16) / np.maximum(variance, 1e-16)
            return float(np.mean(ratio - np.log(ratio) - 1.0))

        constant = np.full(len(actual), float(np.var(dataset.training_set().y)))
        assert qlike(forecast.variance) < qlike(constant)

    def test_the_world_declares_both_facts(self) -> None:
        world = build_world("VOLATILITY_CLUSTERING")
        assert world.predictable_mean is False
        assert world.predictable_variance is True


class TestNonlinearity:
    def test_a_tree_can_bend_where_a_line_cannot(self) -> None:
        """A threshold process: the sensitivity flips with the sign of the
        previous return, which a single linear coefficient cannot represent.

        Asserted as 'the tree finds something', not as 'the tree beats the
        line' -- a ranking assertion would be encoding an expectation.
        """
        assert skill_of("NONLINEAR", "decision_tree") > 0.0


class TestTheWorldsThemselves:
    @pytest.mark.parametrize("name", WORLD_NAMES)
    def test_each_world_builds_a_valid_frame(self, name) -> None:
        world = build_world(name)
        assert world.name == name
        assert len(world.frame) > 0
        assert bool(np.isfinite(world.frame["close"]).all())
        assert bool((world.frame["close"] > 0).all())

    @pytest.mark.parametrize("name", WORLD_NAMES)
    def test_each_world_states_what_it_expects(self, name) -> None:
        """A world without a stated expectation is a random series."""
        assert len(build_world(name).expectation) > 20

    @pytest.mark.parametrize("name", WORLD_NAMES)
    def test_each_world_is_deterministic(self, name) -> None:
        assert np.array_equal(
            build_world(name).frame["close"].to_numpy(),
            build_world(name).frame["close"].to_numpy(),
        )

    def test_there_are_seven(self) -> None:
        assert len(WORLD_NAMES) == 7

    def test_an_unknown_world_raises_with_the_list(self) -> None:
        with pytest.raises(KeyError, match="known:"):
            build_world("SOMETHING_ELSE")
