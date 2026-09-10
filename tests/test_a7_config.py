"""The A7 configuration: what a benchmark is, fixed before it runs.

These tests pin the property the whole track depends on: a result cannot be
improved after the fact by moving a threshold, dropping a baseline or narrowing
a horizon, because every one of those is part of a digest that changes when it
moves.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.walk_forward.config import (
    BASELINE,
    CURATED_MODELS,
    DEFAULT_WINDOWS,
    DRIFT_BASELINE,
    GateThresholds,
    WalkForwardConfig,
    WindowSpec,
)


class TestTheDefaultBenchmark:
    def test_it_asks_the_questions_the_track_poses(self) -> None:
        config = WalkForwardConfig()
        assert config.horizons == (1, 3, 7, 30)
        assert [w.label for w in config.windows] == [
            "rolling-250",
            "rolling-500",
            "rolling-1000",
            "rolling-2000",
            "expanding",
        ]
        assert config.n_blocks >= 3

    def test_both_baselines_are_present(self) -> None:
        config = WalkForwardConfig()
        assert BASELINE in config.models and DRIFT_BASELINE in config.models
        assert config.baseline == BASELINE

    def test_every_curated_model_resolves_through_the_a6_registry(self) -> None:
        """The registry decides what exists; A7 keeps no second model list that
        could drift from it."""
        for model_id in CURATED_MODELS:
            assert registry.get(model_id).model_id == model_id

    def test_no_state_can_express_promotion(self) -> None:
        text = WalkForwardConfig().canonical_json().lower()
        assert "promot" not in text and "trading" not in text


class TestTheConfigurationRefusesBadDesigns:
    def test_the_naive_baseline_cannot_be_dropped(self) -> None:
        with pytest.raises(ValueError, match="required"):
            WalkForwardConfig(models=tuple(m for m in CURATED_MODELS if m != BASELINE))

    def test_the_drift_baseline_cannot_be_dropped(self) -> None:
        with pytest.raises(ValueError, match="required"):
            WalkForwardConfig(models=tuple(m for m in CURATED_MODELS if m != DRIFT_BASELINE))

    def test_the_comparison_baseline_cannot_be_swapped(self) -> None:
        with pytest.raises(ValueError, match="comparison baseline"):
            WalkForwardConfig(baseline=DRIFT_BASELINE)

    @pytest.mark.parametrize("horizons", [(), (0, 1), (3, 1), (1, 1, 7)])
    def test_horizons_must_be_positive_unique_ascending(self, horizons) -> None:
        with pytest.raises(ValueError):
            WalkForwardConfig(horizons=horizons)

    def test_two_refits_is_the_minimum(self) -> None:
        with pytest.raises(ValueError, match="stability"):
            WalkForwardConfig(n_refits=1)

    def test_early_middle_and_late_are_the_minimum(self) -> None:
        with pytest.raises(ValueError, match="early, middle and late"):
            WalkForwardConfig(n_blocks=2)

    def test_duplicate_windows_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            WalkForwardConfig(windows=(WindowSpec("rolling", 250), WindowSpec("rolling", 250)))


class TestWindows:
    def test_a_tiny_rolling_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            WindowSpec("rolling", 50)

    def test_an_expanding_window_has_no_row_count(self) -> None:
        with pytest.raises(ValueError, match="no fixed row count"):
            WindowSpec("expanding", 500)

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown window kind"):
            WindowSpec("sliding", 500)  # type: ignore[arg-type]

    @pytest.mark.parametrize("window", DEFAULT_WINDOWS)
    def test_labels_round_trip(self, window) -> None:
        assert WindowSpec.parse(window.label) == window

    def test_a_malformed_label_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot parse"):
            WindowSpec.parse("rolling-many")


class TestTheGateIsNotPermissive:
    def test_a_coin_flip_fold_share_is_refused(self) -> None:
        with pytest.raises(ValueError, match="coin"):
            GateThresholds(min_positive_fold_fraction=0.5)

    def test_a_negative_skill_floor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="worse than naive"):
            GateThresholds(min_aggregate_skill=-0.01)

    def test_a_loose_alpha_is_refused(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            GateThresholds(alpha=0.5)

    def test_the_practical_floor_sits_above_the_aggregate_floor(self) -> None:
        with pytest.raises(ValueError, match="practical"):
            GateThresholds(min_aggregate_skill=0.02, practical_min_skill=0.01)

    def test_the_declared_defaults(self) -> None:
        """Written out so a change to them is a visible diff in a test, not only
        in a digest."""
        assert GateThresholds().as_dict() == {
            "alpha": 0.05,
            "min_aggregate_skill": 0.0,
            "min_late_block_skill": 0.0,
            "min_positive_fold_fraction": 0.75,
            "practical_min_skill": 0.01,
            "raw_signal_alpha": 0.05,
        }


class TestTheDigest:
    def test_it_is_deterministic(self) -> None:
        assert WalkForwardConfig().digest() == WalkForwardConfig().digest()

    def test_the_canonical_form_is_sorted_and_compact(self) -> None:
        text = WalkForwardConfig().canonical_json()
        assert " " not in text and "\n" not in text
        assert list(json.loads(text)) == sorted(json.loads(text))

    @pytest.mark.parametrize(
        "change",
        [
            {"n_refits": 13},
            {"warmup_bars": 61},
            {"dev_fraction": 0.25},
            {"horizons": (1, 3, 7)},
            {"gates": GateThresholds(min_positive_fold_fraction=0.8)},
            {"gates": GateThresholds(practical_min_skill=0.02)},
        ],
    )
    def test_moving_anything_moves_the_digest(self, change) -> None:
        """The property that stops a threshold being adjusted after the fact."""
        assert replace(WalkForwardConfig(), **change).digest() != WalkForwardConfig().digest()

    def test_it_round_trips(self) -> None:
        config = WalkForwardConfig()
        again = WalkForwardConfig.from_dict(json.loads(config.canonical_json()))
        assert again == config and again.digest() == config.digest()

    def test_it_carries_no_run_identity(self) -> None:
        """No timestamp, host or run id: those would make two identical runs
        disagree about what they were."""
        keys = set(json.loads(WalkForwardConfig().canonical_json()))
        assert not keys & {"generated_at", "created_at", "run_id", "host", "timestamp"}
