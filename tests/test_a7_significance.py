"""Significance, and the gate that decides what a result is allowed to be called.

The gate is the part of A7 that stands between a lucky configuration and a
headline, so it is tested one gate at a time: a configuration that clears six
gates and fails the seventh must fail, and must say which one it failed.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.evaluation.inference import diebold_mariano
from btc_forecaster.research.walk_forward.config import GateThresholds, WalkForwardConfig
from btc_forecaster.research.walk_forward.scoring import ConfigScores
from btc_forecaster.research.walk_forward.significance import (
    DEGENERATE,
    INSUFFICIENT,
    OK,
    Comparison,
    compare,
    correct,
    hac_lags_for,
    newey_west_bandwidth,
)
from btc_forecaster.research.walk_forward.stability import (
    DECISIONS,
    FRAGILE_SIGNAL,
    GATE_NAMES,
    LEAKAGE_FAILED,
    LEAKAGE_PASSED,
    ROBUST_RESEARCH_CANDIDATE,
    ROBUSTLY_UNINTERESTING,
    decide,
    evaluate_gates,
)

CONFIG = WalkForwardConfig(bootstrap_resamples=200)
GATES = GateThresholds()
RNG = np.random.default_rng(7)
ACTUAL = RNG.normal(0.0, 0.02, 600)
NAIVE = np.zeros_like(ACTUAL)


class TestTheA2TestIsExtendedNotChanged:
    """A2's Diebold-Mariano gained an optional lag count. Its default is A2's."""

    loss_a = np.abs(RNG.normal(size=300))
    loss_b = np.abs(RNG.normal(size=300))

    def test_the_default_lags_still_follow_the_horizon(self) -> None:
        assert diebold_mariano(self.loss_a, self.loss_b, horizon=1).hac_lags == 0
        assert diebold_mariano(self.loss_a, self.loss_b, horizon=10).hac_lags == 9

    def test_asking_for_the_default_explicitly_changes_nothing(self) -> None:
        implicit = diebold_mariano(self.loss_a, self.loss_b, horizon=5)
        explicit = diebold_mariano(self.loss_a, self.loss_b, horizon=5, hac_lags=4)
        assert implicit.to_dict() == explicit.to_dict()

    def test_more_lags_are_used_and_reported(self) -> None:
        wider = diebold_mariano(self.loss_a, self.loss_b, horizon=5, hac_lags=12)
        assert wider.hac_lags == 12
        assert wider.statistic != diebold_mariano(self.loss_a, self.loss_b, horizon=5).statistic

    def test_fewer_lags_than_the_overlap_needs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="horizon - 1"):
            diebold_mariano(self.loss_a, self.loss_b, horizon=5, hac_lags=2)


class TestTheLagCount:
    def test_the_newey_west_bandwidth(self) -> None:
        assert newey_west_bandwidth(100) == 4
        assert newey_west_bandwidth(1400) == 7
        assert newey_west_bandwidth(0) == 0

    def test_it_is_the_larger_of_bandwidth_and_overlap(self) -> None:
        assert hac_lags_for(1, 1400) == 7
        assert hac_lags_for(30, 1400) == 29


def run(predicted: np.ndarray, *, actual: np.ndarray = ACTUAL, key=("m", 1, "rolling-250")) -> Comparison:
    return compare(actual, predicted, np.zeros_like(actual), key=key, baseline_id="naive_last_value", config=CONFIG)


class TestAComparison:
    def test_a_real_signal_is_better_and_says_so(self) -> None:
        c = run(ACTUAL * 0.5)
        assert c.status == OK and c.skill > 0 and c.mean_loss_difference < 0
        assert c.p_value_better < 1e-6 and c.skill_ci_lower > 0

    def test_a_harmful_model_is_worse_and_says_so(self) -> None:
        c = run(-ACTUAL * 0.5)
        assert c.status == OK and c.skill < 0 and c.mean_loss_difference > 0
        assert c.p_value_better > 0.5

    def test_a_forecast_identical_to_naive_is_not_a_test(self) -> None:
        assert run(NAIVE).status == DEGENERATE

    def test_too_few_origins_is_reported_not_tested(self) -> None:
        c = run(ACTUAL[:50] * 0.5, actual=ACTUAL[:50])
        assert c.status == INSUFFICIENT and c.p_value is None and c.skill > 0

    def test_it_is_deterministic(self) -> None:
        assert run(ACTUAL * 0.3 + RNG.normal(0, 0.001, 600) * 0) == run(ACTUAL * 0.3)


class TestTheCorrection:
    def comparisons(self) -> list[Comparison]:
        rng = np.random.default_rng(11)
        out = [
            run(rng.normal(0, 0.02, 600), key=(f"noise_{i}", 1 + i % 2, "rolling-250"))
            for i in range(20)
        ]
        out.append(run(ACTUAL * 0.5, key=("signal", 1, "rolling-250")))
        out.append(run(ACTUAL[:50], actual=ACTUAL[:50], key=("short", 1, "rolling-250")))
        return out

    def test_q_values_are_never_below_p_values(self) -> None:
        corrected, _ = correct(self.comparisons(), alpha=0.05)
        for c in corrected:
            if c.testable:
                assert c.q_value >= c.p_value - 1e-15
                assert c.q_value_within_horizon is not None

    def test_the_family_counts_only_testable_comparisons(self) -> None:
        _, summary = correct(self.comparisons(), alpha=0.05)
        assert summary["family_size"] == 21
        assert summary["not_testable"][INSUFFICIENT] == 1
        assert summary["expected_false_positives_at_alpha"] == pytest.approx(1.05)

    def test_the_real_signal_survives_and_is_classified_better(self) -> None:
        corrected, summary = correct(self.comparisons(), alpha=0.05)
        signal = next(c for c in corrected if c.model_id == "signal")
        assert signal.statistically_better(0.05)
        assert signal.practically_useful(0.05, 0.01)
        assert summary["significantly_better_after_bh"] >= 1


def score(skill=0.05, folds=None, late=0.04, model_id="m") -> ConfigScores:
    return ConfigScores(
        model_id=model_id, horizon=1, window="rolling-250", n_origins=1400,
        mae=0.02, rmse=0.03, mase=1.0 - skill, skill_vs_naive=skill, bias=0.0,
        directional_accuracy=0.5, balanced_accuracy=0.5, mcc=0.0, constant_direction_accuracy=0.5,
        fold_skill=folds if folds is not None else dict.fromkeys(range(12), 0.05),
        block_skill={"early": 0.05, "middle": 0.05, "late": late},
    )


def comparison(q=0.01, diff=-0.001, p_better=0.0005, skill=0.05, model_id="m") -> Comparison:
    return Comparison(
        model_id=model_id, horizon=1, window="rolling-250", n_origins=1400, status=OK, skill=skill,
        mean_loss_difference=diff, statistic=-3.5, p_value=0.001, p_value_better=p_better,
        hac_lags=7, q_value=q, q_value_within_horizon=q,
    )


def gate(s=None, c=None, leakage=LEAKAGE_PASSED, clean=True, model_id="m"):
    s = s or score(model_id=model_id)
    c = c or comparison(model_id=model_id)
    return evaluate_gates(
        [s], [c], gates=GATES, late_block="late",
        leakage={model_id: leakage}, clean={(model_id, 1, "rolling-250"): clean},
        baseline="naive_last_value",
    )[0]


class TestTheGateOneGateAtATime:
    def test_a_configuration_that_clears_everything_is_a_candidate(self) -> None:
        result = gate()
        assert result.passed and result.failed == []
        assert decide([result])["decision"] == ROBUST_RESEARCH_CANDIDATE

    @pytest.mark.parametrize(
        ("gate_name", "kwargs", "expected_failures"),
        [
            ("aggregate_skill", {"s": score(skill=-0.01), "c": comparison(skill=-0.01)}, {"aggregate_skill", "practical"}),
            ("bh_better", {"c": comparison(q=0.2)}, {"bh_better"}),
            ("late_block", {"s": score(late=-0.01)}, {"late_block"}),
            ("fold_majority", {"s": score(folds={i: (0.05 if i < 8 else -0.01) for i in range(12)})}, {"fold_majority"}),
            ("practical", {"s": score(skill=0.005), "c": comparison(skill=0.005)}, {"practical"}),
            ("leakage", {"leakage": LEAKAGE_FAILED}, {"leakage"}),
            ("resources", {"clean": False}, {"resources"}),
        ],
    )
    def test_failing_one_gate_fails_and_names_it(self, gate_name, kwargs, expected_failures) -> None:
        result = gate(**kwargs)
        assert not result.passed
        assert set(result.failed) == expected_failures
        assert gate_name in result.failed

    def test_every_gate_is_exercised_above(self) -> None:
        """Guard the guard: a gate with no failing case would be decoration."""
        assert set(GATE_NAMES) == {
            "aggregate_skill", "bh_better", "late_block", "fold_majority", "practical", "leakage", "resources",
        }

    def test_an_unchecked_model_has_not_passed_leakage(self) -> None:
        result = evaluate_gates(
            [score()], [comparison()], gates=GATES, late_block="late",
            leakage={}, clean={("m", 1, "rolling-250"): True}, baseline="naive_last_value",
        )[0]
        assert result.failed == ["leakage"]

    def test_the_baseline_is_not_gated(self) -> None:
        results = evaluate_gates(
            [score(model_id="naive_last_value")], [], gates=GATES, late_block="late",
            leakage={}, clean={}, baseline="naive_last_value",
        )
        assert results == []


class TestTheDecision:
    def test_significance_that_fails_a_gate_is_fragile(self) -> None:
        result = gate(s=score(late=-0.02))
        assert result.fragile
        assert decide([result])["decision"] == FRAGILE_SIGNAL

    def test_a_raw_signal_that_fails_correction_is_fragile(self) -> None:
        result = gate(c=comparison(q=0.3, p_better=0.01))
        assert result.raw_signal and not result.statistically_better
        assert decide([result])["decision"] == FRAGILE_SIGNAL

    def test_better_but_not_useful_is_not_useful(self) -> None:
        result = gate(s=score(skill=0.004), c=comparison(skill=0.004))
        assert result.statistically_better and not result.practically_useful

    def test_nothing_that_looks_like_a_signal_is_uninteresting(self) -> None:
        result = gate(s=score(skill=-0.03), c=comparison(q=0.9, diff=0.002, p_better=0.97, skill=-0.03))
        assert not result.fragile
        assert decide([result])["decision"] == ROBUSTLY_UNINTERESTING

    def test_there_is_no_promoted_state(self) -> None:
        verdict = decide([gate()])
        assert set(DECISIONS) == {ROBUSTLY_UNINTERESTING, FRAGILE_SIGNAL, ROBUST_RESEARCH_CANDIDATE}
        assert verdict["promoted_models"] == [] and verdict["live_trading_enabled"] is False
        assert not any("PROMOT" in d or "TRAD" in d for d in DECISIONS)
