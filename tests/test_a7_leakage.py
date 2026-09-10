"""Walk-forward leakage adversaries, the synthetic worlds, and the leaky oracle.

Two claims are tested here, and both are about the engine rather than about
BTC:

**It cannot see the future.** Every curated model is attacked five ways at one
fold -- future rows, future targets, future feature cells, a bar outside its
declared extent, and a bar inside its window -- through the same code path the
benchmark uses. The first four must change nothing; the fifth must change the
forecast, or the first four prove nothing.

**Spectacular numbers do not survive a leak.** A model that reads the realised
target forecasts perfectly. It is scored, it clears every numeric gate, and it
is still refused -- because the adversaries catch it.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.walk_forward.config import (
    GateThresholds,
    WalkForwardConfig,
    WindowSpec,
)
from btc_forecaster.research.walk_forward.evaluator import (
    InformationCache,
    JobSpec,
    run_job,
    schedule_for,
)
from btc_forecaster.research.walk_forward.leakage import (
    ADVERSARIES,
    FUTURE_TARGET,
    OUT_OF_WINDOW,
    VALID_PAST,
    run_adversaries,
)
from btc_forecaster.research.walk_forward.scoring import score_configs
from btc_forecaster.research.walk_forward.significance import compare_all, correct
from btc_forecaster.research.walk_forward.stability import (
    LEAKAGE_FAILED,
    LEAKAGE_PASSED,
    evaluate_gates,
)
from btc_forecaster.research.walk_forward.worlds import (
    LEAKY_ORACLE_ID,
    WORLDS,
    build,
    known_signal,
    regime_shift,
    registered_leaky_oracle,
)


def available(*model_ids: str) -> list[str]:
    return [m for m in model_ids if registry.get(m).is_available()]


FRAME = build("AUTOCORRELATION").frame
ROLL, EXP = WindowSpec("rolling", 150), WindowSpec("expanding")
ATTACKED = available(
    "naive_last_value", "random_walk_drift", "ar_p", "arima", "theta",
    "local_level", "local_linear_trend", "ridge", "xgboost", "mlp",
)
CONFIG = WalkForwardConfig(
    models=("naive_last_value", "random_walk_drift"),
    horizons=(1, 3),
    windows=(ROLL, EXP),
    n_refits=3,
    bootstrap_resamples=100,
    minimum_paired_origins=30,
)
SCHEDULE = schedule_for(FRAME, CONFIG)
LAST = len(SCHEDULE.folds) - 1


def attack(model_id: str, *, horizon: int = 3, window: WindowSpec = ROLL):
    return run_adversaries(
        model_id, FRAME, SCHEDULE, CONFIG, horizon=horizon, window=window, fold_index=LAST
    )


class TestEveryCuratedModelIsBlindToTheFuture:
    @pytest.mark.parametrize("model_id", ATTACKED)
    def test_it_passes_all_five_adversaries(self, model_id) -> None:
        report = attack(model_id)
        failed = [c.as_dict() for c in report.checks if not c.passed]
        assert report.passed, failed
        assert tuple(c.name for c in report.checks) == ADVERSARIES

    @pytest.mark.parametrize("model_id", available("ar_p", "ridge"))
    def test_and_at_one_bar_ahead(self, model_id) -> None:
        assert attack(model_id, horizon=1).passed

    def test_the_out_of_window_adversary_actually_ran(self) -> None:
        """The last fold's rolling extent starts inside the data, so there is a
        bar before it to poison -- this is not a vacuous pass."""
        check = next(c for c in attack("ar_p").checks if c.name == OUT_OF_WINDOW)
        assert check.applicable and not check.changed


class TestTheAdversariesAreSensitive:
    @pytest.mark.parametrize("model_id", [m for m in ATTACKED if m != "naive_last_value"])
    def test_poisoning_the_valid_past_moves_the_forecast(self, model_id) -> None:
        check = next(c for c in attack(model_id).checks if c.name == VALID_PAST)
        assert check.applicable and check.changed

    def test_the_naive_forecast_is_insensitive_by_design_and_says_so(self) -> None:
        check = next(c for c in attack("naive_last_value").checks if c.name == VALID_PAST)
        assert not check.applicable and "by design" in check.note

    def test_an_expanding_window_has_no_outside(self) -> None:
        check = next(c for c in attack("ar_p", window=EXP).checks if c.name == OUT_OF_WINDOW)
        assert not check.applicable and "first bar" in check.note


class TestTheLeakyOracle:
    def test_it_is_caught(self) -> None:
        with registered_leaky_oracle() as oracle:
            report = attack(oracle)
        assert not report.passed
        target = next(c for c in report.checks if c.name == FUTURE_TARGET)
        assert target.changed and not target.passed

    def test_it_wins_on_every_number_and_is_still_refused(self) -> None:
        """The whole point of the leakage gate, end to end."""
        with registered_leaky_oracle() as oracle:
            config = WalkForwardConfig(
                models=("naive_last_value", "random_walk_drift", oracle),
                horizons=(1,), windows=(ROLL,), n_refits=4,
                bootstrap_resamples=100, minimum_paired_origins=30,
            )
            schedule = schedule_for(FRAME, config)
            cache = InformationCache(FRAME, config)
            jobs = [run_job(JobSpec(m, 1, ROLL), schedule, config, cache) for m in config.models]
            import pandas as pd

            records = pd.concat([j.records for j in jobs], ignore_index=True)
            leak_report = run_adversaries(oracle, FRAME, schedule, config, horizon=1, window=ROLL, fold_index=3)

        scores = score_configs(records, schedule, baseline="naive_last_value")
        comparisons, _ = correct(compare_all(records, config), alpha=0.05)
        oracle_score = next(s for s in scores if s.model_id == oracle)
        assert oracle_score.skill_vs_naive == pytest.approx(1.0)

        result = next(
            r for r in evaluate_gates(
                scores, comparisons, gates=GateThresholds(), late_block="late",
                leakage={oracle: LEAKAGE_PASSED if leak_report.passed else LEAKAGE_FAILED},
                clean={j.spec.key: j.clean for j in jobs}, baseline="naive_last_value",
            )
            if r.model_id == oracle
        )
        assert result.failed == ["leakage"]
        assert not result.passed

    def test_it_cannot_outlive_its_block(self) -> None:
        with registered_leaky_oracle():
            assert registry.get(LEAKY_ORACLE_ID)
        with pytest.raises(KeyError):
            registry.get(LEAKY_ORACLE_ID)


class TestTheWorlds:
    @pytest.mark.parametrize("name", sorted(WORLDS))
    def test_each_builds_deterministically(self, name) -> None:
        assert np.array_equal(build(name).frame["close"].to_numpy(), build(name).frame["close"].to_numpy())
        assert len(build(name).expectation) > 20

    def test_the_known_signal_must_be_stationary(self) -> None:
        with pytest.raises(ValueError, match="stationary"):
            known_signal(beta=1.0)

    def test_the_regime_shift_loses_its_autocorrelation(self) -> None:
        returns = np.log(regime_shift().frame["close"]).diff().dropna().to_numpy()
        half = len(returns) // 2
        early = np.corrcoef(returns[: half - 1], returns[1:half])[0, 1]
        late = np.corrcoef(returns[half:-1], returns[half + 1 :])[0, 1]
        assert early > 0.3 and abs(late) < 0.1

    def test_an_unknown_world_is_refused(self) -> None:
        with pytest.raises(KeyError, match="known"):
            build("SOMETHING")
