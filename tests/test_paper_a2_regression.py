from pathlib import Path

from btc_forecaster.paper.a2 import current_live_permission, promoted_models
from btc_forecaster.paper.decision import LIVE_TRADING_ELIGIBLE


def test_actual_a2_state_is_load_bearing_fail_closed():
    evidence = Path("research/runs/2026-08-29-a2-benchmark/promotion.csv")
    assert promoted_models(evidence) == ()
    assert current_live_permission(evidence) == {
        "promoted_models": (),
        "live_candidates": 0,
        "allowed_live_leverage": 0.0,
        "live_trading_eligible": False,
    }
    assert LIVE_TRADING_ELIGIBLE is False
