from datetime import UTC, datetime, timedelta

import pytest

from btc_forecaster.paper.analysis import (
    block_bootstrap,
    no_trade_analysis,
    ruin_proxy,
    validate_temporal_isolation,
)
from btc_forecaster.paper.calibration import confidence_band, diagnostics, fit_platt
from btc_forecaster.paper.contracts import (
    ConfidenceBand,
    ForecastDistribution,
    PromotionStatus,
    Side,
)
from btc_forecaster.paper.costs import CostModel
from btc_forecaster.paper.decision import decide
from btc_forecaster.paper.execution import MarketBar, TradeState, execute, transition
from btc_forecaster.paper.journal import AppendOnlyJournal
from btc_forecaster.paper.risk import RiskLimits, RiskSnapshot, RiskState, risk_state, size_position

ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)
COSTS = CostModel("test-v1", 0.0001, 0.0002, 0.0001, 0.0001)


def forecast(**overrides):
    values = {
        "forecast_id": "f1",
        "forecast_origin": ORIGIN,
        "instrument": "BTC-USD",
        "horizon": timedelta(hours=24),
        "model_id": "synthetic",
        "model_version": "1",
        "expected_return": 0.04,
        "return_quantiles": {0.1: -0.01, 0.5: 0.04, 0.9: 0.08},
        "direction_probability_raw": 0.75,
        "direction_probability_calibrated": 0.72,
        "calibration_version": "cal-v1",
        "promotion_status": PromotionStatus.PROMOTED,
        "stability_status": "STABLE",
        "data_freshness": timedelta(minutes=5),
        "feature_snapshot_id": "s1",
    }
    values.update(overrides)
    return ForecastDistribution(**values)


def make_decision(**forecast_overrides):
    return decide(
        forecast(**forecast_overrides),
        origin_price=100.0,
        now=ORIGIN,
        costs=COSTS,
        risk=RiskSnapshot(10_000, 10_000),
    )


def test_strong_promoted_signal_is_paper_candidate():
    decision = make_decision()
    assert decision.side is Side.LONG_CANDIDATE
    assert decision.research_only and decision.allowed_leverage <= 1


@pytest.mark.parametrize(
    "overrides,reason",
    [
        (
            {"expected_return": 0.0001, "return_quantiles": {0.1: -0.001, 0.5: 0.0001, 0.9: 0.001}},
            "EV_BELOW_THRESHOLD",
        ),
        ({"return_quantiles": {0.1: -0.20, 0.5: 0.04, 0.9: 0.30}}, "UNCERTAINTY_EXCESSIVE"),
        ({"data_freshness": timedelta(days=1)}, "DATA_STALE"),
        ({"promotion_status": PromotionStatus.UNPROMOTED}, "MODEL_NOT_PROMOTED"),
    ],
)
def test_fail_closed_worlds(overrides, reason):
    decision = make_decision(**overrides)
    assert decision.side is Side.NO_TRADE and reason in decision.reason_codes
    assert decision.allowed_leverage == 0


def test_halted_risk_rejects():
    decision = decide(
        forecast(),
        origin_price=100,
        now=ORIGIN,
        costs=COSTS,
        risk=RiskSnapshot(10_000, 10_000, model_integrity=False),
    )
    assert decision.side is Side.NO_TRADE and "RISK_NOT_ACTIVE" in decision.reason_codes


def test_calibration_and_reliability():
    p = [0.2] * 25 + [0.8] * 25
    y = [0] * 20 + [1] * 5 + [0] * 5 + [1] * 20
    calibrator = fit_platt(p, y, version="v1")
    assert 0 <= calibrator.predict(0.8) <= 1
    report = diagnostics(p, y, n_bins=5)
    assert report.brier_score >= 0 and report.expected_calibration_error >= 0
    assert confidence_band(0.55) is ConfidenceBand.LOW
    with pytest.raises(ValueError):
        fit_platt([0.5], [1], version="bad")


def test_position_sizing_caps_and_live_permission():
    result = size_position(
        RiskSnapshot(1000, 500),
        RiskLimits(max_risk_per_trade=0.01, paper_leverage_cap=1.5),
        requested_risk_fraction=0.05,
        stop_distance_fraction=0.01,
        live_eligible=True,
    )
    assert (
        result.risk_amount == 10
        and result.position_notional == 1000
        and result.allowed_leverage == 1.5
    )
    assert (
        size_position(
            RiskSnapshot(1000, 500),
            RiskLimits(),
            requested_risk_fraction=0.01,
            stop_distance_fraction=0.01,
        ).allowed_leverage
        == 0
    )


def test_risk_states():
    limits = RiskLimits()
    assert risk_state(RiskSnapshot(1000, 1000), limits) is RiskState.ACTIVE
    assert risk_state(RiskSnapshot(1000, 1000, rolling_drawdown=0.06), limits) is RiskState.REDUCED
    assert risk_state(RiskSnapshot(1000, 1000, consecutive_losses=3), limits) is RiskState.HALTED


def test_state_machine_rejects_impossible_transition():
    assert transition(TradeState.PROPOSED, TradeState.OPEN) is TradeState.OPEN
    with pytest.raises(ValueError):
        transition(TradeState.CLOSED_TP, TradeState.OPEN)


def test_tp_sl_and_ambiguity_worlds():
    decision = make_decision()
    tp = execute(decision, [MarketBar(ORIGIN, 100, 105, 99.5, 104)], COSTS)
    sl = execute(decision, [MarketBar(ORIGIN, 100, 100.5, 98, 99)], COSTS)
    ambiguous = execute(decision, [MarketBar(ORIGIN, 100, 105, 98, 100)], COSTS)
    assert tp.state is TradeState.CLOSED_TP
    assert sl.state is TradeState.CLOSED_SL
    assert (
        ambiguous.state is TradeState.CLOSED_SL and ambiguous.exit_reason == "AMBIGUOUS_STOP_FIRST"
    )


def test_time_stop():
    decision = make_decision()
    bar = MarketBar(ORIGIN + timedelta(hours=24), 100, 101, 99.5, 100.5)
    assert execute(decision, [bar], COSTS).state is TradeState.CLOSED_TIME


def test_append_only_hash_chain_detects_tampering(tmp_path):
    journal = AppendOnlyJournal(tmp_path / "journal.jsonl")
    journal.append("decision", make_decision())
    journal.append("outcome", {"pnl": -1})
    assert len(journal.read_verified()) == 2
    text = journal.path.read_text().replace('"pnl": -1', '"pnl": 99')
    journal.path.write_text(text)
    with pytest.raises(ValueError, match="integrity"):
        journal.read_verified()


def test_no_trade_bootstrap_ruin_and_temporal_isolation():
    rejected = make_decision(promotion_status=PromotionStatus.UNPROMOTED)
    assert no_trade_analysis([rejected])["candidate_forecasts_rejected"] == 1
    assert (
        block_bootstrap([1, -1, 2, -2], seed=7) == block_bootstrap([1, -1, 2, -2], seed=7)
    ).all()
    assert 0 <= ruin_proxy([0.01, -0.02] * 5, drawdown_limit=0.2, samples=20) <= 1
    with pytest.raises(ValueError):
        validate_temporal_isolation(10, 10)
