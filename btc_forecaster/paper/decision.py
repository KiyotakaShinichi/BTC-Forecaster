"""Fail-closed transformation from forecasts to paper TradeDecision objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from .calibration import confidence_band
from .contracts import ConfidenceBand, ForecastDistribution, PromotionStatus, Side, TradeDecision
from .costs import CostModel
from .policies import EntryPolicy, StopPolicy, TargetPolicy, entry, stop, target
from .risk import RiskLimits, RiskSnapshot, RiskState, risk_state, size_position

LIVE_TRADING_ELIGIBLE = False


@dataclass(frozen=True)
class DecisionPolicy:
    version: str = "c0-decision-v1"
    max_forecast_age_seconds: float = 7200
    min_ev_after_costs: float = 0.001
    max_uncertainty_width: float = 0.12
    requested_risk_fraction: float = 0.005
    entry_policy: EntryPolicy = EntryPolicy.MARKET_AT_ORIGIN_V1
    stop_policy: StopPolicy = StopPolicy.FORECAST_ADVERSE_QUANTILE_V1
    target_policy: TargetPolicy = TargetPolicy.FORECAST_MEDIAN_V1


def distribution_ev(forecast: ForecastDistribution, side: Side, costs: CostModel) -> float:
    """Approximate integral of the discrete quantile function, net of round-trip costs."""
    returns = list(forecast.return_quantiles.values())
    mean_return = sum(returns) / len(returns) if returns else forecast.expected_return
    signed = mean_return if side is Side.LONG_CANDIDATE else -mean_return
    return signed - costs.round_trip_fraction()


def decide(
    forecast: ForecastDistribution,
    *,
    origin_price: float,
    now: datetime,
    costs: CostModel,
    risk: RiskSnapshot,
    limits: RiskLimits = RiskLimits(),
    policy: DecisionPolicy = DecisionPolicy(),
) -> TradeDecision:
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    calibrated = forecast.direction_probability_calibrated
    band = confidence_band(calibrated) if calibrated is not None else ConfidenceBand.LOW
    desired_side = Side.LONG_CANDIDATE if forecast.expected_return >= 0 else Side.SHORT_CANDIDATE
    interval = (min(forecast.return_quantiles.values()), max(forecast.return_quantiles.values()))
    ev = distribution_ev(forecast, desired_side, costs)
    promotion_gate = forecast.promotion_status is PromotionStatus.PROMOTED
    calibration_gate = calibrated is not None and forecast.calibration_version is not None
    data_gate = forecast.data_freshness.total_seconds() <= policy.max_forecast_age_seconds
    uncertainty_gate = interval[1] - interval[0] <= policy.max_uncertainty_width
    risk_gate = risk_state(risk, limits) is RiskState.ACTIVE
    reasons: list[str] = []
    gates = (
        (promotion_gate, "MODEL_NOT_PROMOTED"),
        (calibration_gate, "CALIBRATION_INVALID"),
        (data_gate, "DATA_STALE"),
        (uncertainty_gate, "UNCERTAINTY_EXCESSIVE"),
        (ev >= policy.min_ev_after_costs, "EV_BELOW_THRESHOLD"),
        (band is not ConfidenceBand.LOW, "CONFIDENCE_LOW"),
        (risk_gate, "RISK_NOT_ACTIVE"),
        (risk.model_integrity, "INTEGRITY_BLOCKER"),
    )
    reasons.extend(code for passed, code in gates if not passed)
    side = desired_side if not reasons else Side.NO_TRADE
    ent = entry(policy.entry_policy, origin_price)
    adverse = interval[0] if desired_side is Side.LONG_CANDIDATE else interval[1]
    stp = stop(policy.stop_policy, desired_side, ent.price, adverse_return=adverse)
    targets = target(
        policy.target_policy,
        desired_side,
        ent.price,
        forecast_return=forecast.expected_return,
        stop_distance=stp.distance_fraction,
    )
    size = size_position(
        risk,
        limits,
        requested_risk_fraction=policy.requested_risk_fraction,
        stop_distance_fraction=stp.distance_fraction,
        live_eligible=side is not Side.NO_TRADE,
    )
    # C0 candidates are paper-only. Live permission is independently and permanently zero here.
    allowed = size.allowed_leverage if side is not Side.NO_TRADE else 0.0
    digest = sha256(
        f"{forecast.forecast_id}|{policy.version}|{now.isoformat()}".encode()
    ).hexdigest()[:20]
    return TradeDecision(
        decision_id=f"c0-{digest}",
        forecast_origin=forecast.forecast_origin,
        instrument=forecast.instrument,
        side=side,
        forecast_horizon=forecast.horizon,
        entry_policy=policy.entry_policy.value,
        entry_price=ent.price if side is not Side.NO_TRADE else None,
        entry_zone_low=ent.zone_low if side is not Side.NO_TRADE else None,
        entry_zone_high=ent.zone_high if side is not Side.NO_TRADE else None,
        expected_return=forecast.expected_return,
        expected_return_interval=interval,
        raw_probability=forecast.direction_probability_raw,
        calibrated_probability=calibrated,
        confidence_band=band,
        stop_price=stp.price if side is not Side.NO_TRADE else None,
        stop_distance_fraction=stp.distance_fraction if side is not Side.NO_TRADE else None,
        stop_policy=policy.stop_policy.value,
        take_profit_levels=targets if side is not Side.NO_TRADE else (),
        time_stop=forecast.forecast_origin + forecast.horizon,
        expected_value_after_costs=ev,
        expected_R=ev / stp.distance_fraction if side is not Side.NO_TRADE else None,
        risk_fraction=size.risk_fraction if side is not Side.NO_TRADE else 0.0,
        risk_amount=size.risk_amount if side is not Side.NO_TRADE else 0.0,
        position_notional=size.position_notional if side is not Side.NO_TRADE else 0.0,
        required_leverage=size.required_leverage if side is not Side.NO_TRADE else 0.0,
        allowed_leverage=allowed,
        promotion_gate=promotion_gate,
        risk_gate=risk_gate,
        data_gate=data_gate,
        reason_codes=tuple(reasons) if reasons else ("ALL_GATES_PASS",),
        invalidation_reasons=(
            "MODEL_PROMOTION_REVOKED",
            "DATA_OR_CALIBRATION_INVALID",
            "RISK_HALTED",
        ),
        forecast_id=forecast.forecast_id,
        model_id=forecast.model_id,
        feature_snapshot_id=forecast.feature_snapshot_id,
        cost_model_version=costs.version,
    )
