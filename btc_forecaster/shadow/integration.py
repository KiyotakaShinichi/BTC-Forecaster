"""One-way A3-to-C0 seam; shadow models remain unpromoted."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..paper.contracts import ForecastDistribution, PromotionStatus


def to_c0_forecast(record: dict[str, Any]) -> ForecastDistribution:
    raw_quantiles = record.get("return_quantiles") or {}
    quantiles = {
        float(key): float(value)
        for key, value in sorted(raw_quantiles.items(), key=lambda item: float(item[0]))
    }
    if not quantiles:
        expected = float(record["expected_return"])
        quantiles = {0.25: expected, 0.5: expected, 0.75: expected}
    return ForecastDistribution(
        forecast_id=str(record["forecast_id"]),
        forecast_origin=__import__("datetime").datetime.fromisoformat(record["forecast_origin"]),
        instrument=str(record["instrument"]),
        horizon=timedelta(days=int(record["horizon_bars"])),
        model_id=str(record["model_id"]),
        model_version=str(record["model_version"]),
        expected_return=float(record["expected_return"]),
        return_quantiles=quantiles,
        direction_probability_raw=float(record["raw_direction_probability"] or 0.5),
        direction_probability_calibrated=None,
        calibration_version=None,
        promotion_status=PromotionStatus.UNPROMOTED,
        stability_status="FORWARD_OBSERVATION_ONLY",
        data_freshness=timedelta(seconds=max(0, float(record["data_freshness_seconds"]))),
        feature_snapshot_id=str(record["feature_snapshot_hash"]),
        regime_support=str(record["support_state"]),
    )
