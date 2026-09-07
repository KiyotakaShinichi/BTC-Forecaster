from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.cli import build_parser
from btc_forecaster.data.snapshot import MarketSnapshot
from btc_forecaster.paper.contracts import Side
from btc_forecaster.paper.costs import CostModel
from btc_forecaster.paper.decision import LIVE_TRADING_ELIGIBLE, decide
from btc_forecaster.paper.risk import RiskSnapshot
from btc_forecaster.shadow.contracts import (
    DataProvenance,
    ForecastMode,
    ForwardForecastRecord,
    Timeliness,
)
from btc_forecaster.shadow.evaluation import (
    MIN_FORWARD_FORECASTS,
    block_bootstrap_skill,
    forward_report,
)
from btc_forecaster.shadow.integration import to_c0_forecast
from btc_forecaster.shadow.ledger import EvidenceLedger, IntegrityError, LockHeldError
from btc_forecaster.shadow.registry import ShadowRegistry
from btc_forecaster.shadow.reporting import audit_forecast, export, status, verify
from btc_forecaster.shadow.service import (
    completed_frame,
    issue_run,
    record_missed,
    score_pending,
    timeliness,
)


def market_frame(periods: int = 240) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC", name="date")
    close = 50_000 * np.exp(np.linspace(0, 0.12, periods) + 0.01 * np.sin(np.arange(periods) / 5))
    return pd.DataFrame(
        {
            "close": close,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": 1000.0,
        },
        index=index,
    )


def snapshot(periods: int = 200) -> MarketSnapshot:
    frame = market_frame(periods)
    return MarketSnapshot.build(
        frame,
        ticker="BTC-USD",
        provider="synthetic",
        retrieved_at=frame.index[-1] + pd.Timedelta(days=1),
        normalise=False,
    )


def on_time(periods: int = 200) -> datetime:
    return (market_frame(periods).index[-1] + pd.Timedelta(days=1, minutes=30)).to_pydatetime()


def fast_registry() -> ShadowRegistry:
    registry = ShadowRegistry.load()
    return replace(registry, models=(registry.models[0],))


def test_registry_is_frozen_small_and_honest():
    registry = ShadowRegistry.load()
    assert registry.horizon_bars == 30
    assert [(model.model_id, model.baseline) for model in registry.models] == [
        ("random_walk", True),
        ("arima", False),
    ]
    assert len(registry.registry_hash) == 64


def test_typed_contracts_reject_non_utc_and_backdating(tmp_path):
    now = datetime(2025, 1, 2, tzinfo=UTC)
    provenance = DataProvenance(
        "synthetic", now, now - timedelta(days=1), now - timedelta(days=1), 10, "a" * 64
    )
    assert provenance.row_count == 10
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        DataProvenance("bad", datetime(2025, 1, 2), now, now, 1, "x")
    result = issue_run(
        snapshot(),
        EvidenceLedger(tmp_path),
        registry=fast_registry(),
        now=on_time(),
        code_sha="1" * 40,
        dry_run=True,
    )
    payload = dict(result["forecasts"][0])
    payload["forecast_mode"] = ForecastMode.FORWARD_SHADOW.value
    payload["created_at"] = (
        datetime.fromisoformat(payload["forecast_origin"]) - timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(ValueError, match="backdated"):
        ForwardForecastRecord.from_dict(payload)


def test_completed_bar_cutoff_defeats_future_rows():
    snap = snapshot()
    now = on_time()
    poisoned = snap.frame.copy()
    poisoned.loc[pd.Timestamp(now).normalize(), "close"] = 1
    revised = MarketSnapshot.build(
        poisoned, ticker="BTC-USD", provider="synthetic", retrieved_at=now, normalise=False
    )
    permitted = completed_frame(revised, now=now)
    assert permitted.index.max() < pd.Timestamp(now).normalize()
    assert float(permitted.iloc[-1]["close"]) != 1


def test_dry_run_is_uncommitted(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger")
    result = issue_run(
        snapshot(), ledger, registry=fast_registry(), now=on_time(), code_sha="a" * 40, dry_run=True
    )
    assert result["mode"] == ForecastMode.UNCOMMITTED_DRY_RUN.value
    assert result["committed"] is False and not ledger.forecasts_path.exists()


def test_issue_idempotency_manifest_last_and_conflict(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger")
    first = issue_run(
        snapshot(), ledger, registry=fast_registry(), now=on_time(), code_sha="b" * 40
    )
    second = issue_run(
        snapshot(),
        ledger,
        registry=fast_registry(),
        now=on_time() + timedelta(minutes=1),
        code_sha="b" * 40,
    )
    assert len(ledger.forecast_payloads()) == 1
    assert first["manifest"]["forecast_ids"] == second["manifest"]["forecast_ids"]
    assert len(ledger.records(ledger.manifests_path)) == 2
    conflict = dict(ledger.forecast_payloads()[0])
    conflict["point_forecast"] += 1
    with pytest.raises(IntegrityError, match="conflicting forecast"):
        ledger.append_forecast(conflict)


def test_late_and_missed_are_evidence(tmp_path):
    registry = ShadowRegistry.load()
    origin = datetime(2025, 1, 1, tzinfo=UTC)
    assert (
        timeliness(registry, origin=origin, created_at=origin + timedelta(days=1, minutes=30))
        is Timeliness.ON_TIME
    )
    assert (
        timeliness(registry, origin=origin, created_at=origin + timedelta(days=2))
        is Timeliness.LATE_FORECAST
    )
    ledger = EvidenceLedger(tmp_path)
    record_missed(
        ledger, scheduled_origin=origin, model_ids=("random_walk",), reason="scheduler unavailable"
    )
    assert status(ledger)["missed_forecasts"] == 1


def test_lock_prevents_overlapping_official_runs(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    with ledger.lock():
        with pytest.raises(LockHeldError):
            with ledger.lock():
                pass
        with pytest.raises(LockHeldError):
            issue_run(
                snapshot(), ledger, registry=fast_registry(), now=on_time(), code_sha="0" * 40
            )


def test_forward_clock_maturity_and_scoring(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger")
    issued = issue_run(
        snapshot(200), ledger, registry=fast_registry(), now=on_time(200), code_sha="c" * 40
    )
    target = datetime.fromisoformat(issued["forecasts"][0]["target_timestamp"])
    full = snapshot(230)
    half = score_pending(full, ledger, now=target - timedelta(days=1))
    assert half == {"scored": 0, "pending": 1, "unavailable": 0}
    mature = score_pending(full, ledger, now=target + timedelta(days=1))
    assert mature == {"scored": 1, "pending": 0, "unavailable": 0}
    assert len(ledger.outcome_payloads()) == 1
    report = forward_report(ledger)["models"]["random_walk"]
    assert report["scored"] == 1 and report["paired_baseline_n"] == 1
    assert score_pending(full, ledger, now=target + timedelta(days=2))["scored"] == 0


def test_missing_target_bar_stays_unavailable(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger")
    issued = issue_run(
        snapshot(200), ledger, registry=fast_registry(), now=on_time(200), code_sha="d" * 40
    )
    target = datetime.fromisoformat(issued["forecasts"][0]["target_timestamp"])
    missing = market_frame(230).drop(pd.Timestamp(target) - pd.Timedelta(days=3))
    outcome = MarketSnapshot.build(
        missing,
        ticker="BTC-USD",
        provider="synthetic",
        retrieved_at=target + timedelta(days=1),
        normalise=False,
    )
    result = score_pending(outcome, ledger, now=target + timedelta(days=1))
    assert result["unavailable"] == 1 and not ledger.outcome_payloads()


def test_integrity_audit_export_and_tamper_detection(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger")
    result = issue_run(
        snapshot(), ledger, registry=fast_registry(), now=on_time(), code_sha="e" * 40
    )
    forecast_id = result["manifest"]["forecast_ids"][0]
    assert audit_forecast(ledger, forecast_id)["integrity_status"] == "VERIFIED"
    assert verify(ledger)["forecasts"] == 1
    destination = export(ledger, tmp_path / "export.json")
    assert json.loads(destination.read_text())["registry"]["version"] == "a3-shadow-v1"
    ledger.forecasts_path.write_text(
        ledger.forecasts_path.read_text().replace(
            '"origin_price":', '"origin_price": 1, "old_origin_price":'
        ),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError):
        ledger.verify()


def test_shadow_to_c0_remains_no_trade(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger")
    result = issue_run(
        snapshot(), ledger, registry=fast_registry(), now=on_time(), code_sha="f" * 40
    )
    c0_forecast = to_c0_forecast(result["forecasts"][0])
    decision = decide(
        c0_forecast,
        origin_price=result["forecasts"][0]["origin_price"],
        now=on_time(),
        costs=CostModel("test", 0.0001, 0.0002, 0.0001, 0.0001),
        risk=RiskSnapshot(10_000, 10_000),
    )
    assert decision.side is Side.NO_TRADE
    assert "MODEL_NOT_PROMOTED" in decision.reason_codes
    assert decision.allowed_leverage == 0 and LIVE_TRADING_ELIGIBLE is False


def test_paired_metrics_minimum_n_and_interval_policy(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    assert forward_report(ledger)["minimum_forward_forecasts"] == MIN_FORWARD_FORECASTS == 100
    assert block_bootstrap_skill([1.0], [2.0]) == "INSUFFICIENT_N_FOR_INTERVAL"
    with pytest.raises(ValueError, match="paired"):
        block_bootstrap_skill([1.0], [1.0, 2.0])
    assert block_bootstrap_skill([1.0] * 50, [2.0] * 50, samples=20, seed=7) == (1.0, 1.0)


def test_cli_surface_and_timezone_invariance(monkeypatch):
    parser = build_parser()
    for command in (
        "shadow-run",
        "shadow-score",
        "shadow-status",
        "shadow-audit",
        "shadow-verify",
        "shadow-export",
    ):
        suffix = (
            ["id"]
            if command == "shadow-audit"
            else (["--output", "x"] if command == "shadow-export" else [])
        )
        assert parser.parse_args([command, *suffix]).command == command
    registry = ShadowRegistry.load()
    origin = datetime(2025, 1, 1, tzinfo=UTC)
    values = []
    for zone in ("UTC", "Pacific/Auckland", "America/New_York"):
        monkeypatch.setenv("TZ", zone)
        values.append(timeliness(registry, origin=origin, created_at=origin + timedelta(days=1)))
    assert values == [Timeliness.ON_TIME] * 3
