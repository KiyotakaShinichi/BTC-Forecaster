"""B4.1 / B4.2 / B4.3 / B4.40 — the data contract, targets and leakage audit.

No network. Every series here is constructed in-process, so a test can put a bar
exactly on a boundary or exactly one microsecond past it, which is the only way
to pin down availability semantics.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.b4.audit import (
    coverage_bias,
    inventory_intelligence_store,
    inventory_market_series,
    unavailable_entry,
    write_inventory,
)
from market_intelligence.b4.contracts import (
    B4DataError,
    DataAvailability,
    EvidenceTier,
    LeakageError,
    MarketBar,
    MarketSeries,
    SourceDomain,
    TimestampConvention,
    require_utc,
)
from market_intelligence.b4.market_data import (
    CachingMarketDataProvider,
    FixtureMarketDataProvider,
    MarketDataProvider,
    close_as_of,
    first_close_after,
    last_available_index,
    series_to_records,
)
from market_intelligence.b4.targets import (
    assert_targets_are_future_only,
    base_close_matches_feature_view,
    build_target_manifest,
    build_targets,
    horizon_delta,
    join_targets_by_origin,
    target_result_hash,
)
from market_intelligence.storage import IntelligenceStore

BASE = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
HOUR = 3_600


def bar(offset_hours: int, close: float, *, period_seconds: int = HOUR) -> MarketBar:
    return MarketBar(
        period_start=BASE + timedelta(hours=offset_hours),
        period_seconds=period_seconds,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def series(closes: list[float], *, period_seconds: int = HOUR, series_id: str = "test") -> MarketSeries:
    step = period_seconds // HOUR
    return MarketSeries(
        series_id=series_id,
        ticker="TEST-USD",
        domain=SourceDomain.BTC_MARKET,
        provider="fixture",
        timestamp_convention=TimestampConvention.PERIOD_OPEN,
        evidence_tier=EvidenceTier.PIT_VALIDATED,
        source_timezone="UTC",
        bars=tuple(bar(index * step, close, period_seconds=period_seconds) for index, close in enumerate(closes)),
    )


class TestMarketBarContract:
    def test_a_naive_timestamp_is_rejected_not_assumed_utc(self) -> None:
        with pytest.raises(B4DataError, match="timezone-aware"):
            MarketBar(
                period_start=datetime(2026, 1, 5, 0, 0),
                period_seconds=HOUR,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=0.0,
            )

    def test_a_non_utc_timestamp_is_converted_not_rejected(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        converted = MarketBar(
            period_start=datetime(2026, 1, 5, 0, 0, tzinfo=eastern),
            period_seconds=HOUR,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=0.0,
        )
        assert converted.period_start == datetime(2026, 1, 5, 5, 0, tzinfo=timezone.utc)

    @pytest.mark.parametrize(
        ("open_", "high", "low", "close"),
        [
            (1.0, 0.5, 2.0, 1.0),  # high below low
            (3.0, 2.0, 1.0, 1.5),  # open above high
            (1.5, 2.0, 1.0, 9.0),  # close above high
        ],
    )
    def test_incoherent_ohlc_is_rejected(self, open_: float, high: float, low: float, close: float) -> None:
        with pytest.raises(B4DataError):
            MarketBar(
                period_start=BASE,
                period_seconds=HOUR,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=0.0,
            )

    def test_availability_is_the_end_of_the_period_not_its_label(self) -> None:
        """The single most load-bearing line in B4's data contract."""
        daily = bar(0, 100.0, period_seconds=86_400)
        assert daily.period_start == BASE
        assert daily.available_at == BASE + timedelta(days=1)


class TestMarketSeriesContract:
    def test_out_of_order_bars_are_rejected(self) -> None:
        with pytest.raises(B4DataError, match="strictly increasing"):
            MarketSeries(
                series_id="s",
                ticker="T",
                domain=SourceDomain.BTC_MARKET,
                provider="fixture",
                timestamp_convention=TimestampConvention.PERIOD_OPEN,
                evidence_tier=EvidenceTier.PIT_VALIDATED,
                source_timezone="UTC",
                bars=(bar(2, 1.0), bar(1, 1.0)),
            )

    def test_mixed_bar_periods_are_rejected(self) -> None:
        with pytest.raises(B4DataError, match="mixed bar periods"):
            MarketSeries(
                series_id="s",
                ticker="T",
                domain=SourceDomain.BTC_MARKET,
                provider="fixture",
                timestamp_convention=TimestampConvention.PERIOD_OPEN,
                evidence_tier=EvidenceTier.PIT_VALIDATED,
                source_timezone="UTC",
                bars=(bar(0, 1.0), bar(24, 1.0, period_seconds=86_400)),
            )

    def test_gaps_are_allowed_because_exchanges_close(self) -> None:
        weekend_gap = MarketSeries(
            series_id="s",
            ticker="T",
            domain=SourceDomain.CROSS_ASSET,
            provider="fixture",
            timestamp_convention=TimestampConvention.PERIOD_OPEN,
            evidence_tier=EvidenceTier.PIT_VALIDATED,
            source_timezone="America/New_York",
            bars=(bar(0, 1.0), bar(72, 1.1)),
        )
        assert len(weekend_gap) == 2

    def test_the_fingerprint_reacts_to_values_and_to_provenance(self) -> None:
        first = series([100.0, 101.0])
        assert first.fingerprint() == series([100.0, 101.0]).fingerprint()
        assert first.fingerprint() != series([100.0, 101.5]).fingerprint()
        relabelled = first.model_copy(update={"evidence_tier": EvidenceTier.RETROSPECTIVE_ONLY})
        assert first.fingerprint() != relabelled.fingerprint()

    def test_quarantined_rows_change_the_fingerprint(self) -> None:
        """A series that silently dropped rows is a different dataset."""
        clean = series([100.0, 101.0])
        quarantined = clean.model_copy(update={"rejected_bar_count": 3})
        assert clean.fingerprint() != quarantined.fingerprint()


class TestAvailabilityLookup:
    def test_a_bar_available_exactly_at_the_origin_counts(self) -> None:
        prices = series([100.0, 101.0, 102.0])
        # bar 0 covers [00:00, 01:00) and is available at 01:00 exactly.
        assert close_as_of(prices, BASE + timedelta(hours=1)) == 100.0

    def test_a_bar_one_microsecond_short_of_availability_does_not(self) -> None:
        prices = series([100.0, 101.0, 102.0])
        just_before = BASE + timedelta(hours=1) - timedelta(microseconds=1)
        assert close_as_of(prices, just_before) is None

    def test_an_origin_before_the_series_has_no_base(self) -> None:
        prices = series([100.0, 101.0])
        assert last_available_index(prices, BASE) == -1
        assert close_as_of(prices, BASE) is None

    def test_first_close_after_is_strictly_after(self) -> None:
        prices = series([100.0, 101.0, 102.0])
        origin = BASE + timedelta(hours=1)
        found = first_close_after(prices, origin)
        assert found is not None
        available_at, close = found
        assert available_at > origin
        assert close == 101.0

    def test_first_close_after_the_end_is_none(self) -> None:
        assert first_close_after(series([100.0]), BASE + timedelta(days=5)) is None

    def test_records_expose_both_timestamps(self) -> None:
        records = series_to_records(series([100.0]))
        assert records[0]["period_start"] != records[0]["available_at"]


class TestTargets:
    def test_the_base_close_is_what_a_feature_would_legitimately_see(self) -> None:
        prices = series([100.0, 110.0, 121.0, 133.1])
        origins = [BASE + timedelta(hours=2)]
        rows = build_targets(prices, origins, horizons=["1h"])
        assert rows[0].base_close == 110.0
        assert base_close_matches_feature_view(prices, origins[0], rows[0].base_close)

    def test_forward_return_uses_a_strictly_future_close(self) -> None:
        prices = series([100.0, 110.0, 121.0])
        rows = build_targets(prices, [BASE + timedelta(hours=2)], horizons=["1h"])
        assert rows[0].forward_return["1h"] == pytest.approx(121.0 / 110.0 - 1.0)
        assert rows[0].forward_direction["1h"] == 1

    def test_a_horizon_running_off_the_data_is_none_not_zero(self) -> None:
        prices = series([100.0, 110.0])
        rows = build_targets(prices, [BASE + timedelta(hours=2)], horizons=["1h", "24h"])
        assert rows[0].forward_return["24h"] is None
        assert rows[0].forward_abs_return["24h"] is None
        assert rows[0].forward_direction["24h"] is None

    def test_origins_before_the_series_are_dropped_not_defaulted(self) -> None:
        prices = series([100.0, 110.0, 121.0])
        rows = build_targets(prices, [BASE, BASE + timedelta(hours=2)], horizons=["1h"])
        assert [row.forecast_origin for row in rows] == [BASE + timedelta(hours=2)]

    def test_a_flat_move_has_no_direction_rather_than_a_fabricated_one(self) -> None:
        prices = series([100.0, 100.0, 100.0])
        rows = build_targets(prices, [BASE + timedelta(hours=1)], horizons=["1h"])
        assert rows[0].forward_return["1h"] == 0.0
        assert rows[0].forward_direction["1h"] is None

    def test_absolute_return_and_realized_volatility_are_reported(self) -> None:
        prices = series([100.0 + (index % 7) * 3.0 for index in range(100)])
        rows = build_targets(prices, [BASE + timedelta(hours=1)], horizons=["6h"])
        assert rows[0].forward_abs_return["6h"] == abs(rows[0].forward_return["6h"] or 0.0)
        volatility = rows[0].realized_volatility["6h"]
        assert volatility is not None and volatility > 0.0

    def test_a_horizon_longer_than_the_series_is_none_rather_than_truncated(self) -> None:
        """Silently shortening a 72h horizon to whatever data exists would put a
        differently-defined outcome in the same column."""
        prices = series([100.0, 110.0, 121.0, 133.0, 146.0])
        rows = build_targets(prices, [BASE + timedelta(hours=1)], horizons=["72h"])
        assert rows[0].forward_return["72h"] is None

    def test_an_unknown_horizon_is_rejected(self) -> None:
        with pytest.raises(B4DataError, match="unknown horizon"):
            horizon_delta("13m")

    def test_targets_from_an_empty_series_are_refused(self) -> None:
        empty = MarketSeries(
            series_id="s",
            ticker="T",
            domain=SourceDomain.BTC_MARKET,
            provider="fixture",
            timestamp_convention=TimestampConvention.PERIOD_OPEN,
            evidence_tier=EvidenceTier.PIT_VALIDATED,
            source_timezone="UTC",
            bars=(),
        )
        with pytest.raises(B4DataError):
            build_targets(empty, [BASE], horizons=["1h"])

    def test_the_result_hash_is_order_sensitive_and_deterministic(self) -> None:
        prices = series([100.0, 110.0, 121.0, 133.0])
        first = build_targets(prices, [BASE + timedelta(hours=1), BASE + timedelta(hours=2)], horizons=["1h"])
        again = build_targets(prices, [BASE + timedelta(hours=1), BASE + timedelta(hours=2)], horizons=["1h"])
        reversed_rows = list(reversed(first))
        assert target_result_hash(first) == target_result_hash(again)
        assert target_result_hash(first) != target_result_hash(reversed_rows)


class TestLeakageAudit:
    """B4.40. The audit must catch leakage that `build_targets` did not create."""

    def test_a_clean_dataset_passes(self) -> None:
        prices = series([100.0, 110.0, 121.0, 133.0])
        origins = [BASE + timedelta(hours=index) for index in (1, 2, 3)]
        rows = build_targets(prices, origins, horizons=["1h"])
        assert_targets_are_future_only(prices, rows, ["1h"])

    def test_a_base_price_from_the_future_is_caught(self) -> None:
        prices = series([100.0, 110.0, 121.0])
        rows = build_targets(prices, [BASE + timedelta(hours=2)], horizons=["1h"])
        tampered = [
            type(rows[0])(
                forecast_origin=rows[0].forecast_origin,
                base_close=rows[0].base_close,
                base_available_at=rows[0].forecast_origin + timedelta(seconds=1),
                forward_return=rows[0].forward_return,
                forward_abs_return=rows[0].forward_abs_return,
                forward_direction=rows[0].forward_direction,
                realized_volatility=rows[0].realized_volatility,
            )
        ]
        with pytest.raises(LeakageError, match="not available at the origin"):
            assert_targets_are_future_only(prices, tampered, ["1h"])

    def test_an_outcome_that_does_not_reproduce_from_the_series_is_caught(self) -> None:
        """An adversarial row whose stored return was quietly edited."""
        prices = series([100.0, 110.0, 121.0])
        rows = build_targets(prices, [BASE + timedelta(hours=2)], horizons=["1h"])
        tampered = [
            type(rows[0])(
                forecast_origin=rows[0].forecast_origin,
                base_close=rows[0].base_close,
                base_available_at=rows[0].base_available_at,
                forward_return={"1h": 0.99},
                forward_abs_return=rows[0].forward_abs_return,
                forward_direction=rows[0].forward_direction,
                realized_volatility=rows[0].realized_volatility,
            )
        ]
        with pytest.raises(LeakageError, match="does not reproduce"):
            assert_targets_are_future_only(prices, tampered, ["1h"])

    def test_every_target_bar_is_strictly_after_its_origin(self) -> None:
        prices = series([100.0 + index for index in range(48)])
        origins = [BASE + timedelta(hours=index) for index in range(1, 24)]
        rows = build_targets(prices, origins, horizons=["1h", "6h"])
        for row in rows:
            assert row.base_available_at <= row.forecast_origin
            for horizon in ("1h", "6h"):
                if row.forward_return[horizon] is None:
                    continue
                end = row.forecast_origin + horizon_delta(horizon)
                index = last_available_index(prices, end)
                assert prices.bars[index].available_at > row.forecast_origin


class TestTargetManifest:
    def test_the_manifest_records_series_and_target_provenance(self) -> None:
        prices = series([100.0, 110.0, 121.0, 133.0])
        origins = [BASE + timedelta(hours=index) for index in (1, 2)]
        rows = build_targets(prices, origins, horizons=["1h"])
        manifest = build_target_manifest(
            prices, rows, ["1h"], git_sha="abc123", created_at=datetime(2026, 8, 31, tzinfo=timezone.utc)
        )
        assert manifest.series_fingerprint == prices.fingerprint()
        assert manifest.result_hash == target_result_hash(rows)
        assert manifest.evidence_tier is EvidenceTier.PIT_VALIDATED
        assert manifest.origin_count == 2
        assert "forward_return_H" in manifest.target_definitions

    def test_an_empty_target_set_cannot_be_manifested(self) -> None:
        prices = series([100.0])
        with pytest.raises(B4DataError):
            build_target_manifest(
                prices, [], ["1h"], git_sha="abc", created_at=datetime(2026, 8, 31, tzinfo=timezone.utc)
            )


class TestJoin:
    def test_features_and_targets_join_on_origin_only(self) -> None:
        prices = series([100.0, 110.0, 121.0])
        origin = BASE + timedelta(hours=2)
        rows = build_targets(prices, [origin], horizons=["1h"])
        joined = join_targets_by_origin([{"forecast_origin": origin, "event_count_24h": 3.0}], rows)
        assert len(joined) == 1
        assert joined[0]["event_count_24h"] == 3.0
        assert joined[0]["forward_return_1h"] == pytest.approx(121.0 / 110.0 - 1.0)

    def test_an_origin_missing_on_either_side_is_dropped_not_filled(self) -> None:
        prices = series([100.0, 110.0, 121.0])
        rows = build_targets(prices, [BASE + timedelta(hours=2)], horizons=["1h"])
        joined = join_targets_by_origin(
            [
                {"forecast_origin": BASE + timedelta(hours=2)},
                {"forecast_origin": BASE + timedelta(hours=99)},
            ],
            rows,
        )
        assert len(joined) == 1

    def test_a_feature_row_without_an_origin_is_an_error(self) -> None:
        with pytest.raises(B4DataError, match="no forecast_origin"):
            join_targets_by_origin([{"event_count_24h": 1.0}], [])


class TestProviders:
    def test_the_fixture_provider_windows_by_period_start(self) -> None:
        provider = FixtureMarketDataProvider({"test": series([100.0, 110.0, 121.0, 133.0])})
        windowed = provider.fetch(
            "test",
            "TEST-USD",
            SourceDomain.BTC_MARKET,
            "1h",
            start=BASE + timedelta(hours=1),
            end=BASE + timedelta(hours=3),
        )
        assert [b.close for b in windowed.bars] == [110.0, 121.0]

    def test_an_unknown_fixture_series_is_an_error(self) -> None:
        with pytest.raises(B4DataError, match="no fixture series"):
            FixtureMarketDataProvider({}).fetch("nope", "T", SourceDomain.BTC_MARKET, "1h")

    def test_the_cache_serves_the_second_fetch_without_the_inner_provider(self, tmp_path: Path) -> None:
        inner = _CountingProvider(series([100.0, 110.0]))
        cached = CachingMarketDataProvider(inner, tmp_path / "cache")
        first = cached.fetch("test", "TEST-USD", SourceDomain.BTC_MARKET, "1h")
        second = cached.fetch("test", "TEST-USD", SourceDomain.BTC_MARKET, "1h")
        assert inner.calls == 1
        assert first.fingerprint() == second.fingerprint()


class _CountingProvider(MarketDataProvider):
    name = "counting"

    def __init__(self, payload: MarketSeries) -> None:
        self.payload = payload
        self.calls = 0

    def fetch(
        self,
        series_id: str,
        ticker: str,
        domain: SourceDomain,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> MarketSeries:
        self.calls += 1
        return self.payload


class TestDataAvailabilityAudit:
    def test_an_empty_intelligence_store_reports_data_unavailable(self, tmp_path: Path) -> None:
        """The finding this whole track turns on, asserted rather than assumed."""
        store = IntelligenceStore(tmp_path / "empty.duckdb")
        try:
            entries = inventory_intelligence_store(store)
        finally:
            store.close()
        assert entries, "the audit must still produce rows for an empty store"
        assert all(entry.availability is DataAvailability.DATA_UNAVAILABLE for entry in entries)
        assert all(not entry.suitable_for_historical_study for entry in entries)
        assert all(entry.evidence_tier is None for entry in entries)

    def test_a_market_series_inventory_records_span_and_provenance(self) -> None:
        entry = inventory_market_series(
            series([100.0, 110.0, 121.0]), data_type="test series", license_status="research only"
        )
        assert entry.availability is DataAvailability.AVAILABLE
        assert entry.suitable_for_historical_study
        assert entry.observation_count == 3
        assert "fingerprint=" in entry.provenance

    def test_quarantined_rows_appear_in_missingness(self) -> None:
        noisy = series([100.0, 110.0]).model_copy(
            update={"rejected_bar_count": 2, "rejection_summary": ("2001-02-13: close outside range",)}
        )
        entry = inventory_market_series(noisy, data_type="t", license_status="l")
        assert "quarantined" in entry.missingness

    def test_an_unavailable_entry_records_the_reason(self) -> None:
        entry = unavailable_entry(
            "whales", SourceDomain.ONCHAIN_WHALE, "large transfers", "no provider implementation exists"
        )
        assert entry.availability is DataAvailability.DATA_UNAVAILABLE
        assert entry.as_record()["suitable_for_historical_study"] == "NO"
        assert "no provider implementation" in entry.notes

    def test_the_inventory_is_written_atomically(self, tmp_path: Path) -> None:
        entry = unavailable_entry("x", SourceDomain.NEWS_WEB, "d", "r")
        path = write_inventory([entry], tmp_path / "nested" / "inventory.json")
        assert path.exists()
        assert not list(path.parent.glob("*.tmp"))


class TestCoverageBias:
    def test_a_single_coverage_bucket_is_reported_as_insufficient(self) -> None:
        rows = [
            {"forecast_origin": BASE, "provider_coverage_ratio": 1.0, "forward_return_24h": 0.01},
            {"forecast_origin": BASE, "provider_coverage_ratio": 1.0, "forward_return_24h": -0.02},
        ]
        result = coverage_bias(rows, outcome_field="forward_return_24h")
        assert result.low_coverage_origins == 0
        assert result.verdict.startswith("INSUFFICIENT")

    def test_a_large_dispersion_gap_between_buckets_is_flagged(self) -> None:
        rows = [
            {"forecast_origin": BASE, "provider_coverage_ratio": 1.0, "forward_return_24h": 0.10},
            {"forecast_origin": BASE, "provider_coverage_ratio": 1.0, "forward_return_24h": -0.10},
            {"forecast_origin": BASE, "provider_coverage_ratio": 0.2, "forward_return_24h": 0.001},
            {"forecast_origin": BASE, "provider_coverage_ratio": 0.2, "forward_return_24h": -0.001},
        ]
        result = coverage_bias(rows, outcome_field="forward_return_24h")
        assert result.verdict.startswith("COVERAGE BIAS PRESENT")
        assert result.absolute_difference is not None
        assert math.isclose(result.absolute_difference, 0.099, abs_tol=1e-9)

    def test_similar_buckets_are_not_flagged(self) -> None:
        rows = [
            {"forecast_origin": BASE, "provider_coverage_ratio": 1.0, "forward_return_24h": 0.010},
            {"forecast_origin": BASE, "provider_coverage_ratio": 0.5, "forward_return_24h": 0.011},
        ]
        result = coverage_bias(rows, outcome_field="forward_return_24h")
        assert result.verdict.startswith("NO MATERIAL COVERAGE BIAS")


def test_require_utc_rejects_naive_and_normalises_offsets() -> None:
    with pytest.raises(B4DataError):
        require_utc(datetime(2026, 1, 1))
    tokyo = timezone(timedelta(hours=9))
    assert require_utc(datetime(2026, 1, 1, 9, tzinfo=tokyo)) == datetime(2026, 1, 1, tzinfo=timezone.utc)
