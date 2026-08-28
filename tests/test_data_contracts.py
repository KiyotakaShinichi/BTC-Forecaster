"""Tests for the market data contract and reproducible snapshots.

None of these touch the network: every frame comes from
:mod:`btc_forecaster.testing`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.data.contracts import (
    REQUIRED_COLUMNS,
    SchemaViolation,
    bars_span,
    normalise_market_frame,
    reindex_to_regular_daily,
    slice_to_window,
    validate_market_frame,
)
from btc_forecaster.data.providers import InMemoryProvider, SnapshotProvider
from btc_forecaster.data.snapshot import (
    MarketSnapshot,
    SnapshotIntegrityError,
    SnapshotManifest,
    canonical_bytes,
    frame_digest,
)
from btc_forecaster.testing import synthetic_market_frame
from btc_forecaster.timebase import UTC


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_market_frame(periods=200, seed=7)


class TestNormalisation:
    def test_naive_index_becomes_utc(self):
        raw = pd.DataFrame(
            {"Close": [1.0, 2.0, 3.0], "Volume": [10.0, 11.0, 12.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )
        out = normalise_market_frame(raw)
        assert str(out.index.tz) == "UTC"
        validate_market_frame(out)

    def test_multiindex_columns_are_flattened(self):
        """yfinance returns a MultiIndex whenever the ticker argument is a list."""
        index = pd.to_datetime(["2024-01-01", "2024-01-02"])
        raw = pd.DataFrame(
            [[100.0, 5.0], [101.0, 6.0]],
            index=index,
            columns=pd.MultiIndex.from_tuples([("Close", "BTC-USD"), ("Volume", "BTC-USD")]),
        )
        out = normalise_market_frame(raw)
        assert list(out.columns) == ["close", "volume"]

    def test_plain_close_is_preferred_over_adj_close(self):
        """The old substring match could silently pick either one."""
        index = pd.to_datetime(["2024-01-01", "2024-01-02"])
        raw = pd.DataFrame(
            {
                "Adj Close": [999.0, 999.0],
                "Close": [100.0, 101.0],
                "Volume": [5.0, 6.0],
            },
            index=index,
        )
        out = normalise_market_frame(raw)
        assert out["close"].tolist() == [100.0, 101.0]

    def test_adj_close_is_used_when_close_is_absent(self):
        index = pd.to_datetime(["2024-01-01", "2024-01-02"])
        raw = pd.DataFrame({"Adj Close": [100.0, 101.0], "Volume": [5.0, 6.0]}, index=index)
        assert normalise_market_frame(raw)["close"].tolist() == [100.0, 101.0]

    def test_optional_ohlc_columns_are_carried_through(self):
        index = pd.to_datetime(["2024-01-01", "2024-01-02"])
        raw = pd.DataFrame(
            {
                "Open": [99.0, 100.0],
                "High": [102.0, 103.0],
                "Low": [98.0, 99.0],
                "Close": [100.0, 101.0],
                "Volume": [5.0, 6.0],
            },
            index=index,
        )
        out = normalise_market_frame(raw)
        assert set(out.columns) == {"close", "volume", "open", "high", "low"}

    def test_unsorted_input_is_sorted(self):
        index = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        raw = pd.DataFrame({"Close": [3.0, 1.0, 2.0], "Volume": [1.0, 1.0, 1.0]}, index=index)
        out = normalise_market_frame(raw)
        assert out.index.is_monotonic_increasing
        assert out["close"].tolist() == [1.0, 2.0, 3.0]

    def test_missing_close_column_is_rejected(self):
        raw = pd.DataFrame({"Volume": [1.0]}, index=pd.to_datetime(["2024-01-01"]))
        with pytest.raises(SchemaViolation, match="missing required column"):
            normalise_market_frame(raw)


class TestValidation:
    def test_a_synthetic_frame_satisfies_the_contract(self, frame):
        report = validate_market_frame(frame)
        assert report.rows == 200
        assert report.is_gapless
        assert set(REQUIRED_COLUMNS) <= set(frame.columns)

    def test_naive_index_is_rejected(self, frame):
        bad = frame.copy()
        bad.index = bad.index.tz_localize(None)
        with pytest.raises(SchemaViolation, match="timezone-aware"):
            validate_market_frame(bad)

    def test_non_utc_index_is_rejected(self, frame):
        bad = frame.copy()
        bad.index = bad.index.tz_convert("America/New_York")
        with pytest.raises(SchemaViolation, match="UTC"):
            validate_market_frame(bad)

    def test_unsorted_index_is_rejected(self, frame):
        with pytest.raises(SchemaViolation, match="sorted"):
            validate_market_frame(frame.iloc[::-1])

    def test_duplicate_bars_are_rejected(self, frame):
        bad = pd.concat([frame, frame.iloc[[5]]]).sort_index()
        with pytest.raises(SchemaViolation, match="duplicate"):
            validate_market_frame(bad)

    def test_nan_close_is_rejected(self, frame):
        bad = frame.copy()
        bad.iloc[10, bad.columns.get_loc("close")] = np.nan
        with pytest.raises(SchemaViolation, match="NaN"):
            validate_market_frame(bad)

    def test_non_positive_close_is_rejected(self, frame):
        """Log price is undefined; catching it here beats a NaN 200 lines later."""
        bad = frame.copy()
        bad.iloc[10, bad.columns.get_loc("close")] = 0.0
        with pytest.raises(SchemaViolation, match="non-positive"):
            validate_market_frame(bad)

    def test_empty_frame_is_rejected(self):
        with pytest.raises(SchemaViolation, match="empty"):
            validate_market_frame(pd.DataFrame())

    def test_unnormalised_bar_labels_are_rejected(self, frame):
        bad = frame.copy()
        bad.index = bad.index + pd.Timedelta(hours=9)
        with pytest.raises(SchemaViolation, match="normalised"):
            validate_market_frame(bad)

    def test_gaps_are_reported_but_tolerated_by_default(self, frame):
        gappy = frame.drop(frame.index[[50, 51]])
        report = validate_market_frame(gappy)
        assert len(report.missing_bars) == 2
        assert not report.is_gapless

    def test_gaps_can_be_made_fatal(self, frame):
        gappy = frame.drop(frame.index[[50]])
        with pytest.raises(SchemaViolation, match="missing daily bar"):
            validate_market_frame(gappy, allow_gaps=False)


class TestGapFilling:
    def test_reindexing_forward_fills_price_and_zeroes_volume(self, frame):
        gappy = frame.drop(frame.index[[50, 51]])
        filled = reindex_to_regular_daily(gappy)

        assert validate_market_frame(filled, allow_gaps=False).is_gapless
        assert filled.loc[frame.index[50], "close"] == gappy.loc[frame.index[49], "close"]
        # Carrying stale volume forward would fabricate trading activity.
        assert filled.loc[frame.index[50], "volume"] == 0.0

    def test_backfill_is_refused_as_look_ahead(self, frame):
        with pytest.raises(ValueError, match="point-in-time safe"):
            reindex_to_regular_daily(frame, method="bfill")


class TestWindowing:
    def test_slice_is_inclusive_at_both_ends(self, frame):
        window = slice_to_window(frame, frame.index[10], frame.index[20])
        assert len(window) == 11
        assert window.index[0] == frame.index[10]
        assert window.index[-1] == frame.index[20]

    def test_slice_accepts_naive_strings_as_utc(self, frame):
        window = slice_to_window(frame, "2019-01-11", "2019-01-20")
        assert window.index[0] == pd.Timestamp("2019-01-11", tz=UTC)

    def test_bars_span_counts_both_endpoints(self, frame):
        assert bars_span(frame) == 200


class TestSnapshotProvenance:
    def test_manifest_records_everything_needed_to_identify_the_data(self, frame):
        snap = MarketSnapshot.build(
            frame, ticker="BTC-USD", provider="test", provider_version="1.2.3", normalise=False
        )
        m = snap.manifest
        assert m.ticker == "BTC-USD"
        assert m.provider == "test"
        assert m.provider_version == "1.2.3"
        assert m.rows == 200
        assert m.timezone == "UTC"
        assert m.frequency == "D"
        assert len(m.sha256) == 64
        assert m.start.startswith("2019-01-01")
        assert pd.Timestamp(m.retrieved_at).tz is not None

    def test_digest_is_deterministic_across_calls(self, frame):
        assert frame_digest(frame) == frame_digest(frame.copy())

    def test_digest_is_independent_of_column_and_row_order(self, frame):
        shuffled = frame[["volume", "close"]].sort_index(ascending=False).sort_index()
        assert frame_digest(shuffled) == frame_digest(frame)

    def test_digest_changes_when_a_single_price_changes(self, frame):
        tampered = frame.copy()
        tampered.iloc[100, tampered.columns.get_loc("close")] *= 1.000001
        assert frame_digest(tampered) != frame_digest(frame)

    def test_roundtrip_preserves_data_and_verifies(self, frame, tmp_path):
        original = MarketSnapshot.build(frame, ticker="BTC-USD", provider="test", normalise=False)
        original.save(tmp_path / "snap")

        loaded = MarketSnapshot.load(tmp_path / "snap")

        assert loaded.manifest.sha256 == original.manifest.sha256
        pd.testing.assert_frame_equal(
            loaded.frame, original.frame, check_exact=False, rtol=1e-11
        )

    def test_tampered_snapshot_fails_verification(self, frame, tmp_path):
        MarketSnapshot.build(frame, ticker="BTC-USD", provider="test", normalise=False).save(
            tmp_path / "snap"
        )

        data_path = tmp_path / "snap" / "data.csv"
        lines = data_path.read_text(encoding="utf-8").splitlines()
        lines[5] = lines[5].replace(lines[5].split(",")[1], "99999")
        data_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(SnapshotIntegrityError, match="does not match its manifest"):
            MarketSnapshot.load(tmp_path / "snap")

    def test_missing_snapshot_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MarketSnapshot.load(tmp_path / "nope")

    def test_manifest_json_roundtrips(self, frame):
        snap = MarketSnapshot.build(frame, ticker="BTC-USD", provider="test", normalise=False)
        restored = SnapshotManifest.from_dict(json.loads(snap.manifest.to_json()))
        assert restored == snap.manifest

    def test_manifest_tolerates_unknown_future_fields(self):
        payload = {
            "ticker": "BTC-USD",
            "provider": "test",
            "retrieved_at": "2024-01-01T00:00:00+00:00",
            "start": "2019-01-01T00:00:00+00:00",
            "end": "2019-07-19T00:00:00+00:00",
            "rows": 200,
            "columns": ["close", "volume"],
            "timezone": "UTC",
            "frequency": "D",
            "sha256": "0" * 64,
            "some_field_added_later": True,
        }
        assert SnapshotManifest.from_dict(payload).ticker == "BTC-USD"

    def test_canonical_bytes_are_stable_utf8_with_lf_endings(self, frame):
        raw = canonical_bytes(frame)
        assert b"\r\n" not in raw, "CRLF would make the hash platform-dependent"
        assert raw.decode("utf-8").startswith("date,close,volume")


class TestProviders:
    def test_in_memory_provider_serves_windows(self, frame):
        provider = InMemoryProvider(frame, ticker="BTC-USD")
        window = provider.fetch("BTC-USD", start=frame.index[10], end=frame.index[19])
        assert len(window) == 10

    def test_in_memory_provider_returns_a_copy(self, frame):
        provider = InMemoryProvider(frame, ticker="BTC-USD")
        fetched = provider.fetch("BTC-USD")
        fetched.iloc[0, 0] = -1.0
        assert provider.frame.iloc[0, 0] != -1.0

    def test_snapshot_provider_is_offline_and_verified(self, frame, tmp_path):
        MarketSnapshot.build(frame, ticker="BTC-USD", provider="test", normalise=False).save(
            tmp_path / "snap"
        )
        provider = SnapshotProvider(tmp_path / "snap")
        assert len(provider.fetch("BTC-USD")) == 200

    def test_snapshot_provider_refuses_the_wrong_ticker(self, frame, tmp_path):
        MarketSnapshot.build(frame, ticker="BTC-USD", provider="test", normalise=False).save(
            tmp_path / "snap"
        )
        with pytest.raises(ValueError, match="ETH-USD"):
            SnapshotProvider(tmp_path / "snap").fetch("ETH-USD")


class TestImportSafety:
    def test_core_imports_without_optional_heavy_dependencies(self):
        """The unit suite must run without prophet/xgboost/arch/yfinance."""
        import importlib
        import sys

        for module in ["btc_forecaster", "btc_forecaster.data", "btc_forecaster.timebase"]:
            importlib.import_module(module)

        forbidden = {"prophet", "xgboost", "arch", "yfinance", "matplotlib.pyplot"}
        loaded = forbidden & set(sys.modules)
        assert not loaded, f"importing the core pulled in heavy dependencies: {loaded}"
