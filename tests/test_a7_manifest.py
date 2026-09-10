"""Immutable inputs and canonical outputs.

A result is reproducible only if two things hold: the input is the recorded
input, byte for byte, and the output is written the same way every time. These
tests pin both -- and pin that a changed input is refused as a reproduction
rather than quietly accepted as "the same data, roughly".
"""

from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.data.snapshot import MarketSnapshot, canonical_bytes
from btc_forecaster.research.snapshot import EXIT_CHANGED, EXIT_INTEGRITY, EXIT_MATCH, main
from btc_forecaster.research.walk_forward.manifest import (
    DATA_LICENCE,
    INPUT_CHANGED,
    INPUT_MATCHES_RECORD,
    NO_RECORDED_INPUT,
    InputChangedError,
    canonical_csv,
    canonical_json,
    deterministic_gzip,
    input_manifest,
    input_status,
    load_snapshot,
    require_recorded_input,
    result_digest,
)
from btc_forecaster.research.walk_forward.targets import TARGET_DEFINITION
from btc_forecaster.testing import synthetic_market_frame


@pytest.fixture
def saved(tmp_path):
    frame = synthetic_market_frame(periods=300, seed=13)
    snapshot = MarketSnapshot.build(
        frame,
        ticker="SYN-USD",
        provider="synthetic",
        retrieved_at=pd.Timestamp("2026-01-02", tz="UTC"),
    )
    return snapshot.save(tmp_path / "snap"), snapshot


def describe(snapshot: MarketSnapshot, file_bytes: bytes | None = None, version: str = "v1"):
    return input_manifest(
        snapshot,
        file_bytes=file_bytes,
        preprocessing_version=version,
        feature_specs=("lag_ret_1",),
        target=TARGET_DEFINITION,
    )


class TestCanonicalJson:
    def test_it_is_sorted_and_compact(self) -> None:
        assert canonical_json({"b": 1, "a": {"d": 2, "c": 3}}) == '{"a":{"c":3,"d":2},"b":1}'

    def test_float_text_depends_only_on_the_value(self) -> None:
        """0.1 + 0.2 is 0.30000000000000004 in repr; at twelve digits it is 0.3."""
        assert canonical_json({"x": 0.1 + 0.2}) == '{"x":0.3}'

    def test_non_finite_values_become_null(self) -> None:
        assert canonical_json([float("nan"), float("inf"), -float("inf")]) == "[null,null,null]"

    def test_numpy_and_timestamps_are_serialised(self) -> None:
        text = canonical_json(
            {"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True), "t": pd.Timestamp("2026-01-02", tz="UTC")}
        )
        assert json.loads(text) == {"b": True, "f": 0.5, "i": 3, "t": "2026-01-02T00:00:00+00:00"}

    def test_an_unknown_type_is_refused_rather_than_stringified(self) -> None:
        with pytest.raises(TypeError, match="canonically"):
            canonical_json({"s": {1, 2}})

    def test_insertion_order_is_irrelevant(self) -> None:
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


class TestCanonicalCsvAndGzip:
    def test_equal_content_gives_equal_bytes(self) -> None:
        frame = pd.DataFrame({"model": ["a", "b"], "value": [1 / 3, 2.0]})
        assert canonical_csv(frame) == canonical_csv(frame.copy())

    def test_line_endings_and_float_text_are_fixed(self) -> None:
        text = canonical_csv(pd.DataFrame({"v": [1 / 3]})).decode("utf-8")
        assert text == "v\n0.333333333333\n"

    def test_gzip_carries_no_timestamp(self) -> None:
        data = b"x" * 1000
        assert deterministic_gzip(data) == deterministic_gzip(data)
        assert gzip.decompress(deterministic_gzip(data)) == data


class TestTheResultDigest:
    def test_part_order_is_irrelevant(self) -> None:
        assert result_digest({"a": b"1", "b": b"2"}) == result_digest({"b": b"2", "a": b"1"})

    def test_any_content_change_moves_it(self) -> None:
        assert result_digest({"a": b"1", "b": b"2"}) != result_digest({"a": b"1", "b": b"3"})

    def test_a_renamed_part_moves_it(self) -> None:
        assert result_digest({"a": b"1"}) != result_digest({"b": b"1"})


class TestTheInputManifest:
    def test_it_records_the_verified_hashes(self, saved) -> None:
        directory, snapshot = saved
        loaded, file_bytes = load_snapshot(directory)
        manifest = describe(loaded, file_bytes)
        assert manifest.frame_sha256 == snapshot.manifest.sha256
        assert manifest.file_sha256 == hashlib.sha256((directory / "data.csv").read_bytes()).hexdigest()

    def test_it_records_where_the_data_came_from(self, saved) -> None:
        payload = describe(saved[1]).as_dict()
        assert payload["source"]["ticker"] == "SYN-USD"
        assert payload["source"]["retrieved_at"].startswith("2026-01-02")
        assert payload["rows"] == 300 and "close" in payload["columns"]
        assert payload["range"]["start"] < payload["range"]["end"]

    def test_it_does_not_pretend_the_provider_is_immutable(self, saved) -> None:
        payload = describe(saved[1]).as_dict()
        assert payload["provider_revises_history"] is True
        assert payload["licence"] == DATA_LICENCE and "not committed" in DATA_LICENCE

    def test_its_digest_is_deterministic_and_versioned(self, saved) -> None:
        assert describe(saved[1]).digest() == describe(saved[1]).digest()
        assert describe(saved[1], version="v2").digest() != describe(saved[1]).digest()

    def test_a_frame_that_no_longer_matches_its_manifest_is_refused(self, saved) -> None:
        snapshot = saved[1]
        altered = snapshot.frame.copy()
        altered.iloc[10, altered.columns.get_loc("close")] *= 1.01
        with pytest.raises(InputChangedError, match="its own manifest"):
            describe(MarketSnapshot(frame=altered, manifest=snapshot.manifest))


class TestAChangedInputIsNotAReproduction:
    def test_the_three_states(self) -> None:
        assert input_status(None, "abc") == NO_RECORDED_INPUT
        assert input_status("abc", "abc") == INPUT_MATCHES_RECORD
        assert input_status("abc", "abd") == INPUT_CHANGED

    def test_proceeding_as_a_reproduction_is_refused(self) -> None:
        with pytest.raises(InputChangedError, match="different experiment"):
            require_recorded_input("abc", "abd")


class TestTheCommand:
    def test_fingerprint_prints_the_manifest_deterministically(self, saved, capsys) -> None:
        directory, snapshot = saved
        assert main(["fingerprint", str(directory)]) == EXIT_MATCH
        first = capsys.readouterr().out
        assert main(["fingerprint", str(directory)]) == EXIT_MATCH
        assert capsys.readouterr().out == first
        assert json.loads(first)["frame_sha256"] == snapshot.manifest.sha256

    def test_verify_accepts_the_recorded_input(self, saved, capsys) -> None:
        directory, snapshot = saved
        assert main(["verify", str(directory), "--expect", snapshot.manifest.sha256]) == EXIT_MATCH
        assert INPUT_MATCHES_RECORD in capsys.readouterr().out

    def test_verify_refuses_a_different_input(self, saved, capsys) -> None:
        directory, _ = saved
        assert main(["verify", str(directory), "--expect", "0" * 64]) == EXIT_CHANGED
        assert "not a reproduction" in capsys.readouterr().err

    def test_a_hand_edited_snapshot_is_an_integrity_failure(self, saved, capsys) -> None:
        directory, snapshot = saved
        edited = snapshot.frame.copy()
        edited.iloc[5, edited.columns.get_loc("close")] *= 1.5
        (directory / "data.csv").write_bytes(canonical_bytes(edited))
        assert main(["fingerprint", str(directory)]) == EXIT_INTEGRITY
        assert "INTEGRITY FAILURE" in capsys.readouterr().err

    def test_a_missing_snapshot_is_reported(self, tmp_path, capsys) -> None:
        assert main(["fingerprint", str(tmp_path / "absent")]) == EXIT_INTEGRITY
        assert "no snapshot" in capsys.readouterr().err
