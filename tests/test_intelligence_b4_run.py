"""B4.48 – B4.50 — run artifacts, telemetry, and manifest integrity.

The manifest is the completion marker for a B4 run, exactly as it is for a B3.1
dataset. These tests pin the two properties that gives it: a run without a
manifest is scratch, and a manifest that does not match its artifacts is a
detected corruption rather than a silent one.

No network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from market_intelligence.b4.contracts import B4DataError, EvidenceTier
from market_intelligence.b4.runner import (
    MANIFEST_FILENAME,
    RunRecorder,
    evidence_tier_counts,
    make_run_id,
    read_manifest,
    software_versions,
    verify_run,
)


def finish(recorder: RunRecorder, **overrides: object) -> object:
    payload: dict[str, object] = {
        "source_git_sha": "abc123",
        "intelligence_dataset_id": None,
        "intelligence_row_count": 0,
        "target_dataset_fingerprint": "f" * 64,
        "target_contract_version": "b4-targets-v1",
        "preregistration_hash": "p" * 64,
        "study_spec_hashes": {"study": "s" * 64},
        "market_series_fingerprints": {"btc": "b" * 64},
        "origin_count": 100,
        "event_counts": {"regulation": 0},
        "evidence_tier_counts": {"PIT_VALIDATED": 0, "RETROSPECTIVE_ONLY": 0, "UNKNOWN": 1},
        "market_bar_count": 4000,
    }
    payload.update(overrides)
    return recorder.finish(**payload)  # type: ignore[arg-type]


class TestRunRecorder:
    def test_artifacts_are_hashed_as_they_are_written(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        recorder.write_json("results.json", {"a": 1})
        assert "results.json" in recorder.result_hashes
        assert (tmp_path / "run" / "results.json").exists()
        assert not list((tmp_path / "run").glob("*.tmp"))

    def test_stage_timings_and_peak_memory_are_recorded(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        with recorder.stage("work"):
            _ = [object() for _ in range(20_000)]
        assert recorder.stage_seconds["work"] >= 0.0
        assert recorder.peak_python_mb is not None and recorder.peak_python_mb > 0.0

    def test_a_stage_records_its_timing_even_when_it_raises(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        with pytest.raises(ValueError), recorder.stage("failing"):
            raise ValueError("boom")
        assert "failing" in recorder.stage_seconds

    def test_a_completed_run_directory_is_never_reopened(self, tmp_path: Path) -> None:
        """B4.49. Negative results that can be quietly replaced are worth little."""
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        recorder.write_json("results.json", {"a": 1})
        finish(recorder)
        with pytest.raises(B4DataError, match="never overwritten"):
            RunRecorder(tmp_path / "run", run_id="run-1")

    def test_the_manifest_records_every_artifact_hash(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        recorder.write_json("results.json", {"a": 1})
        recorder.write_json("registry.json", {"b": 2})
        manifest = finish(recorder)
        assert set(manifest.result_hashes) == {"results.json", "registry.json"}  # type: ignore[attr-defined]

    def test_datetimes_and_models_serialise_in_artifacts(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        path = recorder.write_json("t.json", {"when": datetime(2026, 1, 1, tzinfo=timezone.utc)})
        assert json.loads(path.read_text(encoding="utf-8"))["when"].startswith("2026-01-01")

    def test_an_unserialisable_artifact_is_an_error_not_a_silent_string(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        with pytest.raises(TypeError, match="cannot serialise"):
            recorder.write_json("t.json", {"bad": object()})


class TestManifestIntegrity:
    def test_a_run_without_a_manifest_is_scratch_not_results(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        recorder.write_json("results.json", {"a": 1})
        with pytest.raises(B4DataError, match="did not complete"):
            verify_run(tmp_path / "run")

    def test_a_complete_run_verifies(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        recorder.write_json("results.json", {"a": 1})
        finish(recorder)
        assert verify_run(tmp_path / "run").run_id == "run-1"

    def test_an_edited_artifact_is_detected(self, tmp_path: Path) -> None:
        """Re-hashing the files is the only version of this check worth having."""
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        recorder.write_json("results.json", {"a": 1})
        finish(recorder)
        (tmp_path / "run" / "results.json").write_text('{"a": 999}', encoding="utf-8")
        with pytest.raises(B4DataError, match="does not match the hash"):
            verify_run(tmp_path / "run")

    def test_a_missing_artifact_is_detected(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        recorder.write_json("results.json", {"a": 1})
        finish(recorder)
        (tmp_path / "run" / "results.json").unlink()
        with pytest.raises(B4DataError, match="which is missing"):
            verify_run(tmp_path / "run")

    def test_an_edited_manifest_is_detected(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path / "run", run_id="run-1")
        recorder.write_json("results.json", {"a": 1})
        finish(recorder)
        path = tmp_path / "run" / MANIFEST_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["origin_count"] = 999_999
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(B4DataError, match="manifest hash does not match"):
            read_manifest(path)


class TestRunIdentity:
    def test_the_same_inputs_produce_the_same_run_id(self) -> None:
        assert make_run_id("a", "b") == make_run_id("a", "b")
        assert make_run_id("a", "b") != make_run_id("a", "c")

    def test_evidence_tiers_are_counted_including_the_unknowns(self) -> None:
        """The headline number: a run where nothing is PIT_VALIDATED validated nothing."""
        counts = evidence_tier_counts(
            [EvidenceTier.PIT_VALIDATED, None, None, EvidenceTier.RETROSPECTIVE_ONLY]
        )
        assert counts == {"PIT_VALIDATED": 1, "RETROSPECTIVE_ONLY": 1, "UNKNOWN": 2}

    def test_software_versions_are_recorded_for_reproduction(self) -> None:
        versions = software_versions()
        assert "python" in versions
        assert "pydantic" in versions
