"""Preserved research evidence must stay exactly as recorded.

These tests exist to catch the natural failure mode of a forecasting project:
tuning until the number looks acceptable, then reporting the last number. Each
retune is individually defensible, so nothing catches the drift. A frozen
reference that a test reads back does.
"""

from __future__ import annotations

import json

import pytest

from btc_forecaster.evidence import (
    LEAKAGE_CORRECTED,
    LEAKAGE_CORRECTED_REFERENCE,
    LEGACY_INVALID,
    LEGACY_INVALID_RESULT,
    PRESERVED,
    EvidenceTampering,
    by_label,
    summary_table,
    verify,
    verify_all,
)


class TestArchivesAreIntact:
    def test_both_preserved_runs_still_exist(self):
        for result in PRESERVED:
            assert result.path.is_dir(), f"{result.label} archive is missing"
            assert (result.path / "README.md").exists()

    def test_every_recorded_number_still_matches_the_archive(self):
        verified = verify_all()
        assert set(verified) == {LEGACY_INVALID_RESULT, LEAKAGE_CORRECTED_REFERENCE}

    def test_the_invalid_legacy_headline_is_preserved_verbatim(self):
        """0.6897 / p=0.031 must remain readable as the thing that was wrong."""
        observed = verify(LEGACY_INVALID)
        assert observed["directional_accuracy"] == pytest.approx(0.6896551724137931)
        assert observed["p_value"] == pytest.approx(0.03071417286992073)

    def test_the_corrected_negative_result_is_preserved_verbatim(self):
        observed = verify(LEAKAGE_CORRECTED)
        assert observed["hybrid_directional_accuracy"] == pytest.approx(0.2444444444, abs=1e-9)
        assert observed["hybrid_mae_skill_vs_random_walk"] < -1.6
        assert observed["hybrid_interval_coverage"] < 0.3

    def test_the_corrected_hybrid_is_worse_than_the_random_walk(self):
        """The load-bearing fact. If this ever passes, something was overwritten."""
        observed = verify(LEAKAGE_CORRECTED)
        assert observed["hybrid_mae"] > observed["random_walk_mae"] * 2.5


class TestTamperDetection:
    def test_a_drifted_number_is_detected_and_named(self, tmp_path, monkeypatch):
        import btc_forecaster.evidence as evidence

        fake = tmp_path / "runs" / LEAKAGE_CORRECTED.run_dir
        fake.mkdir(parents=True)

        original = json.loads(
            (LEAKAGE_CORRECTED.path / "forecast_summary.json").read_text(encoding="utf-8")
        )
        for row in original["model_comparison"]:
            if row["model"] == "prophet_xgb_hybrid":
                row["directional_accuracy"] = 0.71  # the tempting edit
        (fake / "forecast_summary.json").write_text(json.dumps(original), encoding="utf-8")

        monkeypatch.setattr(evidence, "RESEARCH_RUNS", tmp_path / "runs")

        with pytest.raises(EvidenceTampering, match="hybrid_directional_accuracy changed"):
            verify(LEAKAGE_CORRECTED)

    def test_a_deleted_archive_is_detected(self, tmp_path, monkeypatch):
        import btc_forecaster.evidence as evidence

        monkeypatch.setattr(evidence, "RESEARCH_RUNS", tmp_path / "empty")
        with pytest.raises(EvidenceTampering, match="must not be deleted"):
            verify(LEGACY_INVALID)


class TestLabelling:
    def test_labels_resolve(self):
        assert by_label(LEGACY_INVALID_RESULT) is LEGACY_INVALID
        assert by_label(LEAKAGE_CORRECTED_REFERENCE) is LEAKAGE_CORRECTED

    def test_unknown_label_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="LEGACY_INVALID_RESULT"):
            by_label("nope")

    def test_the_legacy_result_is_labelled_invalid(self):
        assert "INVALID" in LEGACY_INVALID.status
        assert "do not cite" in LEGACY_INVALID.status

    def test_the_corrected_result_is_labelled_valid_and_negative(self):
        assert "VALID" in LEAKAGE_CORRECTED.status
        assert "negative" in LEAKAGE_CORRECTED.status

    def test_each_result_records_why_and_what_not_to_do(self):
        for result in PRESERVED:
            assert len(result.why) > 100, f"{result.label} needs a real explanation"
            assert result.do_not, f"{result.label} must say what is forbidden"

    def test_the_retune_prohibition_names_the_challenger_instead(self):
        """A2.4: retuning is allowed, but as a new named model."""
        assert any("XGBOOST_CAUSAL_RETUNED" in item for item in LEAKAGE_CORRECTED.do_not)

    def test_results_are_serialisable(self):
        for result in PRESERVED:
            json.dumps(result.to_dict())

    def test_summary_table_states_both_results(self):
        text = summary_table()
        assert LEGACY_INVALID_RESULT in text
        assert LEAKAGE_CORRECTED_REFERENCE in text
        assert "0.689655" in text
