"""B5 Gate 1 output: canonical, verifiable, and driven by the committed plan.

* Two audits of the same store at the same instant write the same bytes, and
  none of them carries a path, a timing or a newline.
* `verify` recomputes every hash and names a touched or a missing file, and an
  edited preregistration fails it.
* The command applies the policy in the preregistration file it is given,
  checked against that file's own hash -- not a default in the code.

No network.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.b5.__main__ import EXIT_INTEGRITY, EXIT_OK, main
from market_intelligence.b5.audit import audit_corpus, open_read_only
from market_intelligence.b5.contracts import SufficiencyPolicy, default_preregistration
from market_intelligence.b5.runner import CANONICAL_FILES, verify_run, write_run
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.models import Direction, EventSignal, EventType, ExtractionMethod, SignalCategory
from market_intelligence.operations import RunManifest, RunStatus
from market_intelligence.storage.store import IntelligenceStore

BASE = datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc)
AS_OF = "2026-09-11T00:00:00+00:00"


def small_store(path: Path, *, confidence: float = 0.35) -> Path:
    """A few events over two days -- the shape of the corpus actually collected."""
    store = IntelligenceStore(path)
    try:
        documents, events = [], []
        for index in range(4):
            published = BASE + timedelta(hours=5 * index)
            doc = fixture_document(
                url=f"https://pub-{index % 3}.example.com/{index}",
                title=f"Announcement {index}",
                retrieved_at=published + timedelta(hours=18),
                publisher=f"pub-{index % 3}",
                published_at=published,
            )
            documents.append(doc)
            events.append(
                EventSignal(
                    event_id=f"event-{index}",
                    event_time=published,
                    available_time=doc.available_at,
                    source_ids=(doc.document_id,),
                    category=SignalCategory.WEB_EVENT,
                    entity="SEC",
                    event_type=EventType.REGULATION,
                    sentiment=0.0,
                    btc_relevance=0.55,
                    novelty=0.5,
                    confidence=confidence,
                    expected_horizon_hours=24,
                    summary=f"event {index}",
                    direction=Direction.UNKNOWN,
                    extractor_version="rules-v1",
                    extraction_method=ExtractionMethod.RULE_BASED,
                )
            )
        store.put_documents(documents)
        store.put_signals(events)
        store.put_run(
            RunManifest(
                run_id="run-1",
                started_at=BASE + timedelta(hours=18),
                finished_at=BASE + timedelta(hours=18, minutes=5),
                configuration_fingerprint="fixture",
                providers_attempted=1,
                queries_attempted=1,
                documents_accepted=4,
                documents_rejected=0,
                events_accepted=4,
                events_rejected=0,
                quality_summary={},
                watermark_changes=0,
                software_source_sha="fixture",
                status=RunStatus.DEGRADED,
                provider_ids=("fixture",),
            )
        )
    finally:
        store.close()
    return path


def written(store: Path, out: Path) -> dict:
    connection = open_read_only(store)
    try:
        result = audit_corpus(connection, as_of=datetime.fromisoformat(AS_OF))
    finally:
        connection.close()
    return write_run(result, default_preregistration(), out, input_label="fixture", input_sha256="0" * 64)


class TestCanonicalOutput:
    def test_two_writes_are_byte_identical(self, tmp_path: Path) -> None:
        store = small_store(tmp_path / "s.duckdb")
        first = written(store, tmp_path / "a")
        second = written(store, tmp_path / "b")
        assert first == second
        for name in (*CANONICAL_FILES, "manifest.json", "README.md"):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name

    def test_canonical_files_carry_no_path_timing_or_newline(self, tmp_path: Path) -> None:
        store = small_store(tmp_path / "s.duckdb")
        written(store, tmp_path / "out")
        for name in (*CANONICAL_FILES, "manifest.json"):
            data = (tmp_path / "out" / name).read_bytes()
            assert b"\n" not in data and b"\r" not in data, name
            text = data.decode("utf-8")
            assert str(tmp_path) not in text and tmp_path.name not in text, name
            for token in ("generated_at", "seconds", "hostname"):
                assert token not in text, (name, token)

    def test_the_decision_is_recorded_and_nothing_trades(self, tmp_path: Path) -> None:
        manifest = written(small_store(tmp_path / "s.duckdb"), tmp_path / "out")
        assert manifest["decision"] == "INTELLIGENCE_CORPUS_INSUFFICIENT" and not manifest["gate1_passed"]
        assert manifest["produces_trading_signal"] is False and manifest["live_trading_enabled"] is False


class TestVerification:
    def test_an_untouched_run_verifies(self, tmp_path: Path) -> None:
        written(small_store(tmp_path / "s.duckdb"), tmp_path / "out")
        assert verify_run(tmp_path / "out")["verified"]

    def test_a_touched_file_is_named(self, tmp_path: Path) -> None:
        written(small_store(tmp_path / "s.duckdb"), tmp_path / "out")
        path = tmp_path / "out" / "corpus_audit.json"
        path.write_bytes(path.read_bytes().replace(b'"documents":4', b'"documents":40'))
        result = verify_run(tmp_path / "out")
        assert not result["verified"] and result["mismatched_files"] == ["corpus_audit.json"]

    def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        written(small_store(tmp_path / "s.duckdb"), tmp_path / "out")
        (tmp_path / "out" / "event_catalog.json").unlink()
        result = verify_run(tmp_path / "out")
        assert result["absent_files"] == ["event_catalog.json"] and not result["verified"]

    def test_an_edited_preregistration_fails_even_with_its_manifest_hash_updated(self, tmp_path: Path) -> None:
        """The file hash can be forged in the manifest; the plan's own content hash cannot."""
        written(small_store(tmp_path / "s.duckdb"), tmp_path / "out")
        path = tmp_path / "out" / "preregistration.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["plan"]["policy"]["minimum_confidence"] = 0.1
        path.write_text(json.dumps(record), encoding="utf-8")
        manifest_path = tmp_path / "out" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["preregistration.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = verify_run(tmp_path / "out")
        assert "preregistration.json" not in result["mismatched_files"]
        assert not result["preregistration_intact"] and not result["verified"]


class TestTheCommand:
    def test_audit_then_verify(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        store = small_store(tmp_path / "s.duckdb")
        plan = default_preregistration().write(tmp_path / "plan.json")
        code = main(["audit", "--db", str(store), "--as-of", AS_OF, "--input-label", "fixture", "--output", str(tmp_path / "out"), "--preregistration", str(plan)])
        assert code == EXIT_OK
        out = capsys.readouterr().out
        assert "INTELLIGENCE_CORPUS_INSUFFICIENT" in out and "no trading signal" in out
        assert main(["verify", str(tmp_path / "out")]) == EXIT_OK

    def test_the_policy_comes_from_the_plan_file(self, tmp_path: Path) -> None:
        store = small_store(tmp_path / "s.duckdb")
        default = default_preregistration().write(tmp_path / "default.json")
        lax = default_preregistration(SufficiencyPolicy(minimum_confidence=0.3)).write(tmp_path / "lax.json")
        for plan, out in ((default, "d"), (lax, "l")):
            assert main(["audit", "--db", str(store), "--as-of", AS_OF, "--input-label", "fixture", "--output", str(tmp_path / out), "--preregistration", str(plan)]) == EXIT_OK
        funnels = [json.loads((tmp_path / out / "corpus_audit.json").read_text(encoding="utf-8"))["funnel"]["5_quality"] for out in ("d", "l")]
        assert funnels == [0, 4]

    def test_an_edited_plan_is_refused(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        store = small_store(tmp_path / "s.duckdb")
        plan = default_preregistration().write(tmp_path / "plan.json")
        record = json.loads(plan.read_text(encoding="utf-8"))
        record["plan"]["policy"]["minimum_confidence"] = 0.1
        plan.write_text(json.dumps(record), encoding="utf-8")
        code = main(["audit", "--db", str(store), "--as-of", AS_OF, "--input-label", "fixture", "--output", str(tmp_path / "out"), "--preregistration", str(plan)])
        assert code == EXIT_INTEGRITY and "edited after it was frozen" in capsys.readouterr().err
        assert not (tmp_path / "out" / "manifest.json").exists()

    def test_an_audit_instant_without_a_timezone_is_refused(self, tmp_path: Path) -> None:
        store = small_store(tmp_path / "s.duckdb")
        code = main(["audit", "--db", str(store), "--as-of", "2026-09-11T00:00:00", "--input-label", "x", "--output", str(tmp_path / "out"), "--preregistration", str(default_preregistration().write(tmp_path / "p.json"))])
        assert code == EXIT_INTEGRITY

    def test_a_missing_store_is_refused(self, tmp_path: Path) -> None:
        code = main(["audit", "--db", str(tmp_path / "none.duckdb"), "--as-of", AS_OF, "--input-label", "x", "--output", str(tmp_path / "out"), "--preregistration", str(default_preregistration().write(tmp_path / "p.json"))])
        assert code == EXIT_INTEGRITY

    def test_verifying_nothing_is_an_integrity_failure(self, tmp_path: Path) -> None:
        assert main(["verify", str(tmp_path / "absent")]) == EXIT_INTEGRITY
