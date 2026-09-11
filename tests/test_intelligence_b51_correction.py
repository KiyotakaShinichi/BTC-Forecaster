"""B5.1 — the rules-v1 CFTC misclassification, corrected append-only.

B5's forensics found one rules-v1 event typed MONETARY_POLICY because "interest
rate" occurs, as a substring, in the title of a CFTC derivatives-clearing rule.
rules-v2 types the same document REGULATION. The correction is the existing
append-only ledger's, and this file pins that it behaves as that ledger always
has:

* the file names exactly one event, literally, and cannot widen;
* the original rules-v1 observation stays exactly as written, and research
  stops counting it;
* the corrected classification is a separate rules-v2 event -- the correction
  never rewrites a type -- and the two are told apart by extractor version;
* applying twice records once, a dry run records nothing, and the first
  correction file is untouched.

No network. The collected store is never opened here.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.b5.audit import Exclusion, audit_corpus, open_read_only
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.corrections import REDISCOVERY_DUPLICATE_PRE_FIX, CorrectionStatus, load_corrections
from market_intelligence.extractors import EvidenceRuleExtractor, RuleBasedExtractor
from market_intelligence.models import Direction, EventSignal, EventType, ExtractionMethod, SignalCategory
from market_intelligence.storage import IntelligenceStore

ROOT = Path(__file__).resolve().parents[1]
CFTC_FILE = ROOT / "corrections" / "2026-09-11-rules-v1-cftc-misclassification.json"
FIRST_FILE = ROOT / "corrections" / "2026-09-03-rediscovery-duplicates.json"
REASON = "MISCLASSIFIED_EVENT_TYPE_RULES_V1"
ORIGINAL = "6579615b1873c979e48cd506b911d5338ddcb06cdccaeddd2333904ae9402cda"
RULES_V2_EVENT = "2262b65a8dc36182c9127d2a470b4642bf71d89508e73be553727a0dc4191917"
TITLE = (
    "CFTC Issues Final Rule to Modify Clearing Requirement for Canadian Dollar- and "
    "Mexican Peso-Denominated Interest Rate Swaps"
)
PUBLISHED = datetime(2026, 9, 2, 16, 4, 52, tzinfo=timezone.utc)
SEEN = datetime(2026, 9, 3, 9, 15, 13, tzinfo=timezone.utc)
AFTER = datetime(2026, 9, 12, tzinfo=timezone.utc)


def raw() -> dict:
    return json.loads(CFTC_FILE.read_text(encoding="utf-8"))


class TestTheFile:
    def test_it_names_exactly_one_event_literally(self) -> None:
        entries = raw()["corrections"]
        assert entries == [{"event_id": ORIGINAL}], "a correction names ids, never a rule"

    def test_it_is_an_invalidation_for_the_misclassification(self) -> None:
        (correction,) = load_corrections(raw())
        assert correction.status is CorrectionStatus.INVALIDATED
        assert correction.reason == REASON
        assert correction.invalidated_by_version.startswith("be5886b")
        assert correction.invalidated_at == datetime(2026, 9, 11, tzinfo=timezone.utc)

    def test_it_names_the_defect_and_the_corrected_classification(self) -> None:
        (correction,) = load_corrections(raw())
        assert "interest rate" in correction.source_bug and "Interest Rate Swaps" in correction.source_bug
        assert RULES_V2_EVENT in correction.notes and "REGULATION" in correction.notes
        assert "never from any market outcome" in correction.notes

    def test_its_id_is_stable_across_machines(self) -> None:
        assert [c.correction_id for c in load_corrections(raw())] == [c.correction_id for c in load_corrections(raw())]

    def test_the_first_correction_file_still_names_its_three_events(self) -> None:
        first = load_corrections(json.loads(FIRST_FILE.read_text(encoding="utf-8")))
        assert len(first) == 3 and {c.reason for c in first} == {REDISCOVERY_DUPLICATE_PRE_FIX}
        assert ORIGINAL not in {c.event_id for c in first}


def cftc_document():
    return fixture_document(
        url="https://www.cftc.gov/PressRoom/PressReleases/fixture",
        title=TITLE,
        retrieved_at=SEEN,
        publisher="Commodity Futures Trading Commission",
        published_at=PUBLISHED,
        primary_source=True,
        official_source=True,
    )


class TestThePremise:
    def test_rules_v1_types_the_document_monetary_policy_and_rules_v2_regulation(self) -> None:
        document = cftc_document()
        (v1,) = RuleBasedExtractor({"CFTC": ("Commodity Futures Trading Commission",)}).extract([document])
        (v2,) = EvidenceRuleExtractor(
            {"CFTC": ("Commodity Futures Trading Commission",)}, {"CFTC": (EventType.REGULATION,)}
        ).extract([document])
        assert v1.event_type is EventType.MONETARY_POLICY and v1.extractor_version == "rules-v1"
        assert v2.event_type is EventType.REGULATION and v2.extractor_version == "rules-v2"
        assert v1.event_id != v2.event_id


def a_rules_v1_event(event_id: str, document_id: str, available: datetime) -> EventSignal:
    """The rules-v1 observation as written: its fixed scores, its wrong type."""
    return EventSignal(
        event_id=event_id,
        event_time=PUBLISHED,
        available_time=available,
        source_ids=(document_id,),
        category=SignalCategory.WEB_EVENT,
        entity="CFTC",
        event_type=EventType.MONETARY_POLICY,
        sentiment=0.0,
        btc_relevance=0.55,
        novelty=0.5,
        confidence=0.35,
        expected_horizon_hours=24,
        summary=f"Rule-based extraction: {TITLE}",
        direction=Direction.UNKNOWN,
        extractor_version="rules-v1",
        extraction_method=ExtractionMethod.RULE_BASED,
    )


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """The original event, a later unrelated rules-v1 MONETARY_POLICY event, and the rules-v2 re-extraction."""
    path = tmp_path / "corpus.duckdb"
    document = cftc_document()
    other = fixture_document(
        url="https://www.federalreserve.gov/fixture",
        title="Federal Reserve issues FOMC statement",
        retrieved_at=SEEN + timedelta(days=1),
        publisher="Board of Governors of the Federal Reserve System",
        published_at=PUBLISHED + timedelta(days=1),
        primary_source=True,
        official_source=True,
    )
    rules_v2 = EvidenceRuleExtractor(
        {"CFTC": ("Commodity Futures Trading Commission",)}, {"CFTC": (EventType.REGULATION,)}
    ).extract([document])
    store = IntelligenceStore(path)
    try:
        store.put_documents([document, other])
        store.put_signals(
            [
                a_rules_v1_event(ORIGINAL, document.document_id, SEEN),
                a_rules_v1_event("unrelated-" + "0" * 54, other.document_id, SEEN + timedelta(days=1)),
                *rules_v2,
            ]
        )
    finally:
        store.close()
    return path


def correct(path: Path, *extra: str) -> int:
    from market_intelligence.cli import main

    return main(["--db", str(path), "corpus-correct", "--file", str(CFTC_FILE), *extra])


def read(path: Path):
    store = IntelligenceStore(path)
    try:
        raw_view = {event.event_id: event for event in store.signals_as_of(AFTER)}
        eligible = {event.event_id for event in store.eligible_signals_as_of(AFTER)}
        return raw_view, eligible, store.correction_count()
    finally:
        store.close()


class TestApplyingIt:
    def test_a_dry_run_records_nothing(self, store_path: Path) -> None:
        assert correct(store_path, "--dry-run") == 0
        assert read(store_path)[2] == 0

    def test_the_original_is_preserved_physically_and_excluded_scientifically(self, store_path: Path) -> None:
        before, eligible_before, _ = read(store_path)
        assert ORIGINAL in eligible_before
        assert correct(store_path) == 0
        after, eligible_after, count = read(store_path)
        assert count == 1
        assert after[ORIGINAL] == before[ORIGINAL], "the historical observation was rewritten"
        assert after[ORIGINAL].event_type is EventType.MONETARY_POLICY, "a correction never rewrites a type"
        assert ORIGINAL not in eligible_after
        assert eligible_after == eligible_before - {ORIGINAL}

    def test_it_cannot_widen_to_other_monetary_policy_events(self, store_path: Path) -> None:
        correct(store_path)
        _, eligible, _ = read(store_path)
        assert "unrelated-" + "0" * 54 in eligible

    def test_applying_twice_records_once(self, store_path: Path) -> None:
        assert correct(store_path) == 0 and correct(store_path) == 0
        assert read(store_path)[2] == 1

    def test_the_audit_tells_the_original_from_the_corrected_classification(self, store_path: Path) -> None:
        correct(store_path)
        connection = open_read_only(store_path)
        try:
            original = audit_corpus(connection, as_of=AFTER, extractor_version="rules-v1")
            corrected = audit_corpus(connection, as_of=AFTER, extractor_version="rules-v2")
        finally:
            connection.close()
        record = next(r for r in original.catalog if r.event_id == ORIGINAL)
        assert record.correction_reason == REASON and Exclusion.INVALIDATED in record.exclusions
        assert record.event_type == "MONETARY_POLICY"
        (v2,) = corrected.catalog
        assert v2.event_type == "REGULATION" and v2.correction_reason is None
        assert Exclusion.INVALIDATED not in v2.exclusions
