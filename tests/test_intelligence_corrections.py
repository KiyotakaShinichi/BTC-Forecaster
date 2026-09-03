"""Corrections — preserved physically, excluded scientifically.

Three events in the live corpus were manufactured by a defect: re-extracting an
unchanged document produced a new event on every cycle. The tempting fix is to
delete them, and it is wrong. A deleted row leaves no trace, so nobody later can
tell a corpus that never held an observation from one that quietly dropped it;
the manifests and snapshots that counted them still exist and would now be
referencing nothing; and a corpus whose contents can change without leaving
evidence is not evidence.

So the ledger is append-only, and everything below tests the two views that
follow from it: an invalidated observation is exactly where it always was, and
is absent from anything that counts.

No network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.collection.clustering import cluster_events
from market_intelligence.collection.corpus import (
    RAW_ELIGIBILITY_CONTRACT,
    CorpusCatalog,
    build_snapshot,
)
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.collection.readiness import AdequacyPolicy, assess_family
from market_intelligence.corrections import (
    ELIGIBILITY_CONTRACT_VERSION,
    REDISCOVERY_DUPLICATE_PRE_FIX,
    CorrectionStatus,
    EligibilityMode,
    EventCorrection,
    eligible,
    ineligible_ids,
    load_corrections,
    resolve,
)
from market_intelligence.models import (
    Direction,
    EventSignal,
    EventType,
    SignalCategory,
)
from market_intelligence.ops.integrity import IntegrityStatus, verify
from market_intelligence.storage import IntelligenceStore

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
FIXED = datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc)
#: After every fixture document, before the correction.
HORIZON = datetime(2026, 9, 6, 0, 0, tzinfo=timezone.utc)
BUG = "event identity derived from the re-retrieved document's available_at"
VERSION = "ec4d1c85980174f005deeed91b07257d600c5a95"

#: The correction actually applied to the live corpus.
COMMITTED = (
    Path(__file__).resolve().parents[1] / "corrections" / "2026-09-03-rediscovery-duplicates.json"
)


# ------------------------------------------------------------------- fixtures


def an_event(document_id: str, available: datetime, *, extractor: str = "rules-v1") -> EventSignal:
    return EventSignal(
        event_id=EventSignal.stable_id([document_id], EventType.REGULATION, available, extractor),
        event_time=available,
        available_time=available,
        source_ids=(document_id,),
        category=SignalCategory.WEB_EVENT,
        entity="SEC",
        event_type=EventType.REGULATION,
        direction=Direction.UNKNOWN,
        sentiment=0.0,
        btc_relevance=0.55,
        novelty=0.5,
        confidence=0.5,
        expected_horizon_hours=24,
        summary=f"regulation notice from {document_id[:8]}",
        extractor_version=extractor,
    )


def invalidation(event_id: str, *, at: datetime = FIXED, notes: str = "") -> EventCorrection:
    return EventCorrection(
        event_id=event_id,
        status=CorrectionStatus.INVALIDATED,
        reason=REDISCOVERY_DUPLICATE_PRE_FIX,
        invalidated_at=at,
        invalidated_by_version=VERSION,
        source_bug=BUG,
        notes=notes,
    )


@pytest.fixture
def corpus(tmp_path: Path):
    """A store shaped like the live one: genuine observations plus duplicates.

    Each document carries one real event and one artefact, exactly as the defect
    produced them -- same source, same extractor, a later availability.
    """
    store = IntelligenceStore(tmp_path / "corpus.duckdb")
    documents = [
        fixture_document(
            url=f"https://sec.example.gov/news/{index}",
            title=f"SEC regulation notice {index}",
            retrieved_at=NOW + timedelta(days=index),
            publisher=f"publisher-{index}.example.gov",
            primary_source=True,
            official_source=True,
            body=f"body {index}",
        )
        for index in range(3)
    ]
    genuine = [an_event(document.document_id, document.available_at) for document in documents]
    spurious = [
        an_event(document.document_id, document.available_at + timedelta(hours=3))
        for document in documents
    ]
    store.persist_cycle(documents, genuine + spurious, [])
    try:
        yield store, documents, genuine, spurious
    finally:
        store.close()


# ------------------------------------------------ preserved, and excluded


class TestPreservedPhysicallyExcludedScientifically:
    def test_an_invalidated_event_is_still_physically_present(self, corpus) -> None:
        """The point of the whole design. Deleting would leave a corpus that
        cannot explain its own history."""
        store, _, _, spurious = corpus
        store.put_corrections([invalidation(event.event_id) for event in spurious])

        rows = store.connection.execute("SELECT COUNT(*) FROM signals").fetchone()
        assert rows is not None and rows[0] == 6
        raw = {event.event_id for event in store.signals_as_of(HORIZON)}
        assert all(event.event_id in raw for event in spurious)

    def test_the_raw_view_still_counts_them(self, corpus) -> None:
        store, _, _, spurious = corpus
        store.put_corrections([invalidation(event.event_id) for event in spurious])
        assert len(store.signals_as_of(HORIZON)) == 6

    def test_the_eligible_view_does_not(self, corpus) -> None:
        store, _, genuine, spurious = corpus
        store.put_corrections([invalidation(event.event_id) for event in spurious])

        research = store.eligible_signals_as_of(HORIZON)
        assert len(research) == 3
        assert {event.event_id for event in research} == {event.event_id for event in genuine}

    def test_unaffected_events_remain_eligible(self, corpus) -> None:
        """A correction excludes exactly what it names and nothing adjacent."""
        store, _, genuine, spurious = corpus
        store.put_corrections([invalidation(spurious[0].event_id)])

        research = {e.event_id for e in store.eligible_signals_as_of(HORIZON)}
        assert spurious[0].event_id not in research
        assert spurious[1].event_id in research, "an unnamed event was swept up"
        assert spurious[2].event_id in research
        assert all(event.event_id in research for event in genuine)

    def test_an_invalidated_event_stays_visible_to_audit(self, corpus) -> None:
        store, _, _, spurious = corpus
        store.put_corrections([invalidation(spurious[0].event_id)])

        recorded = store.corrections([spurious[0].event_id])
        assert len(recorded) == 1
        assert recorded[0].reason == REDISCOVERY_DUPLICATE_PRE_FIX
        assert recorded[0].source_bug == BUG
        assert recorded[0].invalidated_by_version == VERSION


# ------------------------------------------------------------------ readiness


class TestResearchEligibility:
    def test_readiness_does_not_count_invalidated_observations(self, corpus) -> None:
        """The consequence that made this urgent: the gate counts events, so
        duplicates of one announcement could open it."""
        store, _, _, spurious = corpus
        policy = AdequacyPolicy(
            minimum_events=5,
            minimum_effective_events=4,
            minimum_publishers=1,
            minimum_providers=1,
            minimum_span_days=1,
            horizon_hours=1,
            minimum_coverage_fraction=0.0,
        )

        documents = store.documents_as_of(HORIZON)
        raw = cluster_events(store.signals_as_of(HORIZON), documents, window_hours=1)
        assert assess_family("regulation", raw, policy=policy, coverage_fraction=1.0).ready, (
            "the raw corpus must clear this bar, or the test proves nothing"
        )

        store.put_corrections([invalidation(event.event_id) for event in spurious])
        corrected = cluster_events(
            store.eligible_signals_as_of(HORIZON), documents, window_hours=1
        )
        outcome = assess_family("regulation", corrected, policy=policy, coverage_fraction=1.0)
        assert not outcome.ready
        assert any("3 events" in reason for reason in outcome.unmet)

    def test_event_study_queries_exclude_invalidated_observations(self, corpus) -> None:
        """Whatever a study reaches for, it reaches through the eligible view."""
        from market_intelligence.cycle import ReplayService

        store, _, genuine, spurious = corpus
        store.put_corrections([invalidation(event.event_id) for event in spurious])

        horizon = HORIZON
        snapshot = ReplayService(store).replay(horizon, {"syndication": "1"}, "fingerprint")
        assert len(snapshot.event_ids) == 3
        assert set(snapshot.event_ids) == {event.event_id for event in genuine}

    def test_feature_aggregation_excludes_them(self, corpus) -> None:
        from market_intelligence.services import IntelligenceReadService

        store, _, _, spurious = corpus
        # The aggregate features are 24h and 72h windows, so the origin has to
        # sit close to the events or both sides are zero and prove nothing.
        origin = NOW + timedelta(days=2, hours=4)
        before = IntelligenceReadService(store).aggregate(origin)
        assert before["event_count_24h"] == 2, "the window must see both events first"

        store.put_corrections([invalidation(event.event_id) for event in spurious])
        after = IntelligenceReadService(store).aggregate(origin)
        assert after["event_count_24h"] == 1
        assert before != after, "aggregation still counted the invalidated events"

    def test_the_feature_matrix_fingerprint_moves_with_eligibility(self, corpus) -> None:
        """Otherwise an incremental extension silently keeps excluded rows."""
        from market_intelligence.feature_matrix import source_fingerprint

        store, _, _, spurious = corpus
        horizon = HORIZON
        before = source_fingerprint(store, horizon)
        store.put_corrections([invalidation(event.event_id) for event in spurious])
        assert source_fingerprint(store, horizon) != before


# ------------------------------------------------------------------ integrity


class TestCorpusIntegrityStillSucceeds:
    def test_verification_passes_after_a_correction(self, corpus) -> None:
        store, _, _, spurious = corpus
        store.put_corrections([invalidation(event.event_id) for event in spurious])

        report = verify(store, as_of=HORIZON)
        assert report.status is IntegrityStatus.OK, report.human_readable()

    def test_verification_still_sees_every_event(self, corpus) -> None:
        """Integrity reads the raw view deliberately: an invalidated event must
        still satisfy the invariants it always satisfied."""
        store, _, _, spurious = corpus
        store.put_corrections([invalidation(event.event_id) for event in spurious])
        assert verify(store, as_of=HORIZON).events == 6


# ------------------------------------------------------ snapshots and replay


class TestSnapshotEligibilityContract:
    def test_an_old_snapshot_is_not_reinterpreted(self, corpus) -> None:
        """A snapshot frozen before corrections existed was built from every
        event in the store. Checking it against a corrected view would report a
        corpus as damaged for having been corrected."""
        store, documents, _, spurious = corpus
        horizon = HORIZON
        raw_snapshot = build_snapshot(
            store.documents_as_of(horizon),
            store.signals_as_of(horizon),
            as_of=horizon,
            extractor_version="rules-v1",
            created_at=horizon,
        )
        assert raw_snapshot.eligibility_contract == RAW_ELIGIBILITY_CONTRACT
        assert raw_snapshot.event_count == 6
        CorpusCatalog(store.connection).register(raw_snapshot)

        store.put_corrections([invalidation(event.event_id) for event in spurious])
        assert verify(store, as_of=horizon).status is IntegrityStatus.OK

    def test_a_corrected_snapshot_is_a_distinct_artefact(self, corpus) -> None:
        """Not a silent replacement of the raw one at the same instant."""
        store, _, _, spurious = corpus
        horizon = HORIZON
        store.put_corrections([invalidation(event.event_id) for event in spurious])

        raw_snapshot = build_snapshot(
            store.documents_as_of(horizon),
            store.signals_as_of(horizon),
            as_of=horizon,
            extractor_version="rules-v1",
            created_at=horizon,
        )
        corrected = build_snapshot(
            store.documents_as_of(horizon),
            store.eligible_signals_as_of(horizon),
            as_of=horizon,
            extractor_version="rules-v1",
            created_at=horizon,
            eligibility_contract=ELIGIBILITY_CONTRACT_VERSION,
        )
        assert corrected.corpus_id != raw_snapshot.corpus_id
        assert corrected.membership_hash != raw_snapshot.membership_hash
        assert corrected.event_count == 3
        assert corrected.eligibility_contract == ELIGIBILITY_CONTRACT_VERSION

    def test_both_snapshots_verify_side_by_side(self, corpus) -> None:
        """The old one and the corrected one coexist, each checked under the
        rule it declares."""
        store, _, _, spurious = corpus
        horizon = HORIZON
        catalog = CorpusCatalog(store.connection)
        catalog.register(
            build_snapshot(
                store.documents_as_of(horizon),
                store.signals_as_of(horizon),
                as_of=horizon,
                extractor_version="rules-v1",
                created_at=horizon,
            )
        )
        store.put_corrections([invalidation(event.event_id) for event in spurious])
        catalog.register(
            build_snapshot(
                store.documents_as_of(horizon),
                store.eligible_signals_as_of(horizon),
                as_of=horizon,
                extractor_version="rules-v1",
                created_at=horizon,
                eligibility_contract=ELIGIBILITY_CONTRACT_VERSION,
            )
        )
        report = verify(store, as_of=horizon)
        assert report.status is IntegrityStatus.OK, report.human_readable()
        assert report.snapshots == 2

    def test_as_believed_mode_ignores_later_corrections(self, corpus) -> None:
        """Auditing what a past run saw is a different question from what
        research may now count, and it must not be answered with hindsight."""
        store, _, _, spurious = corpus
        store.put_corrections([invalidation(event.event_id, at=FIXED) for event in spurious])

        before_the_correction = HORIZON
        assert before_the_correction < FIXED
        assert (
            len(
                store.eligible_signals_as_of(
                    before_the_correction, mode=EligibilityMode.AS_BELIEVED
                )
            )
            == 6
        )
        assert len(store.eligible_signals_as_of(FIXED + timedelta(days=1))) == 3

    def test_as_believed_requires_an_instant(self) -> None:
        with pytest.raises(ValueError, match="instant"):
            ineligible_ids([], mode=EligibilityMode.AS_BELIEVED)


# ------------------------------------------------------------ the ledger itself


class TestTheLedgerIsAppendOnly:
    def test_recording_the_same_correction_twice_writes_one_row(self, corpus) -> None:
        store, _, _, spurious = corpus
        correction = invalidation(spurious[0].event_id)
        assert store.put_corrections([correction]) == 1
        assert store.put_corrections([correction]) == 0
        assert store.correction_count() == 1

    def test_a_correction_never_overwrites_another(self, corpus) -> None:
        """Two statements about one event both survive; only their standing
        differs."""
        store, _, _, spurious = corpus
        event_id = spurious[0].event_id
        store.put_corrections([invalidation(event_id, notes="first")])
        store.put_corrections([invalidation(event_id, notes="second")])
        assert store.correction_count() == 2
        assert {item.notes for item in store.corrections([event_id])} == {"first", "second"}

    def test_a_later_correction_stands(self, corpus) -> None:
        store, _, genuine, spurious = corpus
        event_id = spurious[0].event_id
        store.put_corrections([invalidation(event_id, at=FIXED)])
        assert event_id in store.ineligible_event_ids()

        store.put_corrections(
            [
                EventCorrection(
                    event_id=event_id,
                    status=CorrectionStatus.REINSTATED,
                    reason="the invalidation was itself wrong",
                    invalidated_at=FIXED + timedelta(days=1),
                    invalidated_by_version=VERSION,
                    source_bug=BUG,
                )
            ]
        )
        assert event_id not in store.ineligible_event_ids()
        assert store.correction_count() == 2, "reinstating erased the original record"

    def test_contradictory_corrections_at_the_same_instant_still_resolve(self) -> None:
        """The one case where 'latest wins' is not enough. Without a tiebreak,
        eligibility would depend on row order -- which is precisely the
        inconsistent state an append-only ledger exists to rule out."""
        shared = {
            "event_id": "e1",
            "reason": "r",
            "invalidated_at": FIXED,
            "invalidated_by_version": VERSION,
            "source_bug": BUG,
        }
        a = EventCorrection(status=CorrectionStatus.INVALIDATED, **shared)
        b = EventCorrection(status=CorrectionStatus.REINSTATED, **shared)

        forwards = resolve([a, b])
        backwards = resolve([b, a])
        assert forwards["e1"].correction_id == backwards["e1"].correction_id
        assert forwards["e1"].status is backwards["e1"].status

    def test_the_correction_id_is_content_addressed(self) -> None:
        assert invalidation("e1").correction_id == invalidation("e1").correction_id
        assert invalidation("e1").correction_id != invalidation("e2").correction_id
        assert invalidation("e1").correction_id != invalidation("e1", notes="x").correction_id

    def test_eligible_helper_matches_the_store(self, corpus) -> None:
        store, _, genuine, spurious = corpus
        corrections = [invalidation(event.event_id) for event in spurious]
        store.put_corrections(corrections)

        horizon = HORIZON
        assert {event.event_id for event in eligible(store.signals_as_of(horizon), corrections)} == {
            event.event_id for event in store.eligible_signals_as_of(horizon)
        }


# ---------------------------------------------------- the committed correction


class TestTheCommittedCorrection:
    def test_the_committed_file_parses(self) -> None:
        corrections = load_corrections(json.loads(COMMITTED.read_text(encoding="utf-8")))
        assert len(corrections) == 3
        assert {item.reason for item in corrections} == {REDISCOVERY_DUPLICATE_PRE_FIX}
        assert {item.status for item in corrections} == {CorrectionStatus.INVALIDATED}

    def test_it_names_exactly_three_events_and_cannot_widen(self) -> None:
        """The ids are literal. A correction expressed as a rule would keep
        matching new events forever, which is a filter, not a correction."""
        raw = json.loads(COMMITTED.read_text(encoding="utf-8"))
        assert len(raw["corrections"]) == 3
        assert all(set(entry) == {"event_id"} for entry in raw["corrections"])
        assert len({entry["event_id"] for entry in raw["corrections"]}) == 3

    def test_it_records_the_fix_that_made_it_knowable(self) -> None:
        corrections = load_corrections(json.loads(COMMITTED.read_text(encoding="utf-8")))
        assert all(item.invalidated_by_version.startswith("ec4d1c8") for item in corrections)
        assert all(item.source_bug for item in corrections)
        assert all(item.notes for item in corrections)

    def test_correction_ids_are_stable_across_machines(self) -> None:
        """Applying the committed file twice, anywhere, records three rows once."""
        first = load_corrections(json.loads(COMMITTED.read_text(encoding="utf-8")))
        second = load_corrections(json.loads(COMMITTED.read_text(encoding="utf-8")))
        assert [item.correction_id for item in first] == [item.correction_id for item in second]

    def test_a_file_with_no_corrections_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="corrections"):
            load_corrections({"reason": "x"})


# -------------------------------------------------------------------- the CLI


class TestCorrectionCommands:
    def _store(self, tmp_path: Path):
        store = IntelligenceStore(tmp_path / "cli.duckdb")
        document = fixture_document(
            url="https://sec.example.gov/news/0",
            title="SEC regulation notice",
            retrieved_at=NOW,
            body="b",
        )
        events = [an_event(document.document_id, NOW), an_event(document.document_id, NOW + timedelta(hours=3))]
        store.persist_cycle([document], events, [])
        return store, events

    def _file(self, tmp_path: Path, event_ids: list[str]) -> Path:
        target = tmp_path / "correction.json"
        target.write_text(
            json.dumps(
                {
                    "reason": REDISCOVERY_DUPLICATE_PRE_FIX,
                    "invalidated_at": FIXED.isoformat(),
                    "invalidated_by_version": VERSION,
                    "source_bug": BUG,
                    "corrections": [{"event_id": event_id} for event_id in event_ids],
                }
            ),
            encoding="utf-8",
        )
        return target

    def test_a_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        from market_intelligence.cli import main

        store, events = self._store(tmp_path)
        path = self._file(tmp_path, [events[1].event_id])
        store.close()

        assert main(["--db", str(tmp_path / "cli.duckdb"), "corpus-correct", "--file", str(path), "--dry-run"]) == 0
        reopened = IntelligenceStore(tmp_path / "cli.duckdb")
        try:
            assert reopened.correction_count() == 0
        finally:
            reopened.close()

    def test_applying_then_reapplying_records_once(self, tmp_path: Path) -> None:
        from market_intelligence.cli import main

        store, events = self._store(tmp_path)
        path = self._file(tmp_path, [events[1].event_id])
        store.close()
        argv = ["--db", str(tmp_path / "cli.duckdb"), "corpus-correct", "--file", str(path)]
        assert main(argv) == 0
        assert main(argv) == 0

        reopened = IntelligenceStore(tmp_path / "cli.duckdb")
        try:
            assert reopened.correction_count() == 1
            assert len(reopened.eligible_signals_as_of(HORIZON)) == 1
        finally:
            reopened.close()

    def test_a_correction_naming_an_unknown_event_is_refused(self, tmp_path: Path) -> None:
        """A row pointing at nothing is indistinguishable from a typo, and it
        would sit in the ledger looking authoritative."""
        from market_intelligence.cli import main

        store, _ = self._store(tmp_path)
        path = self._file(tmp_path, ["0" * 64])
        store.close()

        code = main(["--db", str(tmp_path / "cli.duckdb"), "corpus-correct", "--file", str(path)])
        assert code == 2
        reopened = IntelligenceStore(tmp_path / "cli.duckdb")
        try:
            assert reopened.correction_count() == 0, "a refused correction was recorded anyway"
        finally:
            reopened.close()

    def test_the_report_shows_both_counts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from market_intelligence.cli import main

        store, events = self._store(tmp_path)
        path = self._file(tmp_path, [events[1].event_id])
        store.close()
        main(["--db", str(tmp_path / "cli.duckdb"), "corpus-correct", "--file", str(path)])
        capsys.readouterr()

        assert main(["--db", str(tmp_path / "cli.duckdb"), "corpus-corrections", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["events_raw"] == 2
        assert payload["events_eligible"] == 1
        assert payload["events_excluded"] == 1
        assert payload["eligibility_contract"] == ELIGIBILITY_CONTRACT_VERSION
