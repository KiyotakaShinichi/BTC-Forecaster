"""B5 Gate 1 audit: synthetic corpora with known answers, and point-in-time attacks.

An audit that reports "insufficient" on the real corpus is only informative if
it can also report "sufficient" when the evidence is there, and if it cannot be
fooled into counting what it must not. So every corpus here has a known answer:

* a corpus that meets every clause passes, and an empty one does not;
* a replica of the corpus actually collected reproduces its verdict;
* the same occurrence reported by three publishers is one event, and a
  rediscovered document is a sighting, not an observation;
* one publisher, low extraction confidence or heavy correction each close the gate;
* impossible timestamps and retrieval-aligned event times each close it on
  their own, even beside a family that is otherwise ready.

And the four point-in-time attacks: an event moved past the audit instant
disappears; an event time that moves changes alignment and nothing else; a
future article cannot touch an earlier audit; a later re-extraction cannot change
what an earlier audit counted.

No network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from market_intelligence.b5.audit import (
    Exclusion,
    PitViolation,
    TimePrecision,
    TimeSource,
    audit_corpus,
    file_sha256,
    open_read_only,
)
from market_intelligence.b5.contracts import Decision, canonical_json
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.corrections import CorrectionStatus, EventCorrection
from market_intelligence.errors import StorageError
from market_intelligence.models import (
    Direction,
    Document,
    EventSignal,
    EventType,
    ExtractionMethod,
    SignalCategory,
    TransferContext,
)
from market_intelligence.operations import RunManifest, RunStatus
from market_intelligence.storage.store import IntelligenceStore

BASE = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
AS_OF = BASE + timedelta(days=400)
PUBLISHERS = ("pub-a", "pub-b", "pub-c")


# -- building corpora ----------------------------------------------------------


def document(index: int, *, publisher: str, published: datetime | None, retrieved: datetime) -> Document:
    return fixture_document(
        url=f"https://{publisher}.example.com/{index}",
        title=f"Announcement {index} from {publisher}",
        retrieved_at=retrieved,
        publisher=publisher,
        published_at=published,
        primary_source=True,
        official_source=True,
    )


def event(
    key: str,
    *,
    sources: list[str],
    event_time: datetime,
    available: datetime,
    event_type: EventType = EventType.REGULATION,
    entity: str | None = "SEC",
    relevance: float = 0.9,
    confidence: float = 0.8,
    version: str = "rules-v1",
    context: TransferContext | None = None,
) -> EventSignal:
    return EventSignal(
        event_id=f"event-{key}-{version}",
        event_time=event_time,
        available_time=available,
        source_ids=tuple(sources),
        category=SignalCategory.WEB_EVENT,
        entity=entity,
        event_type=event_type,
        sentiment=0.1,
        btc_relevance=relevance,
        novelty=0.8,
        confidence=confidence,
        expected_horizon_hours=24,
        summary=f"event {key}",
        direction=Direction.UNKNOWN,
        transfer_context=context,
        extractor_version=version,
        extraction_method=ExtractionMethod.RULE_BASED,
    )


def run(started: datetime, status: RunStatus = RunStatus.SUCCESS) -> RunManifest:
    return RunManifest(
        run_id=f"run-{started:%Y%m%dT%H%M%S}",
        started_at=started,
        finished_at=started + timedelta(minutes=5),
        configuration_fingerprint="fixture",
        providers_attempted=1,
        queries_attempted=1,
        documents_accepted=1,
        documents_rejected=0,
        events_accepted=1,
        events_rejected=0,
        quality_summary={},
        watermark_changes=0,
        software_source_sha="fixture",
        status=status,
        provider_ids=("fixture",),
    )


def invalidate(event_id: str) -> EventCorrection:
    return EventCorrection(
        event_id=event_id,
        status=CorrectionStatus.INVALIDATED,
        reason="REDISCOVERY_DUPLICATE_PRE_FIX",
        invalidated_at=BASE + timedelta(days=380),
        invalidated_by_version="fixture",
        source_bug="a rediscovered document produced a new event id",
    )


class Corpus:
    """A store under construction, written through the store's own API."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.documents: list[Document] = []
        self.events: list[EventSignal] = []
        self.runs: list[RunManifest] = []
        self.corrections: list[EventCorrection] = []

    def occurrences(
        self,
        count: int,
        *,
        spacing_days: int = 8,
        publishers: tuple[str, ...] = PUBLISHERS,
        reports_per_occurrence: int = 1,
        confidence: float = 0.8,
        start: datetime = BASE,
        key: str = "o",
    ) -> "Corpus":
        """`count` real-world occurrences, each reported by one or more publishers."""
        for index in range(count):
            published = start + timedelta(days=spacing_days * index)
            for report in range(reports_per_occurrence):
                publisher = publishers[(index + report) % len(publishers)]
                retrieved = published + timedelta(minutes=30 + 40 * report)
                doc = document(len(self.documents), publisher=publisher, published=published, retrieved=retrieved)
                self.documents.append(doc)
                self.events.append(
                    event(f"{key}{index}-{report}", sources=[doc.document_id], event_time=published, available=retrieved, confidence=confidence)
                )
        return self

    def daily_runs(self, days: int, *, start: datetime = BASE, status: RunStatus = RunStatus.SUCCESS) -> "Corpus":
        self.runs.extend(run(start + timedelta(days=day), status) for day in range(days))
        return self

    def write(self, rediscover: int = 0) -> Path:
        store = IntelligenceStore(self.path)
        try:
            store.put_documents(self.documents)
            for later in range(rediscover):
                store.put_documents(
                    [d.model_copy(update={"retrieved_at": d.retrieved_at + timedelta(days=later + 1)}) for d in self.documents]
                )
            store.put_signals(self.events)
            for manifest in self.runs:
                store.put_run(manifest)
            if self.corrections:
                store.put_corrections(self.corrections)
        finally:
            store.close()
        return self.path


def audit(path: Path, as_of: datetime = AS_OF):
    connection = open_read_only(path)
    try:
        return audit_corpus(connection, as_of=as_of)
    finally:
        connection.close()


def sufficient(path: Path) -> Corpus:
    """40 occurrences 8 days apart from three publishers, collected every day."""
    return Corpus(path).occurrences(40).daily_runs(40 * 8 + 1)


# -- corpora with known answers --------------------------------------------------


class TestKnownAnswers:
    def test_a_corpus_meeting_every_clause_passes(self, tmp_path: Path) -> None:
        result = audit(sufficient(tmp_path / "c.duckdb").write())
        assert result.gate.passed and result.gate.decision is None
        assert result.gate.ready_families == ("event_type:REGULATION",)
        assert result.audit.funnel["7_independent_events"] == 40
        assert result.audit.funnel["8_effective_events"] == 40
        regulation = next(f for f in result.audit.families if f.family == "event_type:REGULATION")
        assert (regulation.events, regulation.publishers, regulation.coverage_fraction) == (40, 3, 1.0)

    def test_an_empty_store_is_insufficient_on_every_clause(self, tmp_path: Path) -> None:
        result = audit(Corpus(tmp_path / "e.duckdb").write())
        assert not result.gate.passed
        assert result.gate.decision is Decision.INTELLIGENCE_CORPUS_INSUFFICIENT
        assert not any(clause.met for clause in result.gate.clauses)
        assert all(family.events == 0 for family in result.audit.families)
        assert len(result.audit.families) == 15

    def test_a_replica_of_the_collected_corpus_reproduces_its_verdict(self, tmp_path: Path) -> None:
        """Six documents, seven events, three invalidated rediscoveries, confidence 0.35, two days."""
        corpus = Corpus(tmp_path / "r.duckdb")
        first = datetime(2026, 9, 2, 8, 39, 22, tzinfo=timezone.utc)
        sec = [
            document(i, publisher="U.S. Securities and Exchange Commission".replace(" ", "-").replace(".", ""), published=published, retrieved=first + timedelta(seconds=19 * (i == 2)))
            for i, published in enumerate(
                [datetime(2026, 9, 1, 14, 57, tzinfo=timezone.utc), datetime(2026, 9, 1, 17, 52, 32, tzinfo=timezone.utc), datetime(2026, 9, 1, 19, 17, 39, tzinfo=timezone.utc)]
            )
        ]
        cftc = [
            document(10 + i, publisher="cftc", published=datetime(2026, 9, 2, 16, 4, 52, tzinfo=timezone.utc) + timedelta(hours=i), retrieved=datetime(2026, 9, 3, 9, 15, 13, tzinfo=timezone.utc))
            for i in range(2)
        ]
        bls = [document(20, publisher="bls", published=datetime(2026, 9, 2, 12, 30, tzinfo=timezone.utc), retrieved=datetime(2026, 9, 3, 9, 15, 13, tzinfo=timezone.utc))]
        corpus.documents = sec + cftc + bls
        for doc in sec:
            corpus.events.append(event(doc.document_id[:8], sources=[doc.document_id], event_time=doc.published_at, available=doc.available_at, relevance=0.55, confidence=0.35))  # type: ignore[arg-type]
            duplicate = event(doc.document_id[:8] + "-dup", sources=[doc.document_id], event_time=doc.published_at, available=doc.available_at + timedelta(minutes=22), relevance=0.55, confidence=0.35)  # type: ignore[arg-type]
            corpus.events.append(duplicate)
            corpus.corrections.append(invalidate(duplicate.event_id))
        corpus.events.append(event("cftc", sources=[cftc[0].document_id], event_time=cftc[0].published_at, available=cftc[0].available_at, event_type=EventType.MONETARY_POLICY, entity="CFTC", relevance=0.55, confidence=0.35))  # type: ignore[arg-type]
        corpus.runs = [run(first + timedelta(minutes=m), RunStatus.DEGRADED) for m in (0, 22, 39)] + [
            run(datetime(2026, 9, 3, 9, 15, tzinfo=timezone.utc) + timedelta(minutes=m), RunStatus.DEGRADED) for m in (0, 32)
        ]
        result = audit(corpus.write(), as_of=datetime(2026, 9, 11, tzinfo=timezone.utc))
        assert result.gate.decision is Decision.INTELLIGENCE_CORPUS_INSUFFICIENT
        assert (result.audit.documents, result.audit.raw_events, result.audit.invalidated_events) == (6, 7, 3)
        assert result.audit.clusters_after_corrections == 2
        assert result.audit.funnel == {
            "1_collected": 7,
            "2_not_invalidated": 4,
            "3_sources_resolved": 4,
            "4_point_in_time_valid": 4,
            "5_quality": 0,
            "6_publication_time": 0,
            "7_independent_events": 0,
            "8_effective_events": 0,
        }
        assert result.audit.collection_days == 2 and result.audit.days_since_last_collection == 8
        failed = [clause.name for clause in result.gate.clauses if not clause.met]
        assert failed == ["ready_families"]


class TestNoPseudoreplication:
    def test_three_publishers_reporting_one_occurrence_are_one_event(self, tmp_path: Path) -> None:
        result = audit(Corpus(tmp_path / "d.duckdb").occurrences(40, reports_per_occurrence=3).daily_runs(321).write())
        assert result.audit.raw_events == 120
        assert result.audit.funnel["7_independent_events"] == 40
        assert result.gate.passed

    def test_a_rediscovered_document_is_a_sighting_not_an_observation(self, tmp_path: Path) -> None:
        once = audit(sufficient(tmp_path / "once.duckdb").write())
        twice = audit(sufficient(tmp_path / "twice.duckdb").write(rediscover=2))
        assert twice.audit.documents == once.audit.documents == 40
        assert twice.audit.retrieved_copies == 120 and twice.audit.duplicate_rate == pytest.approx(80 / 120, abs=1e-6)
        assert twice.audit.funnel == once.audit.funnel


class TestEachClauseCanCloseTheGate:
    def test_one_publisher_is_not_enough(self, tmp_path: Path) -> None:
        result = audit(Corpus(tmp_path / "p.duckdb").occurrences(40, publishers=("pub-a",)).daily_runs(321).write())
        assert result.gate.decision is Decision.INTELLIGENCE_CORPUS_INSUFFICIENT
        regulation = next(f for f in result.audit.families if f.family == "event_type:REGULATION")
        assert any("1 publishers, need 3" in clause for clause in regulation.unmet)

    def test_events_the_extractor_was_unsure_of_are_not_counted(self, tmp_path: Path) -> None:
        result = audit(Corpus(tmp_path / "q.duckdb").occurrences(40, confidence=0.35).daily_runs(321).write())
        assert result.audit.funnel["5_quality"] == 0
        assert all(Exclusion.BELOW_QUALITY in record.exclusions for record in result.catalog)
        assert result.gate.decision is Decision.INTELLIGENCE_CORPUS_INSUFFICIENT

    def test_corrected_events_stay_in_the_catalog_and_out_of_the_count(self, tmp_path: Path) -> None:
        corpus = sufficient(tmp_path / "x.duckdb")
        corpus.corrections = [invalidate(e.event_id) for e in corpus.events[:15]]
        result = audit(corpus.write())
        assert result.audit.invalidated_events == 15 and len(result.catalog) == 40
        assert result.audit.funnel["7_independent_events"] == 25
        invalidated = [r for r in result.catalog if Exclusion.INVALIDATED in r.exclusions]
        assert len(invalidated) == 15 and {r.correction_reason for r in invalidated} == {"REDISCOVERY_DUPLICATE_PRE_FIX"}
        assert result.gate.decision is Decision.INTELLIGENCE_CORPUS_INSUFFICIENT

    def test_impossible_timestamps_close_the_gate_beside_a_ready_family(self, tmp_path: Path) -> None:
        corpus = sufficient(tmp_path / "f.duckdb")
        for index in range(10):
            doc = document(1000 + index, publisher="pub-a", published=BASE + timedelta(days=3 + index), retrieved=BASE + timedelta(days=3 + index, hours=1))
            corpus.documents.append(doc)
            corpus.events.append(event(f"leak{index}", sources=[doc.document_id], event_time=doc.available_at + timedelta(hours=6), available=doc.available_at, entity="CFTC"))
        result = audit(corpus.write())
        assert result.gate.ready_families == ("event_type:REGULATION",)
        leaks = [r for r in result.catalog if r.event_id.startswith("event-leak")]
        assert all(PitViolation.EVENT_AFTER_AVAILABILITY in r.pit_violations and not r.counted for r in leaks)
        clause = next(c for c in result.gate.clauses if c.name == "point_in_time_integrity")
        assert not clause.met and result.gate.decision is Decision.INTELLIGENCE_CORPUS_INSUFFICIENT

    def test_event_times_taken_from_retrieval_close_the_gate_on_their_own(self, tmp_path: Path) -> None:
        corpus = sufficient(tmp_path / "t.duckdb")
        for index in range(40):
            doc = document(2000 + index, publisher="pub-b", published=BASE + timedelta(days=4 + 8 * index), retrieved=BASE + timedelta(days=4 + 8 * index, hours=5))
            corpus.documents.append(doc)
            corpus.events.append(event(f"ret{index}", sources=[doc.document_id], event_time=doc.available_at, available=doc.available_at, entity="Fed"))
        result = audit(corpus.write())
        assert result.audit.time_sources == {"PUBLICATION": 40, "RETRIEVAL": 40}
        clause = next(c for c in result.gate.clauses if c.name == "event_time_from_publication")
        assert not clause.met and result.gate.decision is Decision.INTELLIGENCE_CORPUS_INSUFFICIENT

    def test_a_source_claiming_publication_after_its_retrieval_is_a_violation(self, tmp_path: Path) -> None:
        corpus = Corpus(tmp_path / "s.duckdb")
        doc = document(1, publisher="pub-a", published=BASE + timedelta(hours=3), retrieved=BASE)
        corpus.documents.append(doc)
        corpus.events.append(event("s", sources=[doc.document_id], event_time=BASE - timedelta(hours=1), available=BASE))
        record = audit(corpus.write()).catalog[0]
        assert PitViolation.SOURCE_PUBLISHED_AFTER_RETRIEVAL in record.pit_violations and not record.counted


class TestPointInTimeAttacks:
    def test_an_event_moved_past_the_audit_instant_disappears(self, tmp_path: Path) -> None:
        corpus = sufficient(tmp_path / "a.duckdb")
        late = document(3000, publisher="pub-a", published=AS_OF + timedelta(hours=1), retrieved=AS_OF + timedelta(hours=2))
        corpus.documents.append(late)
        corpus.events.append(event("late", sources=[late.document_id], event_time=late.published_at, available=late.available_at))  # type: ignore[arg-type]
        path = corpus.write()
        assert "event-late-rules-v1" not in {r.event_id for r in audit(path).catalog}
        assert "event-late-rules-v1" in {r.event_id for r in audit(path, as_of=AS_OF + timedelta(days=1)).catalog}

    def test_moving_an_event_time_changes_alignment_and_nothing_else(self, tmp_path: Path) -> None:
        base = sufficient(tmp_path / "b.duckdb")
        moved = sufficient(tmp_path / "m.duckdb")
        target = moved.events[5]
        moved.events[5] = target.model_copy(update={"event_time": target.event_time - timedelta(hours=3)})
        before = {r.event_id: r for r in audit(base.write()).catalog}[target.event_id]
        after = {r.event_id: r for r in audit(moved.write()).catalog}[target.event_id]
        assert after.available_time == before.available_time
        assert after.event_time == before.event_time - timedelta(hours=3)
        assert after.retrieval_lag_hours == pytest.approx(before.retrieval_lag_hours + 3)
        assert (before.time_source, after.time_source) == (TimeSource.PUBLICATION, TimeSource.OTHER)
        assert after.exclusions == (Exclusion.NO_PUBLICATION_TIME,)

    def test_a_future_article_cannot_touch_an_earlier_audit(self, tmp_path: Path) -> None:
        without = sufficient(tmp_path / "w.duckdb").write()
        corpus = sufficient(tmp_path / "f.duckdb")
        future = document(4000, publisher="pub-z", published=AS_OF + timedelta(days=2), retrieved=AS_OF + timedelta(days=2, hours=1))
        corpus.documents.append(future)
        corpus.events.append(event("future", sources=[future.document_id], event_time=future.published_at, available=future.available_at))  # type: ignore[arg-type]
        with_future = corpus.write()
        assert canonical_json(audit(without).model_dump(mode="json")) == canonical_json(audit(with_future).model_dump(mode="json"))

    def test_a_later_re_extraction_cannot_change_what_an_earlier_audit_counted(self, tmp_path: Path) -> None:
        """A re-extraction is a second version beside the first. An earlier audit is
        untouched; a later one refuses to pool the two versions, and pinned to the
        original version counts exactly what the original store counts."""
        base = sufficient(tmp_path / "v1.duckdb").write()
        corpus = sufficient(tmp_path / "v2.duckdb")
        original = corpus.events[0]
        corpus.events.append(
            event("o0-0", sources=list(original.source_ids), event_time=original.event_time, available=AS_OF + timedelta(days=3), event_type=EventType.MONETARY_POLICY, version="rules-v2")
        )
        path = corpus.write()
        assert audit(path).catalog == audit(base).catalog
        later = AS_OF + timedelta(days=4)
        with pytest.raises(ValueError, match="pin one with --extractor-version"):
            audit(path, as_of=later)
        connection = open_read_only(path)
        try:
            pinned_v1 = audit_corpus(connection, as_of=later, extractor_version="rules-v1")
            pinned_v2 = audit_corpus(connection, as_of=later, extractor_version="rules-v2")
        finally:
            connection.close()
        assert pinned_v1.catalog == audit(base, as_of=later).catalog
        assert [r.event_id for r in pinned_v2.catalog] == ["event-o0-0-rules-v2"]
        assert pinned_v2.catalog[0].event_type == "MONETARY_POLICY"


class TestTheAuditItself:
    def test_it_never_writes_to_the_store(self, tmp_path: Path) -> None:
        path = sufficient(tmp_path / "ro.duckdb").write()
        before = file_sha256(path)
        audit(path)
        assert file_sha256(path) == before

    def test_it_refuses_a_missing_store(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_read_only(tmp_path / "absent.duckdb")

    def test_it_refuses_another_schema_rather_than_migrating(self, tmp_path: Path) -> None:
        path = tmp_path / "old.duckdb"
        connection = duckdb.connect(str(path))
        connection.execute("CREATE TABLE schema_metadata (component VARCHAR PRIMARY KEY, version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_metadata VALUES ('market_intelligence', 2)")
        connection.close()
        with pytest.raises(StorageError, match="refuses to migrate"):
            open_read_only(path)

    def test_two_audits_are_identical(self, tmp_path: Path) -> None:
        path = sufficient(tmp_path / "det.duckdb").write()
        first, second = audit(path), audit(path)
        assert canonical_json(first.model_dump(mode="json")) == canonical_json(second.model_dump(mode="json"))

    def test_whale_transfers_are_judged_per_context(self, tmp_path: Path) -> None:
        corpus = Corpus(tmp_path / "wh.duckdb")
        doc = document(1, publisher="pub-a", published=BASE, retrieved=BASE + timedelta(minutes=10))
        corpus.documents.append(doc)
        corpus.events.append(event("wh", sources=[doc.document_id], event_time=BASE, available=doc.available_at, event_type=EventType.WHALE_TRANSFER, entity=None, context=TransferContext.EXCHANGE_INFLOW))
        result = audit(corpus.write())
        assert result.catalog[0].family == "whale:EXCHANGE_INFLOW"
        assert {f.family for f in result.audit.families} >= {f"whale:{c.value}" for c in TransferContext}

    def test_date_only_timestamps_are_labelled(self, tmp_path: Path) -> None:
        corpus = Corpus(tmp_path / "day.duckdb")
        midnight = datetime(2026, 3, 2, tzinfo=timezone.utc)
        doc = document(1, publisher="pub-a", published=midnight, retrieved=midnight + timedelta(hours=9))
        corpus.documents.append(doc)
        corpus.events.append(event("day", sources=[doc.document_id], event_time=midnight, available=doc.available_at))
        result = audit(corpus.write())
        assert result.catalog[0].time_precision is TimePrecision.DATE_ONLY
        assert result.audit.time_precision == {"DATE_ONLY": 1}

    def test_every_catalogued_event_can_say_what_was_known_and_why_it_counted(self, tmp_path: Path) -> None:
        result = audit(sufficient(tmp_path / "prov.duckdb").write())
        for record in result.catalog:
            assert record.source_ids and record.publishers and record.earliest_source_available_at is not None
            assert record.available_time >= record.earliest_source_available_at
            assert record.cluster_id is not None if record.counted else record.cluster_id is None
        assert json.loads(canonical_json([r.model_dump(mode="json") for r in result.catalog]))
