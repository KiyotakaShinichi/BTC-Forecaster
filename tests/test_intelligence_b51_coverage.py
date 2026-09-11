"""B5.1 — readiness coverage is computed from what collection actually did, once.

B5 found the readiness report could never open: `corpus-status` passed no
coverage, so every family read "collection covered 0% of days". And the gate's
`coverage_fraction()` counted days outside the span it was asked about, so it
could have fabricated coverage in the other direction.

Pinned here:

* no collection record is `NO_COLLECTION`, never 0%, and an unmeasured family
  says "not measured";
* a covered day is a UTC day with a successful provider attempt: partial and full
  coverage, degraded runs, failed runs, a provider that did not answer and a
  retired feed each count exactly as they should;
* days are UTC whatever zone the database hands them back in, spans are
  inclusive, and a day outside a span never counts;
* `corpus-status`, B5's Gate 1 audit and the collection service give the same
  answer, because they call the same function;
* the deployed path -- `collect-scheduled` -- records what coverage reads, and a
  cycle that could not take the run lock records nothing.

No network.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from market_intelligence.b5.audit import audit_corpus, open_read_only
from market_intelligence.collection.backoff import FailureClass, ProviderFailure
from market_intelligence.collection.coverage import (
    COVERAGE_CONTRACT_VERSION,
    CoverageState,
    collection_coverage,
    span_coverage,
    successful_days,
    utc_day,
)
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.collection.readiness import AdequacyPolicy, assess_family, coverage_fraction
from market_intelligence.collection.service import ForwardCollector
from market_intelligence.extractors import CURRENT_RULE_EXTRACTOR_VERSION, EvidenceRuleExtractor
from market_intelligence.models import EventType
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import CollectionProfile, collect_once
from market_intelligence.ops.runlock import RunLock
from market_intelligence.ops.scheduled import EXIT_LOCK_HELD
from market_intelligence.reports import corpus_status
from market_intelligence.retrieval import ProviderAttempt
from market_intelligence.storage.store import IntelligenceStore
from tests.test_intelligence_silent_empty import NOW as SCHEDULED
from tests.test_intelligence_silent_empty import minimal, offline

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
MANILA = timezone(timedelta(hours=8))


def attempts(store: IntelligenceStore, run_id: str, observed: datetime, *outcomes: tuple[str, bool]) -> None:
    store.put_provider_attempts(
        run_id,
        [ProviderAttempt(provider_id=provider, query_id="q", success=ok, attempts=1, latency_ms=1.0) for provider, ok in outcomes],
        observed,
    )


@pytest.fixture
def store(tmp_path: Path):
    handle = IntelligenceStore(tmp_path / "corpus.duckdb")
    yield handle
    handle.close()


class TestNoCollectionIsNotZero:
    def test_an_empty_store_says_not_measurable(self, store: IntelligenceStore) -> None:
        record = collection_coverage(store.connection, as_of=T0)
        assert record.state is CoverageState.NO_COLLECTION and record.coverage_fraction is None
        assert (record.elapsed_days, record.expected_days, record.successful_days) == (0, 0, 0)
        assert "not yet measurable" in record.describe()
        assert record.contract == COVERAGE_CONTRACT_VERSION

    def test_the_status_report_never_shows_an_unmeasured_zero(self, store: IntelligenceStore) -> None:
        status = corpus_status(store, CURRENT_RULE_EXTRACTOR_VERSION, ["SEC"])
        text = status.human_readable()
        assert status.collection is not None and status.collection.state is CoverageState.NO_COLLECTION
        assert "not yet measurable" in text and "covered 0%" not in text
        assert all(family.coverage_fraction is None for family in status.families)
        assert all(any("not measured" in reason for reason in family.unmet) for family in status.families)

    def test_an_unmeasured_family_fails_the_clause_and_says_why(self) -> None:
        family = assess_family("event_type:REGULATION", [])
        assert family.coverage_fraction is None and not family.ready
        assert any("coverage not measured" in reason for reason in family.unmet)
        assert not any("covered 0%" in reason for reason in family.unmet)

    def test_a_policy_with_no_coverage_floor_asks_nothing_of_coverage(self) -> None:
        family = assess_family("x", [], policy=AdequacyPolicy(minimum_coverage_fraction=0.0))
        assert not any("coverage" in reason for reason in family.unmet)


class TestWhatCountsAsCovered:
    def test_partial_coverage(self, store: IntelligenceStore) -> None:
        for day in (0, 3, 6):
            attempts(store, f"r{day}", T0 + timedelta(days=day), ("syndication", True))
        record = collection_coverage(store.connection, as_of=T0 + timedelta(days=9))
        assert (record.elapsed_days, record.successful_days) == (10, 3)
        assert record.coverage_fraction == pytest.approx(0.3)

    def test_full_coverage(self, store: IntelligenceStore) -> None:
        for day in range(5):
            attempts(store, f"r{day}", T0 + timedelta(days=day), ("syndication", True))
        assert collection_coverage(store.connection, as_of=T0 + timedelta(days=4)).coverage_fraction == 1.0

    def test_a_degraded_run_that_retrieved_counts(self, store: IntelligenceStore) -> None:
        """DEGRADED means every query succeeded and a quality check failed: still collection."""
        attempts(store, "degraded", T0, ("syndication", True), ("syndication", True))
        assert successful_days(store.connection, as_of=T0) == (T0.date(),)

    def test_a_run_whose_attempts_all_failed_does_not(self, store: IntelligenceStore) -> None:
        attempts(store, "failed", T0, ("syndication", False), ("news-api", False))
        record = collection_coverage(store.connection, as_of=T0)
        assert record.state is CoverageState.MEASURED
        assert (record.successful_days, record.coverage_fraction, record.failed_attempts) == (0, 0.0, 2)
        assert record.last_success is None and "no successful attempt yet" in record.describe()

    def test_a_provider_that_did_not_answer_is_not_a_success(self, store: IntelligenceStore) -> None:
        attempts(store, "a", T0, ("news-api", False), ("syndication", True))
        attempts(store, "b", T0 + timedelta(days=1), ("news-api", False))
        assert successful_days(store.connection, as_of=T0 + timedelta(days=1)) == (T0.date(),)

    def test_a_retired_feed_failing_neither_helps_nor_hurts(self, store: IntelligenceStore) -> None:
        for day in range(3):
            attempts(store, f"r{day}", T0 + timedelta(days=day), ("syndication:sec-litigation", False), ("syndication:sec-press", True))
        attempts(store, "retired-only", T0 + timedelta(days=3), ("syndication:sec-litigation", False))
        record = collection_coverage(store.connection, as_of=T0 + timedelta(days=3))
        assert (record.successful_days, record.elapsed_days) == (3, 4)

    def test_cycles_are_counted_by_run_at_the_declared_cadence(self, store: IntelligenceStore) -> None:
        attempts(store, "c1", T0, ("syndication", True))
        attempts(store, "c2", T0 + timedelta(hours=3), ("syndication", False))
        attempts(store, "c3", T0 + timedelta(hours=6), ("syndication", True), ("news-api", False))
        record = collection_coverage(store.connection, as_of=T0 + timedelta(hours=9))
        assert (record.expected_cycles, record.attempted_cycles, record.successful_cycles) == (4, 3, 2)
        assert (record.first_success, record.last_success) == (T0, T0 + timedelta(hours=6))

    def test_attempts_after_the_instant_are_not_counted(self, store: IntelligenceStore) -> None:
        attempts(store, "later", T0 + timedelta(days=2), ("syndication", True))
        assert collection_coverage(store.connection, as_of=T0).state is CoverageState.NO_COLLECTION


class TestTimeBoundaries:
    def test_an_instant_is_placed_on_its_utc_day(self) -> None:
        late = datetime(2026, 3, 1, 23, 30, tzinfo=timezone.utc)
        assert utc_day(late.astimezone(MANILA)) == date(2026, 3, 1)
        assert utc_day(datetime(2026, 3, 2, 0, 10)) == date(2026, 3, 2)

    def test_the_database_session_zone_does_not_move_a_day(self, store: IntelligenceStore) -> None:
        attempts(store, "late", datetime(2026, 3, 1, 23, 30, tzinfo=timezone.utc), ("syndication", True))
        try:
            store.connection.execute("SET TimeZone = 'Asia/Manila'")
        except duckdb.Error as error:  # pragma: no cover - a build without time zone support
            pytest.skip(f"this DuckDB build cannot set a session time zone: {error}")
        assert successful_days(store.connection, as_of=datetime(2026, 3, 5, tzinfo=timezone.utc)) == (date(2026, 3, 1),)

    def test_both_ends_of_a_span_count(self) -> None:
        start, end = date(2026, 3, 1), date(2026, 3, 4)
        assert span_coverage([start, end], start, end) == pytest.approx(0.5)

    def test_a_day_outside_the_span_never_counts(self) -> None:
        """The old helper counted every day it was given and capped at 1.0."""
        start = datetime(2026, 3, 10, tzinfo=timezone.utc)
        elsewhere = [datetime(2026, 1, day, tzinfo=timezone.utc) for day in range(1, 31)]
        assert coverage_fraction(elsewhere, start, start + timedelta(days=9)) == 0.0

    def test_a_one_day_span_with_collection_is_covered(self) -> None:
        """The old helper returned 0.0 for any span that did not cross midnight."""
        moment = datetime(2026, 3, 10, 9, tzinfo=timezone.utc)
        assert coverage_fraction([moment], moment, moment + timedelta(hours=5)) == 1.0

    def test_a_span_ending_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="before it starts"):
            span_coverage([], date(2026, 3, 2), date(2026, 3, 1))


def regulation_corpus(path: Path, *, collected_days: tuple[int, ...]) -> Path:
    """Three SEC regulation events a week apart, and collection on the given days."""
    store = IntelligenceStore(path)
    try:
        extractor = EvidenceRuleExtractor({"SEC": ("Securities and Exchange Commission",)}, {"SEC": (EventType.REGULATION,)})
        documents = [
            fixture_document(
                url=f"https://sec.example.gov/{index}",
                title=f"SEC Charges Adviser Number {index}",
                retrieved_at=T0 + timedelta(days=7 * index, hours=1),
                publisher="U.S. Securities and Exchange Commission",
                published_at=T0 + timedelta(days=7 * index),
                primary_source=True,
                official_source=True,
            )
            for index in range(3)
        ]
        store.put_documents(documents)
        store.put_signals(extractor.extract(documents))
        for day in collected_days:
            attempts(store, f"r{day}", T0 + timedelta(days=day, hours=1), ("syndication", True))
    finally:
        store.close()
    return path


class TestOnePlaceComputesIt:
    def test_corpus_status_and_the_gate_1_audit_agree(self, tmp_path: Path) -> None:
        path = regulation_corpus(tmp_path / "c.duckdb", collected_days=(0, 1, 2, 7, 14))
        store = IntelligenceStore(path)
        try:
            status = corpus_status(store, CURRENT_RULE_EXTRACTOR_VERSION, ["SEC"])
        finally:
            store.close()
        connection = open_read_only(path)
        try:
            result = audit_corpus(connection, as_of=datetime.now(timezone.utc))
        finally:
            connection.close()
        from_status = next(f for f in status.families if f.family == "event_type:REGULATION")
        from_audit = next(f for f in result.audit.families if f.family == "event_type:REGULATION")
        # The family spans days 0-14: 15 days, collection on 5 of them.
        assert from_status.coverage_fraction == from_audit.coverage_fraction == pytest.approx(5 / 15)
        assert status.collection is not None and result.audit.collection_coverage is not None
        assert status.collection.successful_days == result.audit.collection_coverage.successful_days == 5
        assert result.audit.collection_days == 5

    def test_the_status_report_states_the_collection_record(self, tmp_path: Path) -> None:
        path = regulation_corpus(tmp_path / "c.duckdb", collected_days=(0, 7, 14))
        store = IntelligenceStore(path)
        try:
            text = corpus_status(store, CURRENT_RULE_EXTRACTOR_VERSION, ["SEC"]).human_readable()
        finally:
            store.close()
        assert "elapsed day(s) with a successful provider attempt" in text
        assert "collection covered 20% of days, need 80%" in text

    def test_the_collection_service_reads_the_same_days(self, store: IntelligenceStore) -> None:
        """It used to read each watermark's latest retrieval, forgetting every earlier day."""
        attempts(store, "a", T0, ("syndication", True))
        attempts(store, "b", T0 + timedelta(days=2), ("syndication", True))
        attempts(store, "c", T0 + timedelta(days=3), ("syndication", False))
        collector = ForwardCollector(store, now=lambda: T0 + timedelta(days=5))
        assert collector.successful_run_days() == [
            datetime(2026, 3, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 3, tzinfo=timezone.utc),
        ]


class TestScheduledCollectionIsWhatCounts:
    """The deployed path, offline: `collect-scheduled` records what coverage reads."""

    def coverage(self, paths: StoragePaths, as_of: datetime):
        store = IntelligenceStore(paths.database)
        try:
            return collection_coverage(store.connection, as_of=as_of)
        finally:
            store.close()

    def test_scheduled_cycles_cover_their_day(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        offline(monkeypatch)
        paths = StoragePaths.from_environment(tmp_path / "s").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        for cycle in range(3):
            collect_once(paths, profile, now=lambda c=cycle: SCHEDULED + timedelta(hours=3 * c), require_free_bytes=1)
        record = self.coverage(paths, SCHEDULED + timedelta(days=1))
        assert record.state is CoverageState.MEASURED
        assert (record.successful_days, record.elapsed_days, record.attempted_cycles) == (1, 2, 3)
        assert record.coverage_fraction == pytest.approx(0.5)

    def test_a_cycle_whose_feed_failed_covers_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from market_intelligence.collection import syndication

        def retired(url: str, timeout: float, agent: str | None = None) -> bytes:
            raise ProviderFailure(FailureClass.PERMANENT, f"HTTP 404 for {url}")

        monkeypatch.setattr(syndication, "_default_opener", retired)
        paths = StoragePaths.from_environment(tmp_path / "s").ensure()
        collect_once(paths, CollectionProfile.from_mapping(minimal()), now=lambda: SCHEDULED, require_free_bytes=1)
        record = self.coverage(paths, SCHEDULED)
        assert record.state is CoverageState.MEASURED and record.failed_attempts >= 1
        assert (record.successful_days, record.coverage_fraction) == (0, 0.0)

    def test_a_cycle_that_could_not_take_the_lock_records_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        offline(monkeypatch)
        paths = StoragePaths.from_environment(tmp_path / "s").ensure()
        with RunLock(paths.lock, purpose="another collector", now=lambda: SCHEDULED):
            outcome = collect_once(
                paths, CollectionProfile.from_mapping(minimal()), now=lambda: SCHEDULED, require_free_bytes=1
            )
        assert outcome.exit_code == EXIT_LOCK_HELD
        assert self.coverage(paths, SCHEDULED).state is CoverageState.NO_COLLECTION
