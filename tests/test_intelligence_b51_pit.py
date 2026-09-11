"""B5.1 — point-in-time timing stays strict, and collection lag stays descriptive.

    collection_lag = first_seen_at - published_at

B5 reported per-event `retrieval_lag_hours` but nothing measured how stale the
collector's view of each document was. This adds that measure, and pins what it
must never become: a reason to move a timestamp.

Pinned here:

* the lag is first sighting minus publication, falls back to first retrieval
  when there is no sighting, and does not move on rediscovery;
* a missing publication time is counted and never guessed; a publication after
  first sight is reported as impossible and never clipped;
* the session's time zone does not change it;
* computing it writes nothing, and extraction timing ignores it: an event's
  event_time is its publication and its available_time its first retrieval,
  however late that was;
* the frozen timing rules hold: publication is the event time where there is
  one, a re-crawl with an earlier clock cannot move availability earlier,
  later sightings cannot qualify an event at an earlier origin, a correction
  never rewrites availability, and a provider's delay stays visible;
* every point-in-time query admits exactly what was available at or before the
  origin -- inclusive at the origin, nothing a microsecond after;
* corpus-status and the Gate 1 audit report the same lag, from one function.

No network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from market_intelligence.b5.audit import audit_corpus, open_read_only
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.collection.lag import LAG_CONTRACT_VERSION, LagBasis, document_lag, lag_summary, summarize
from market_intelligence.corrections import CorrectionStatus, EventCorrection
from market_intelligence.extractors import CURRENT_RULE_EXTRACTOR_VERSION, EvidenceRuleExtractor
from market_intelligence.models import Document, EventType
from market_intelligence.reports import corpus_status
from market_intelligence.storage import IntelligenceStore, queries

PUBLISHED = datetime(2026, 9, 1, 14, 57, tzinfo=timezone.utc)
SEEN = datetime(2026, 9, 2, 8, 39, 22, tzinfo=timezone.utc)
LAG_HOURS = (SEEN - PUBLISHED).total_seconds() / 3600.0
TICK = timedelta(microseconds=1)
EXTRACTOR = EvidenceRuleExtractor({"SEC": ("Securities and Exchange Commission",)}, {"SEC": (EventType.REGULATION,)})


def sec(index: int = 0, *, published: datetime | None = PUBLISHED, seen: datetime = SEEN) -> Document:
    return fixture_document(
        url=f"https://sec.example.gov/{index}",
        title=f"SEC Charges Adviser Number {index}",
        retrieved_at=seen,
        publisher="U.S. Securities and Exchange Commission",
        published_at=published,
        primary_source=True,
        official_source=True,
    )


@pytest.fixture
def store(tmp_path: Path):
    handle = IntelligenceStore(tmp_path / "corpus.duckdb")
    yield handle
    handle.close()


class TestTheLag:
    def test_it_is_first_sighting_minus_publication(self, store: IntelligenceStore) -> None:
        document = sec()
        store.put_documents([document])
        summary = lag_summary(store.connection, [document])
        assert summary.median_hours == pytest.approx(LAG_HOURS, abs=1e-6)
        assert summary.basis == {LagBasis.FIRST_SEEN.value: 1}
        assert summary.contract == LAG_CONTRACT_VERSION

    def test_with_no_sighting_it_falls_back_to_first_retrieval(self) -> None:
        lag = document_lag(sec(), None)
        assert lag.basis is LagBasis.AVAILABLE_AT and lag.first_seen_at == SEEN
        assert lag.lag_hours == pytest.approx(LAG_HOURS, abs=1e-6)

    def test_rediscovery_does_not_move_it(self, store: IntelligenceStore) -> None:
        document = sec()
        store.put_documents([document])
        store.put_documents([document.model_copy(update={"retrieved_at": SEEN + timedelta(days=2)})])
        later = store.documents_as_of(SEEN + timedelta(days=3))
        assert lag_summary(store.connection, later).max_hours == pytest.approx(LAG_HOURS, abs=1e-6)

    def test_a_missing_publication_time_is_counted_never_guessed(self, store: IntelligenceStore) -> None:
        document = sec(published=None)
        store.put_documents([document])
        summary = lag_summary(store.connection, [document])
        assert (summary.documents, summary.measured, summary.without_publication_time) == (1, 0, 1)
        assert summary.median_hours is None
        assert "carries a publication time" in summary.describe()

    def test_an_impossible_timestamp_is_reported_not_clipped(self) -> None:
        lag = document_lag(sec(published=SEEN + timedelta(hours=2)), None)
        assert lag.lag_hours == pytest.approx(-2.0) and lag.impossible
        summary = summarize([lag, document_lag(sec(1), None)])
        assert summary.impossible == 1 and summary.min_hours == pytest.approx(-2.0)

    def test_an_empty_corpus_is_not_measurable(self) -> None:
        summary = summarize([])
        assert summary.documents == 0 and summary.median_hours is None and summary.p90_hours is None
        assert "not yet measurable" in summary.describe()

    def test_percentiles_are_values_that_occurred(self) -> None:
        lags = [document_lag(sec(i, published=SEEN - timedelta(hours=i)), None) for i in range(1, 11)]
        summary = summarize(lags)
        assert summary.median_hours == pytest.approx(5.5)
        assert summary.p90_hours == pytest.approx(9.0) and summary.max_hours == pytest.approx(10.0)

    def test_the_session_zone_does_not_change_it(self, store: IntelligenceStore) -> None:
        document = sec()
        store.put_documents([document])
        before = lag_summary(store.connection, [document])
        try:
            store.connection.execute("SET TimeZone = 'Asia/Manila'")
        except duckdb.Error as error:  # pragma: no cover - a build without time zone support
            pytest.skip(f"this DuckDB build cannot set a session time zone: {error}")
        assert lag_summary(store.connection, [document]) == before


def table_contents(store: IntelligenceStore) -> dict[str, list[tuple]]:
    return {
        table: sorted(map(tuple, store.connection.execute(f"SELECT * FROM {table}").fetchall()), key=repr)
        for table in ("documents", "document_sightings", "signals")
    }


class TestItRedefinesNothing:
    def test_computing_it_writes_nothing(self, store: IntelligenceStore) -> None:
        documents = [sec(0), sec(1, seen=SEEN + timedelta(days=3))]
        store.put_documents(documents)
        store.put_signals(EXTRACTOR.extract(documents))
        before = table_contents(store)
        lag_summary(store.connection, documents)
        assert table_contents(store) == before

    def test_extraction_timing_ignores_how_late_a_document_was_seen(self) -> None:
        late = sec(published=PUBLISHED, seen=PUBLISHED + timedelta(days=30))
        (event,) = EXTRACTOR.extract([late])
        assert event.event_time == PUBLISHED
        assert event.available_time == late.available_at == PUBLISHED + timedelta(days=30)

    def test_a_late_seen_document_is_not_back_dated_to_its_publication(self, store: IntelligenceStore) -> None:
        late = sec(seen=PUBLISHED + timedelta(days=30))
        store.put_documents([late])
        store.put_signals(EXTRACTOR.extract([late]))
        assert store.documents_as_of(PUBLISHED + timedelta(days=1)) == []
        assert store.signals_as_of(PUBLISHED + timedelta(days=1)) == []

    def test_every_point_in_time_query_is_inclusive_at_the_origin_and_no_further(
        self, store: IntelligenceStore
    ) -> None:
        document = sec()
        store.put_documents([document])
        store.put_signals(EXTRACTOR.extract([document]))
        origin = document.available_at
        for query in (store.documents_as_of, store.signals_as_of, store.eligible_signals_as_of):
            assert len(query(origin)) == 1, f"{query.__name__} excluded what was available at the origin"
            assert query(origin - TICK) == [], f"{query.__name__} admitted what was not yet available"
        assert len(queries.signals_as_of(store.connection, origin)) == 1
        assert queries.signals_as_of(store.connection, origin - TICK) == []


class TestTheFrozenTimingRules:
    """deploy/COLLECTION_FREEZE.md, sections 1 and 2, as the lag must leave them."""

    def test_publication_is_the_event_time_where_there_is_one(self) -> None:
        dated, undated = sec(0), sec(1, published=None)
        (a,) = EXTRACTOR.extract([dated])
        (b,) = EXTRACTOR.extract([undated])
        assert a.event_time == PUBLISHED
        assert b.event_time == undated.available_at == SEEN
        assert a.available_time == b.available_time == SEEN

    def test_a_recrawl_with_an_earlier_clock_cannot_move_availability_earlier(self, store: IntelligenceStore) -> None:
        document = sec()
        store.put_documents([document])
        skewed = SEEN - timedelta(hours=6)
        earlier = document.model_copy(update={"retrieved_at": skewed, "available_at": skewed})
        store.put_documents([earlier])
        (stored,) = store.documents_as_of(SEEN)
        assert stored.available_at == SEEN
        assert store.documents_as_of(SEEN - TICK) == []
        (canonical,) = store.canonical_availability([earlier])
        (event,) = EXTRACTOR.extract([canonical])
        assert event.available_time == SEEN

    def test_later_sightings_cannot_qualify_an_event_at_an_earlier_origin(self, store: IntelligenceStore) -> None:
        first_seen = SEEN + timedelta(days=1)
        document = sec(seen=first_seen)
        store.put_documents([document])
        store.put_documents([document.model_copy(update={"retrieved_at": first_seen + timedelta(days=4)})])
        store.put_signals(EXTRACTOR.extract(store.canonical_availability([document])))
        origin = SEEN  # after publication, before the collector first saw it
        assert store.documents_as_of(origin) == []
        assert store.signals_as_of(origin) == [] and store.eligible_signals_as_of(origin) == []
        assert len(store.signals_as_of(first_seen)) == 1

    def test_a_correction_never_rewrites_availability(self, store: IntelligenceStore) -> None:
        document = sec()
        store.put_documents([document])
        (event,) = EXTRACTOR.extract([document])
        store.put_signals([event])
        store.put_corrections(
            [
                EventCorrection(
                    event_id=event.event_id,
                    status=CorrectionStatus.INVALIDATED,
                    reason="FIXTURE",
                    invalidated_at=SEEN + timedelta(days=10),
                    invalidated_by_version="fixture",
                    source_bug="fixture",
                )
            ]
        )
        assert store.signals_as_of(SEEN) == [event], "the observation moved or changed"
        assert store.documents_as_of(SEEN)[0].available_at == SEEN
        assert store.eligible_signals_as_of(SEEN) == []

    def test_a_provider_delay_stays_visible(self, store: IntelligenceStore) -> None:
        delayed = sec(seen=PUBLISHED + timedelta(hours=40))
        store.put_documents([delayed])
        (event,) = EXTRACTOR.extract([delayed])
        assert event.available_time - event.event_time == timedelta(hours=40)
        assert lag_summary(store.connection, [delayed]).max_hours == pytest.approx(40.0)


class TestOnePlaceComputesIt:
    def test_corpus_status_and_the_gate_1_audit_report_the_same_lag(self, tmp_path: Path) -> None:
        path = tmp_path / "c.duckdb"
        documents = [sec(0), sec(1, published=None), sec(2, seen=SEEN + timedelta(days=7))]
        store = IntelligenceStore(path)
        try:
            store.put_documents(documents)
            store.put_signals(EXTRACTOR.extract(documents))
            status = corpus_status(store, CURRENT_RULE_EXTRACTOR_VERSION, ["SEC"])
        finally:
            store.close()
        connection = open_read_only(path)
        try:
            result = audit_corpus(connection, as_of=datetime.now(timezone.utc))
        finally:
            connection.close()
        assert status.collection_lag is not None and status.collection_lag == result.audit.collection_lag
        assert (status.collection_lag.documents, status.collection_lag.measured) == (3, 2)
        assert "first seen - published" in status.human_readable()
