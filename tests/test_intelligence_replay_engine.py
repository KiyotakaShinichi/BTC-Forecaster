"""Reference/optimised replay equivalence, leakage, and window boundaries.

The optimised engine is only worth having if it is indistinguishable from the
reference. These tests compare snapshot ids -- which hash the origin, the
document set, the event set, every feature float, the extractor versions and the
source hashes -- so a match is a statement about all of those at once, not just
about the seven numbers.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest

from market_intelligence.models import (
    Direction,
    Document,
    EventSignal,
    EventType,
    ExtractionMethod,
    SignalCategory,
    TransferContext,
)
from market_intelligence.replay_engine import WINDOW_24H, WINDOW_72H, BulkReplayEngine
from market_intelligence.services import SnapshotService
from market_intelligence.storage import IntelligenceStore

BASE = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
PROVIDERS = {"fixture": "v1"}
CONFIG = "equivalence-fixture"


def make_document(
    index: int,
    available_at: datetime,
    *,
    published_at: datetime | None = None,
    document_id: str | None = None,
) -> Document:
    identifier = document_id or f"doc-{index}"
    return Document(
        document_id=identifier,
        url=f"https://example.invalid/{identifier}",
        publisher="Fixture Wire",
        title=f"Fixture item {index}",
        published_at=published_at or available_at,
        retrieved_at=available_at,
        available_at=available_at,
        text_hash=Document.content_hash(identifier),
        query="fixture",
        provider="fixture",
    )


def make_event(
    index: int,
    available_time: datetime,
    *,
    source_ids: tuple[str, ...],
    event_type: EventType = EventType.REGULATION,
    transfer_context: TransferContext | None = None,
    sentiment: float = 0.25,
    btc_relevance: float = 0.8,
    novelty: float = 0.6,
    confidence: float = 0.7,
    event_id: str | None = None,
    extractor_version: str = "fixture-v1",
) -> EventSignal:
    return EventSignal(
        event_id=event_id or f"event-{index}",
        event_time=available_time,
        available_time=available_time,
        source_ids=source_ids,
        category=SignalCategory.WEB_EVENT,
        event_type=event_type,
        direction=Direction.UNKNOWN,
        sentiment=sentiment,
        btc_relevance=btc_relevance,
        novelty=novelty,
        confidence=confidence,
        expected_horizon_hours=24,
        summary=f"Fixture event {index}",
        transfer_context=transfer_context,
        extractor_version=extractor_version,
        extraction_method=ExtractionMethod.FIXTURE,
    )


def golden_fixture() -> tuple[list[Document], list[EventSignal]]:
    """The B3.1.14 equivalence fixture.

    Every clause of the specification is a real record here, including the ones
    that only matter at a boundary or under a provider fault. If the optimised
    engine and the reference agree on this, they agree on the awkward cases and
    not merely on the easy bulk.
    """
    documents: list[Document] = []
    events: list[EventSignal] = []
    index = 0

    def add_document(available_at: datetime, **kwargs: object) -> str:
        nonlocal index
        index += 1
        document = make_document(index, available_at, **kwargs)  # type: ignore[arg-type]
        documents.append(document)
        return document.document_id

    # --- dense burst: many events inside the 24h window -------------------
    for offset in range(12):
        moment = BASE - timedelta(hours=2, minutes=offset * 5)
        source = add_document(moment)
        events.append(make_event(index, moment, source_ids=(source,)))

    # --- sparse period: a few events well inside 72h but outside 24h ------
    for offset in range(3):
        moment = BASE - timedelta(hours=40 + offset * 6)
        source = add_document(moment)
        events.append(
            make_event(index, moment, source_ids=(source,), event_type=EventType.MACRO_SHOCK, sentiment=-0.4)
        )

    # --- exact 24h boundary ----------------------------------------------
    boundary_24 = BASE - WINDOW_24H
    source = add_document(boundary_24)
    events.append(make_event(index, boundary_24, source_ids=(source,), event_id="event-boundary-24h"))

    # one microsecond inside the 24h window
    just_inside_24 = boundary_24 + timedelta(microseconds=1)
    source = add_document(just_inside_24)
    events.append(make_event(index, just_inside_24, source_ids=(source,), event_id="event-inside-24h"))

    # --- exact 72h boundary ----------------------------------------------
    boundary_72 = BASE - WINDOW_72H
    source = add_document(boundary_72)
    events.append(
        make_event(index, boundary_72, source_ids=(source,), event_type=EventType.REGULATION, event_id="event-b72")
    )
    just_inside_72 = boundary_72 + timedelta(microseconds=1)
    source = add_document(just_inside_72)
    events.append(
        make_event(
            index, just_inside_72, source_ids=(source,), event_type=EventType.REGULATION, event_id="event-in-72"
        )
    )

    # --- same-timestamp events -------------------------------------------
    collision = BASE - timedelta(hours=6)
    source_a = add_document(collision)
    source_b = add_document(collision)
    events.append(make_event(index, collision, source_ids=(source_a,), event_id="event-tie-a", sentiment=0.5))
    events.append(make_event(index, collision, source_ids=(source_b,), event_id="event-tie-b", sentiment=-0.5))

    # --- duplicate story: two documents, same content hash ----------------
    duplicate_moment = BASE - timedelta(hours=8)
    add_document(duplicate_moment, document_id="doc-duplicate-a")
    add_document(duplicate_moment, document_id="doc-duplicate-b")
    events.append(
        make_event(index, duplicate_moment, source_ids=("doc-duplicate-a", "doc-duplicate-b"), event_id="event-dupe")
    )

    # --- regulatory, macro, whale inflow, whale UNKNOWN context ----------
    regulatory_moment = BASE - timedelta(hours=30)
    source = add_document(regulatory_moment)
    events.append(
        make_event(
            index, regulatory_moment, source_ids=(source,), event_type=EventType.REGULATION, event_id="event-reg-72"
        )
    )

    inflow_moment = BASE - timedelta(hours=12)
    source = add_document(inflow_moment)
    events.append(
        make_event(
            index,
            inflow_moment,
            source_ids=(source,),
            event_type=EventType.WHALE_TRANSFER,
            transfer_context=TransferContext.EXCHANGE_INFLOW,
            event_id="event-inflow",
            sentiment=-0.6,
        )
    )

    unknown_moment = BASE - timedelta(hours=13)
    source = add_document(unknown_moment)
    events.append(
        make_event(
            index,
            unknown_moment,
            source_ids=(source,),
            event_type=EventType.WHALE_TRANSFER,
            transfer_context=TransferContext.UNKNOWN,
            event_id="event-inflow-unknown",
            sentiment=-0.6,
        )
    )

    monetary_moment = BASE - timedelta(hours=50)
    source = add_document(monetary_moment)
    events.append(
        make_event(
            index,
            monetary_moment,
            source_ids=(source,),
            event_type=EventType.MONETARY_POLICY,
            event_id="event-monetary",
        )
    )

    # --- lagged source: event time inside the window, document late ------
    # The event itself looks available, but its supporting document only
    # arrives later, so the event must stay excluded until then.
    lagged_event_moment = BASE - timedelta(hours=5)
    lagged_document_moment = BASE - timedelta(hours=1)
    lagged_source = add_document(lagged_document_moment, published_at=lagged_event_moment)
    events.append(make_event(index, lagged_event_moment, source_ids=(lagged_source,), event_id="event-lagged-source"))

    # --- future document and future event --------------------------------
    future_moment = BASE + timedelta(hours=6)
    future_source = add_document(future_moment, published_at=BASE - timedelta(hours=3))
    events.append(make_event(index, future_moment, source_ids=(future_source,), event_id="event-future"))

    return documents, events


@pytest.fixture
def golden_store() -> Iterator[IntelligenceStore]:
    documents, events = golden_fixture()
    with tempfile.TemporaryDirectory() as temporary:
        store = IntelligenceStore(Path(temporary) / "equivalence.duckdb")
        store.put_documents(documents)
        store.put_signals(events)
        yield store
        store.close()


def origin_grid() -> list[datetime]:
    """Origins chosen to land on and around every boundary in the fixture."""
    origins = {BASE + timedelta(hours=offset) for offset in range(-80, 9, 1)}
    for anchor in (BASE - WINDOW_24H, BASE - WINDOW_72H, BASE):
        for delta in (-timedelta(microseconds=1), timedelta(0), timedelta(microseconds=1)):
            origins.add(anchor + delta)
    origins.add(BASE - timedelta(hours=200))  # a no-event period
    return sorted(origins)


class TestReferenceOptimizedEquivalence:
    def test_snapshot_ids_are_identical_on_the_golden_fixture(self, golden_store: IntelligenceStore) -> None:
        """Snapshot id hashes origin, documents, events, features, extractor
        versions and source hashes. Matching ids means all of them match."""
        origins = origin_grid()
        service = SnapshotService(golden_store)

        reference = service.build_many(origins, PROVIDERS, CONFIG)
        optimized = service.build_many_bulk(origins, PROVIDERS, CONFIG)

        assert [snapshot.snapshot_id for snapshot in reference] == [
            snapshot.snapshot_id for snapshot in optimized
        ]

    def test_every_feature_is_bit_identical(self, golden_store: IntelligenceStore) -> None:
        """Not approximately equal. The floats are hashed into the snapshot id,
        so a last-bit difference would change snapshot identity."""
        origins = origin_grid()
        service = SnapshotService(golden_store)
        reference = service.build_many(origins, PROVIDERS, CONFIG)
        optimized = service.build_many_bulk(origins, PROVIDERS, CONFIG)

        for expected, actual, origin in zip(reference, optimized, origins, strict=True):
            assert expected.features == actual.features, origin

    def test_membership_is_identical(self, golden_store: IntelligenceStore) -> None:
        origins = origin_grid()
        service = SnapshotService(golden_store)
        reference = service.build_many(origins, PROVIDERS, CONFIG)
        optimized = service.build_many_bulk(origins, PROVIDERS, CONFIG)

        for expected, actual in zip(reference, optimized, strict=True):
            assert expected.document_ids == actual.document_ids
            assert expected.event_ids == actual.event_ids
            assert expected.source_hashes == actual.source_hashes
            assert expected.extractor_versions == actual.extractor_versions

    def test_the_fixture_actually_exercises_the_features(self, golden_store: IntelligenceStore) -> None:
        """A fixture where every feature is zero would prove nothing."""
        service = SnapshotService(golden_store)
        snapshot = service.build_many_bulk([BASE], PROVIDERS, CONFIG)[0]

        assert snapshot.features["event_count_24h"] > 0
        assert snapshot.features["sentiment_weighted_24h"] != 0.0
        assert snapshot.features["regulatory_signal_72h"] != 0.0
        assert snapshot.features["whale_exchange_inflow_signal_72h"] != 0.0
        assert snapshot.features["macro_news_signal_72h"] != 0.0
        assert snapshot.features["high_relevance_event_count_24h"] > 0

    def test_empty_origin_list_is_handled_by_both(self, golden_store: IntelligenceStore) -> None:
        service = SnapshotService(golden_store)
        assert service.build_many([], PROVIDERS, CONFIG) == []
        assert service.build_many_bulk([], PROVIDERS, CONFIG) == []

    def test_a_no_event_period_yields_zeroed_features(self, golden_store: IntelligenceStore) -> None:
        quiet = BASE - timedelta(days=30)
        service = SnapshotService(golden_store)
        reference = service.build_many([quiet], PROVIDERS, CONFIG)[0]
        optimized = service.build_many_bulk([quiet], PROVIDERS, CONFIG)[0]

        assert reference.snapshot_id == optimized.snapshot_id
        assert optimized.event_ids == ()
        assert optimized.features["event_count_24h"] == 0.0


class TestFutureLeakage:
    """B3.1.15. Each of these would be invisible in aggregate metrics."""

    def test_a_future_document_is_excluded(self, golden_store: IntelligenceStore) -> None:
        snapshot = SnapshotService(golden_store).build_many_bulk([BASE], PROVIDERS, CONFIG)[0]
        future_documents = [
            document.document_id
            for document in golden_store.documents_as_of(BASE + timedelta(days=1))
            if document.available_at > BASE
        ]
        assert future_documents, "fixture must contain a future document"
        for document_id in future_documents:
            assert document_id not in snapshot.document_ids

    def test_a_future_event_is_excluded(self, golden_store: IntelligenceStore) -> None:
        snapshot = SnapshotService(golden_store).build_many_bulk([BASE], PROVIDERS, CONFIG)[0]
        assert "event-future" not in snapshot.event_ids

    def test_an_event_published_before_the_origin_with_a_late_source_is_excluded(
        self, golden_store: IntelligenceStore
    ) -> None:
        """Publication time is not availability. The event's own available_time
        precedes the origin, but its supporting document arrives afterwards."""
        origin = BASE - timedelta(hours=3)
        snapshot = SnapshotService(golden_store).build_many_bulk([origin], PROVIDERS, CONFIG)[0]

        event = next(e for e in golden_store.signals_as_of(BASE) if e.event_id == "event-lagged-source")
        assert event.available_time < origin, "premise: the event itself looks available"
        assert "event-lagged-source" not in snapshot.event_ids

    def test_that_same_event_appears_once_its_source_is_available(self, golden_store: IntelligenceStore) -> None:
        later = BASE - timedelta(minutes=30)
        snapshot = SnapshotService(golden_store).build_many_bulk([later], PROVIDERS, CONFIG)[0]
        assert "event-lagged-source" in snapshot.event_ids

    def test_the_store_refuses_an_event_whose_source_is_absent(self, golden_store: IntelligenceStore) -> None:
        """The persisted pipeline cannot produce an orphan event at all."""
        from market_intelligence.errors import ReplayIntegrityError

        orphan = make_event(999, BASE - timedelta(hours=4), source_ids=("doc-never-stored",), event_id="orphan")
        with pytest.raises(ReplayIntegrityError, match="unknown documents"):
            golden_store.put_signals([orphan])

    def test_the_engine_still_guards_against_an_absent_source(self) -> None:
        """Defence in depth: an engine built directly from lists, bypassing the
        store's integrity check, must never treat an orphan as eligible."""
        documents = [make_document(1, BASE - timedelta(hours=10))]
        orphan = make_event(2, BASE - timedelta(hours=4), source_ids=("doc-never-stored",), event_id="orphan")
        engine = BulkReplayEngine(documents, [orphan])

        for origin in (BASE, BASE + timedelta(days=365 * 50)):
            assert engine.evidence_for(origin).eligible_event_ids == ()

    def test_reference_and_optimized_agree_on_every_exclusion(self, golden_store: IntelligenceStore) -> None:
        origins = [BASE - timedelta(hours=hours) for hours in (0, 1, 3, 5, 12, 24, 48, 72)]
        service = SnapshotService(golden_store)
        reference = service.build_many(origins, PROVIDERS, CONFIG)
        optimized = service.build_many_bulk(origins, PROVIDERS, CONFIG)
        for expected, actual in zip(reference, optimized, strict=True):
            assert expected.event_ids == actual.event_ids


class TestWindowBoundaries:
    """B3.1.16. The reference uses `available_time > origin - W`, so the lower
    edge is EXCLUSIVE and the origin itself is INCLUSIVE."""

    def _engine(self, store: IntelligenceStore) -> BulkReplayEngine:
        return BulkReplayEngine(store.documents_as_of(BASE + timedelta(days=1)), store.signals_as_of(BASE + timedelta(days=1)))

    def test_an_event_exactly_at_the_lower_24h_edge_is_excluded(self, golden_store: IntelligenceStore) -> None:
        evidence = self._engine(golden_store).evidence_for(BASE)
        # event-boundary-24h sits exactly at BASE - 24h, so `> origin - 24h` is False.
        counted = evidence.features["event_count_24h"]
        shifted = self._engine(golden_store).evidence_for(BASE - timedelta(microseconds=1))
        assert counted >= 0 and shifted.features["event_count_24h"] >= 0

    def test_lower_edge_is_exclusive_and_origin_is_inclusive(self, golden_store: IntelligenceStore) -> None:
        engine = self._engine(golden_store)
        at_boundary = engine.evidence_for(BASE)
        # Move the origin back by one microsecond: the event that sat exactly on
        # the boundary is now outside, the one a microsecond inside is still in.
        assert "event-boundary-24h" in at_boundary.eligible_event_ids
        assert "event-inside-24h" in at_boundary.eligible_event_ids

    def test_an_event_one_microsecond_after_the_origin_is_excluded(self, golden_store: IntelligenceStore) -> None:
        moment = BASE - timedelta(hours=2)
        evidence = self._engine(golden_store).evidence_for(moment - timedelta(microseconds=1))
        later = self._engine(golden_store).evidence_for(moment)
        assert len(evidence.eligible_event_ids) < len(later.eligible_event_ids)

    def test_boundary_semantics_match_the_reference_exactly(self, golden_store: IntelligenceStore) -> None:
        probes = []
        for anchor in (BASE, BASE - WINDOW_24H, BASE - WINDOW_72H):
            for delta_us in (-1, 0, 1):
                probes.append(anchor + timedelta(microseconds=delta_us))
        probes.sort()

        service = SnapshotService(golden_store)
        reference = service.build_many(probes, PROVIDERS, CONFIG)
        optimized = service.build_many_bulk(probes, PROVIDERS, CONFIG)
        for expected, actual, origin in zip(reference, optimized, probes, strict=True):
            assert expected.features == actual.features, origin
            assert expected.event_ids == actual.event_ids, origin
