"""B4.1.14 – B4.1.39 — clustering, corpus, readiness, and time adversaries.

The adversarial-time class is the one that matters. Every case in it is a way a
plausible implementation fabricates historical availability, and each is
constructed so that getting it wrong produces confident, wrong evidence rather
than an error.

No network.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from market_intelligence.collection.clustering import (
    ClusterStore,
    cluster_events,
    effective_non_overlapping,
    write_clusters,
)
from market_intelligence.collection.corpus import (
    CorpusCatalog,
    build_snapshot,
    membership_hash,
)
from market_intelligence.collection.fixtures import (
    FIXTURE_BASE,
    AuthFailingProvider,
    EmptyProvider,
    OutageProvider,
    RateLimitedProvider,
    SchemaInvalidProvider,
    StaticFixtureProvider,
    cross_provider_duplicate,
    distinct_primary_sources,
    fixture_document,
    future_timestamped_document,
    late_source,
    primary_and_secondary,
)
from market_intelligence.collection.readiness import (
    AdequacyPolicy,
    Readiness,
    assess_entities,
    assess_family,
    assess_sentiment,
    assess_whale_contexts,
    coverage_fraction,
    days_until_ready,
    overall_readiness,
)
from market_intelligence.collection.service import cadence_due, redacted_configuration
from market_intelligence.collection.status import build_status, daily_coverage, measure_storage
from market_intelligence.errors import ReplayIntegrityError
from market_intelligence.extractors import RuleBasedExtractor
from market_intelligence.models import (
    Direction,
    Document,
    EventSignal,
    EventType,
    SignalCategory,
    TransferContext,
)
from market_intelligence.retrieval import deduplicate_across_providers
from market_intelligence.storage import IntelligenceStore

NOW = FIXTURE_BASE


def event(
    index: int,
    *,
    available: datetime,
    event_type: EventType = EventType.REGULATION,
    entity: str | None = "SEC",
    sources: tuple[str, ...] = ("doc-1",),
    version: str = "rules-v1",
    context: TransferContext | None = None,
    relevance: float = 0.9,
) -> EventSignal:
    return EventSignal(
        event_id=f"event-{index}",
        event_time=available - timedelta(minutes=5),
        available_time=available,
        source_ids=sources,
        category=SignalCategory.WEB_EVENT,
        entity=entity,
        event_type=event_type,
        sentiment=0.1,
        btc_relevance=relevance,
        novelty=0.8,
        confidence=0.8,
        expected_horizon_hours=24,
        summary=f"event {index}",
        transfer_context=context,
        direction=Direction.UNKNOWN,
        extractor_version=version,
    )


def document(index: int, *, retrieved: datetime, primary: bool = False, publisher: str = "a.example.com"):
    return fixture_document(
        url=f"https://{publisher}/{index}",
        title=f"Bitcoin item {index}",
        retrieved_at=retrieved,
        publisher=publisher,
        primary_source=primary,
        official_source=primary,
        body=f"body {index}",
    )


# ------------------------------------------------------------------ time


class TestAdversarialTime:
    """B4.1.39. Every case where a plausible implementation fabricates history."""

    def test_an_old_article_retrieved_today_is_available_today(self) -> None:
        stale = late_source(NOW)
        assert stale.published_at is not None
        assert stale.published_at < NOW - timedelta(days=20)
        assert stale.available_at == NOW, "publication date is not availability"

    def test_a_future_published_document_is_still_available_at_retrieval(self) -> None:
        """A publisher's embargo stamp must not push availability forward either."""
        embargoed = future_timestamped_document(NOW)
        assert embargoed.published_at is not None and embargoed.published_at > NOW
        assert embargoed.available_at == NOW

    def test_availability_after_retrieval_is_rejected_by_the_contract(self) -> None:
        with pytest.raises(Exception, match="available_at cannot be after retrieved_at"):
            Document(
                document_id="x",
                url="https://example.com/a",
                publisher="example.com",
                title="t",
                retrieved_at=NOW,
                available_at=NOW + timedelta(seconds=1),
                text_hash="h" * 64,
                query="bitcoin",
                provider="fixture",
            )

    def test_the_same_article_rediscovered_keeps_its_first_availability(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            first = document(1, retrieved=NOW)
            store.put_documents([first])
            store.put_documents([document(1, retrieved=NOW + timedelta(days=7))])
            assert store.documents_as_of(NOW + timedelta(days=8))[0].available_at == NOW
        finally:
            store.close()

    def test_a_provider_timestamp_correction_does_not_move_availability(self, tmp_path: Path) -> None:
        """A feed re-publishing an item with a corrected date is still the same
        evidence, first seen when it was first seen."""
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            original = document(2, retrieved=NOW)
            store.put_documents([original])
            corrected = original.model_copy(
                update={
                    "published_at": NOW - timedelta(days=1),
                    "retrieved_at": NOW + timedelta(hours=6),
                    "available_at": NOW + timedelta(hours=6),
                }
            )
            store.put_documents([corrected])
            stored = store.documents_as_of(NOW + timedelta(days=1))[0]
            assert stored.available_at == NOW
        finally:
            store.close()

    def test_a_late_source_document_cannot_backdate_its_event(self) -> None:
        """An event is available no earlier than the document it came from."""
        source = late_source(NOW)
        extracted = RuleBasedExtractor().extract([source])
        for signal in extracted:
            assert signal.available_time >= source.available_at

    def test_events_are_never_available_before_their_documents(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            source = document(3, retrieved=NOW)
            store.put_documents([source])
            store.put_signals([event(1, available=NOW, sources=(source.document_id,))])
            at_origin = NOW - timedelta(seconds=1)
            assert store.documents_as_of(at_origin) == []
            assert store.signals_as_of(at_origin) == []
        finally:
            store.close()


# ------------------------------------------------------------------ dedup


class TestDeduplication:
    def test_one_story_from_two_providers_collapses_to_one_document(self) -> None:
        merged = deduplicate_across_providers(cross_provider_duplicate(NOW))
        assert len(merged) == 1

    def test_dedup_keeps_the_earliest_sighting(self) -> None:
        merged = deduplicate_across_providers(cross_provider_duplicate(NOW))
        assert merged[0].retrieved_at == NOW

    def test_two_distinct_primary_sources_are_never_merged(self) -> None:
        """B4.1.13. Identical headlines from different regulators are two events."""
        merged = deduplicate_across_providers(distinct_primary_sources(NOW))
        assert len(merged) == 2
        assert {item.publisher for item in merged} == {"sec.example.gov", "cftc.example.gov"}


# ------------------------------------------------------------- clustering


class TestClustering:
    def test_forty_articles_about_one_decision_are_one_cluster(self) -> None:
        documents = [document(index, retrieved=NOW) for index in range(40)]
        events = [
            event(index, available=NOW + timedelta(minutes=3 * index), sources=(documents[index].document_id,))
            for index in range(40)
        ]
        clusters = cluster_events(events, documents, window_hours=24)
        assert len(clusters) == 1
        assert clusters[0].document_count == 40
        assert clusters[0].publisher_count == 1

    def test_clustering_is_independent_of_arrival_order(self) -> None:
        documents = [document(index, retrieved=NOW) for index in range(6)]
        events = [
            event(index, available=NOW + timedelta(hours=index * 30), sources=(documents[index].document_id,))
            for index in range(6)
        ]
        forward = [cluster.cluster_id for cluster in cluster_events(events, documents)]
        backward = [cluster.cluster_id for cluster in cluster_events(list(reversed(events)), documents)]
        assert forward == backward

    def test_different_entities_never_share_a_cluster(self) -> None:
        documents = [document(0, retrieved=NOW), document(1, retrieved=NOW)]
        events = [
            event(0, available=NOW, sources=(documents[0].document_id,), entity="SEC"),
            event(1, available=NOW, sources=(documents[1].document_id,), entity="CFTC"),
        ]
        assert len(cluster_events(events, documents)) == 2

    def test_a_cluster_records_that_a_primary_source_is_present(self) -> None:
        """B4.1.7. An announcement inside a cluster must remain findable."""
        official, coverage = primary_and_secondary(NOW)
        events = [
            event(0, available=NOW, sources=(official.document_id,)),
            event(1, available=NOW + timedelta(minutes=40), sources=(coverage.document_id,)),
        ]
        cluster = cluster_events(events, [official, coverage])[0]
        assert cluster.primary_source_present
        assert cluster.primary_document_ids == (official.document_id,)
        assert cluster.publisher_count == 2
        assert cluster.provider_count == 2

    def test_a_cluster_with_no_primary_source_says_so(self) -> None:
        documents = [document(0, retrieved=NOW), document(1, retrieved=NOW)]
        events = [event(index, available=NOW, sources=(documents[index].document_id,)) for index in range(2)]
        assert not cluster_events(events, documents)[0].primary_source_present

    def test_effective_count_falls_below_the_raw_count_when_windows_overlap(self) -> None:
        documents = [document(index, retrieved=NOW) for index in range(10)]
        events = [
            event(
                index,
                available=NOW + timedelta(hours=index),
                sources=(documents[index].document_id,),
                entity=f"entity-{index}",
            )
            for index in range(10)
        ]
        clusters = cluster_events(events, documents)
        assert len(clusters) == 10
        assert effective_non_overlapping(clusters, horizon_hours=168) == 1
        assert effective_non_overlapping(clusters, horizon_hours=1) == 10

    def test_clusters_round_trip_through_the_store(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            documents = [document(index, retrieved=NOW) for index in range(3)]
            events = [
                event(index, available=NOW + timedelta(hours=index * 40), sources=(documents[index].document_id,))
                for index in range(3)
            ]
            clusters = cluster_events(events, documents)
            cluster_store = ClusterStore(store.connection)
            assert cluster_store.replace_all(clusters) == len(clusters)
            assert [item.cluster_id for item in cluster_store.all()] == [c.cluster_id for c in clusters]
            assert cluster_store.count() == len(clusters)
            assert cluster_store.counts_by_type() == {"REGULATION": len(clusters)}
            assert cluster_store.counts_by_entity() == {"SEC": len(clusters)}
        finally:
            store.close()

    def test_replacing_clusters_does_not_accumulate(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            documents = [document(0, retrieved=NOW)]
            events = [event(0, available=NOW, sources=(documents[0].document_id,))]
            cluster_store = ClusterStore(store.connection)
            cluster_store.replace_all(cluster_events(events, documents))
            cluster_store.replace_all(cluster_events(events, documents))
            assert cluster_store.count() == 1
        finally:
            store.close()

    def test_clusters_can_be_written_atomically(self, tmp_path: Path) -> None:
        documents = [document(0, retrieved=NOW)]
        events = [event(0, available=NOW, sources=(documents[0].document_id,))]
        path = write_clusters(cluster_events(events, documents), tmp_path / "nested" / "c.json")
        assert path.exists() and not list(path.parent.glob("*.tmp"))


# ----------------------------------------------------------------- corpus


class TestCorpusSnapshot:
    def _corpus(self) -> tuple[list[Document], list[EventSignal]]:
        documents = [document(index, retrieved=NOW + timedelta(hours=index)) for index in range(4)]
        events = [
            event(index, available=NOW + timedelta(hours=index), sources=(documents[index].document_id,))
            for index in range(4)
        ]
        return documents, events

    def test_a_snapshot_freezes_an_as_of_boundary(self) -> None:
        documents, events = self._corpus()
        snapshot = build_snapshot(
            documents,
            events,
            as_of=NOW + timedelta(hours=1),
            extractor_version="rules-v1",
            created_at=NOW,
        )
        assert snapshot.document_count == 2
        assert snapshot.event_count == 2

    def test_a_snapshot_freezes_one_extractor_version(self) -> None:
        """B4.1.16 / B4.1.27. A later extractor cannot change a published result."""
        documents, events = self._corpus()
        events.append(
            event(99, available=NOW, sources=(documents[0].document_id,), version="rules-v2")
        )
        snapshot = build_snapshot(
            documents, events, as_of=NOW + timedelta(days=1), extractor_version="rules-v1", created_at=NOW
        )
        assert snapshot.extractor_version == "rules-v1"
        assert snapshot.event_count == 4, "the v2 event is outside this snapshot"

    def test_an_event_whose_document_is_outside_the_boundary_is_excluded(self) -> None:
        documents, _ = self._corpus()
        orphaned = event(50, available=NOW, sources=("not-in-corpus",))
        snapshot = build_snapshot(
            documents, [orphaned], as_of=NOW + timedelta(days=1), extractor_version="rules-v1", created_at=NOW
        )
        assert snapshot.event_count == 0

    def test_the_same_request_produces_the_same_corpus_id(self) -> None:
        documents, events = self._corpus()
        arguments = {
            "as_of": NOW + timedelta(days=1),
            "extractor_version": "rules-v1",
            "created_at": NOW,
        }
        first = build_snapshot(documents, events, **arguments)  # type: ignore[arg-type]
        again = build_snapshot(list(reversed(documents)), list(reversed(events)), **arguments)  # type: ignore[arg-type]
        assert first.corpus_id == again.corpus_id
        assert first.membership_hash == again.membership_hash

    def test_a_different_membership_produces_a_different_id(self) -> None:
        documents, events = self._corpus()
        arguments = {"as_of": NOW + timedelta(days=1), "extractor_version": "rules-v1", "created_at": NOW}
        full = build_snapshot(documents, events, **arguments)  # type: ignore[arg-type]
        partial = build_snapshot(documents[:-1], events[:-1], **arguments)  # type: ignore[arg-type]
        assert full.corpus_id != partial.corpus_id

    def test_coverage_maps_are_populated(self) -> None:
        official, coverage = primary_and_secondary(NOW)
        events = [
            event(0, available=NOW, sources=(official.document_id,)),
            event(1, available=NOW, sources=(coverage.document_id,), event_type=EventType.ETF_FLOW),
        ]
        snapshot = build_snapshot(
            [official, coverage],
            events,
            as_of=NOW + timedelta(days=1),
            extractor_version="rules-v1",
            created_at=NOW,
        )
        assert snapshot.primary_source_documents == 1
        assert set(snapshot.event_type_coverage) == {"REGULATION", "ETF_FLOW"}
        assert set(snapshot.provider_coverage) == {"syndication", "news-api"}

    def test_membership_hash_is_order_independent_within_each_set(self) -> None:
        assert membership_hash(["b", "a"], ["y", "x"]) == membership_hash(["a", "b"], ["x", "y"])
        assert membership_hash(["a"], ["b"]) != membership_hash(["b"], ["a"])

    def test_a_naive_as_of_is_rejected(self) -> None:
        documents, events = self._corpus()
        with pytest.raises(ReplayIntegrityError, match="timezone-aware"):
            build_snapshot(
                documents,
                events,
                as_of=datetime(2026, 6, 2),
                extractor_version="rules-v1",
                created_at=NOW,
            )


class TestCorpusCatalog:
    def _snapshot(self, store: IntelligenceStore, hours: int = 24):
        documents = [document(index, retrieved=NOW + timedelta(hours=index)) for index in range(3)]
        events = [
            event(index, available=NOW + timedelta(hours=index), sources=(documents[index].document_id,))
            for index in range(3)
        ]
        return build_snapshot(
            documents,
            events,
            as_of=NOW + timedelta(hours=hours),
            extractor_version="rules-v1",
            created_at=NOW,
        )

    def test_a_snapshot_round_trips(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            catalog = CorpusCatalog(store.connection)
            snapshot = self._snapshot(store)
            catalog.register(snapshot)
            assert catalog.get(snapshot.corpus_id) is not None
            assert catalog.latest() is not None
        finally:
            store.close()

    def test_re_registering_identical_contents_is_a_no_op(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            catalog = CorpusCatalog(store.connection)
            snapshot = self._snapshot(store)
            catalog.register(snapshot)
            catalog.register(snapshot)
            assert len(catalog.list_snapshots()) == 1
        finally:
            store.close()

    def test_different_contents_under_one_id_are_refused(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            catalog = CorpusCatalog(store.connection)
            snapshot = self._snapshot(store)
            catalog.register(snapshot)
            tampered = snapshot.model_copy(update={"document_count": 999})
            with pytest.raises(ReplayIntegrityError, match="never overwritten"):
                catalog.register(tampered)
        finally:
            store.close()


# -------------------------------------------------------------- readiness


class TestReadiness:
    def _clusters(self, count: int, *, publishers: int = 3, span_days: int = 200):
        documents = [document(index, retrieved=NOW) for index in range(count)]
        step = timedelta(days=span_days / max(1, count - 1)) if count > 1 else timedelta()
        events = [
            event(index, available=NOW + step * index, sources=(documents[index].document_id,))
            for index in range(count)
        ]
        clusters = cluster_events(events, documents, window_hours=1)
        return [cluster.model_copy(update={"publisher_count": publishers}) for cluster in clusters]

    def test_an_empty_family_is_not_ready(self) -> None:
        result = assess_family("regulation", [])
        assert result.readiness is Readiness.NOT_READY
        assert not result.ready
        assert any("0 events" in reason for reason in result.unmet)

    def test_a_family_meeting_every_clause_is_ready(self) -> None:
        result = assess_family("regulation", self._clusters(40), coverage_fraction=0.95)
        assert result.readiness is Readiness.READY_FOR_VALIDATION
        assert result.unmet == ()

    def test_a_short_span_blocks_readiness_however_many_events(self) -> None:
        """A study spanning a month cannot separate an effect from the month."""
        result = assess_family("regulation", self._clusters(60, span_days=20), coverage_fraction=0.95)
        assert not result.ready
        assert any("day span" in reason for reason in result.unmet)

    def test_one_publisher_blocks_readiness(self) -> None:
        result = assess_family(
            "regulation", self._clusters(40, publishers=1), coverage_fraction=0.95
        )
        assert not result.ready
        assert any("publishers" in reason for reason in result.unmet)

    def test_poor_collection_coverage_blocks_readiness(self) -> None:
        result = assess_family("regulation", self._clusters(40), coverage_fraction=0.2)
        assert not result.ready
        assert any("covered 20%" in reason for reason in result.unmet)

    def test_partial_readiness_is_never_a_licence_to_run(self) -> None:
        result = assess_family("regulation", self._clusters(20), coverage_fraction=0.95)
        assert result.readiness is Readiness.PARTIALLY_READY
        assert not result.ready

    def test_every_named_entity_gets_a_row_including_the_empty_ones(self) -> None:
        """B4.1.31. A missing row would read as 'not asked about'."""
        results = assess_entities(self._clusters(5), ["SEC", "Donald Trump", "Elon Musk"])
        assert [item.family for item in results] == [
            "entity:SEC",
            "entity:Donald Trump",
            "entity:Elon Musk",
        ]
        assert results[1].events == 0 and results[2].events == 0

    def test_whale_contexts_are_assessed_separately_and_never_pooled(self) -> None:
        """B4.1.32. Categories are not merged to reach a minimum."""
        results = assess_whale_contexts({TransferContext.EXCHANGE_INFLOW.value: self._clusters(40)})
        by_family = {item.family: item for item in results}
        assert len(results) == len(TransferContext)
        assert by_family["whale:EXCHANGE_INFLOW"].events == 40
        assert by_family["whale:UNKNOWN"].events == 0
        assert by_family["whale:EXCHANGE_OUTFLOW"].events == 0

    def test_sentiment_composites_are_each_assessed(self) -> None:
        results = assess_sentiment(self._clusters(5), ["sentiment", "sentiment x relevance"])
        assert [item.family for item in results] == [
            "sentiment:sentiment",
            "sentiment:sentiment x relevance",
        ]

    def test_the_corpus_is_only_as_ready_as_its_families(self) -> None:
        ready = assess_family("a", self._clusters(40), coverage_fraction=0.95)
        empty = assess_family("b", [])
        assert overall_readiness([ready]) is Readiness.READY_FOR_VALIDATION
        assert overall_readiness([ready, empty]) is Readiness.PARTIALLY_READY
        assert overall_readiness([empty]) is Readiness.NOT_READY
        assert overall_readiness([]) is Readiness.NOT_READY

    def test_coverage_fraction_counts_distinct_days_not_runs(self) -> None:
        """Ten runs in one afternoon is one day of coverage."""
        runs = [NOW + timedelta(hours=hour) for hour in range(10)]
        assert coverage_fraction(runs, NOW, NOW + timedelta(days=9)) == pytest.approx(0.1)

    def test_a_projection_is_offered_but_is_not_a_criterion(self) -> None:
        family = assess_family("a", self._clusters(10), coverage_fraction=0.95)
        assert days_until_ready(family, events_per_day=1.0, policy=AdequacyPolicy()) is not None
        assert days_until_ready(family, events_per_day=0.0, policy=AdequacyPolicy()) is None
        ready = assess_family("a", self._clusters(40), coverage_fraction=0.95)
        assert days_until_ready(ready, events_per_day=1.0, policy=AdequacyPolicy()) == 0


# ----------------------------------------------------------------- status


class TestStatusReport:
    def _corpus(self):
        documents = [document(index, retrieved=NOW + timedelta(days=index)) for index in range(3)]
        events = [
            event(index, available=NOW + timedelta(days=index), sources=(documents[index].document_id,))
            for index in range(3)
        ]
        clusters = cluster_events(events, documents, window_hours=1)
        return documents, events, clusters

    def test_missing_categories_are_listed_rather_than_absent(self) -> None:
        """The distinction B4 ended on HOLD for."""
        documents, events, clusters = self._corpus()
        status = build_status(
            documents,
            events,
            clusters,
            [assess_family("regulation", clusters)],
            generated_at=NOW,
            expected_event_types=["REGULATION", "ETF_FLOW", "WHALE_TRANSFER"],
            expected_entities=["SEC", "Donald Trump"],
            providers_enabled=2,
        )
        assert status.missing_event_types == ("ETF_FLOW", "WHALE_TRANSFER")
        assert status.missing_entities == ("Donald Trump",)
        assert len(status.missing_whale_contexts) == len(TransferContext)

    def test_a_collection_gap_is_distinct_from_a_quiet_day(self) -> None:
        documents, events, clusters = self._corpus()
        rows = daily_coverage(
            documents,
            events,
            clusters,
            providers_enabled=2,
            successful_run_days=[NOW, NOW + timedelta(days=2)],
        )
        by_day = {row.day: row for row in rows}
        assert by_day[NOW.date()].collection_gap is False
        assert by_day[(NOW + timedelta(days=1)).date()].collection_gap is True, (
            "no successful provider that day, even though a document exists"
        )

    def test_daily_rows_include_empty_days(self) -> None:
        documents = [document(0, retrieved=NOW), document(1, retrieved=NOW + timedelta(days=4))]
        rows = daily_coverage(documents, [], [], providers_enabled=1)
        assert len(rows) == 5, "holes are the part a coverage-bias check needs"

    def test_storage_growth_is_measured_from_real_bytes(self) -> None:
        documents, events, _ = self._corpus()
        growth = measure_storage(documents, events, raw_evidence_bytes=1234)
        assert growth.document_bytes > 0 and growth.event_bytes > 0
        assert growth.total_bytes == growth.document_bytes + growth.event_bytes + 1234
        assert growth.bytes_per_1k_documents == pytest.approx(
            growth.document_bytes / len(documents) * 1000
        )
        assert growth.projected_bytes(1000, 1000) > 0
        assert "extrapolation" in growth.projection_note

    def test_the_human_readable_report_states_readiness_and_reasons(self) -> None:
        documents, events, clusters = self._corpus()
        status = build_status(
            documents,
            events,
            clusters,
            [assess_family("regulation", clusters)],
            generated_at=NOW,
            providers_enabled=1,
        )
        text = status.human_readable()
        assert "READINESS    NOT_READY" in text
        assert "events, need 30" in text

    def test_the_report_round_trips_to_disk(self, tmp_path: Path) -> None:
        from market_intelligence.collection.status import load_status

        documents, events, clusters = self._corpus()
        status = build_status(
            documents, events, clusters, [], generated_at=NOW, providers_enabled=1
        )
        path = status.write(tmp_path / "status.json")
        assert load_status(path).documents == status.documents
        assert not list(path.parent.glob("*.tmp"))


# --------------------------------------------------------------- providers


class TestFixtureProviders:
    def test_a_static_provider_restamps_on_each_retrieval(self) -> None:
        """As a real feed does. Frozen fixtures would never exercise rediscovery."""
        moment = {"now": NOW}
        engine = StaticFixtureProvider([document(0, retrieved=NOW)], now=lambda: moment["now"])
        first = engine.search("bitcoin", NOW - timedelta(hours=1), NOW + timedelta(days=30))
        moment["now"] = NOW + timedelta(days=1)
        second = engine.search("bitcoin", NOW - timedelta(hours=1), NOW + timedelta(days=30))
        assert first[0].available_at == NOW
        assert second[0].available_at == NOW + timedelta(days=1)
        assert first[0].document_id == second[0].document_id

    def test_an_empty_provider_succeeds_with_nothing(self) -> None:
        assert EmptyProvider().search("bitcoin", NOW, NOW) == []

    def test_a_rate_limited_provider_raises_a_classified_failure(self) -> None:
        from market_intelligence.collection.backoff import FailureClass, ProviderFailure

        with pytest.raises(ProviderFailure) as caught:
            RateLimitedProvider().search("bitcoin", NOW, NOW)
        assert caught.value.failure_class is FailureClass.RATE_LIMIT

    def test_an_auth_failure_is_classified_as_auth(self) -> None:
        from market_intelligence.collection.backoff import FailureClass, ProviderFailure

        with pytest.raises(ProviderFailure) as caught:
            AuthFailingProvider().search("bitcoin", NOW, NOW)
        assert caught.value.failure_class is FailureClass.AUTH

    def test_a_schema_invalid_provider_raises_something_classifiable(self) -> None:
        from market_intelligence.collection.backoff import FailureClass, classify_exception

        with pytest.raises(KeyError) as caught:
            SchemaInvalidProvider().search("bitcoin", NOW, NOW)
        assert classify_exception(caught.value) is FailureClass.SCHEMA

    def test_an_outage_is_transient_and_therefore_retried(self) -> None:
        from market_intelligence.collection.backoff import FailureClass, ProviderFailure

        with pytest.raises(ProviderFailure) as caught:
            OutageProvider().search("bitcoin", NOW, NOW)
        assert caught.value.failure_class is FailureClass.TRANSIENT


class TestCadence:
    def test_a_provider_with_no_history_is_due(self) -> None:
        assert cadence_due(None, 900, NOW)

    def test_a_provider_polled_recently_is_not_due(self) -> None:
        assert not cadence_due(NOW - timedelta(seconds=100), 900, NOW)
        assert cadence_due(NOW - timedelta(seconds=901), 900, NOW)

    def test_configuration_in_a_manifest_is_redacted(self) -> None:
        redacted = redacted_configuration({"endpoint": "https://x", "api_key": "sk-1"})
        assert redacted["api_key"] == "<redacted>"
        assert redacted["endpoint"] == "https://x"
