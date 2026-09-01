"""B4.1.28 / B4.1.29 / B4.1.38 — collection through to a B3.1 feature matrix.

The claim this file exists to check: **evidence collected forward flows through
the *unchanged* B3.1 replay path.** Not a parallel implementation that happens
to agree, and not a second definition of a feature — the same
`HistoricalDatasetService` B3.1 shipped, reading the same store the collector
wrote, producing a matrix a B4 study could join to.

If that broke, collection would be accumulating evidence into a shape nothing
downstream can read, and nobody would find out until the corpus was large enough
to matter.

No network.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from market_intelligence.collection.clustering import cluster_events
from market_intelligence.collection.corpus import build_snapshot
from market_intelligence.collection.fixtures import (
    FIXTURE_BASE,
    OutageProvider,
    StaticFixtureProvider,
    fixture_document,
    primary_and_secondary,
)
from market_intelligence.collection.service import ForwardCollector
from market_intelligence.configuration import (
    EntityType,
    ProviderCategory,
    ProviderConfig,
    QueryPlanner,
    WatchEntity,
)
from market_intelligence.extractors import RuleBasedExtractor
from market_intelligence.historical import HistoricalDatasetService
from market_intelligence.models import EventType
from market_intelligence.origins import OriginFrequency, generate_origins
from market_intelligence.retrieval import MultiProviderRetriever
from market_intelligence.storage import IntelligenceStore

NOW = FIXTURE_BASE


def configs(*provider_ids: str) -> dict[str, ProviderConfig]:
    """The retriever needs a config per provider; these are the test defaults."""
    return {
        provider_id: ProviderConfig(
            id=provider_id,
            type="fixture",
            source_category=ProviderCategory.GENERAL_WEB,
            timeout=5.0,
        )
        for provider_id in provider_ids
    }


def watchlist() -> list[WatchEntity]:
    return [
        WatchEntity(
            canonical_name="SEC",
            aliases=("Securities and Exchange Commission",),
            entity_type=EntityType.REGULATOR,
            topics=("bitcoin regulation",),
            expected_event_types=(EventType.REGULATION,),
        )
    ]


def documents_for(queries: list[str], moment: datetime) -> list:
    """Realistic evidence: an official announcement plus media coverage.

    Built an hour before the cycle runs, because coverage lands after the
    announcement it covers -- 40 minutes later in the fixture -- and evidence
    stamped in a cycle's future is correctly refused by the retriever.
    """
    published = moment - timedelta(hours=1)
    official, coverage = primary_and_secondary(published)
    return [
        official.model_copy(update={"query": queries[0]}),
        coverage.model_copy(update={"query": queries[0]}),
        fixture_document(
            url="https://news.example.com/bitcoin-market-note",
            title="Bitcoin market note mentions SEC regulation",
            retrieved_at=published,
            publisher="news.example.com",
            provider="news-api",
            query=queries[0],
            body="A market note referencing SEC regulation of bitcoin.",
        ),
    ]


class TestForwardCollectionCycle:
    def _run(self, tmp_path: Path, moment: datetime, *, include_outage: bool = False):
        store = IntelligenceStore(tmp_path / "corpus.duckdb")
        planner = QueryPlanner()
        queries = planner.plan(watchlist(), moment)
        sources = {
            "fixture": StaticFixtureProvider(
                documents_for([query.query for query in queries], moment),
                name="fixture",
                now=lambda: moment,
                restamp=False,
            )
        }
        if include_outage:
            sources["outage"] = OutageProvider()

        collector = ForwardCollector(store, now=lambda: moment)
        result = collector.collect(
            queries,
            MultiProviderRetriever(sources, configs(*sources)),
            RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}),
            {"watchlist": "test"},
            source_sha="test-sha",
        )
        return store, collector, result

    def test_a_cycle_persists_evidence_and_registers_a_corpus(self, tmp_path: Path) -> None:
        store, collector, result = self._run(tmp_path, NOW)
        try:
            assert result.manifest.documents_retrieved > 0
            assert result.manifest.documents_new == result.manifest.documents_retrieved
            assert result.snapshot is not None
            assert collector.catalog.get(result.snapshot.corpus_id) is not None
        finally:
            store.close()

    def test_a_second_cycle_rediscovers_rather_than_duplicating(self, tmp_path: Path) -> None:
        """The steady state of forward collection: feeds re-serve their items."""
        store, collector, _ = self._run(tmp_path, NOW)
        try:
            before = {
                document.document_id: document.available_at
                for document in store.documents_as_of(NOW)
            }
            assert before, "the first cycle must have collected something"
            later = NOW + timedelta(hours=6)
            queries = QueryPlanner().plan(watchlist(), later)
            sources = {
                "fixture": StaticFixtureProvider(
                    documents_for([query.query for query in queries], NOW),
                    name="fixture",
                    now=lambda: later,
                    restamp=False,
                )
            }
            collector._now = lambda: later  # noqa: SLF001 -- advancing the clock is the test
            second = collector.collect(
                queries, MultiProviderRetriever(sources, configs(*sources)), RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}), {"watchlist": "test"}
            )
            assert second.manifest.documents_new == 0
            assert second.manifest.documents_rediscovered == second.manifest.documents_retrieved
            # And every document's original availability survived the rediscovery.
            after = {
                document.document_id: document.available_at
                for document in store.documents_as_of(later)
            }
            assert after == before
        finally:
            store.close()

    def test_a_provider_outage_does_not_lose_the_working_provider(self, tmp_path: Path) -> None:
        """B4.1.17. Partial failure must not corrupt successful work."""
        store, _, result = self._run(tmp_path, NOW, include_outage=True)
        try:
            assert result.manifest.documents_retrieved > 0
            assert result.manifest.provider_success.get("outage") is False
            assert any("outage" in error for error in result.manifest.errors)
        finally:
            store.close()

    def test_the_manifest_is_written_last_and_never_overwritten(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "corpus.duckdb")
        try:
            queries = QueryPlanner().plan(watchlist(), NOW)
            sources = {
                "fixture": StaticFixtureProvider(
                    documents_for([query.query for query in queries], NOW), now=lambda: NOW, restamp=False
                )
            }
            collector = ForwardCollector(store, now=lambda: NOW)
            path = tmp_path / "manifests" / "run.json"
            collector.collect(
                queries,
                MultiProviderRetriever(sources, configs(*sources)),
                RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}),
                {"watchlist": "test"},
                manifest_path=path,
            )
            assert path.exists()
            assert not list(path.parent.glob("*.tmp"))
            with pytest.raises(FileExistsError, match="never overwrite"):
                collector.collect(
                    queries,
                    MultiProviderRetriever(sources, configs(*sources)),
                    RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}),
                    {"watchlist": "test"},
                    manifest_path=path,
                )
        finally:
            store.close()

    def test_clusters_are_persisted_and_keep_the_primary_source_findable(self, tmp_path: Path) -> None:
        store, collector, _ = self._run(tmp_path, NOW)
        try:
            clusters = collector.clusters.all()
            assert clusters
            assert any(cluster.primary_source_present for cluster in clusters)
        finally:
            store.close()


class TestB31ReplayCompatibility:
    """B4.1.28. Collected evidence through the unchanged B3.1 path."""

    def _collected_store(self, tmp_path: Path) -> IntelligenceStore:
        store = IntelligenceStore(tmp_path / "corpus.duckdb")
        queries = QueryPlanner().plan(watchlist(), NOW)
        sources = {
            "fixture": StaticFixtureProvider(
                documents_for([query.query for query in queries], NOW), now=lambda: NOW, restamp=False
            )
        }
        ForwardCollector(store, now=lambda: NOW).collect(
            queries, MultiProviderRetriever(sources, configs(*sources)), RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}), {"watchlist": "test"}
        )
        return store

    def test_collected_evidence_produces_a_b31_feature_matrix(self, tmp_path: Path) -> None:
        store = self._collected_store(tmp_path)
        try:
            origins = generate_origins(
                NOW - timedelta(hours=2), NOW + timedelta(hours=6), OriginFrequency.HOURLY
            )
            result = HistoricalDatasetService(store, chunk_size=4).build(
                origins,
                tmp_path / "features.csv",
                tmp_path / "features.manifest.json",
                {"b41": "test"},
                "B41_TEST",
                export_format="csv",
                git_sha="test",
            )
            assert result.manifest.row_count == len(origins)
            assert len(result.manifest.columns) == 11
            assert result.manifest.file_hash
        finally:
            store.close()

    def test_features_are_zero_before_the_evidence_and_non_zero_after(self, tmp_path: Path) -> None:
        """The point-in-time contract, surviving all the way to the matrix."""
        import csv

        store = self._collected_store(tmp_path)
        try:
            origins = generate_origins(
                NOW - timedelta(hours=3), NOW + timedelta(hours=3), OriginFrequency.HOURLY
            )
            result = HistoricalDatasetService(store, chunk_size=8).build(
                origins,
                tmp_path / "features.csv",
                tmp_path / "features.manifest.json",
                {"b41": "test"},
                "B41_TEST",
                export_format="csv",
                git_sha="test",
            )
            with result.output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            # Availability is inclusive: evidence available at exactly T is
            # visible to an origin at T. "Before" therefore means strictly before
            # the first document became available.
            first_available = min(
                document.available_at for document in store.documents_as_of(NOW + timedelta(days=1))
            )
            before = [
                row for row in rows if datetime.fromisoformat(row["forecast_origin"]) < first_available
            ]
            after = [
                row for row in rows if datetime.fromisoformat(row["forecast_origin"]) >= first_available
            ]
            assert all(float(row["event_count_24h"]) == 0.0 for row in before), (
                "no evidence may be visible at or before its own availability"
            )
            assert any(float(row["event_count_24h"]) > 0.0 for row in after), (
                "and it must become visible afterwards, or nothing was collected"
            )
        finally:
            store.close()

    def test_a_frozen_corpus_snapshot_matches_what_replay_sees(self, tmp_path: Path) -> None:
        """B4.1.29. The snapshot a study cites and the rows replay reads agree."""
        store = self._collected_store(tmp_path)
        try:
            as_of = NOW + timedelta(hours=1)
            documents = store.documents_as_of(as_of)
            events = store.signals_as_of(as_of)
            snapshot = build_snapshot(
                documents,
                events,
                as_of=as_of,
                extractor_version="rules-v1",
                created_at=as_of,
            )
            assert snapshot.document_count == len(documents)
            assert snapshot.event_count == len(
                [event for event in events if event.extractor_version == "rules-v1"]
            )
            assert snapshot.cluster_count == len(cluster_events(events, documents))
        finally:
            store.close()

    def test_a_corpus_snapshot_is_stable_across_repeated_collection(self, tmp_path: Path) -> None:
        """Re-collecting the same feed must not change a published corpus id."""
        store = self._collected_store(tmp_path)
        try:
            as_of = NOW + timedelta(hours=1)
            first = build_snapshot(
                store.documents_as_of(as_of),
                store.signals_as_of(as_of),
                as_of=as_of,
                extractor_version="rules-v1",
                created_at=as_of,
            )
            queries = QueryPlanner().plan(watchlist(), NOW + timedelta(minutes=30))
            sources = {
                "fixture": StaticFixtureProvider(
                    documents_for([query.query for query in queries], NOW),
                    now=lambda: NOW + timedelta(minutes=30),
                    restamp=False,
                )
            }
            ForwardCollector(store, now=lambda: NOW + timedelta(minutes=30)).collect(
                queries, MultiProviderRetriever(sources, configs(*sources)), RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}), {"watchlist": "test"}
            )
            again = build_snapshot(
                store.documents_as_of(as_of),
                store.signals_as_of(as_of),
                as_of=as_of,
                extractor_version="rules-v1",
                created_at=as_of,
            )
            assert first.corpus_id == again.corpus_id
        finally:
            store.close()


class TestCorpusStatusSurfaces:
    """B4.1.34 / B4.1.35. One implementation behind the CLI and the API."""

    def test_the_cli_and_the_api_report_the_same_corpus(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from market_intelligence.api import create_app
        from market_intelligence.cli import _corpus_status

        database = tmp_path / "corpus.duckdb"
        store = IntelligenceStore(database)
        queries = QueryPlanner().plan(watchlist(), NOW)
        sources = {
            "fixture": StaticFixtureProvider(
                documents_for([query.query for query in queries], NOW), now=lambda: NOW, restamp=False
            )
        }
        ForwardCollector(store, now=lambda: NOW).collect(
            queries, MultiProviderRetriever(sources, configs(*sources)), RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}), {"watchlist": "test"}
        )
        direct = _corpus_status(store, "rules-v1", ["SEC"])
        store.close()

        client = TestClient(create_app(database))
        served = client.get("/corpus/status", params={"entities": "SEC"}).json()
        assert served["documents"] == direct.documents
        assert served["events"] == direct.events
        assert served["clusters"] == direct.clusters
        assert served["readiness"] == direct.readiness.value

    def test_the_status_report_never_carries_document_text(self, tmp_path: Path) -> None:
        """A non-redistributable provider's content must not leak through a
        status endpoint."""
        from market_intelligence.cli import _corpus_status

        store = self_store = IntelligenceStore(tmp_path / "corpus.duckdb")
        try:
            queries = QueryPlanner().plan(watchlist(), NOW)
            sources = {
                "fixture": StaticFixtureProvider(
                    documents_for([query.query for query in queries], NOW), now=lambda: NOW, restamp=False
                )
            }
            ForwardCollector(store, now=lambda: NOW).collect(
                queries, MultiProviderRetriever(sources, configs(*sources)), RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}), {"watchlist": "test"}
            )
            payload = _corpus_status(self_store, "rules-v1", ["SEC"]).model_dump_json()
            assert "Official announcement text" not in payload
            assert "Media interpretation" not in payload
        finally:
            store.close()

    def test_readiness_is_not_ready_on_a_corpus_this_young(self, tmp_path: Path) -> None:
        """B4.1.30 / B4.1.46. Three documents is not a licence to run B4."""
        from market_intelligence.cli import _corpus_status

        store = IntelligenceStore(tmp_path / "corpus.duckdb")
        try:
            queries = QueryPlanner().plan(watchlist(), NOW)
            sources = {
                "fixture": StaticFixtureProvider(
                    documents_for([query.query for query in queries], NOW), now=lambda: NOW, restamp=False
                )
            }
            ForwardCollector(store, now=lambda: NOW).collect(
                queries, MultiProviderRetriever(sources, configs(*sources)), RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}), {"watchlist": "test"}
            )
            status = _corpus_status(store, "rules-v1", ["SEC"])
            assert status.readiness.value == "NOT_READY"
            assert all(not family.ready for family in status.families)
        finally:
            store.close()
