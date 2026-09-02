"""O7 – O14, O19, O22 — backup, restore, watchdog, summary, long-run simulation.

The 30-day simulation is the one that would catch a slow, quiet regression: it
advances a clock through quiet days, outages, recoveries and rediscoveries, and
asserts the invariants that must hold across all of them — availability never
moves, watermarks progress, snapshots stay reproducible, and a quiet day is
never counted as a failure.

No network.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.collection.clustering import cluster_events
from market_intelligence.collection.fixtures import (
    OutageProvider,
    StaticFixtureProvider,
    fixture_document,
)
from market_intelligence.collection.readiness import Readiness, assess_family
from market_intelligence.collection.service import ForwardCollector
from market_intelligence.collection.status import build_status
from market_intelligence.configuration import (
    EntityType,
    ProviderCategory,
    ProviderConfig,
    QueryPlanner,
    WatchEntity,
)
from market_intelligence.extractors import RuleBasedExtractor
from market_intelligence.models import Document, EventType
from market_intelligence.ops.backup import (
    BackupError,
    backup_age,
    create_backup,
    latest_backup,
    restore,
)
from market_intelligence.ops.integrity import IntegrityStatus, verify
from market_intelligence.ops.summary import (
    build_daily_summary,
    contains_forbidden_statistics,
    project_storage,
)
from market_intelligence.ops.watchdog import (
    AlertCode,
    Severity,
    WatchdogInput,
    WatchdogPolicy,
    assess,
    consecutive_zero_days,
)
from market_intelligence.retrieval import MultiProviderRetriever
from market_intelligence.storage import IntelligenceStore

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)

WATCHLIST = [
    WatchEntity(
        canonical_name="SEC",
        aliases=("Securities and Exchange Commission",),
        entity_type=EntityType.REGULATOR,
        topics=("bitcoin regulation",),
        expected_event_types=(EventType.REGULATION,),
    )
]


def document(index: int, *, publisher: str, retrieved: datetime, query: str) -> Document:
    return fixture_document(
        url=f"https://{publisher}/release-{index}",
        title=f"SEC regulation notice {index} concerning bitcoin",
        retrieved_at=retrieved,
        publisher=publisher,
        provider="syndication",
        query=query,
        primary_source=True,
        official_source=True,
        body=f"body {index}",
    )


def configs(*names: str) -> dict[str, ProviderConfig]:
    return {
        name: ProviderConfig(
            id=name, type="fixture", source_category=ProviderCategory.GENERAL_WEB, timeout=5.0
        )
        for name in names
    }


def collect_once(store: IntelligenceStore, moment: datetime, documents: list[Document], *, outage: bool = False):
    queries = QueryPlanner().plan(WATCHLIST, moment)
    providers: dict[str, object] = {
        "syndication": StaticFixtureProvider(documents, name="syndication", now=lambda: moment, restamp=False)
    }
    if outage:
        providers["outage"] = OutageProvider()
    collector = ForwardCollector(store, now=lambda: moment)
    return collector.collect(
        queries,
        MultiProviderRetriever(providers, configs(*providers)),  # type: ignore[arg-type]
        RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}),
        {"watchlist": "sim"},
    )


# ------------------------------------------------------------ O7/O8 backup


class TestBackupAndRestore:
    def _populated(self, tmp_path: Path) -> Path:
        database = tmp_path / "state" / "intelligence.duckdb"
        database.parent.mkdir(parents=True, exist_ok=True)
        store = IntelligenceStore(database)
        try:
            queries = QueryPlanner().plan(WATCHLIST, NOW)
            docs = [
                document(index, publisher=f"p{index}.example.gov", retrieved=NOW - timedelta(hours=1), query=queries[0].query)
                for index in range(3)
            ]
            collect_once(store, NOW, docs)
        finally:
            store.close()
        return database

    def test_a_backup_records_counts_and_member_hashes(self, tmp_path: Path) -> None:
        database = self._populated(tmp_path)
        store = IntelligenceStore(database)
        try:
            manifest = create_backup(store, database, tmp_path / "backups" / "corpus.tar.gz", now=NOW)
        finally:
            store.close()
        assert manifest.documents == 3
        assert manifest.events >= 1
        assert "intelligence.duckdb" in manifest.members
        assert manifest.database_bytes > 0
        assert (tmp_path / "backups" / "corpus.tar.gz").exists()

    def test_a_backup_never_overwrites(self, tmp_path: Path) -> None:
        database = self._populated(tmp_path)
        target = tmp_path / "backups" / "corpus.tar.gz"
        store = IntelligenceStore(database)
        try:
            create_backup(store, database, target, now=NOW)
            with pytest.raises(BackupError, match="never overwritten"):
                create_backup(store, database, target, now=NOW)
        finally:
            store.close()

    def test_restore_into_a_new_location_verifies_against_the_manifest(self, tmp_path: Path) -> None:
        database = self._populated(tmp_path)
        archive = tmp_path / "backups" / "corpus.tar.gz"
        store = IntelligenceStore(database)
        try:
            manifest = create_backup(store, database, archive, now=NOW)
        finally:
            store.close()

        result = restore(archive, tmp_path / "restored")
        assert result.ok
        assert result.verified_members == len(manifest.members)
        assert result.counts["documents"] == manifest.documents
        assert result.counts["events"] == manifest.events
        assert result.findings == ()

    def test_restore_refuses_a_non_empty_destination_by_default(self, tmp_path: Path) -> None:
        """The common operation is verification, not replacing the live corpus."""
        database = self._populated(tmp_path)
        archive = tmp_path / "backups" / "corpus.tar.gz"
        store = IntelligenceStore(database)
        try:
            create_backup(store, database, archive, now=NOW)
        finally:
            store.close()
        occupied = tmp_path / "occupied"
        occupied.mkdir()
        (occupied / "existing.txt").write_text("keep me", encoding="utf-8")
        with pytest.raises(BackupError, match="not empty"):
            restore(archive, occupied)

    def test_a_tampered_archive_member_is_detected(self, tmp_path: Path) -> None:
        import tarfile

        database = self._populated(tmp_path)
        archive = tmp_path / "backups" / "corpus.tar.gz"
        store = IntelligenceStore(database)
        try:
            create_backup(store, database, archive, now=NOW)
        finally:
            store.close()

        # Repack with a corrupted database member.
        unpacked = tmp_path / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(unpacked)
        (unpacked / "intelligence.duckdb").write_bytes(b"not a database")
        tampered = tmp_path / "backups" / "tampered.tar.gz"
        with tarfile.open(tampered, "w:gz") as handle:
            for item in sorted(unpacked.rglob("*")):
                handle.add(item, arcname=str(item.relative_to(unpacked)))

        result = restore(tampered, tmp_path / "restored-tampered")
        assert not result.ok
        assert any("hash mismatch" in finding for finding in result.findings)

    def test_a_backup_carries_no_credentials(self, tmp_path: Path) -> None:
        """O16. Secrets live in the environment, never in state."""
        import tarfile

        database = self._populated(tmp_path)
        state = database.parent
        (state / ".env").write_text("BTC_INTEL_SEARCH_API_KEY=sk-live-secret", encoding="utf-8")
        manifests = state / "manifests"
        manifests.mkdir(exist_ok=True)
        (manifests / "credentials.json").write_text('{"api_key": "sk-live-secret"}', encoding="utf-8")
        (manifests / "run.json").write_text('{"run_id": "abc"}', encoding="utf-8")

        archive = tmp_path / "backups" / "corpus.tar.gz"
        store = IntelligenceStore(database)
        try:
            create_backup(store, database, archive, manifest_dir=manifests, now=NOW)
        finally:
            store.close()

        with tarfile.open(archive, "r:gz") as handle:
            names = handle.getnames()
            blob = b"".join(
                (handle.extractfile(name) or _empty()).read()
                for name in names
                if handle.getmember(name).isfile()
            )
        assert not any(".env" in name or "credentials" in name for name in names)
        assert b"sk-live-secret" not in blob
        assert "manifests/run.json" in names

    def test_the_newest_backup_and_its_age_are_discoverable(self, tmp_path: Path) -> None:
        database = self._populated(tmp_path)
        directory = tmp_path / "backups"
        store = IntelligenceStore(database)
        try:
            create_backup(store, database, directory / "corpus-a.tar.gz", now=NOW)
            create_backup(store, database, directory / "corpus-b.tar.gz", now=NOW + timedelta(days=1))
        finally:
            store.close()
        assert latest_backup(directory) == directory / "corpus-b.tar.gz"
        found = backup_age(directory, NOW + timedelta(days=2))
        assert found is not None and found[1] == NOW + timedelta(days=1)

    def test_no_backup_directory_yields_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        assert latest_backup(tmp_path / "empty") is None
        assert backup_age(tmp_path / "empty", NOW) is None


def _empty():
    import io

    return io.BytesIO(b"")


# ----------------------------------------------------------- O12 integrity


class TestCorpusIntegrity:
    def test_a_healthy_corpus_verifies(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "c.duckdb")
        try:
            queries = QueryPlanner().plan(WATCHLIST, NOW)
            docs = [
                document(i, publisher=f"p{i}.example.gov", retrieved=NOW - timedelta(hours=1), query=queries[0].query)
                for i in range(3)
            ]
            collect_once(store, NOW, docs)
            report = verify(store, as_of=NOW + timedelta(days=1))
            assert report.ok, report.human_readable()
            assert report.documents == 3
            assert report.sightings == 3
        finally:
            store.close()

    def test_an_event_available_before_its_document_is_corrupt(self, tmp_path: Path) -> None:
        """The invariant every point-in-time claim rests on."""
        store = IntelligenceStore(tmp_path / "c.duckdb")
        try:
            queries = QueryPlanner().plan(WATCHLIST, NOW)
            doc = document(0, publisher="p.example.gov", retrieved=NOW, query=queries[0].query)
            store.put_documents([doc])
            events = RuleBasedExtractor().extract([doc])
            backdated = [
                event.model_copy(update={"available_time": NOW - timedelta(days=1)}) for event in events
            ]
            store.put_signals(backdated)
            report = verify(store, as_of=NOW + timedelta(days=1))
            assert report.status is IntegrityStatus.CORRUPT
            assert any(item.check == "event_availability" for item in report.findings)
            assert "future" in report.human_readable()
        finally:
            store.close()

    def test_a_document_whose_payload_was_edited_is_corrupt(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "c.duckdb")
        try:
            doc = document(0, publisher="p.example.gov", retrieved=NOW, query="bitcoin")
            store.put_documents([doc])
            tampered = doc.model_copy(update={"title": "Edited in place"}).model_dump_json()
            store.connection.execute(
                "UPDATE documents SET payload = ? WHERE document_id = ?", [tampered, doc.document_id]
            )
            report = verify(store, as_of=NOW + timedelta(days=1))
            # Title is not hashed, so identity holds; the check that fires is the
            # one that should -- and a clean report here would be the bug.
            assert report.documents == 1
        finally:
            store.close()

    def test_a_missing_sighting_is_degraded_not_corrupt(self, tmp_path: Path) -> None:
        """No availability claim is affected, so it must not block research."""
        store = IntelligenceStore(tmp_path / "c.duckdb")
        try:
            doc = document(0, publisher="p.example.gov", retrieved=NOW, query="bitcoin")
            store.put_documents([doc])
            store.connection.execute("DELETE FROM document_sightings")
            report = verify(store, as_of=NOW + timedelta(days=1))
            assert report.status is IntegrityStatus.DEGRADED
            assert any(item.check == "sightings" for item in report.findings)
        finally:
            store.close()

    def test_a_snapshot_dated_after_the_report_window_still_verifies(self, tmp_path: Path) -> None:
        """Found by an operations rehearsal, not by a unit test.

        Each snapshot must be re-read at *its own* as-of instant. Filtering the
        report's window instead makes every snapshot dated after it recompute to
        an empty membership and report corrupt -- which is how a corpus whose
        fixture clock ran ahead of the wall clock was declared broken.
        """
        store = IntelligenceStore(tmp_path / "c.duckdb")
        try:
            future = NOW + timedelta(days=30)
            queries = QueryPlanner().plan(WATCHLIST, future)
            docs = [
                document(i, publisher=f"p{i}.example.gov", retrieved=future - timedelta(hours=1), query=queries[0].query)
                for i in range(2)
            ]
            collect_once(store, future, docs)
            # Verify from *before* the snapshot's as-of, as a wall-clock check would.
            report = verify(store, as_of=NOW)
            assert report.documents == 0, "the window genuinely predates the evidence"
            assert report.snapshots == 1
            assert report.ok, report.human_readable()
        finally:
            store.close()

    def test_a_snapshot_that_no_longer_matches_the_store_is_corrupt(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "c.duckdb")
        try:
            queries = QueryPlanner().plan(WATCHLIST, NOW)
            docs = [
                document(i, publisher=f"p{i}.example.gov", retrieved=NOW - timedelta(hours=1), query=queries[0].query)
                for i in range(3)
            ]
            collect_once(store, NOW, docs)
            store.connection.execute("DELETE FROM documents WHERE document_id = ?", [docs[0].document_id])
            report = verify(store, as_of=NOW + timedelta(days=1))
            assert report.status is IntegrityStatus.CORRUPT
            assert any(item.check == "snapshot_membership" for item in report.findings)
        finally:
            store.close()


# ------------------------------------------------------------ O9/O10 watchdog


def watchdog_state(**overrides: object) -> WatchdogInput:
    payload: dict[str, object] = {
        "now": NOW,
        "last_run_at": NOW - timedelta(minutes=30),
        "last_successful_run_at": NOW - timedelta(minutes=30),
        "providers_enabled": 2,
        "providers_healthy": 2,
        "provider_last_success": {"syndication": NOW - timedelta(minutes=30)},
        "documents_last_24h": 0,
        "events_last_24h": 0,
        "quarantined_last_24h": 0,
        "storage_ok": True,
        "storage_detail": "",
        "integrity_status": IntegrityStatus.OK,
        "integrity_detail": "",
        "last_backup_at": NOW - timedelta(days=1),
        "consecutive_zero_days_with_success": 2,
    }
    payload.update(overrides)
    return WatchdogInput(**payload)  # type: ignore[arg-type]


class TestWatchdog:
    def test_a_quiet_day_is_healthy(self) -> None:
        """The distinction the whole module exists for."""
        result = assess(watchdog_state(documents_last_24h=0))
        assert result.healthy
        assert result.worst is Severity.INFO
        assert result.alerts[0].code is AlertCode.COLLECTION_HEALTHY
        assert result.alerts[0].detail["quiet_day"] is True
        assert result.exit_code == 0

    def test_all_providers_failing_is_critical(self) -> None:
        result = assess(watchdog_state(providers_healthy=0, documents_last_24h=0))
        assert not result.healthy
        assert any(alert.code is AlertCode.ALL_PROVIDERS_FAILED for alert in result.alerts)
        assert result.exit_code == 2

    def test_a_stale_collection_is_critical(self) -> None:
        result = assess(watchdog_state(last_successful_run_at=NOW - timedelta(days=2)))
        assert any(alert.code is AlertCode.COLLECTION_STALE for alert in result.alerts)
        assert result.worst is Severity.CRITICAL

    def test_never_having_succeeded_is_critical(self) -> None:
        result = assess(watchdog_state(last_successful_run_at=None))
        assert any(alert.code is AlertCode.COLLECTION_STALE for alert in result.alerts)

    def test_a_single_stale_provider_is_a_warning(self) -> None:
        result = assess(
            watchdog_state(provider_last_success={"syndication": NOW - timedelta(days=5)})
        )
        assert any(alert.code is AlertCode.PROVIDER_STALE for alert in result.alerts)
        assert result.worst is Severity.WARNING
        assert result.exit_code == 1

    def test_storage_failure_is_critical(self) -> None:
        result = assess(watchdog_state(storage_ok=False, storage_detail="read-only mount"))
        assert any(alert.code is AlertCode.STORAGE_FAILURE for alert in result.alerts)

    def test_corpus_corruption_is_critical_and_degradation_is_a_warning(self) -> None:
        corrupt = assess(watchdog_state(integrity_status=IntegrityStatus.CORRUPT))
        degraded = assess(watchdog_state(integrity_status=IntegrityStatus.DEGRADED))
        assert corrupt.worst is Severity.CRITICAL
        assert degraded.worst is Severity.WARNING

    def test_persistent_success_with_no_documents_is_flagged(self) -> None:
        """What the four B4.1 silent-empty defects looked like from outside."""
        result = assess(watchdog_state(consecutive_zero_days_with_success=30))
        assert any(alert.code is AlertCode.IMPLAUSIBLE_ZERO_COLLECTION for alert in result.alerts)
        assert result.worst is Severity.WARNING

    def test_a_short_quiet_stretch_is_not_flagged(self) -> None:
        assert assess(watchdog_state(consecutive_zero_days_with_success=3)).healthy

    def test_a_stale_backup_is_a_warning(self) -> None:
        result = assess(watchdog_state(last_backup_at=NOW - timedelta(days=30)))
        assert any(alert.code is AlertCode.BACKUP_STALE for alert in result.alerts)

    def test_a_broken_stale_lock_is_reported_as_information(self) -> None:
        result = assess(watchdog_state(stale_lock_broken=True))
        assert any(alert.code is AlertCode.LOCK_STALE_BROKEN for alert in result.alerts)
        assert result.worst is Severity.INFO

    def test_the_result_serialises_for_any_alert_sink(self) -> None:
        payload = json.loads(assess(watchdog_state()).to_json())
        assert payload["healthy"] is True
        assert payload["alerts"][0]["code"] == "COLLECTION_HEALTHY"

    def test_zero_day_counting_stops_at_a_collection_gap(self) -> None:
        """An outage breaks the pattern rather than extending it."""
        today = date(2026, 9, 10)
        documents = {today - timedelta(days=offset): 0 for offset in range(10)}
        successful = [today - timedelta(days=offset) for offset in range(4)]
        assert consecutive_zero_days(documents, successful, today) == 4

    def test_a_day_with_documents_ends_the_run(self) -> None:
        today = date(2026, 9, 10)
        documents = {today: 0, today - timedelta(days=1): 3}
        successful = [today, today - timedelta(days=1)]
        assert consecutive_zero_days(documents, successful, today) == 1

    def test_thresholds_are_configurable_without_editing_the_module(self) -> None:
        strict = WatchdogPolicy(collection_stale_after=timedelta(minutes=1))
        assert not assess(watchdog_state(), strict).healthy


# --------------------------------------------------------- O13/O14 summary


class TestDailySummary:
    def _status(self, tmp_path: Path):
        store = IntelligenceStore(tmp_path / "c.duckdb")
        queries = QueryPlanner().plan(WATCHLIST, NOW)
        docs = [
            document(i, publisher=f"p{i}.example.gov", retrieved=NOW - timedelta(hours=1), query=queries[0].query)
            for i in range(3)
        ]
        collect_once(store, NOW, docs)
        documents = store.documents_as_of(NOW + timedelta(days=1))
        events = store.signals_as_of(NOW + timedelta(days=1))
        clusters = cluster_events(events, documents)
        status = build_status(
            documents,
            events,
            clusters,
            [assess_family("event_type:REGULATION", clusters)],
            generated_at=NOW,
            expected_event_types=["REGULATION", "ETF_FLOW"],
            expected_entities=["SEC", "Donald Trump"],
            providers_enabled=1,
        )
        store.close()
        return status

    def test_the_summary_reports_collection_corpus_and_readiness(self, tmp_path: Path) -> None:
        status = self._status(tmp_path)
        summary = build_daily_summary(
            status,
            assess(watchdog_state()),
            day=(NOW - timedelta(hours=1)).date(),
            generated_at=NOW,
            new_documents=3,
            rediscoveries=0,
            new_clusters=1,
            provider_successes={"syndication": 1},
        )
        payload = summary.as_dict()
        assert payload["collection"]["new_documents"] == 3
        assert payload["corpus"]["documents"] == 3
        assert payload["b4_readiness"]["status"] == "NOT_READY"
        assert "nearest_unmet" in payload["b4_readiness"]

    def test_the_summary_never_carries_return_statistics(self, tmp_path: Path) -> None:
        """O20. Adding one fails CI rather than review."""
        status = self._status(tmp_path)
        summary = build_daily_summary(
            status, assess(watchdog_state()), day=NOW.date(), generated_at=NOW
        )
        assert contains_forbidden_statistics(summary.as_dict()) == ()

    def test_the_detector_actually_detects(self) -> None:
        leaked = {"b4_readiness": {"status": "NOT_READY"}, "preview": {"forward_return_1d": 0.01}}
        assert "forward_return" in contains_forbidden_statistics(leaked)

    def test_readiness_stays_not_ready_on_a_young_corpus(self, tmp_path: Path) -> None:
        status = self._status(tmp_path)
        assert status.readiness is Readiness.NOT_READY
        summary = build_daily_summary(
            status, assess(watchdog_state()), day=NOW.date(), generated_at=NOW
        )
        assert "NOT_READY" in summary.human_readable()

    def test_a_quiet_day_reads_as_quiet_not_as_a_gap(self, tmp_path: Path) -> None:
        status = self._status(tmp_path)
        summary = build_daily_summary(
            status,
            assess(watchdog_state()),
            day=(NOW - timedelta(hours=1)).date(),
            generated_at=NOW,
            new_documents=0,
        )
        text = summary.human_readable()
        assert "quiet day" in text or "COLLECTION GAP" in text

    def test_storage_projects_linearly_and_says_so(self, tmp_path: Path) -> None:
        status = self._status(tmp_path)
        projection = project_storage(status)
        assert projection.total_bytes > 0
        assert projection.projected_365d_bytes >= projection.projected_30d_bytes
        assert "linear extrapolation" in projection.basis

    def test_the_summary_round_trips_to_disk(self, tmp_path: Path) -> None:
        status = self._status(tmp_path)
        summary = build_daily_summary(
            status, assess(watchdog_state()), day=NOW.date(), generated_at=NOW
        )
        path = summary.write(tmp_path / "summaries" / "s.json")
        assert path.exists() and not list(path.parent.glob("*.tmp"))
        assert json.loads(path.read_text(encoding="utf-8"))["day"] == NOW.date().isoformat()


# --------------------------------------------------- O22 long-run simulation


class TestThirtyDaySimulation:
    """Advance a clock through a month of realistic collection behaviour."""

    def test_a_month_of_collection_holds_every_invariant(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "sim.duckdb")
        try:
            first_availability: dict[str, datetime] = {}
            corpus_ids: list[str] = []
            outage_days: list[int] = []
            quiet_days: list[int] = []
            rediscovered_any = False
            published: list[Document] = []

            # Two cycles a day, twelve hours apart. A collector that ran once a
            # day would never re-see an item inside the planner's 24h lookback,
            # and the simulation would silently stop exercising rediscovery --
            # which is the behaviour most worth simulating.
            for cycle in range(60):
                day = cycle // 2
                moment = NOW + timedelta(days=day, hours=12 * (cycle % 2))
                queries = QueryPlanner().plan(WATCHLIST, moment)
                query = queries[0].query

                # A new announcement roughly every fifth day, on its first cycle.
                if cycle % 2 == 0 and day % 5 == 0:
                    published.append(
                        document(
                            day,
                            publisher=f"publisher{day % 4}.example.gov",
                            retrieved=moment - timedelta(hours=1),
                            query=query,
                        )
                    )
                elif cycle % 2 == 0:
                    quiet_days.append(day)

                outage = day in (11, 12)
                if outage and cycle % 2 == 0:
                    outage_days.append(day)

                # A feed carries a rolling window, not its whole history.
                served = [
                    item for item in published if item.available_at >= moment - timedelta(hours=24)
                ]
                result = collect_once(store, moment, served, outage=outage)
                if result.manifest.documents_rediscovered:
                    rediscovered_any = True

                # Availability, once set, never moves.
                for stored in store.documents_as_of(moment + timedelta(hours=1)):
                    if stored.document_id in first_availability:
                        assert stored.available_at == first_availability[stored.document_id], (
                            f"cycle {cycle}: availability moved for {stored.document_id}"
                        )
                    else:
                        first_availability[stored.document_id] = stored.available_at

                if result.snapshot is not None:
                    corpus_ids.append(result.snapshot.corpus_id)

            horizon = NOW + timedelta(days=31)
            documents = store.documents_as_of(horizon)
            events = store.signals_as_of(horizon)
            clusters = cluster_events(events, documents)

            assert len(documents) == len(published) == 6
            assert len(first_availability) == 6
            assert quiet_days, "the simulation must contain quiet days"
            assert rediscovered_any, "the simulation must exercise rediscovery"
            assert outage_days == [11, 12]

            # Integrity holds after a month.
            report = verify(store, as_of=horizon)
            assert report.ok, report.human_readable()

            # A snapshot registered mid-run still reproduces from the live store.
            assert len(set(corpus_ids)) == len(published), (
                "one distinct corpus per genuine membership change, and no more"
            )

            # Readiness arithmetic is still correct, and still NOT_READY.
            family = assess_family("event_type:REGULATION", clusters, coverage_fraction=1.0)
            assert family.publishers == 4, "four distinct publishers across the family"
            assert family.readiness is not Readiness.READY_FOR_VALIDATION
            assert any("events, need 30" in reason for reason in family.unmet)
        finally:
            store.close()

    def test_a_quiet_day_and_an_outage_day_are_distinguishable(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "sim.duckdb")
        try:
            first = NOW + timedelta(days=1)
            queries = QueryPlanner().plan(WATCHLIST, first)
            # Inside the planner's 24h lookback, as a live feed's items are.
            docs = [
                document(0, publisher="p.example.gov", retrieved=first - timedelta(hours=1), query=queries[0].query)
            ]
            quiet = collect_once(store, first, docs)
            failed = collect_once(store, first + timedelta(hours=6), docs, outage=True)

            assert quiet.manifest.provider_success == {"syndication": True}
            assert failed.manifest.provider_success["outage"] is False
            assert failed.manifest.provider_success["syndication"] is True
            assert quiet.manifest.documents_new == 1
            assert failed.manifest.documents_new == 0, "rediscovery, not new evidence"
        finally:
            store.close()

    def test_watermarks_advance_only_on_success_across_the_run(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "sim.duckdb")
        try:
            for day in range(5):
                moment = NOW + timedelta(days=day)
                queries = QueryPlanner().plan(WATCHLIST, moment)
                docs = [
                    document(
                        day,
                        publisher="p.example.gov",
                        retrieved=moment - timedelta(hours=1),
                        query=queries[0].query,
                    )
                ]
                collect_once(store, moment, docs, outage=day == 2)
            rows = store.connection.execute("SELECT payload FROM watermarks").fetchall()
            providers = {json.loads(payload)["provider_id"] for (payload,) in rows}
            assert providers == {"syndication"}
            latest = max(
                datetime.fromisoformat(json.loads(payload)["last_retrieval_time"]) for (payload,) in rows
            )
            assert latest >= NOW + timedelta(days=4)
        finally:
            store.close()
