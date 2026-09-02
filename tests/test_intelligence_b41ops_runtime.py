"""O2 / O5 / O6 / O21 — lock, storage, crash recovery, and the publisher audit.

The publisher-count class is the one worth reading first: it is a real defect
that would have kept the readiness gate closed forever, and it was only visible
because O21 asked the question.

No network.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.collection.clustering import cluster_events
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.collection.readiness import AdequacyPolicy, assess_family
from market_intelligence.collection.service import ForwardCollector
from market_intelligence.configuration import (
    EntityType,
    ProviderCategory,
    ProviderConfig,
    QueryPlanner,
    WatchEntity,
)
from market_intelligence.errors import ConfigurationError
from market_intelligence.extractors import RuleBasedExtractor
from market_intelligence.models import Document, EventType
from market_intelligence.ops.paths import (
    StoragePaths,
    looks_ephemeral,
    require_usable,
    validate,
)
from market_intelligence.ops.runlock import (
    LockHeld,
    LockRecord,
    RunLock,
    held_by,
)
from market_intelligence.retrieval import MultiProviderRetriever
from market_intelligence.storage import IntelligenceStore

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def document(index: int, *, publisher: str, retrieved: datetime) -> Document:
    return fixture_document(
        url=f"https://{publisher}/release-{index}",
        title=f"SEC regulation notice {index} concerning bitcoin",
        retrieved_at=retrieved,
        publisher=publisher,
        provider="syndication",
        primary_source=True,
        official_source=True,
        body=f"body {index}",
    )


# ------------------------------------------------------- O21 publisher audit


class TestPublisherCountAudit:
    """O21. The reported symptom, and the real defect underneath it."""

    def _clusters(self, publishers: list[str], *, span_days: int = 300):
        documents = [
            document(index, publisher=name, retrieved=NOW + timedelta(days=index * span_days // max(1, len(publishers))))
            for index, name in enumerate(publishers)
        ]
        events = RuleBasedExtractor().extract(documents)
        return cluster_events(events, documents, window_hours=1)

    def test_one_document_reports_one_publisher_not_zero(self) -> None:
        """The reported symptom does not reproduce on a non-empty corpus.

        `0 publishers` came from an *empty* store, where the count is correctly
        zero. One real document reports one publisher.
        """
        clusters = self._clusters(["sec.gov"])
        assert len(clusters) == 1
        result = assess_family("regulation", clusters)
        assert result.publishers == 1
        assert any("1 publishers, need 3" in reason for reason in result.unmet)

    def test_an_empty_family_reports_zero_publishers(self) -> None:
        result = assess_family("regulation", [])
        assert result.publishers == 0
        assert any("0 publishers, need 3" in reason for reason in result.unmet)

    def test_diversity_is_a_union_across_the_family_not_a_per_event_maximum(self) -> None:
        """The real defect.

        Counting `max(cluster.publisher_count)` measures whether any *single*
        event was covered by three publishers. Official feeds publish each
        announcement once, so that number stays at 1 however long collection
        runs -- and the publisher clause could never have been satisfied.
        """
        names = [f"publisher{index % 5}.example.gov" for index in range(40)]
        clusters = self._clusters(names)
        assert len(clusters) == 40
        assert {cluster.publisher_count for cluster in clusters} == {1}, (
            "each announcement has exactly one publisher, as official feeds do"
        )
        result = assess_family("regulation", clusters, coverage_fraction=0.95)
        assert result.publishers == 5, "the family draws on five distinct publishers"
        assert not any("publishers" in reason for reason in result.unmet)

    def test_the_threshold_itself_is_unchanged(self) -> None:
        """O21 says fix the defect, not the bar."""
        assert AdequacyPolicy().minimum_publishers == 3

    def test_a_single_publisher_family_still_fails_the_clause(self) -> None:
        clusters = self._clusters(["sec.gov"] * 40)
        result = assess_family("regulation", clusters, coverage_fraction=0.95)
        assert result.publishers == 1
        assert any("publishers" in reason for reason in result.unmet)

    def test_clusters_carry_publisher_and_provider_identities(self) -> None:
        clusters = self._clusters(["a.example.gov", "b.example.gov"])
        assert all(cluster.publishers for cluster in clusters)
        assert all(cluster.providers == ("syndication",) for cluster in clusters)


# ------------------------------------------------------------------- O2 lock


class TestRunLock:
    def test_a_second_collector_is_refused(self, tmp_path: Path) -> None:
        first = RunLock(tmp_path / "c.lock", now=lambda: NOW).acquire()
        try:
            with pytest.raises(LockHeld, match="is held by pid"):
                RunLock(tmp_path / "c.lock", now=lambda: NOW).acquire()
        finally:
            first.release()

    def test_the_lock_is_released_and_reacquirable(self, tmp_path: Path) -> None:
        with RunLock(tmp_path / "c.lock", now=lambda: NOW):
            assert (tmp_path / "c.lock").exists()
        assert not (tmp_path / "c.lock").exists()
        RunLock(tmp_path / "c.lock", now=lambda: NOW).acquire().release()

    def test_a_dead_holder_does_not_wedge_collection_forever(self, tmp_path: Path) -> None:
        """The failure mode that matters more than exclusion."""
        path = tmp_path / "c.lock"
        stale = LockRecord(
            pid=999_999,
            hostname=_hostname(),
            acquired_at=NOW - timedelta(hours=1),
            heartbeat_at=NOW - timedelta(hours=1),
            purpose="collect",
        )
        path.write_text(stale.as_json(), encoding="utf-8")

        lock = RunLock(path, now=lambda: NOW, is_alive=lambda pid: False).acquire()
        try:
            assert lock.broke_stale_lock
            assert lock.previous_holder is not None and lock.previous_holder.pid == 999_999
        finally:
            lock.release()

    def test_a_live_holder_is_never_broken_even_when_old(self, tmp_path: Path) -> None:
        path = tmp_path / "c.lock"
        record = LockRecord(
            pid=4242,
            hostname=_hostname(),
            acquired_at=NOW - timedelta(minutes=5),
            heartbeat_at=NOW - timedelta(minutes=5),
            purpose="collect",
        )
        path.write_text(record.as_json(), encoding="utf-8")
        with pytest.raises(LockHeld):
            RunLock(path, now=lambda: NOW, is_alive=lambda pid: True).acquire()

    def test_a_heartbeat_older_than_the_stale_window_is_broken(self, tmp_path: Path) -> None:
        """Covers a holder on another host, whose liveness cannot be checked."""
        path = tmp_path / "c.lock"
        record = LockRecord(
            pid=4242,
            hostname="some-other-host",
            acquired_at=NOW - timedelta(hours=4),
            heartbeat_at=NOW - timedelta(hours=4),
            purpose="collect",
        )
        path.write_text(record.as_json(), encoding="utf-8")
        lock = RunLock(path, now=lambda: NOW, is_alive=lambda pid: True).acquire()
        try:
            assert lock.broke_stale_lock
        finally:
            lock.release()

    def test_a_corrupt_lock_file_is_treated_as_stale_not_fatal(self, tmp_path: Path) -> None:
        """A partial write is exactly what a crash leaves behind."""
        path = tmp_path / "c.lock"
        path.write_text('{"pid": 12', encoding="utf-8")
        lock = RunLock(path, now=lambda: NOW).acquire()
        try:
            assert lock.broke_stale_lock
        finally:
            lock.release()

    def test_a_heartbeat_refreshes_without_changing_the_holder(self, tmp_path: Path) -> None:
        moment = {"now": NOW}
        lock = RunLock(tmp_path / "c.lock", now=lambda: moment["now"]).acquire()
        try:
            acquired = held_by(tmp_path / "c.lock")
            moment["now"] = NOW + timedelta(minutes=10)
            lock.heartbeat()
            refreshed = held_by(tmp_path / "c.lock")
            assert acquired is not None and refreshed is not None
            assert refreshed.pid == acquired.pid == os.getpid()
            assert refreshed.acquired_at == acquired.acquired_at
            assert refreshed.heartbeat_at > acquired.heartbeat_at
        finally:
            lock.release()

    def test_releasing_does_not_delete_a_lock_that_was_taken_over(self, tmp_path: Path) -> None:
        """If ours was broken as stale, the file belongs to someone else now."""
        path = tmp_path / "c.lock"
        mine = RunLock(path, now=lambda: NOW).acquire()
        other = LockRecord(
            pid=os.getpid() + 1,
            hostname=_hostname(),
            acquired_at=NOW,
            heartbeat_at=NOW,
            purpose="collect",
        )
        path.write_text(other.as_json(), encoding="utf-8")
        mine.release()
        assert path.exists(), "another collector's lock must survive our release"

    def test_inspecting_an_absent_lock_returns_nothing(self, tmp_path: Path) -> None:
        assert held_by(tmp_path / "absent.lock") is None


def _hostname() -> str:
    import socket

    return socket.gethostname()


# ---------------------------------------------------------------- O5 storage


class TestStoragePaths:
    def test_every_path_is_derived_from_one_root(self, tmp_path: Path) -> None:
        paths = StoragePaths.from_environment(tmp_path / "state")
        assert paths.database.parent == paths.root
        assert paths.manifests.parent == paths.root
        assert set(paths.as_dict()) == {
            "root",
            "database",
            "manifests",
            "corpus",
            "backups",
            "logs",
            "lock",
        }

    def test_individual_paths_can_be_overridden(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("BTC_INTEL_BACKUP_DIR", str(tmp_path / "elsewhere"))
        paths = StoragePaths.from_environment(tmp_path / "state")
        assert paths.backups == tmp_path / "elsewhere"
        assert paths.root == tmp_path / "state"

    def test_the_root_comes_from_the_environment_when_unset(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("BTC_INTEL_STATE_ROOT", str(tmp_path / "from-env"))
        assert StoragePaths.from_environment().root == tmp_path / "from-env"

    def test_validation_proves_writability_by_writing(self, tmp_path: Path) -> None:
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        result = validate(paths, require_free_bytes=1)
        assert result.ok
        assert not list(paths.root.glob(".btc-intel-write-probe"))

    def test_an_unusable_location_refuses_to_start(self, tmp_path: Path) -> None:
        """A collector that cannot persist must not run cleanly and lose everything."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        with pytest.raises(ConfigurationError, match="refusing to start"):
            require_usable(paths, require_free_bytes=1 << 62)

    def test_a_low_disk_check_names_the_shortfall(self, tmp_path: Path) -> None:
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        result = validate(paths, require_free_bytes=1 << 62)
        assert not result.ok
        assert any(check.name == "free_space" for check in result.failures())

    def test_an_ephemeral_root_is_flagged(self) -> None:
        """The most expensive misconfiguration produces no error at all."""
        assert looks_ephemeral(Path("/tmp/btc-intel-state")) is not None
        assert "ephemeral" in (looks_ephemeral(Path("/tmp/btc-intel-state")) or "")
        assert looks_ephemeral(Path("/srv/btc-intel-state")) is None


# --------------------------------------------------------- O6 crash recovery


class TestCrashRecovery:
    """Fault injection at each stage. The next scheduled run is the recovery."""

    def _setup(self, tmp_path: Path):
        store = IntelligenceStore(tmp_path / "c.duckdb")
        watch = [
            WatchEntity(
                canonical_name="SEC",
                aliases=("Securities and Exchange Commission",),
                entity_type=EntityType.REGULATOR,
                topics=("bitcoin regulation",),
                expected_event_types=(EventType.REGULATION,),
            )
        ]
        queries = QueryPlanner().plan(watch, NOW)
        docs = [
            document(0, publisher="sec.example.gov", retrieved=NOW - timedelta(hours=1)).model_copy(
                update={"query": queries[0].query}
            )
        ]
        from market_intelligence.collection.fixtures import StaticFixtureProvider

        providers = {"fixture": StaticFixtureProvider(docs, now=lambda: NOW, restamp=False)}
        configs = {
            "fixture": ProviderConfig(
                id="fixture", type="fixture", source_category=ProviderCategory.GENERAL_WEB, timeout=5.0
            )
        }
        return store, queries, providers, configs

    def test_a_crash_before_the_manifest_leaves_no_completed_run(self, tmp_path: Path) -> None:
        """The manifest is the completion marker; a crash must leave none."""
        store, queries, providers, configs = self._setup(tmp_path)
        try:
            collector = ForwardCollector(store, now=lambda: NOW)
            manifest_path = tmp_path / "manifests" / "run.json"

            original = collector.clusters.replace_all

            def explode(*args: object, **kwargs: object) -> int:
                raise RuntimeError("crash after persistence, before the manifest")

            collector.clusters.replace_all = explode  # type: ignore[method-assign]
            with pytest.raises(RuntimeError, match="crash after persistence"):
                collector.collect(
                    queries,
                    MultiProviderRetriever(providers, configs),
                    RuleBasedExtractor(),
                    {},
                    manifest_path=manifest_path,
                )
            collector.clusters.replace_all = original  # type: ignore[method-assign]

            assert not manifest_path.exists(), "no manifest means no completed run"
            # Evidence that was persisted before the crash is kept, not lost.
            assert store.documents_as_of(NOW + timedelta(days=1)), "persisted evidence survives"
        finally:
            store.close()

    def test_restarting_after_a_crash_completes_without_duplicating(self, tmp_path: Path) -> None:
        store, queries, providers, configs = self._setup(tmp_path)
        try:
            collector = ForwardCollector(store, now=lambda: NOW)
            original = collector.clusters.replace_all
            collector.clusters.replace_all = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign,assignment]
            with pytest.raises(RuntimeError):
                collector.collect(queries, MultiProviderRetriever(providers, configs), RuleBasedExtractor(), {})
            collector.clusters.replace_all = original  # type: ignore[method-assign]

            before = {
                item.document_id: item.available_at
                for item in store.documents_as_of(NOW + timedelta(days=1))
            }
            later = NOW + timedelta(hours=2)
            collector._now = lambda: later  # noqa: SLF001 -- the restart is the point
            result = collector.collect(
                QueryPlanner().plan(
                    [
                        WatchEntity(
                            canonical_name="SEC",
                            aliases=("Securities and Exchange Commission",),
                            entity_type=EntityType.REGULATOR,
                            topics=("bitcoin regulation",),
                            expected_event_types=(EventType.REGULATION,),
                        )
                    ],
                    later,
                ),
                MultiProviderRetriever(providers, configs),
                RuleBasedExtractor(),
                {},
                manifest_path=tmp_path / "manifests" / "restart.json",
            )
            after = {
                item.document_id: item.available_at
                for item in store.documents_as_of(later + timedelta(days=1))
            }
            assert after == before, "restart must not duplicate or move evidence"
            assert result.manifest.documents_new == 0
            assert (tmp_path / "manifests" / "restart.json").exists()
        finally:
            store.close()

    def test_a_failing_provider_does_not_advance_its_watermark(self, tmp_path: Path) -> None:
        import json as json_module

        from market_intelligence.collection.fixtures import OutageProvider

        store, queries, providers, configs = self._setup(tmp_path)
        try:
            providers["outage"] = OutageProvider()
            configs["outage"] = ProviderConfig(
                id="outage", type="fixture", source_category=ProviderCategory.GENERAL_WEB, timeout=5.0
            )
            ForwardCollector(store, now=lambda: NOW).collect(
                queries, MultiProviderRetriever(providers, configs), RuleBasedExtractor(), {}
            )
            rows = store.connection.execute("SELECT payload FROM watermarks").fetchall()
            recorded = {json_module.loads(payload)["provider_id"] for (payload,) in rows}
            assert "fixture" in recorded
            assert "outage" not in recorded, "a failed provider must not claim progress"
        finally:
            store.close()


# ------------------------------------------------------- O3/O4 scheduled entry


class TestScheduledEntrypoint:
    """The single command a scheduler calls, and the codes it must return."""

    def _arguments(self, tmp_path: Path, moment: datetime):
        from market_intelligence.collection.fixtures import StaticFixtureProvider
        from market_intelligence.collection.policy import ProviderDeclaration, ProviderPolicy

        watch = [
            WatchEntity(
                canonical_name="SEC",
                aliases=("Securities and Exchange Commission",),
                entity_type=EntityType.REGULATOR,
                topics=("bitcoin regulation",),
                expected_event_types=(EventType.REGULATION,),
            )
        ]
        queries = QueryPlanner().plan(watch, moment)
        docs = [
            document(0, publisher="sec.example.gov", retrieved=moment - timedelta(hours=1)).model_copy(
                update={"query": queries[0].query}
            )
        ]

        def build_retriever(due: object) -> MultiProviderRetriever:
            provider = StaticFixtureProvider(docs, name="syndication", now=lambda: moment, restamp=False)
            config = ProviderConfig(
                id="syndication",
                type="fixture",
                source_category=ProviderCategory.GENERAL_WEB,
                timeout=5.0,
            )
            return MultiProviderRetriever({"syndication": provider}, {"syndication": config})

        declaration = ProviderDeclaration(
            provider_id="syndication",
            policy=ProviderPolicy.PUBLIC_DOCUMENTED,
            purpose="test",
            data_returned="test",
            minimum_interval_seconds=900,
        )
        return {
            "build_queries": lambda when: QueryPlanner().plan(watch, when),
            "build_retriever": build_retriever,
            "extractor": RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}),
            "configuration": {"test": True},
            "declarations": {"syndication": declaration},
            "require_free_bytes": 1,
        }

    def test_a_cycle_runs_and_writes_its_manifest_last(self, tmp_path: Path) -> None:
        from market_intelligence.ops.scheduled import EXIT_OK, run_scheduled

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        outcome = run_scheduled(paths, now=lambda: NOW, **self._arguments(tmp_path, NOW))
        assert outcome.exit_code == EXIT_OK
        assert outcome.ran
        assert outcome.manifest_path is not None and outcome.manifest_path.exists()
        assert outcome.result is not None and outcome.result.manifest.documents_new == 1
        assert not paths.lock.exists(), "the lock is released on the way out"

    def test_a_second_invocation_inside_the_cadence_floor_does_nothing(self, tmp_path: Path) -> None:
        """O4. Exit 4 is not a failure; treating it as one trains operators to
        ignore the collector's exit code."""
        from market_intelligence.ops.scheduled import EXIT_NOTHING_DUE, run_scheduled

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        run_scheduled(paths, now=lambda: NOW, **self._arguments(tmp_path, NOW))
        again = run_scheduled(
            paths, now=lambda: NOW + timedelta(minutes=5), **self._arguments(tmp_path, NOW)
        )
        assert again.exit_code == EXIT_NOTHING_DUE
        assert not again.ran
        assert again.skipped_providers == ("syndication",)

    def test_a_provider_due_again_after_the_floor_runs(self, tmp_path: Path) -> None:
        from market_intelligence.ops.scheduled import EXIT_OK, run_scheduled

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        run_scheduled(paths, now=lambda: NOW, **self._arguments(tmp_path, NOW))
        later = NOW + timedelta(hours=1)
        again = run_scheduled(paths, now=lambda: later, **self._arguments(tmp_path, later))
        assert again.exit_code == EXIT_OK
        assert again.due_providers == ("syndication",)
        assert again.result is not None and again.result.manifest.documents_new == 0

    def test_a_held_lock_is_reported_rather_than_failing(self, tmp_path: Path) -> None:
        """O2. A scheduler firing during a long cycle is normal."""
        from market_intelligence.ops.scheduled import EXIT_LOCK_HELD, run_scheduled

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        holder = RunLock(paths.lock, now=lambda: NOW).acquire()
        try:
            outcome = run_scheduled(paths, now=lambda: NOW, **self._arguments(tmp_path, NOW))
            assert outcome.exit_code == EXIT_LOCK_HELD
            assert not outcome.ran
            assert "is held by pid" in outcome.reason
        finally:
            holder.release()

    def test_a_stale_lock_is_broken_and_reported(self, tmp_path: Path) -> None:
        from market_intelligence.ops.scheduled import EXIT_OK, run_scheduled

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        RunLock(paths.lock, now=lambda: NOW).acquire()  # deliberately abandoned
        later = NOW + timedelta(hours=2)
        outcome = run_scheduled(
            paths,
            now=lambda: later,
            **self._arguments(tmp_path, later),
        )
        assert outcome.exit_code in (EXIT_OK,)
        assert outcome.stale_lock_broken

    def test_unusable_storage_refuses_to_start(self, tmp_path: Path) -> None:
        from market_intelligence.ops.scheduled import run_scheduled

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        arguments = self._arguments(tmp_path, NOW)
        arguments["require_free_bytes"] = 1 << 62
        with pytest.raises(ConfigurationError, match="refusing to start"):
            run_scheduled(paths, now=lambda: NOW, **arguments)

    def test_an_ephemeral_state_root_is_warned_about_but_still_runs(self, tmp_path: Path) -> None:
        """A warning, not a refusal: an operator may genuinely be testing."""
        from market_intelligence.ops.scheduled import run_scheduled

        ephemeral = Path("/tmp") / "btc-intel-ops-test" / tmp_path.name
        paths = StoragePaths.from_environment(ephemeral).ensure()
        try:
            outcome = run_scheduled(paths, now=lambda: NOW, **self._arguments(tmp_path, NOW))
            assert outcome.ran
            assert any("ephemeral" in warning for warning in outcome.warnings)
        finally:
            import shutil

            shutil.rmtree(ephemeral.parent, ignore_errors=True)

    def test_the_outcome_serialises_for_a_scheduler_log(self, tmp_path: Path) -> None:
        from market_intelligence.ops.scheduled import run_scheduled

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        payload = run_scheduled(paths, now=lambda: NOW, **self._arguments(tmp_path, NOW)).as_dict()
        assert payload["ran"] is True
        assert payload["collection"]["documents_new"] == 1
        assert "corpus_id" in payload["collection"]

    def test_cadence_is_measured_from_completed_retrievals(self, tmp_path: Path) -> None:
        """O4. A failed attempt did not consume the provider's quota in any way
        that matters, so it must not delay the next try."""
        from market_intelligence.collection.policy import ProviderDeclaration, ProviderPolicy
        from market_intelligence.ops.scheduled import due_providers

        declaration = ProviderDeclaration(
            provider_id="p", policy=ProviderPolicy.PUBLIC_DOCUMENTED, purpose="x",
            data_returned="x", minimum_interval_seconds=900,
        )
        never = due_providers({"p": declaration}, {}, NOW)
        assert never == (["p"], [])
        recent = due_providers({"p": declaration}, {"p": NOW - timedelta(seconds=60)}, NOW)
        assert recent == ([], ["p"])
        old = due_providers({"p": declaration}, {"p": NOW - timedelta(hours=2)}, NOW)
        assert old == (["p"], [])
