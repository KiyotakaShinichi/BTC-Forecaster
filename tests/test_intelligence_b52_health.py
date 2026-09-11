"""B5.2 — the health check says what is wrong, and a failed cycle fails.

Before B5.2 three things an operator needs to hear about were invisible:

* a cycle that ran and read nothing from any provider was recorded FAILED but
  exited 0, so its unit reported success;
* the watchdog's ALL_PROVIDERS_FAILED read "has any provider ever succeeded",
  so it could only fire on a deployment that had never collected;
* nothing watched the disk above the collector's own 64 MiB refusal, and
  nothing checked that the deployed configuration still loads.

Pinned here: the watchdog's new checks as a pure function; a failed cycle
exiting 2 without advancing a watermark; `ops-watch` end to end, including that
a quiet day with no events is healthy; and `ops-status` answering what the last
run did.

No network. The end-to-end tests run on the real clock, because `ops-status`
does, and serve feeds from memory.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from market_intelligence.cli import main
from market_intelligence.collection import syndication
from market_intelligence.collection.backoff import FailureClass, ProviderFailure
from market_intelligence.ops import paths as ops_paths
from market_intelligence.ops.backup import create_backup
from market_intelligence.ops.integrity import IntegrityStatus
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import CollectionProfile, collect_once
from market_intelligence.ops.scheduled import EXIT_FAILED, EXIT_NOTHING_DUE, EXIT_OK
from market_intelligence.ops.watchdog import AlertCode, Severity, WatchdogInput, WatchdogPolicy, assess
from market_intelligence.storage import IntelligenceStore
from tests.test_intelligence_deployment import FEED_PAYLOAD, minimal

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
POLICY = WatchdogPolicy()


def healthy(**overrides: Any) -> WatchdogInput:
    base: dict[str, Any] = dict(
        now=NOW,
        last_run_at=NOW,
        last_successful_run_at=NOW,
        providers_enabled=1,
        providers_healthy=1,
        provider_last_success={"syndication": NOW},
        documents_last_24h=0,
        events_last_24h=0,
        quarantined_last_24h=0,
        storage_ok=True,
        storage_detail="",
        integrity_status=IntegrityStatus.OK,
        integrity_detail="",
        last_backup_at=NOW - timedelta(hours=12),
        consecutive_zero_days_with_success=1,
    )
    base.update(overrides)
    return WatchdogInput(**base)


def codes(state: WatchdogInput) -> dict[AlertCode, Severity]:
    return {alert.code: alert.severity for alert in assess(state).alerts}


class TestTheWatchdog:
    def test_a_quiet_day_with_no_events_is_healthy(self) -> None:
        result = assess(healthy(documents_last_24h=0, events_last_24h=0))
        assert result.healthy and result.exit_code == 0

    def test_disk_below_the_warning_line_warns(self) -> None:
        assert codes(healthy(free_bytes=POLICY.disk_warning_free_bytes - 1)) == {AlertCode.DISK_LOW: Severity.WARNING}

    def test_disk_below_the_critical_line_is_critical(self) -> None:
        state = healthy(free_bytes=POLICY.disk_critical_free_bytes - 1)
        assert codes(state) == {AlertCode.DISK_LOW: Severity.CRITICAL} and assess(state).exit_code == 2

    def test_ample_or_unmeasured_disk_raises_nothing(self) -> None:
        assert assess(healthy(free_bytes=10 * POLICY.disk_warning_free_bytes)).healthy
        assert assess(healthy(free_bytes=None)).healthy

    def test_the_disk_lines_sit_above_the_collectors_own_refusal(self) -> None:
        assert POLICY.disk_warning_free_bytes > POLICY.disk_critical_free_bytes > 64 * 1024 * 1024

    def test_a_broken_configuration_is_critical(self) -> None:
        assert codes(healthy(configuration_error="profile not found")) == {
            AlertCode.CONFIGURATION_INVALID: Severity.CRITICAL
        }

    def test_a_failed_last_run_is_an_outage_even_after_past_successes(self) -> None:
        assert codes(healthy(last_run_status="FAILED")) == {AlertCode.ALL_PROVIDERS_FAILED: Severity.CRITICAL}

    def test_a_degraded_last_run_collected_and_is_not_an_outage(self) -> None:
        assert assess(healthy(last_run_status="DEGRADED")).healthy


def serve(monkeypatch: pytest.MonkeyPatch, payload: bytes | None) -> None:
    """Serve `payload` from every feed, or fail every feed when it is None."""

    def fetch(url: str, timeout: float, agent: str | None = None) -> bytes:
        if payload is None:
            raise ProviderFailure(FailureClass.PERMANENT, f"HTTP 403 for {url}")
        return payload

    monkeypatch.setattr(syndication, "_default_opener", fetch)


def published(moment: datetime) -> bytes:
    """The fixture feed, republished at `moment`, so its item is inside the planner's lookback."""
    return FEED_PAYLOAD.replace(b"Wed, 02 Sep 2026 09:30:00 GMT", moment.strftime("%a, %d %b %Y %H:%M:%S GMT").encode())


def state(tmp_path: Path) -> StoragePaths:
    return StoragePaths.from_environment(tmp_path / "state").ensure()


def profile() -> CollectionProfile:
    return CollectionProfile.from_mapping(minimal())


class TestAFailedCycleFails:
    def test_a_cycle_that_read_nothing_exits_2_and_records_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        serve(monkeypatch, None)
        paths = state(tmp_path)
        outcome = collect_once(paths, profile(), now=lambda: NOW, require_free_bytes=1)
        assert (outcome.exit_code, outcome.ran, outcome.run_status) == (EXIT_FAILED, True, "FAILED")
        assert outcome.as_dict()["run_status"] == "FAILED"
        store = IntelligenceStore(paths.database)
        try:
            assert [row[0] for row in store.connection.execute("SELECT status FROM runs").fetchall()] == ["FAILED"]
        finally:
            store.close()

    def test_a_failed_cycle_advances_no_watermark(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The next cycle is still due: nothing was recorded as collected."""
        serve(monkeypatch, None)
        paths = state(tmp_path)
        collect_once(paths, profile(), now=lambda: NOW, require_free_bytes=1)
        serve(monkeypatch, FEED_PAYLOAD)
        retry = collect_once(paths, profile(), now=lambda: NOW + timedelta(minutes=1), require_free_bytes=1)
        assert retry.ran and retry.exit_code == EXIT_OK
        again = collect_once(paths, profile(), now=lambda: NOW + timedelta(minutes=2), require_free_bytes=1)
        assert again.exit_code == EXIT_NOTHING_DUE, "a successful cycle must advance the watermark"

    def test_a_cycle_that_read_something_still_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        serve(monkeypatch, FEED_PAYLOAD)
        outcome = collect_once(state(tmp_path), profile(), now=lambda: NOW, require_free_bytes=1)
        assert outcome.exit_code == EXIT_OK and outcome.run_status in {"SUCCESS", "DEGRADED", "PARTIAL_SUCCESS"}

    def test_the_scheduler_entrypoint_returns_2_and_logs_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        serve(monkeypatch, None)
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(minimal()), encoding="utf-8")
        root = tmp_path / "state"
        code = main(["collect-scheduled", "--profile", str(profile_path), "--state-root", str(root), "--json"])
        assert code == EXIT_FAILED
        (log,) = (root / "logs").glob("collect-*.json")
        recorded = json.loads(log.read_text(encoding="utf-8"))
        assert (recorded["exit_code"], recorded["run_status"]) == (EXIT_FAILED, "FAILED")


def collect_at(paths: StoragePaths, monkeypatch: pytest.MonkeyPatch, moment: datetime, *, failing: bool = False) -> None:
    serve(monkeypatch, None if failing else published(moment - timedelta(hours=1)))
    collect_once(paths, profile(), now=lambda: moment, require_free_bytes=1)


def back_up(paths: StoragePaths) -> None:
    store = IntelligenceStore(paths.database)
    try:
        create_backup(store, paths.database, paths.backups / "corpus-test.tar.gz")
    finally:
        store.close()


def watch(paths: StoragePaths, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, dict[AlertCode, str]]:
    code = main(["--db", str(paths.database), "ops-watch", "--json", "--state-root", str(paths.root), *extra])
    payload = json.loads(capsys.readouterr().out)
    return code, {AlertCode(alert["code"]): alert["severity"] for alert in payload["alerts"]}


class TestTheHealthCheck:
    @pytest.fixture(autouse=True)
    def ample_disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The host's own free space must not decide these tests; the disk tests set their own."""
        monkeypatch.setattr(ops_paths, "_free_bytes", lambda path: 64 * 1024**3)

    def test_a_fresh_healthy_deployment_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = state(tmp_path)
        collect_at(paths, monkeypatch, datetime.now(timezone.utc) - timedelta(minutes=5))
        back_up(paths)
        code, alerts = watch(paths, capsys)
        assert code == 0 and set(alerts) == {AlertCode.COLLECTION_HEALTHY}, alerts

    def test_feeds_read_with_nothing_matched_yet_is_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A fresh deployment on a quiet week: every feed read, no document admitted.

        The fixture item is dated 2026-09-02, outside the planner's lookback from
        today, so the cycle succeeds and stores nothing. That is not an outage and
        not a corrupt corpus, and the health check must not say it is.
        """
        paths = state(tmp_path)
        serve(monkeypatch, FEED_PAYLOAD)
        collect_once(paths, profile(), now=lambda: datetime.now(timezone.utc) - timedelta(minutes=5), require_free_bytes=1)
        store = IntelligenceStore(paths.database)
        try:
            documents = store.connection.execute("SELECT count(*) FROM documents").fetchone()
        finally:
            store.close()
        assert documents is not None and documents[0] == 0, "the fixture was admitted; this test would prove nothing"
        back_up(paths)
        code, alerts = watch(paths, capsys)
        assert code == 0 and set(alerts) == {AlertCode.COLLECTION_HEALTHY}, alerts

    def test_a_collector_that_stopped_is_critical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = state(tmp_path)
        collect_at(paths, monkeypatch, datetime.now(timezone.utc) - timedelta(hours=7))
        back_up(paths)
        code, alerts = watch(paths, capsys)
        assert code == 2 and alerts[AlertCode.COLLECTION_STALE] == "CRITICAL"

    def test_every_feed_failing_is_critical_at_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An hour after a good cycle -- long before the six-hour staleness line."""
        paths = state(tmp_path)
        now = datetime.now(timezone.utc)
        collect_at(paths, monkeypatch, now - timedelta(hours=1))
        collect_at(paths, monkeypatch, now - timedelta(minutes=5), failing=True)
        back_up(paths)
        code, alerts = watch(paths, capsys)
        assert code == 2 and alerts[AlertCode.ALL_PROVIDERS_FAILED] == "CRITICAL"
        assert AlertCode.COLLECTION_STALE not in alerts

    def test_low_disk_is_reported_before_the_collector_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = state(tmp_path)
        collect_at(paths, monkeypatch, datetime.now(timezone.utc) - timedelta(minutes=5))
        back_up(paths)
        monkeypatch.setattr(ops_paths, "_free_bytes", lambda path: 512 * 1024 * 1024)
        code, alerts = watch(paths, capsys)
        assert code == 1 and alerts == {AlertCode.DISK_LOW: "WARNING"}

    def test_unusable_storage_is_critical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = state(tmp_path)
        collect_at(paths, monkeypatch, datetime.now(timezone.utc) - timedelta(minutes=5))
        back_up(paths)
        monkeypatch.setattr(ops_paths, "_free_bytes", lambda path: 1024)
        code, alerts = watch(paths, capsys)
        assert code == 2
        assert alerts[AlertCode.STORAGE_FAILURE] == "CRITICAL" and alerts[AlertCode.DISK_LOW] == "CRITICAL"

    def test_a_broken_configuration_is_critical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = state(tmp_path)
        collect_at(paths, monkeypatch, datetime.now(timezone.utc) - timedelta(minutes=5))
        back_up(paths)
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps(minimal(feeds=["not-a-feed"])), encoding="utf-8")
        code, alerts = watch(paths, capsys, "--profile", str(broken))
        assert code == 2 and alerts[AlertCode.CONFIGURATION_INVALID] == "CRITICAL"


class TestTheStatusAnswersTheOperatorsQuestions:
    def status(self, paths: StoragePaths, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
        assert main(["--db", str(paths.database), "ops-status", "--json", "--state-root", str(paths.root)]) == 0
        return dict(json.loads(capsys.readouterr().out))

    def test_it_reports_what_the_last_run_did(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = state(tmp_path)
        collect_at(paths, monkeypatch, datetime.now(timezone.utc) - timedelta(minutes=5))
        payload = self.status(paths, capsys)
        run = payload["last_run"]
        assert run["status"] in {"SUCCESS", "DEGRADED", "PARTIAL_SUCCESS"} and run["failed_providers"] == []
        assert run["documents_accepted"] >= 1 and run["events_accepted"] >= 1
        for key in ("documents_rejected", "events_rejected", "finished_at"):
            assert key in run
        assert payload["collection"]["successful_days"] == 1
        assert payload["collection_lag"]["measured"] >= 1
        assert payload["free_bytes"] is not None and "backup_age_seconds" in payload

    def test_it_names_the_providers_that_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = state(tmp_path)
        collect_at(paths, monkeypatch, datetime.now(timezone.utc) - timedelta(minutes=5), failing=True)
        run = self.status(paths, capsys)["last_run"]
        assert run["status"] == "FAILED" and run["failed_providers"] == ["syndication"]

    def test_the_human_view_says_the_same(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = state(tmp_path)
        collect_at(paths, monkeypatch, datetime.now(timezone.utc) - timedelta(minutes=5), failing=True)
        assert main(["--db", str(paths.database), "ops-status", "--state-root", str(paths.root)]) == 0
        text = capsys.readouterr().out
        assert "last run               FAILED" in text and "failed providers: syndication" in text
        assert "configuration          (not checked; pass --profile)" in text
