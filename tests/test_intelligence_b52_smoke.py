"""B5.2 — the deployment smoke test, and locking across real processes.

Deployment files existing is not a deployment. `ops-smoke` runs one real cycle
on a host and then checks everything that cycle should leave behind. Here it is
exercised end to end, offline, on the real clock -- which is also the
documented dry run of the deployment package when no host is available.

Pinned here:

* on a healthy feed every check passes: a recorded run, a provider that
  answered, documents, events with confidences that vary, point-in-time timing,
  an advanced watermark, a backup that restores, a restart that polls nothing,
  and a health check that is not critical;
* on a quiet feed documents and events are NOT_OBSERVED and the smoke test
  still passes -- an absence of news is not a failure;
* when every feed fails the smoke test fails, the watermark is left alone, and
  the provider stays due;
* locking holds across processes: a live holder is refused and left unharmed,
  a second collector is refused, and a killed holder is recovered -- at once on
  POSIX, after the heartbeat's stale window on Windows.

No network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.cli import main
from market_intelligence.collection import syndication
from market_intelligence.collection.backoff import FailureClass, ProviderFailure
from market_intelligence.ops import paths as ops_paths
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import CollectionProfile, collect_once
from market_intelligence.ops.runlock import DEFAULT_STALE_AFTER, LockHeld, RunLock
from market_intelligence.ops.scheduled import EXIT_LOCK_HELD
from market_intelligence.ops.smoke import Outcome, run_smoke

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def ample_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host's own free space must not decide a smoke test's health check."""
    monkeypatch.setattr(ops_paths, "_free_bytes", lambda path: 64 * 1024**3)


def feed(*items: tuple[str, datetime]) -> bytes:
    body = "".join(
        f"<item><title>{title}</title><link>https://sec.example.gov/{index}</link>"
        f"<description>{title}</description><pubDate>{moment.strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate></item>"
        for index, (title, moment) in enumerate(items)
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>SEC</title>{body}</channel></rss>'.encode()


def serve(monkeypatch: pytest.MonkeyPatch, payload: bytes | None) -> None:
    def fetch(url: str, timeout: float, agent: str | None = None) -> bytes:
        if payload is None:
            raise ProviderFailure(FailureClass.PERMANENT, f"HTTP 403 for {url}")
        return payload

    monkeypatch.setattr(syndication, "_default_opener", fetch)


def profile_file(tmp_path: Path) -> Path:
    raw = {
        "name": "b52-smoke",
        "user_agent": "BTC-Forecaster Research <${BTC_INTEL_CONTACT}>",
        "feeds": ["sec-press"],
        "watchlist": [
            {
                "canonical_name": "SEC",
                "aliases": ["Securities and Exchange Commission"],
                "entity_type": "REGULATOR",
                "topics": ["bitcoin regulation", "crypto enforcement"],
                "expected_event_types": ["REGULATION"],
            }
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture
def deployed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StoragePaths, Path]:
    monkeypatch.setenv("BTC_INTEL_CONTACT", "collector-ops@example.org")
    return StoragePaths.from_environment(tmp_path / "state").ensure(), profile_file(tmp_path)


def outcomes(report_checks: list[dict[str, str]]) -> dict[str, str]:
    return {check["name"]: check["outcome"] for check in report_checks}


def smoke(deployed: tuple[StoragePaths, Path], capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, dict[str, str]]:
    paths, profile = deployed
    code = main(["ops-smoke", "--profile", str(profile), "--state-root", str(paths.root), "--require-free-mb", "1", "--json", *extra])
    return code, outcomes(json.loads(capsys.readouterr().out)["checks"])


class TestTheSmokeTest:
    def test_a_healthy_deployment_passes_every_check(
        self, deployed: tuple[StoragePaths, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        serve(
            monkeypatch,
            feed(
                ("SEC charges adviser over bitcoin regulation breach", hour_ago),
                ("SEC proposes rule on crypto enforcement and regulation", hour_ago - timedelta(minutes=30)),
            ),
        )
        code, checks = smoke(deployed, capsys)
        assert code == 0, checks
        assert set(checks.values()) == {Outcome.PASS.value}, checks
        paths, _ = deployed
        assert list(paths.backups.glob("corpus-*.tar.gz")) and not paths.lock.exists()

    def test_a_second_run_checks_the_same_run_and_polls_nothing(
        self, deployed: tuple[StoragePaths, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        serve(monkeypatch, feed(("SEC charges adviser over bitcoin regulation breach", datetime.now(timezone.utc) - timedelta(hours=1))))
        assert smoke(deployed, capsys)[0] == 0
        code, checks = smoke(deployed, capsys, "--no-backup")
        assert code == 0 and checks["cycle"] == "PASS" and checks["restart_recovery"] == "PASS"
        assert "backup" not in checks

    def test_a_quiet_week_is_not_observed_and_still_passes(
        self, deployed: tuple[StoragePaths, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        serve(monkeypatch, feed(("SEC announces a roundtable agenda", datetime(2020, 1, 1, tzinfo=timezone.utc))))
        code, checks = smoke(deployed, capsys)
        assert code == 0, checks
        assert checks["documents_stored"] == checks["events_extracted"] == checks["confidence_varies"] == "NOT_OBSERVED"
        assert checks["provider_request"] == checks["watermark"] == checks["restore"] == "PASS"

    def test_every_feed_failing_fails_and_moves_no_watermark(
        self, deployed: tuple[StoragePaths, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        serve(monkeypatch, None)
        code, checks = smoke(deployed, capsys, "--no-backup")
        assert code == 2
        assert checks["cycle"] == checks["run_recorded"] == checks["provider_request"] == "FAIL"
        assert checks["watermark"] == "PASS", "a failed run must not advance the watermark"
        assert checks["restart_recovery"] == "PASS", "a failed run must leave the provider due"
        assert checks["health"] == "FAIL"

    def test_a_profile_without_its_contact_is_refused_before_anything_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("BTC_INTEL_CONTACT", raising=False)
        root = tmp_path / "state"
        code = main(["ops-smoke", "--profile", str(profile_file(tmp_path)), "--state-root", str(root)])
        assert code == 2 and "BTC_INTEL_CONTACT" in capsys.readouterr().err
        assert not (root / "intelligence.duckdb").exists()

    def test_the_report_is_a_function_too(
        self, deployed: tuple[StoragePaths, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        serve(monkeypatch, feed(("SEC charges adviser over bitcoin regulation breach", datetime.now(timezone.utc) - timedelta(hours=1))))
        paths, path = deployed
        report = run_smoke(paths, CollectionProfile.load(path), require_free_bytes=1, backup=False)
        assert report.ok and "smoke test passed" in report.human_readable()


HOLDER = """
import sys, time
from market_intelligence.ops.runlock import RunLock
RunLock(sys.argv[1], purpose="another collector").acquire()
print("held", flush=True)
time.sleep(120)
"""


@pytest.fixture
def holder(tmp_path: Path):
    """A separate process holding the run lock, until the test ends or kills it."""
    lock = StoragePaths.from_environment(tmp_path / "state").ensure().lock
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(lock)],
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(REPO)),
    )
    assert process.stdout is not None and process.stdout.readline().strip() == "held"
    yield process, lock
    if process.poll() is None:
        process.kill()
        process.wait()


class TestLockingAcrossProcesses:
    def test_a_live_holder_is_refused_and_left_unharmed(self, holder) -> None:
        process, lock = holder
        with pytest.raises(LockHeld):
            RunLock(lock).acquire()
        assert process.poll() is None, "asking whether the holder was alive harmed it"

    def test_a_second_collector_is_refused(self, holder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        process, _ = holder
        serve(monkeypatch, feed(("SEC charges adviser over bitcoin regulation breach", datetime.now(timezone.utc))))
        monkeypatch.setenv("BTC_INTEL_CONTACT", "collector-ops@example.org")
        outcome = collect_once(
            StoragePaths.from_environment(tmp_path / "state"),
            CollectionProfile.load(profile_file(tmp_path)),
            require_free_bytes=1,
        )
        assert outcome.exit_code == EXIT_LOCK_HELD and process.poll() is None

    def test_a_killed_holder_is_recovered(self, holder) -> None:
        process, lock = holder
        process.kill()
        process.wait()
        if os.name == "nt":
            # No safe liveness probe on Windows: the heartbeat's stale window decides.
            with pytest.raises(LockHeld):
                RunLock(lock).acquire()
            later = datetime.now(timezone.utc) + DEFAULT_STALE_AFTER + timedelta(minutes=1)
            recovered = RunLock(lock, now=lambda: later).acquire()
        else:
            recovered = RunLock(lock).acquire()
        try:
            assert recovered.broke_stale_lock
        finally:
            recovered.release()
        assert not lock.exists(), "the recovered lock was not cleaned up"
