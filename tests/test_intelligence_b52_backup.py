"""B5.2 — a backup proves itself by restoring, and retention never costs the last good one.

B4.1-Ops could write a verifiable archive and restore one into a new location.
What a production host still lacked: a record of what the archive held that
survives DuckDB rewriting its own file, a checksum an off-host copy can be
checked against with standard tools, a restore rehearsal that runs every time
rather than when somebody remembers, a retention policy, and a way for the
health check to hear that last night's backup failed.

Pinned here:

* the manifest's counts and content fingerprint describe the archived copy,
  and a v1 manifest without a fingerprint still parses;
* verify_backup checks the checksum, restores into a throwaway location, checks
  members, runs the integrity check, compares counts and fingerprint, cleans up
  -- and a damaged archive, a wrong checksum or a wrong fingerprint is caught;
* retention keeps the newest N scheduled archives, deletes their checksums
  with them, never the newest, never an archive an operator named by hand;
* `ops-backup` archives, verifies, prunes and records the attempt, takes the run
  lock, and leaves the live corpus as it found it;
* a failed attempt is BACKUP_FAILURE in the health check;
* `corpus-backup-verify` is the operator's rehearsal.

No network.
"""

from __future__ import annotations

import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.cli import main
from market_intelligence.collection import syndication
from market_intelligence.ops.backup import (
    BACKUP_STATUS_NAME,
    BACKUP_VERSION,
    BackupManifest,
    create_backup,
    prune_backups,
    read_backup_status,
    verify_backup,
    write_backup_status,
    write_checksum,
)
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import CollectionProfile, collect_once
from market_intelligence.ops.runlock import RunLock
from market_intelligence.ops.scheduled import EXIT_LOCK_HELD
from market_intelligence.storage import IntelligenceStore
from tests.test_intelligence_deployment import FEED_PAYLOAD, minimal

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StoragePaths:
    """A state root holding one collected cycle."""
    monkeypatch.setattr(syndication, "_default_opener", lambda url, timeout, agent=None: FEED_PAYLOAD)
    resolved = StoragePaths.from_environment(tmp_path / "state").ensure()
    collect_once(resolved, CollectionProfile.from_mapping(minimal()), now=lambda: NOW, require_free_bytes=1)
    return resolved


def archive_of(paths: StoragePaths, name: str = "corpus-20260902T120000Z.tar.gz") -> tuple[Path, BackupManifest]:
    store = IntelligenceStore(paths.database)
    try:
        manifest = create_backup(store, paths.database, paths.backups / name, manifest_dir=paths.manifests, now=NOW)
    finally:
        store.close()
    return paths.backups / name, manifest


def live_counts(paths: StoragePaths) -> tuple[int, int]:
    store = IntelligenceStore(paths.database)
    try:
        documents = store.connection.execute("SELECT count(*) FROM documents").fetchone()
        events = store.connection.execute("SELECT count(*) FROM signals").fetchone()
    finally:
        store.close()
    assert documents is not None and events is not None
    return int(documents[0]), int(events[0])


def rewrite_manifest(archive: Path, target: Path, **changes: object) -> Path:
    """Copy an archive, altering its manifest -- a forgery the rehearsal must catch."""
    with tarfile.open(archive, "r:gz") as source, tarfile.open(target, "w:gz") as copy:
        for member in source.getmembers():
            handle = source.extractfile(member) if member.isfile() else None
            if member.name == "backup_manifest.json" and handle is not None:
                payload = json.loads(handle.read().decode("utf-8"))
                payload.update(changes)
                data = json.dumps(payload).encode("utf-8")
                member.size = len(data)
                import io

                copy.addfile(member, io.BytesIO(data))
            else:
                copy.addfile(member, handle)
    return target


class TestTheManifest:
    def test_it_is_v2_and_describes_the_archived_copy(self, paths: StoragePaths) -> None:
        _, manifest = archive_of(paths)
        assert manifest.backup_version == BACKUP_VERSION == "b52-ops-backup-v2"
        assert (manifest.documents, manifest.events) == live_counts(paths)
        assert manifest.content_fingerprint and len(manifest.content_fingerprint) == 64

    def test_the_fingerprint_is_stable_across_archives_of_one_corpus(self, paths: StoragePaths) -> None:
        _, first = archive_of(paths, "corpus-20260902T120000Z.tar.gz")
        _, second = archive_of(paths, "corpus-20260902T130000Z.tar.gz")
        assert first.content_fingerprint == second.content_fingerprint

    def test_a_v1_manifest_still_parses_without_a_fingerprint(self, paths: StoragePaths) -> None:
        _, manifest = archive_of(paths)
        legacy = manifest.as_dict()
        legacy.pop("content_fingerprint")
        legacy["backup_version"] = "b41-ops-backup-v1"
        assert BackupManifest.parse(legacy).content_fingerprint is None


class TestTheRehearsal:
    def test_a_good_archive_verifies_and_leaves_nothing_behind(self, paths: StoragePaths, tmp_path: Path) -> None:
        archive, manifest = archive_of(paths)
        write_checksum(archive)
        work = tmp_path / "work"
        work.mkdir()
        result = verify_backup(archive, work_dir=work)
        assert result.ok, result.findings
        assert result.checksum_matches is True and result.integrity_status in {"OK", "DEGRADED"}
        assert result.restore is not None and result.restore.fingerprint_matches is True
        assert result.restore.counts["documents"] == manifest.documents
        assert list(work.iterdir()) == [], "the rehearsal was not cleaned up"

    def test_a_wrong_checksum_is_caught(self, paths: StoragePaths) -> None:
        archive, _ = archive_of(paths)
        write_checksum(archive).write_text("0" * 64 + f"  {archive.name}\n", encoding="utf-8")
        result = verify_backup(archive)
        assert not result.ok and result.checksum_matches is False

    def test_a_damaged_archive_is_caught(self, paths: StoragePaths) -> None:
        archive, _ = archive_of(paths)
        archive.write_bytes(archive.read_bytes()[: len(archive.read_bytes()) // 2])
        result = verify_backup(archive)
        assert not result.ok and any("could not be restored" in finding for finding in result.findings)

    def test_a_forged_fingerprint_is_caught(self, paths: StoragePaths) -> None:
        archive, _ = archive_of(paths)
        forged = rewrite_manifest(archive, archive.with_name("corpus-forged.tar.gz"), content_fingerprint="f" * 64)
        result = verify_backup(forged)
        assert not result.ok and result.restore is not None and result.restore.fingerprint_matches is False

    def test_a_missing_archive_is_a_finding_not_a_crash(self, tmp_path: Path) -> None:
        result = verify_backup(tmp_path / "absent.tar.gz")
        assert not result.ok and "does not exist" in result.findings[0]

    def test_the_live_corpus_is_left_as_it_was(self, paths: StoragePaths) -> None:
        before = live_counts(paths)
        archive, _ = archive_of(paths)
        verify_backup(archive)
        assert live_counts(paths) == before


class TestRetention:
    def names(self, directory: Path) -> list[str]:
        return sorted(path.name for path in directory.iterdir())

    def test_it_keeps_the_newest_and_takes_their_checksums_with_the_rest(self, tmp_path: Path) -> None:
        for day in range(1, 6):
            archive = tmp_path / f"corpus-202609{day:02d}T054000Z.tar.gz"
            archive.write_bytes(b"x")
            write_checksum(archive)
        deleted = prune_backups(tmp_path, retain=2)
        assert [path.name for path in deleted] == [f"corpus-202609{day:02d}T054000Z.tar.gz" for day in (1, 2, 3)]
        assert self.names(tmp_path) == [
            "corpus-20260904T054000Z.tar.gz",
            "corpus-20260904T054000Z.tar.gz.sha256",
            "corpus-20260905T054000Z.tar.gz",
            "corpus-20260905T054000Z.tar.gz.sha256",
        ]

    def test_an_archive_named_by_hand_is_never_pruned(self, tmp_path: Path) -> None:
        (tmp_path / "corpus-before-upgrade.tar.gz").write_bytes(b"x")
        for day in (1, 2):
            (tmp_path / f"corpus-202609{day:02d}T054000Z.tar.gz").write_bytes(b"x")
        prune_backups(tmp_path, retain=1)
        assert "corpus-before-upgrade.tar.gz" in self.names(tmp_path)

    def test_the_last_backup_can_never_be_pruned(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="retain at least one"):
            prune_backups(tmp_path, retain=0)


def ops_backup(paths: StoragePaths, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, dict]:
    code = main(["ops-backup", "--state-root", str(paths.root), "--wait-seconds", "0", *extra])
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else {}


class TestTheScheduledBackup:
    def test_it_archives_verifies_records_and_prunes(self, paths: StoragePaths, capsys: pytest.CaptureFixture[str]) -> None:
        for day in (1, 2, 3):
            (paths.backups / f"corpus-202608{day:02d}T054000Z.tar.gz").write_bytes(b"old")
        before = live_counts(paths)
        code, payload = ops_backup(paths, capsys, "--retain", "2")
        assert code == 0 and payload["verification"]["ok"], payload
        archive = paths.backups / payload["backup"]["archive"]
        assert archive.exists() and archive.with_name(archive.name + ".sha256").exists()
        assert payload["backup"]["pruned"] == ["corpus-20260801T054000Z.tar.gz", "corpus-20260802T054000Z.tar.gz"]
        status = read_backup_status(paths.backups)
        assert status is not None and status["ok"] and status["archive"] == archive.name
        assert live_counts(paths) == before

    def test_it_will_not_run_beside_a_collection_cycle(
        self, paths: StoragePaths, capsys: pytest.CaptureFixture[str]
    ) -> None:
        holder = RunLock(paths.lock, purpose="collect-scheduled").acquire()
        try:
            code, payload = ops_backup(paths, capsys)
        finally:
            holder.release()
        assert code == EXIT_LOCK_HELD and payload["exit_code"] == EXIT_LOCK_HELD
        assert not list(paths.backups.glob("corpus-*.tar.gz"))

    def test_a_failed_attempt_is_critical_in_the_health_check(
        self, paths: StoragePaths, capsys: pytest.CaptureFixture[str]
    ) -> None:
        archive_of(paths, f"corpus-{(datetime.now(timezone.utc) - timedelta(hours=1)).strftime('%Y%m%dT%H%M%SZ')}.tar.gz")
        write_backup_status(paths.backups, {"ok": False, "findings": ["the archive could not be restored: truncated"]})
        main(["--db", str(paths.database), "ops-watch", "--json", "--state-root", str(paths.root)])
        alerts = {alert["code"]: alert for alert in json.loads(capsys.readouterr().out)["alerts"]}
        assert alerts["BACKUP_FAILURE"]["severity"] == "CRITICAL"
        assert "truncated" in alerts["BACKUP_FAILURE"]["detail"]["detail"]

    def test_a_successful_attempt_clears_it(self, paths: StoragePaths, capsys: pytest.CaptureFixture[str]) -> None:
        write_backup_status(paths.backups, {"ok": False, "findings": ["earlier failure"]})
        assert ops_backup(paths, capsys)[0] == 0
        main(["--db", str(paths.database), "ops-watch", "--json", "--state-root", str(paths.root)])
        codes = {alert["code"] for alert in json.loads(capsys.readouterr().out)["alerts"]}
        assert "BACKUP_FAILURE" not in codes and "BACKUP_STALE" not in codes

    def test_an_unreadable_status_reads_as_failed(self, tmp_path: Path) -> None:
        (tmp_path / BACKUP_STATUS_NAME).write_text("{not json", encoding="utf-8")
        status = read_backup_status(tmp_path)
        assert status is not None and status["ok"] is False


class TestTheOperatorsRehearsal:
    def test_the_latest_archive_verifies(self, paths: StoragePaths, capsys: pytest.CaptureFixture[str]) -> None:
        archive_of(paths)
        code = main(["corpus-backup-verify", "--latest", "--state-root", str(paths.root)])
        assert code == 0 and json.loads(capsys.readouterr().out)["ok"]

    def test_a_damaged_archive_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bad = tmp_path / "corpus-20260902T120000Z.tar.gz"
        bad.write_bytes(b"not an archive")
        assert main(["corpus-backup-verify", "--archive", str(bad)]) == 2

    def test_no_archive_at_all_exits_2_with_a_sentence(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = StoragePaths.from_environment(tmp_path / "empty").ensure().root
        assert main(["corpus-backup-verify", "--latest", "--state-root", str(root)]) == 2
        assert capsys.readouterr().err.startswith("error: ")
