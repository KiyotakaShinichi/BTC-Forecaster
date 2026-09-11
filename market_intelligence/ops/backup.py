"""O7 / O8 — deterministic backup, and a restore you can verify.

Restore is the half people skip, and it is the half that matters. A backup
nobody has restored is a hypothesis. So `restore` writes into a **new**
destination by default and refuses to overwrite an existing one — the operation
you actually want at 3am is "prove this archive is good", not "replace the
corpus with it".

What travels: the database, catalogs, manifests, watermarks, sightings and the
raw evidence a licence permits. What never travels: credentials, licence-
restricted payloads, run locks and temporary files. Each exclusion is recorded
by category, so a restored corpus that is missing something says which kind of
something and why.

Every archive carries a hash manifest over its own members, so a corrupted or
partial transfer is detectable before anyone builds on it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from ..collection.corpus import membership_hash
from ..collection.evidence import EvidenceStore
from ..errors import IntelligenceError
from ..storage import IntelligenceStore

BACKUP_MANIFEST_NAME = "backup_manifest.json"
#: v2 (B5.2) adds a content fingerprint and counts taken from the archived copy
#: itself. A v1 archive still parses and restores; it simply has no fingerprint
#: to check.
BACKUP_VERSION = "b52-ops-backup-v2"

#: `sha256sum -c` reads this, so an archive copied off the host can be checked
#: with standard tools and without this code.
CHECKSUM_SUFFIX = ".sha256"

#: Where `ops-backup` records its last attempt, for the health check to read.
BACKUP_STATUS_NAME = "backup-status.json"

#: The only archives retention may ever delete: the scheduled job's own,
#: timestamped names. An archive an operator named by hand is never pruned.
SCHEDULED_ARCHIVE = re.compile(r"^corpus-\d{8}T\d{6}Z\.tar\.gz$")


class BackupError(IntelligenceError):
    """A backup or restore could not be completed safely."""


@dataclass
class BackupManifest:
    """What went in, what did not, and the hash of each member."""

    backup_version: str
    created_at: datetime
    source_root: str
    database_bytes: int
    documents: int
    events: int
    snapshots: int
    watermarks: int
    sightings: int
    members: dict[str, str] = field(default_factory=dict)
    omitted: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    #: Document ids, event ids and correction ids of the archived copy, hashed.
    #: None in a v1 archive.
    content_fingerprint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backup_version": self.backup_version,
            "created_at": self.created_at.isoformat(),
            "source_root": self.source_root,
            "database_bytes": self.database_bytes,
            "documents": self.documents,
            "events": self.events,
            "snapshots": self.snapshots,
            "watermarks": self.watermarks,
            "sightings": self.sightings,
            "members": dict(sorted(self.members.items())),
            "omitted": dict(sorted(self.omitted.items())),
            "notes": list(self.notes),
            "content_fingerprint": self.content_fingerprint,
        }

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> "BackupManifest":
        return cls(
            backup_version=str(payload["backup_version"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            source_root=str(payload["source_root"]),
            database_bytes=int(payload["database_bytes"]),
            documents=int(payload["documents"]),
            events=int(payload["events"]),
            snapshots=int(payload["snapshots"]),
            watermarks=int(payload["watermarks"]),
            sightings=int(payload["sightings"]),
            members=dict(payload.get("members", {})),
            omitted=dict(payload.get("omitted", {})),
            notes=tuple(payload.get("notes", ())),
            content_fingerprint=payload.get("content_fingerprint"),
        )


#: Never copied into a backup, whatever directory they are found in.
EXCLUDED_NAMES = (".env", "credentials.json", "collector.lock")
#: `.env` is a suffix as well as a name. The documented deployment keeps its
#: environment file outside the state root, but an operator who puts
#: `collector.env` beside the database would otherwise have it copied into every
#: archive -- and an archive is the one artefact that gets moved to another
#: machine. Excluding by suffix makes that safe by construction rather than by
#: where somebody happened to put the file.
EXCLUDED_SUFFIXES = (".tmp", ".hb", ".wal", ".lock", ".env")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_backup(
    store: IntelligenceStore,
    database: Path,
    destination: Path,
    *,
    manifest_dir: Path | None = None,
    now: datetime | None = None,
) -> BackupManifest:
    """Write a verifiable archive. Refuses to overwrite an existing one."""
    moment = now or datetime.now(timezone.utc)
    destination = Path(destination)
    if destination.exists():
        raise BackupError(f"{destination} already exists; backups are never overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)

    evidence_counts = EvidenceStore(store.connection).counts_by_retention()
    restricted = sum(
        count for retention, count in evidence_counts.items() if retention != "FULL"
    )

    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    members: dict[str, str] = {}
    omitted: dict[str, int] = {}

    try:
        # DuckDB's own copy, not a filesystem copy of a possibly-open file.
        # `COPY DATABASE` produces a consistent image; copying the file while a
        # writer holds it can produce one that opens and is subtly short.
        database_copy = staging / "intelligence.duckdb"
        _export_database(store, database, database_copy)
        members["intelligence.duckdb"] = _hash_file(database_copy)
        # Counted from the copy, not the live store, so the manifest describes
        # exactly what the archive holds even if a writer committed in between.
        copy = duckdb.connect(str(database_copy), read_only=True)
        try:
            counts = _counts(copy)
            fingerprint = _fingerprint(copy)
        finally:
            copy.close()

        if manifest_dir and manifest_dir.exists():
            target = staging / "manifests"
            target.mkdir(parents=True, exist_ok=True)
            for source in sorted(manifest_dir.glob("*.json")):
                if _excluded(source):
                    omitted["OMITTED_TEMPORARY"] = omitted.get("OMITTED_TEMPORARY", 0) + 1
                    continue
                shutil.copy2(source, target / source.name)
                members[f"manifests/{source.name}"] = _hash_file(target / source.name)

        if restricted:
            omitted["OMITTED_LICENSE"] = restricted

        manifest = BackupManifest(
            backup_version=BACKUP_VERSION,
            created_at=moment,
            source_root=str(database.parent),
            database_bytes=database_copy.stat().st_size,
            documents=counts["documents"],
            events=counts["events"],
            snapshots=counts["snapshots"],
            watermarks=counts["watermarks"],
            sightings=counts["sightings"],
            members=members,
            omitted=omitted,
            content_fingerprint=fingerprint,
            notes=(
                "credentials are never included: they live in the environment, not in state",
                "licence-restricted raw payloads travel as hashes only; the hash still lets a "
                "holder of the original prove it is the one this corpus used",
            ),
        )
        (staging / BACKUP_MANIFEST_NAME).write_text(
            json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

        with tarfile.open(destination, "w:gz") as archive:
            for item in sorted(staging.rglob("*")):
                archive.add(item, arcname=str(item.relative_to(staging)))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return manifest


def _export_database(store: IntelligenceStore, database: Path, target: Path) -> None:
    """Ask DuckDB for a consistent copy of the live database.

    `COPY FROM DATABASE` against an attached target, not a filesystem copy of the
    open file. Copying the file underneath a live connection produces an image
    that opens and is subtly short -- and on Windows it does not even open, the
    file is locked. There is deliberately no fallback for the same reason: a
    backup that might be truncated is worse than a backup that failed loudly.

    The source catalog is whatever DuckDB named it, which for a file-backed
    store is the filename stem rather than `memory`.
    """
    source = store.connection.execute("SELECT current_database()").fetchone()
    if not source or not source[0]:
        raise BackupError("could not determine the source database name")
    catalog = str(source[0]).replace('"', '""')
    escaped = str(target.resolve()).replace("'", "''")
    try:
        store.connection.execute(f"ATTACH '{escaped}' AS backup_target")
    except Exception as error:  # noqa: BLE001 -- surfaced as a backup failure
        raise BackupError(f"could not attach the backup target {target}: {error}") from error
    try:
        store.connection.execute(f'COPY FROM DATABASE "{catalog}" TO backup_target')
    except Exception as error:  # noqa: BLE001
        raise BackupError(f"could not copy {catalog} into the backup target: {error}") from error
    finally:
        try:
            store.connection.execute("DETACH backup_target")
        except Exception:  # noqa: BLE001 -- detach failure must not mask the real error
            pass


def _excluded(path: Path) -> bool:
    return path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES


def _counts(connection: Any) -> dict[str, int]:
    def scalar(sql: str) -> int:
        row = connection.execute(sql).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    return {
        "documents": scalar("SELECT count(*) FROM documents"),
        "events": scalar("SELECT count(*) FROM signals"),
        "snapshots": scalar("SELECT count(*) FROM corpus_catalog"),
        "watermarks": scalar("SELECT count(*) FROM watermarks"),
        "sightings": scalar("SELECT count(*) FROM document_sightings"),
    }


def _fingerprint(connection: Any) -> str:
    """What the corpus holds, independent of how DuckDB laid it out on disk.

    The database file's own hash changes with every checkpoint; this does not.
    Two copies with the same documents, events and corrections have the same
    fingerprint however they were written.
    """
    documents = [row[0] for row in connection.execute("SELECT document_id FROM documents").fetchall()]
    events = [row[0] for row in connection.execute("SELECT event_id FROM signals").fetchall()]
    corrections = sorted(row[0] for row in connection.execute("SELECT correction_id FROM event_corrections").fetchall())
    material = json.dumps(
        {"membership": membership_hash(documents, events), "corrections": corrections}, sort_keys=True
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RestoreResult:
    destination: Path
    manifest: BackupManifest
    verified_members: int
    counts: dict[str, int]
    matches_manifest: bool
    findings: tuple[str, ...]
    #: None when the archive predates fingerprints (v1).
    fingerprint_matches: bool | None = None

    @property
    def ok(self) -> bool:
        return self.matches_manifest and not self.findings

    def as_dict(self) -> dict[str, Any]:
        return {
            "destination": str(self.destination),
            "ok": self.ok,
            "verified_members": self.verified_members,
            "counts": self.counts,
            "matches_manifest": self.matches_manifest,
            "findings": list(self.findings),
            "fingerprint_matches": self.fingerprint_matches,
        }


def restore(archive: Path, destination: Path, *, allow_existing: bool = False) -> RestoreResult:
    """Unpack into a new location and check it against its own manifest.

    Non-destructive by default. The common operation is verification, and a
    restore that overwrites the live corpus to prove an archive is good would
    destroy the thing it was checking.
    """
    archive = Path(archive)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()) and not allow_existing:
        raise BackupError(
            f"{destination} is not empty; restore into a new location, or pass allow_existing"
        )
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as handle:
        _safe_extract(handle, destination)

    manifest_path = destination / BACKUP_MANIFEST_NAME
    if not manifest_path.exists():
        raise BackupError(f"{archive} has no {BACKUP_MANIFEST_NAME}; it is not a corpus backup")
    manifest = BackupManifest.parse(json.loads(manifest_path.read_text(encoding="utf-8")))

    findings: list[str] = []
    verified = 0
    for member, expected in manifest.members.items():
        path = destination / member
        if not path.exists():
            findings.append(f"missing member: {member}")
            continue
        if _hash_file(path) != expected:
            findings.append(f"hash mismatch: {member}")
            continue
        verified += 1

    counts: dict[str, int] = {}
    matches = False
    fingerprint_matches: bool | None = None
    database = destination / "intelligence.duckdb"
    if database.exists() and not findings:
        store = IntelligenceStore(database)
        try:
            counts = _counts(store.connection)
            restored_fingerprint = _fingerprint(store.connection)
        finally:
            store.close()
        if manifest.content_fingerprint is not None:
            fingerprint_matches = restored_fingerprint == manifest.content_fingerprint
            if not fingerprint_matches:
                findings.append("the restored corpus's content fingerprint does not match the manifest")
        matches = (
            counts.get("documents") == manifest.documents
            and counts.get("events") == manifest.events
            and counts.get("snapshots") == manifest.snapshots
            and counts.get("sightings") == manifest.sightings
        )
        if not matches:
            findings.append(
                f"restored counts {counts} do not match the manifest "
                f"(documents={manifest.documents}, events={manifest.events})"
            )

    return RestoreResult(
        destination=destination,
        manifest=manifest,
        verified_members=verified,
        counts=counts,
        matches_manifest=matches,
        findings=tuple(findings),
        fingerprint_matches=fingerprint_matches,
    )


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract, refusing any member that would escape the destination.

    A tar archive can name `../` paths. This one is written by this system, but
    a restore is exactly where an operator points the tool at a file someone
    else handed them.
    """
    root = destination.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if not str(target).startswith(str(root)):
            raise BackupError(f"archive member escapes the destination: {member.name}")
        if member.issym() or member.islnk():
            raise BackupError(f"archive contains a link, which is never expected: {member.name}")
    archive.extractall(destination)  # noqa: S202 -- members validated immediately above


def latest_backup(directory: Path) -> Path | None:
    archives = sorted(Path(directory).glob("*.tar.gz"))
    return archives[-1] if archives else None


def backup_age(directory: Path, now: datetime) -> tuple[Path, datetime] | None:
    """Newest archive and its recorded creation time, for the watchdog."""
    newest = latest_backup(directory)
    if newest is None:
        return None
    try:
        with tarfile.open(newest, "r:gz") as archive:
            member = archive.extractfile(BACKUP_MANIFEST_NAME)
            if member is None:
                return None
            manifest = BackupManifest.parse(json.loads(member.read().decode("utf-8")))
    except (tarfile.TarError, KeyError, ValueError, OSError):
        return None
    return newest, manifest.created_at


def write_checksum(archive: Path) -> Path:
    """`<archive>.sha256`, in the format `sha256sum -c` reads."""
    target = archive.with_name(archive.name + CHECKSUM_SUFFIX)
    target.write_text(f"{_hash_file(archive)}  {archive.name}\n", encoding="utf-8")
    return target


@dataclass(frozen=True)
class BackupVerification:
    """A restore rehearsal: what was checked, and everything that was wrong."""

    archive: Path
    sha256: str | None
    checksum_matches: bool | None
    restore: RestoreResult | None
    integrity_status: str | None
    integrity_detail: str
    findings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.restore is not None and not self.findings

    def as_dict(self) -> dict[str, Any]:
        return {
            "archive": str(self.archive),
            "ok": self.ok,
            "sha256": self.sha256,
            "checksum_matches": self.checksum_matches,
            "restore": self.restore.as_dict() if self.restore is not None else None,
            "integrity_status": self.integrity_status,
            "integrity_detail": self.integrity_detail,
            "findings": list(self.findings),
        }


def verify_backup(archive: Path, *, work_dir: Path | None = None) -> BackupVerification:
    """Prove an archive restores to a sound corpus, without touching the live one.

    The rehearsal an operator should run and usually does not: check the
    archive against its `.sha256`, restore it into a new, empty location, check
    every member against its hash, open the restored corpus and run the
    integrity check on it, compare its counts and content fingerprint with the
    manifest -- then delete the rehearsal. The live corpus is never opened.
    """
    from .integrity import IntegrityStatus
    from .integrity import verify as verify_integrity

    archive = Path(archive)
    if not archive.is_file():
        return BackupVerification(archive, None, None, None, None, "", (f"{archive} does not exist",))
    findings: list[str] = []
    digest = _hash_file(archive)
    checksum_matches: bool | None = None
    sidecar = archive.with_name(archive.name + CHECKSUM_SUFFIX)
    if sidecar.exists():
        recorded = sidecar.read_text(encoding="utf-8").split()
        checksum_matches = bool(recorded) and recorded[0] == digest
        if not checksum_matches:
            findings.append(f"the archive's sha256 does not match {sidecar.name}")

    workspace = Path(tempfile.mkdtemp(prefix="btc-intel-restore-check-", dir=str(work_dir) if work_dir else None))
    restored: RestoreResult | None = None
    integrity_status: str | None = None
    integrity_detail = ""
    try:
        try:
            restored = restore(archive, workspace / "corpus")
        # A truncated or corrupted gzip raises EOFError or zlib.error, neither of
        # which is an OSError: a damaged archive must be a finding, not a crash.
        except (BackupError, tarfile.TarError, OSError, EOFError, zlib.error, ValueError, KeyError) as error:
            findings.append(f"the archive could not be restored: {error}")
        else:
            findings.extend(restored.findings)
            database = workspace / "corpus" / "intelligence.duckdb"
            if restored.ok and database.exists():
                store = IntelligenceStore(database)
                try:
                    report = verify_integrity(store, as_of=datetime.now(timezone.utc))
                finally:
                    store.close()
                integrity_status = report.status.value
                integrity_detail = report.human_readable().splitlines()[0]
                if report.status is IntegrityStatus.CORRUPT:
                    findings.append(f"the restored corpus fails its integrity check: {integrity_detail}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return BackupVerification(
        archive, digest, checksum_matches, restored, integrity_status, integrity_detail, tuple(findings)
    )


def prune_backups(directory: Path, retain: int) -> list[Path]:
    """Keep the newest `retain` scheduled archives; delete older ones, with their checksums.

    Only the scheduled job's own timestamped archives are considered, the newest
    is never deleted, and the caller prunes only after the newest archive has
    been verified: retention must never be what leaves an operator with no good
    backup.
    """
    if retain < 1:
        raise ValueError("retain at least one backup; pruning the last one is never what anyone meant")
    archives = sorted(path for path in Path(directory).glob("corpus-*.tar.gz") if SCHEDULED_ARCHIVE.match(path.name))
    doomed = archives[:-retain] if len(archives) > retain else []
    for archive in doomed:
        archive.unlink()
        sidecar = archive.with_name(archive.name + CHECKSUM_SUFFIX)
        if sidecar.exists():
            sidecar.unlink()
    return doomed


def write_backup_status(directory: Path, payload: dict[str, Any]) -> Path:
    """Record the last backup attempt where the health check reads it. Atomic."""
    target = Path(directory) / BACKUP_STATUS_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def read_backup_status(directory: Path) -> dict[str, Any] | None:
    """The last recorded attempt, or None if none has been. Unreadable reads as failed."""
    target = Path(directory) / BACKUP_STATUS_NAME
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"ok": False, "findings": [f"{BACKUP_STATUS_NAME} is unreadable: {error}"]}
    return dict(payload) if isinstance(payload, dict) else {"ok": False, "findings": ["malformed backup status"]}


__all__ = [
    "BACKUP_MANIFEST_NAME",
    "BACKUP_STATUS_NAME",
    "BACKUP_VERSION",
    "CHECKSUM_SUFFIX",
    "BackupError",
    "BackupManifest",
    "BackupVerification",
    "RestoreResult",
    "backup_age",
    "create_backup",
    "latest_backup",
    "prune_backups",
    "read_backup_status",
    "restore",
    "verify_backup",
    "write_backup_status",
    "write_checksum",
]
