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
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..collection.evidence import EvidenceStore
from ..errors import IntelligenceError
from ..storage import IntelligenceStore

BACKUP_MANIFEST_NAME = "backup_manifest.json"
BACKUP_VERSION = "b41-ops-backup-v1"


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
        )


#: Never copied into a backup, whatever directory they are found in.
EXCLUDED_NAMES = (".env", "credentials.json", "collector.lock")
EXCLUDED_SUFFIXES = (".tmp", ".hb", ".wal", ".lock")


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

    counts = _counts(store)
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


def _counts(store: IntelligenceStore) -> dict[str, int]:
    def scalar(sql: str) -> int:
        row = store.connection.execute(sql).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    return {
        "documents": scalar("SELECT count(*) FROM documents"),
        "events": scalar("SELECT count(*) FROM signals"),
        "snapshots": scalar("SELECT count(*) FROM corpus_catalog"),
        "watermarks": scalar("SELECT count(*) FROM watermarks"),
        "sightings": scalar("SELECT count(*) FROM document_sightings"),
    }


@dataclass(frozen=True)
class RestoreResult:
    destination: Path
    manifest: BackupManifest
    verified_members: int
    counts: dict[str, int]
    matches_manifest: bool
    findings: tuple[str, ...]

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
    database = destination / "intelligence.duckdb"
    if database.exists() and not findings:
        store = IntelligenceStore(database)
        try:
            counts = _counts(store)
        finally:
            store.close()
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


__all__ = [
    "BACKUP_MANIFEST_NAME",
    "BACKUP_VERSION",
    "BackupError",
    "BackupManifest",
    "RestoreResult",
    "backup_age",
    "create_backup",
    "latest_backup",
    "restore",
]
