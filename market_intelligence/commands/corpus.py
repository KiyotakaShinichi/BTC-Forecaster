"""Commands that inspect, verify, snapshot and move the corpus.

Three of these fail closed on purpose. `corpus-verify` and `corpus-restore`
return 2 when the corpus does not check out, because an integrity command that
reports a problem and exits 0 is worse than no integrity command: a scheduler
reads the code, not the sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..collection.corpus import CorpusCatalog
from ..ops.backup import create_backup
from ..ops.backup import restore as restore_backup
from ..ops.integrity import verify as ops_verify
from ..reports import corpus_status as build_corpus_status
from . import CommandContext


def corpus_status(ctx: CommandContext) -> int:
    status = build_corpus_status(
        ctx.store, ctx.args.extractor_version, ctx.args.entities.split(",")
    )
    if ctx.args.output:
        status.write(ctx.args.output)
    print(status.model_dump_json(indent=2) if ctx.args.json else status.human_readable())
    return 0


def corpora(ctx: CommandContext) -> int:
    snapshots = CorpusCatalog(ctx.store.connection).list_snapshots(limit=ctx.args.limit)
    print(json.dumps([item.model_dump(mode="json") for item in snapshots], indent=2))
    return 0


def corpus_verify(ctx: CommandContext) -> int:
    report = ops_verify(ctx.store, as_of=datetime.now(timezone.utc))
    print(json.dumps(report.as_dict(), indent=2) if ctx.args.json else report.human_readable())
    return 0 if report.ok else 2


def corpus_backup(ctx: CommandContext) -> int:
    manifest = create_backup(
        ctx.store,
        Path(ctx.args.db),
        Path(ctx.args.output),
        manifest_dir=Path(ctx.args.manifest_dir) if ctx.args.manifest_dir else None,
    )
    print(json.dumps(manifest.as_dict(), indent=2))
    return 0


def corpus_restore(ctx: CommandContext) -> int:
    outcome = restore_backup(Path(ctx.args.archive), Path(ctx.args.destination))
    print(json.dumps(outcome.as_dict(), indent=2))
    return 0 if outcome.ok else 2



def ops_backup(args: argparse.Namespace) -> int:
    """The scheduled backup: lock, archive, prove it by restoring, then prune.

    Exit 0 archived and verified, 2 failed, 3 a collection cycle held the lock
    for longer than --wait-seconds. It takes the collector's run lock, so a
    backup and a cycle never hold the database at once -- DuckDB allows one
    writer per file, and the loser of that race fails. The archive is verified
    by a full restore rehearsal before any old archive is pruned, and every
    attempt is recorded where the health check reads it.
    """
    import logging
    import time
    from datetime import datetime, timezone

    from ..logs import configure, get_logger
    from ..ops.backup import (
        BackupError,
        create_backup,
        prune_backups,
        verify_backup,
        write_backup_status,
        write_checksum,
    )
    from ..ops.paths import StoragePaths
    from ..ops.runlock import LockHeld, RunLock
    from ..ops.scheduled import EXIT_FAILED, EXIT_LOCK_HELD, EXIT_OK
    from ..storage import IntelligenceStore

    configure()
    log = get_logger("backup")
    paths = StoragePaths.from_environment(args.state_root).ensure()
    moment = datetime.now(timezone.utc)
    archive = paths.backups / f"corpus-{moment.strftime('%Y%m%dT%H%M%SZ')}.tar.gz"

    lock = RunLock(paths.lock, purpose="backup")
    deadline = time.monotonic() + max(0, args.wait_seconds)
    while True:
        try:
            lock.acquire()
            break
        except LockHeld as held:
            if time.monotonic() >= deadline:
                log.emit("backup_skipped", reason=str(held))
                print(json.dumps({"exit_code": EXIT_LOCK_HELD, "reason": str(held)}, indent=2))
                return EXIT_LOCK_HELD
            time.sleep(5)

    try:
        store = IntelligenceStore(paths.database)
        try:
            manifest = create_backup(store, paths.database, archive, manifest_dir=paths.manifests, now=moment)
        finally:
            store.close()
    except BackupError as error:
        write_backup_status(
            paths.backups, {"at": moment.isoformat(), "archive": archive.name, "ok": False, "findings": [str(error)]}
        )
        log.emit("backup_failed", severity=logging.ERROR, detail=str(error))
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        lock.release()

    write_checksum(archive)
    verification = verify_backup(archive)
    pruned = prune_backups(paths.backups, args.retain) if verification.ok else []
    status: dict[str, Any] = {
        "at": moment.isoformat(),
        "archive": archive.name,
        "ok": verification.ok,
        "findings": list(verification.findings),
        "sha256": verification.sha256,
        "documents": manifest.documents,
        "events": manifest.events,
        "content_fingerprint": manifest.content_fingerprint,
        "integrity_status": verification.integrity_status,
        "retained": args.retain,
        "pruned": [path.name for path in pruned],
    }
    write_backup_status(paths.backups, status)
    log.emit(
        "backup_finished",
        severity=logging.INFO if verification.ok else logging.ERROR,
        archive=archive.name,
        verified=verification.ok,
        documents=manifest.documents,
        events=manifest.events,
        pruned=len(pruned),
    )
    print(json.dumps({"backup": status, "verification": verification.as_dict()}, indent=2))
    return EXIT_OK if verification.ok else EXIT_FAILED


def corpus_backup_verify(args: argparse.Namespace) -> int:
    """Rehearse a restore of one archive into a scratch location, and say whether it is sound.

    Exit 0 sound, 2 not. The live corpus is never opened, and the rehearsal is
    deleted afterwards whatever it found.
    """
    from ..ops.backup import latest_backup, verify_backup
    from ..ops.paths import StoragePaths

    if args.archive:
        archive = Path(args.archive)
    else:
        found = latest_backup(StoragePaths.from_environment(args.state_root).backups)
        if found is None:
            print("error: no backup archive found to verify", file=sys.stderr)
            return 2
        archive = found
    verification = verify_backup(archive)
    print(json.dumps(verification.as_dict(), indent=2))
    return 0 if verification.ok else 2


__all__ = [
    "corpora",
    "corpus_backup",
    "corpus_backup_verify",
    "corpus_restore",
    "corpus_status",
    "corpus_verify",
    "ops_backup",
]
