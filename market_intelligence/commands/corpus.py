"""Commands that inspect, verify, snapshot and move the corpus.

Three of these fail closed on purpose. `corpus-verify` and `corpus-restore`
return 2 when the corpus does not check out, because an integrity command that
reports a problem and exits 0 is worse than no integrity command: a scheduler
reads the code, not the sentence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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



__all__ = [
    "corpora",
    "corpus_backup",
    "corpus_restore",
    "corpus_status",
    "corpus_verify",
]
