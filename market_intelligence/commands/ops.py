"""Commands that report on the machine, and write nothing.

Every one of these is safe to run at any time, including while a collection
cycle is in flight. They read the corpus and the filesystem and render what
they find; none of them takes the run lock, because a status check that can
block a collector is a status check nobody will run when it matters.

`ops-watch` is the one with teeth: it exits 0 healthy, 1 warning, 2 critical,
so a scheduler can branch on it without parsing anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ops.paths import StoragePaths, looks_ephemeral
from ..ops.paths import validate as storage_validate
from ..reports import ops_report as build_ops_report
from ..reports import provider_report as build_provider_report
from . import CommandContext


def health(ctx: CommandContext) -> int:
    print(json.dumps([h.model_dump(mode="json") for h in ctx.reads.health()], default=str))
    return 0


def quality(ctx: CommandContext) -> int:
    report = ctx.reads.quality(ctx.args.origin)
    print(report.model_dump_json(indent=2))
    return 0 if report.valid else 2


def providers(ctx: CommandContext) -> int:
    print(json.dumps(build_provider_report(), indent=2))
    return 0


def ops_paths(ctx: CommandContext) -> int:
    resolved = StoragePaths.from_environment(ctx.args.state_root)
    checked = storage_validate(resolved)
    payload: dict[str, Any] = {
        "paths": resolved.as_dict(),
        "validation": checked.as_dict(),
    }
    warning = looks_ephemeral(resolved.root)
    if warning:
        payload["warning"] = warning
    print(json.dumps(payload, indent=2))
    return 0 if checked.ok else 2


def ops_status(ctx: CommandContext) -> int:
    payload = build_ops_report(ctx.store, Path(ctx.args.db), ctx.args.state_root)
    print(json.dumps(payload, indent=2) if ctx.args.json else payload["human"])
    return 0


def ops_watch(ctx: CommandContext) -> int:
    payload = build_ops_report(ctx.store, Path(ctx.args.db), ctx.args.state_root)
    if ctx.args.json:
        print(json.dumps(payload["watchdog"], indent=2))
    else:
        for alert in payload["watchdog"]["alerts"]:
            print(f"[{alert['severity']}] {alert['code']}: {alert['message']}")
    return int(payload["exit_code"])


__all__ = ["health", "ops_paths", "ops_status", "ops_watch", "providers", "quality"]
