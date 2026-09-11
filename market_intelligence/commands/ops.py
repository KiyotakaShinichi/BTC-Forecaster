"""Commands that report on the machine, and write nothing.

Every one of these is safe to run at any time, including while a collection
cycle is in flight. They read the corpus and the filesystem and render what
they find; none of them takes the run lock, because a status check that can
block a collector is a status check nobody will run when it matters.

`ops-watch` is the one with teeth: it exits 0 healthy, 1 warning, 2 critical,
so a scheduler can branch on it without parsing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def ops_config_check(args: argparse.Namespace) -> int:
    """Whether this host's configuration is deployable, printed with nothing secret in it.

    0 deployable, 2 not. It checks what a collector would otherwise discover at
    03:07 on a host nobody is watching: a profile that does not load, a contact
    that is not set, a retired feed switched back on, a state root that cannot
    be written. The contact address itself is never printed.
    """
    from ..errors import ConfigurationError
    from ..ops.profile import CollectionProfile

    problems: list[str] = []
    warnings: list[str] = []
    description: dict[str, Any] | None = None
    try:
        profile = CollectionProfile.load(args.profile)
    except ConfigurationError as error:
        problems.append(str(error))
    else:
        description = profile.describe()
        retired = [feed["feed_id"] for feed in description["feeds"] if feed["retired"]]
        if retired:
            problems.append(
                f"retired feed(s) enabled: {', '.join(retired)}; each answers 404 and would read as a "
                "permanent outage"
            )
        if not profile.user_agent:
            problems.append("no contact User-Agent is configured; the SEC answers 403 to a collector it cannot contact")

    paths = StoragePaths.from_environment(args.state_root)
    checked = storage_validate(paths)
    problems.extend(f"storage {check.name}: {check.reason}" for check in checked.failures())
    ephemeral = looks_ephemeral(paths.root)
    if ephemeral:
        # A warning, as it is to the collector: some hosts mount durable storage
        # under a path that looks temporary. The operator decides.
        warnings.append(ephemeral)

    payload: dict[str, Any] = {
        "deployable": not problems,
        "profile": description,
        "storage": {"paths": paths.as_dict(), "validation": checked.as_dict()},
        "problems": problems,
        "warnings": warnings,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"deployable   {'yes' if not problems else 'NO'}")
        if description is not None:
            print(f"profile      {description['profile']}, {len(description['feeds'])} feed(s)")
            print(f"user agent   {description['user_agent'] or '(default; no contact)'}")
            print(
                f"cadence      no provider polled within {description['minimum_interval_seconds']}s "
                f"(declared floor {description['declared_floor_seconds']}s)"
            )
        print(f"state root   {paths.root} ({'usable' if checked.ok else 'NOT usable'})")
        for problem in problems:
            print(f"  problem: {problem}")
        for warning in warnings:
            print(f"  warning: {warning}")
    return 0 if not problems else 2


def ops_probe(args: argparse.Namespace) -> int:
    """Read each configured feed once, exactly as the collector would, and store nothing.

    0 every feed readable, 1 some not, 2 none -- or a profile that does not
    load. One request per feed and no retries: an operator's question should
    not multiply requests to a publisher, and a probe is not a collection, so no
    document is written and no watermark moves. Run it after installing and after
    a change, not on a timer; the collector's own cadence is the schedule.
    """
    from ..collection.feeds import RETIRED_FEED_IDS
    from ..errors import ConfigurationError
    from ..ops.profile import CollectionProfile

    try:
        profile = CollectionProfile.load(args.profile)
    except ConfigurationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    probes = profile.syndication_provider().probe()
    readable = sum(1 for probe in probes if probe.ok)
    payload: dict[str, Any] = {
        "profile": profile.name,
        "user_agent": profile.describe()["user_agent"],
        "configured": len(probes),
        "readable": readable,
        "feeds": [dict(probe.as_dict(), retired=probe.feed_id in RETIRED_FEED_IDS) for probe in probes],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{readable} of {len(probes)} feed(s) readable, user agent {payload['user_agent'] or '(default)'}")
        for probe in probes:
            marker = "ok  " if probe.ok else "FAIL"
            print(f"  {marker} {probe.feed_id:<26} {probe.entries:>4} entries  {probe.latency_ms:>8.1f} ms  {probe.detail}")
    if readable == len(probes):
        return 0
    return 1 if readable else 2


__all__ = [
    "health",
    "ops_config_check",
    "ops_paths",
    "ops_probe",
    "ops_status",
    "ops_watch",
    "providers",
    "quality",
]
