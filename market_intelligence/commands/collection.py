"""Commands that put new evidence into the corpus.

Everything here writes, and everything it writes is stamped with retrieval time
rather than publication time. That is the whole point-in-time contract in one
sentence, and it is why `collect` and `backfill` share a cycle: a backfill is
not a licence to date evidence earlier, only to run the same cycle over a
window that has already passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..configuration import ProviderConfig, ProviderRegistry, QueryPlanner, WatchEntity
from ..cycle import run_intelligence_cycle
from ..extractors import RuleBasedExtractor
from ..ops.paths import StoragePaths
from ..providers import JsonSearchApiProvider, RssSearchProvider
from ..retrieval import MultiProviderRetriever
from . import CommandContext


def load_config(path: str) -> tuple[list[ProviderConfig], list[WatchEntity], dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        [ProviderConfig.model_validate(p) for p in raw["providers"]],
        [WatchEntity.model_validate(e) for e in raw["watchlist"]],
        raw,
    )


def registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("rss", lambda c: RssSearchProvider(c.settings["feed_urls"], int(c.timeout)))
    registry.register(
        "json_search_api",
        lambda c: JsonSearchApiProvider(
            c.settings["endpoint"], c.credential() or "", c.id, int(c.timeout)
        ),
    )
    return registry


def collect(ctx: CommandContext) -> int:
    configs, watchlist, raw = load_config(ctx.args.config)
    providers = registry().build(configs)
    planner = QueryPlanner()
    queries = planner.plan(watchlist, ctx.args.origin)
    entity_aliases = {e.canonical_name: e.aliases for e in watchlist}
    run_report = run_intelligence_cycle(
        queries,
        MultiProviderRetriever(providers, {c.id: c for c in configs}),
        RuleBasedExtractor(entity_aliases),
        ctx.store,
        raw,
        ctx.args.origin,
        ctx.args.manifest,
    )
    print(run_report.model_dump_json(indent=2))
    return 0


def extract(ctx: CommandContext) -> int:
    documents = ctx.store.documents_as_of(ctx.args.origin)
    events = RuleBasedExtractor().extract(documents)
    ctx.store.put_signals(events)
    print(json.dumps({"events_created": len(events)}))
    return 0


def aggregate(ctx: CommandContext) -> int:
    print(json.dumps(ctx.reads.aggregate(ctx.args.origin), sort_keys=True))
    return 0


def backfill(ctx: CommandContext) -> int:
    from ..backfill import BackfillRunner
    from ..operations import BackfillManifest

    args = ctx.args
    progress = Path(args.progress)
    backfill_manifest = BackfillManifest(
        from_time=args.from_time,
        to_time=args.to_time,
        window_hours=args.window_hours,
        max_windows=args.max_windows,
    )
    configs, watchlist, raw = load_config(args.config)
    providers = registry().build(configs)
    retriever = MultiProviderRetriever(providers, {c.id: c for c in configs})
    aliases = {e.canonical_name: e.aliases for e in watchlist}
    manifests_dir = Path(args.manifests_dir)

    def collect_window(start: datetime, end: datetime) -> None:
        hours = max(1, int((end - start).total_seconds() // 3600))
        planned = QueryPlanner().plan(watchlist, end, lookback_hours=hours)
        name = end.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ.json")
        run_intelligence_cycle(
            planned,
            retriever,
            RuleBasedExtractor(aliases),
            ctx.store,
            raw,
            end,
            manifests_dir / name,
        )

    completed = BackfillRunner(collect_window).run(backfill_manifest, progress)
    print(completed.model_dump_json(indent=2))
    return 0


def collect_scheduled(args: argparse.Namespace) -> int:
    """The one command a scheduler calls.

    Deliberately thin: everything it needs is either in the committed profile or
    in the environment, so what ran can be reconstructed from the repository and
    the unit file alone. The exit code is the whole interface -- 3 and 4 are
    ordinary outcomes, not failures, and a scheduler configured to treat them as
    errors will page someone every night for nothing.

    It manages its own storage, which is why it is registered as a command that
    must not be handed a store.
    """
    from ..ops.profile import CollectionProfile, collect_once

    profile = CollectionProfile.load(args.profile)
    paths = StoragePaths.from_environment(args.state_root)
    outcome = collect_once(
        paths,
        profile,
        require_free_bytes=max(0, args.require_free_mb) * 1024 * 1024,
        source_sha=os.environ.get("BTC_INTEL_SOURCE_SHA"),
    )

    payload = outcome.as_dict()
    payload["profile"] = profile.fingerprint()
    rendered = json.dumps(payload, indent=2)
    print(rendered if args.json else outcome.reason)

    # A scheduler keeps only the last few runs of stdout; the corpus keeps the
    # manifest. Neither is a log of what the collector decided, so write that.
    try:
        paths.logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (paths.logs / f"collect-{stamp}.json").write_text(rendered, encoding="utf-8")
    except OSError as error:  # a log we cannot write must not fail the cycle
        print(f"warning: could not write the run log: {error}", file=sys.stderr)

    for warning in outcome.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return outcome.exit_code


__all__ = [
    "aggregate",
    "backfill",
    "collect",
    "collect_scheduled",
    "extract",
    "load_config",
    "registry",
]
