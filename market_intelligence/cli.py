from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .aggregation import FeatureAggregator
from .configuration import ProviderConfig, ProviderRegistry, QueryPlanner, WatchEntity
from .cycle import ReplayService, run_intelligence_cycle
from .extractors import RuleBasedExtractor
from .providers import JsonSearchApiProvider, RssSearchProvider
from .quality import evaluate_quality
from .retrieval import MultiProviderRetriever
from .storage import IntelligenceStore


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc-intel", description="Point-in-time BTC market-intelligence operations")
    parser.add_argument("--db", default="btc-intelligence.duckdb")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect"); collect.add_argument("--config", required=True); collect.add_argument("--origin", type=_time, required=True); collect.add_argument("--manifest", required=True)
    extract = sub.add_parser("extract"); extract.add_argument("--origin", type=_time, required=True)
    aggregate = sub.add_parser("aggregate"); aggregate.add_argument("--origin", type=_time, required=True)
    replay = sub.add_parser("replay"); replay.add_argument("--origin", type=_time, required=True); replay.add_argument("--config-fingerprint", default="manual")
    backfill = sub.add_parser("backfill"); backfill.add_argument("--from", dest="from_time", type=_time, required=True); backfill.add_argument("--to", dest="to_time", type=_time, required=True); backfill.add_argument("--window-hours", type=int, default=24); backfill.add_argument("--max-windows", type=int, default=31); backfill.add_argument("--config", required=True); backfill.add_argument("--progress", default="backfill-progress.json"); backfill.add_argument("--manifests-dir", default="backfill-manifests")
    sub.add_parser("health")
    quality = sub.add_parser("quality"); quality.add_argument("--origin", type=_time, default=datetime.now(timezone.utc))
    return parser


def _load_config(path: str) -> tuple[list[ProviderConfig], list[WatchEntity], dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ([ProviderConfig.model_validate(p) for p in raw["providers"]],
            [WatchEntity.model_validate(e) for e in raw["watchlist"]], raw)


def _registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("rss", lambda c: RssSearchProvider(c.settings["feed_urls"], int(c.timeout)))
    registry.register("json_search_api", lambda c: JsonSearchApiProvider(c.settings["endpoint"], c.credential() or "", c.id, int(c.timeout)))
    return registry


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = IntelligenceStore(args.db)
    try:
        if args.command == "collect":
            configs, watchlist, raw = _load_config(args.config)
            providers = _registry().build(configs)
            planner = QueryPlanner(); queries = planner.plan(watchlist, args.origin)
            entity_aliases = {e.canonical_name: e.aliases for e in watchlist}
            report = run_intelligence_cycle(queries, MultiProviderRetriever(providers, {c.id: c for c in configs}),
                RuleBasedExtractor(entity_aliases), store, raw, args.origin, args.manifest)
            print(report.model_dump_json(indent=2)); return 0
        if args.command == "extract":
            documents = store.documents_as_of(args.origin)
            events = RuleBasedExtractor().extract(documents); store.put_signals(events)
            print(json.dumps({"events_created": len(events)})); return 0
        if args.command == "aggregate":
            print(json.dumps(FeatureAggregator().aggregate(store.signals_as_of(args.origin), args.origin), sort_keys=True)); return 0
        if args.command == "replay":
            snapshot = ReplayService(store).replay(args.origin, {}, args.config_fingerprint)
            print(snapshot.model_dump_json(indent=2)); return 0
        if args.command == "health":
            print(json.dumps([h.model_dump(mode="json") for h in store.all_health()], default=str)); return 0
        if args.command == "quality":
            report = evaluate_quality(store.documents_as_of(args.origin), store.signals_as_of(args.origin), args.origin)
            print(report.model_dump_json(indent=2)); return 0 if report.valid else 2
        if args.command == "backfill":
            from .backfill import BackfillRunner
            from .operations import BackfillManifest
            progress = Path(args.progress)
            manifest = BackfillManifest(from_time=args.from_time, to_time=args.to_time,
                window_hours=args.window_hours, max_windows=args.max_windows)
            configs, watchlist, raw = _load_config(args.config)
            providers = _registry().build(configs)
            retriever = MultiProviderRetriever(providers, {c.id: c for c in configs})
            aliases = {e.canonical_name: e.aliases for e in watchlist}
            manifests_dir = Path(args.manifests_dir)
            def collect_window(start: datetime, end: datetime) -> None:
                hours = max(1, int((end - start).total_seconds() // 3600))
                planned = QueryPlanner().plan(watchlist, end, lookback_hours=hours)
                name = end.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ.json")
                run_intelligence_cycle(planned, retriever, RuleBasedExtractor(aliases), store, raw, end,
                    manifests_dir / name)
            completed = BackfillRunner(collect_window).run(manifest, progress)
            print(completed.model_dump_json(indent=2)); return 0
    finally:
        store.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
