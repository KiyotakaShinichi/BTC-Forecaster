from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import DatasetCatalog
from .configuration import ProviderConfig, ProviderRegistry, QueryPlanner, WatchEntity
from .cycle import run_intelligence_cycle
from .extractors import RuleBasedExtractor
from .historical import HistoricalDatasetService
from .origins import OriginFrequency, generate_origins
from .providers import JsonSearchApiProvider, RssSearchProvider
from .replay_dataset import ReplayDatasetBuilder, ReplayMode
from .retrieval import MultiProviderRetriever
from .services import IntelligenceReadService, SnapshotService
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
    collect = sub.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--origin", type=_time, required=True)
    collect.add_argument("--manifest", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--origin", type=_time, required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--origin", type=_time, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--origin", type=_time, required=True)
    replay.add_argument("--config-fingerprint", default="manual")
    backfill = sub.add_parser("backfill")
    backfill.add_argument("--from", dest="from_time", type=_time, required=True)
    backfill.add_argument("--to", dest="to_time", type=_time, required=True)
    backfill.add_argument("--window-hours", type=int, default=24)
    backfill.add_argument("--max-windows", type=int, default=31)
    backfill.add_argument("--config", required=True)
    backfill.add_argument("--progress", default="backfill-progress.json")
    backfill.add_argument("--manifests-dir", default="backfill-manifests")
    sub.add_parser("health")
    quality = sub.add_parser("quality")
    quality.add_argument("--origin", type=_time, default=datetime.now(timezone.utc))
    dataset = sub.add_parser("dataset")
    dataset.add_argument("--origins", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--manifest", required=True)
    dataset.add_argument("--config-fingerprint", required=True)
    dataset.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    dataset.add_argument(
        "--mode",
        choices=("OPTIMIZED", "REFERENCE"),
        default="OPTIMIZED",
        help="REFERENCE is the correctness oracle; both produce identical output",
    )

    replay_dataset = sub.add_parser(
        "replay-dataset",
        help="historical intelligence feature matrix over a generated origin schedule",
    )
    replay_dataset.add_argument("--start", type=_time, required=True)
    replay_dataset.add_argument("--end", type=_time, required=True)
    replay_dataset.add_argument(
        "--frequency", choices=("HOURLY", "4H", "DAILY"), default="HOURLY"
    )
    replay_dataset.add_argument("--output", required=True)
    replay_dataset.add_argument("--manifest", required=True)
    replay_dataset.add_argument("--config-fingerprint", required=True)
    replay_dataset.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    replay_dataset.add_argument("--mode", choices=("OPTIMIZED", "REFERENCE"), default="OPTIMIZED")
    replay_dataset.add_argument("--chunk-size", type=int, default=None)
    replay_dataset.add_argument(
        "--extend",
        default=None,
        metavar="DATASET_ID",
        help="append to an existing dataset instead of building a new one",
    )
    replay_dataset.add_argument("--no-resume", action="store_true")

    catalog = sub.add_parser("catalog", help="list registered historical datasets")
    catalog.add_argument("--limit", type=int, default=20)
    demo = sub.add_parser("demo")
    demo.add_argument("--output-dir", default="market-intelligence-demo")
    gold = sub.add_parser("gold-report")
    gold.add_argument("--output", default="gold-evaluation-report.json")
    return parser


def _load_config(path: str) -> tuple[list[ProviderConfig], list[WatchEntity], dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        [ProviderConfig.model_validate(p) for p in raw["providers"]],
        [WatchEntity.model_validate(e) for e in raw["watchlist"]],
        raw,
    )


def _registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("rss", lambda c: RssSearchProvider(c.settings["feed_urls"], int(c.timeout)))
    registry.register(
        "json_search_api",
        lambda c: JsonSearchApiProvider(c.settings["endpoint"], c.credential() or "", c.id, int(c.timeout)),
    )
    return registry


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        from .demo import run_offline_demo

        print(json.dumps(run_offline_demo(args.output_dir), indent=2))
        return 0
    if args.command == "gold-report":
        from .gold import write_gold_evaluation

        report = write_gold_evaluation(
            RuleBasedExtractor({"SEC": (), "Jerome Powell": ("Powell",), "CFTC": (), "Elon Musk": ()}),
            args.output,
        )
        print(report.model_dump_json(indent=2))
        return 0
    store = IntelligenceStore(args.db)
    reads = IntelligenceReadService(store)
    try:
        if args.command == "collect":
            configs, watchlist, raw = _load_config(args.config)
            providers = _registry().build(configs)
            planner = QueryPlanner()
            queries = planner.plan(watchlist, args.origin)
            entity_aliases = {e.canonical_name: e.aliases for e in watchlist}
            run_report = run_intelligence_cycle(
                queries,
                MultiProviderRetriever(providers, {c.id: c for c in configs}),
                RuleBasedExtractor(entity_aliases),
                store,
                raw,
                args.origin,
                args.manifest,
            )
            print(run_report.model_dump_json(indent=2))
            return 0
        if args.command == "extract":
            documents = store.documents_as_of(args.origin)
            events = RuleBasedExtractor().extract(documents)
            store.put_signals(events)
            print(json.dumps({"events_created": len(events)}))
            return 0
        if args.command == "aggregate":
            print(json.dumps(reads.aggregate(args.origin), sort_keys=True))
            return 0
        if args.command == "replay":
            snapshot = SnapshotService(store).build_snapshot(args.origin, {}, args.config_fingerprint)
            print(snapshot.model_dump_json(indent=2))
            return 0
        if args.command == "health":
            print(json.dumps([h.model_dump(mode="json") for h in reads.health()], default=str))
            return 0
        if args.command == "quality":
            quality_report = reads.quality(args.origin)
            print(quality_report.model_dump_json(indent=2))
            return 0 if quality_report.valid else 2
        if args.command == "dataset":
            origins = [
                _time(value.strip())
                for value in Path(args.origins).read_text(encoding="utf-8").splitlines()
                if value.strip()
            ]
            dataset_manifest = ReplayDatasetBuilder(store).build(
                origins,
                args.output,
                args.manifest,
                {},
                args.config_fingerprint,
                args.format,
                mode=ReplayMode(args.mode),
            )
            print(dataset_manifest.model_dump_json(indent=2))
            return 0
        if args.command == "replay-dataset":
            # Same service the API calls, so the two cannot drift.
            service = (
                HistoricalDatasetService(store, chunk_size=args.chunk_size)
                if args.chunk_size
                else HistoricalDatasetService(store)
            )
            origins = generate_origins(args.start, args.end, OriginFrequency(args.frequency))
            if args.extend:
                result = service.extend(
                    args.extend,
                    origins,
                    args.output,
                    args.manifest,
                    {},
                    args.config_fingerprint,
                    export_format=args.format,
                    mode=ReplayMode(args.mode),
                )
            else:
                result = service.build(
                    origins,
                    args.output,
                    args.manifest,
                    {},
                    args.config_fingerprint,
                    export_format=args.format,
                    mode=ReplayMode(args.mode),
                    resume=not args.no_resume,
                )
            print(
                json.dumps(
                    {
                        "dataset_id": result.manifest.dataset_id,
                        "rows": result.manifest.row_count,
                        "rows_appended": result.rows_appended,
                        "chunks": result.chunk_count,
                        "resumed_chunks": result.resumed_chunks,
                        "mode": result.manifest.mode,
                        "origin_frequency": result.catalog_entry.origin_frequency,
                        "output": str(result.output_path),
                        "manifest": str(result.manifest_path),
                        "file_hash": result.manifest.file_hash,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "catalog":
            entries = DatasetCatalog(store.connection).list_datasets(limit=args.limit)
            print(json.dumps([entry.model_dump(mode="json") for entry in entries], indent=2))
            return 0
        if args.command == "backfill":
            from .backfill import BackfillRunner
            from .operations import BackfillManifest

            progress = Path(args.progress)
            backfill_manifest = BackfillManifest(
                from_time=args.from_time,
                to_time=args.to_time,
                window_hours=args.window_hours,
                max_windows=args.max_windows,
            )
            configs, watchlist, raw = _load_config(args.config)
            providers = _registry().build(configs)
            retriever = MultiProviderRetriever(providers, {c.id: c for c in configs})
            aliases = {e.canonical_name: e.aliases for e in watchlist}
            manifests_dir = Path(args.manifests_dir)

            def collect_window(start: datetime, end: datetime) -> None:
                hours = max(1, int((end - start).total_seconds() // 3600))
                planned = QueryPlanner().plan(watchlist, end, lookback_hours=hours)
                name = end.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ.json")
                run_intelligence_cycle(
                    planned, retriever, RuleBasedExtractor(aliases), store, raw, end, manifests_dir / name
                )

            completed = BackfillRunner(collect_window).run(backfill_manifest, progress)
            print(completed.model_dump_json(indent=2))
            return 0
    finally:
        store.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
