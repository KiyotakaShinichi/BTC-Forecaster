from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .catalog import DatasetCatalog
from .collection.clustering import EventCluster, cluster_events
from .collection.corpus import CorpusCatalog
from .collection.feeds import NEWS_API_DECLARATION, SYNDICATION_DECLARATION
from .collection.readiness import assess_entities, assess_family, assess_whale_contexts
from .collection.statements import STATEMENT_DECLARATION
from .collection.status import CorpusStatus, build_status
from .collection.whales import WHALE_DECLARATION
from .configuration import ProviderConfig, ProviderRegistry, QueryPlanner, WatchEntity
from .cycle import run_intelligence_cycle
from .extractors import RuleBasedExtractor
from .historical import HistoricalDatasetService
from .models import EventType
from .ops.backup import backup_age, create_backup
from .ops.backup import restore as restore_backup
from .ops.integrity import verify as ops_verify
from .ops.paths import StoragePaths, looks_ephemeral
from .ops.paths import validate as storage_validate
from .ops.scheduled import last_run_times
from .ops.summary import project_storage
from .ops.watchdog import assess as watchdog_assess
from .ops.watchdog import from_status
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

    corpus_status = sub.add_parser(
        "corpus-status",
        help="what the accumulated intelligence corpus holds, and whether B4 can be re-run",
    )
    corpus_status.add_argument("--json", action="store_true", help="machine-readable output")
    corpus_status.add_argument("--output", default=None, help="also write the JSON report here")
    corpus_status.add_argument("--extractor-version", default="rules-v1")
    corpus_status.add_argument(
        "--entities",
        default="Donald Trump,Elon Musk,Jerome Powell,Michael Saylor,SEC,CFTC",
        help="comma-separated entities to assess for readiness",
    )

    corpora = sub.add_parser("corpora", help="list registered corpus snapshots")
    corpora.add_argument("--limit", type=int, default=20)

    sub.add_parser("providers", help="declared providers and whether they can run")

    verify_parser = sub.add_parser("corpus-verify", help="check corpus integrity; fails closed")
    verify_parser.add_argument("--json", action="store_true")

    backup_parser = sub.add_parser("corpus-backup", help="write a verifiable corpus archive")
    backup_parser.add_argument("--output", required=True)
    backup_parser.add_argument("--manifest-dir", default=None)

    restore_parser = sub.add_parser(
        "corpus-restore", help="restore an archive into a NEW location and verify it"
    )
    restore_parser.add_argument("--archive", required=True)
    restore_parser.add_argument("--destination", required=True)

    ops_status = sub.add_parser(
        "ops-status", help="collection health, storage, backup age and B4 readiness"
    )
    ops_status.add_argument("--json", action="store_true")
    ops_status.add_argument("--state-root", default=None)

    watch = sub.add_parser("ops-watch", help="watchdog assessment; exit 0 ok, 1 warning, 2 critical")
    watch.add_argument("--json", action="store_true")
    watch.add_argument("--state-root", default=None)

    paths_parser = sub.add_parser(
        "ops-paths", help="resolved persistent paths and whether they are usable"
    )
    paths_parser.add_argument("--state-root", default=None)
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
        if args.command == "corpus-status":
            status = _corpus_status(store, args.extractor_version, args.entities.split(","))
            if args.output:
                status.write(args.output)
            print(status.model_dump_json(indent=2) if args.json else status.human_readable())
            return 0
        if args.command == "corpora":
            snapshots = CorpusCatalog(store.connection).list_snapshots(limit=args.limit)
            print(json.dumps([item.model_dump(mode="json") for item in snapshots], indent=2))
            return 0
        if args.command == "corpus-verify":
            integrity_report = ops_verify(store, as_of=datetime.now(timezone.utc))
            print(
                json.dumps(integrity_report.as_dict(), indent=2)
                if args.json
                else integrity_report.human_readable()
            )
            return 0 if integrity_report.ok else 2
        if args.command == "corpus-backup":
            manifest = create_backup(
                store,
                Path(args.db),
                Path(args.output),
                manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
            )
            print(json.dumps(manifest.as_dict(), indent=2))
            return 0
        if args.command == "corpus-restore":
            outcome = restore_backup(Path(args.archive), Path(args.destination))
            print(json.dumps(outcome.as_dict(), indent=2))
            return 0 if outcome.ok else 2
        if args.command == "ops-paths":
            resolved = StoragePaths.from_environment(args.state_root)
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
        if args.command in ("ops-status", "ops-watch"):
            report_payload = _ops_report(store, Path(args.db), args.state_root)
            if args.command == "ops-watch":
                if args.json:
                    print(json.dumps(report_payload["watchdog"], indent=2))
                else:
                    for alert in report_payload["watchdog"]["alerts"]:
                        print(f"[{alert['severity']}] {alert['code']}: {alert['message']}")
                return int(report_payload["exit_code"])
            print(
                json.dumps(report_payload, indent=2)
                if args.json
                else report_payload["human"]
            )
            return 0
        if args.command == "providers":
            print(json.dumps(_provider_report(), indent=2))
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


def _corpus_status(store: IntelligenceStore, extractor_version: str, entities: list[str]) -> CorpusStatus:
    """B4.1.34. Assemble the status report from whatever the store actually holds."""
    horizon = datetime.now(timezone.utc)
    documents = store.documents_as_of(horizon)
    events = [event for event in store.signals_as_of(horizon) if event.extractor_version == extractor_version]
    clusters = cluster_events(events, documents)
    by_context: dict[str, list[EventCluster]] = {}
    for cluster in clusters:
        for event in events:
            if event.event_id in cluster.event_ids and event.transfer_context is not None:
                by_context.setdefault(event.transfer_context.value, []).append(cluster)
                break

    families = [
        assess_family(f"event_type:{name}", [c for c in clusters if c.event_type == name])
        for name in sorted({cluster.event_type for cluster in clusters}) or ["REGULATION"]
    ]
    families.extend(assess_entities(clusters, [entity.strip() for entity in entities if entity.strip()]))
    families.extend(assess_whale_contexts(by_context))

    latest = CorpusCatalog(store.connection).latest()
    return build_status(
        documents,
        events,
        clusters,
        families,
        generated_at=horizon,
        corpus_id=latest.corpus_id if latest else None,
        expected_event_types=[member.value for member in EventType],
        expected_entities=[entity.strip() for entity in entities if entity.strip()],
        providers_enabled=len({document.provider for document in documents}),
    )


def _provider_report() -> list[dict[str, object]]:
    """B4.1.41. Declared providers, their policy, and why any is unavailable.

    Credential *presence* is reported; the credential itself never is.
    """
    declarations = [
        SYNDICATION_DECLARATION,
        NEWS_API_DECLARATION,
        STATEMENT_DECLARATION,
        WHALE_DECLARATION,
    ]
    rows: list[dict[str, object]] = []
    for declaration in declarations:
        present = bool(os.environ.get(declaration.credentials_env)) if declaration.credentials_env else True
        rows.append(
            {
                "provider_id": declaration.provider_id,
                "policy": declaration.policy.value,
                "operable": declaration.operable(present),
                "reason": declaration.disabled_reason(present),
                "credentials_env": declaration.credentials_env,
                "credential_present": present,
                "requires_paid_contract": declaration.requires_paid_contract,
                "minimum_interval_seconds": declaration.minimum_interval_seconds,
                "raw_retention": declaration.raw_retention.value,
                "primary_source": declaration.primary_source,
                "purpose": declaration.purpose,
                "rate_limit_note": declaration.rate_limit_note,
                "terms_note": declaration.terms_note,
            }
        )
    return rows


def _ops_report(store: IntelligenceStore, database: Path, state_root: str | None) -> dict[str, Any]:
    """O11. Everything an operator asks, answered from one place.

    Shared by the CLI and the API so the two cannot drift, exactly as the corpus
    status report is.
    """
    moment = datetime.now(timezone.utc)
    paths = StoragePaths.from_environment(state_root or database.parent)
    checked = storage_validate(paths)
    status = _corpus_status(store, "rules-v1", ["Donald Trump", "Elon Musk", "Jerome Powell", "SEC"])
    integrity = ops_verify(store, as_of=moment)
    last_run, last_success = last_run_times(store)

    provider_last_success: dict[str, datetime | None] = {}
    for row in store.connection.execute("SELECT payload FROM watermarks").fetchall():
        record = json.loads(row[0])
        provider = str(record.get("provider_id", ""))
        retrieval = record.get("last_retrieval_time")
        if provider and retrieval:
            provider_last_success[provider] = datetime.fromisoformat(retrieval).astimezone(timezone.utc)

    found = backup_age(paths.backups, moment) if paths.backups.exists() else None
    watchdog = watchdog_assess(
        from_status(
            status,
            now=moment,
            last_run_at=last_run,
            last_successful_run_at=last_success,
            provider_last_success=provider_last_success,
            storage_ok=checked.ok,
            storage_detail="; ".join(f"{c.name}: {c.reason}" for c in checked.failures()),
            integrity_status=integrity.status,
            integrity_detail=integrity.human_readable().splitlines()[0],
            last_backup_at=found[1] if found else None,
        )
    )
    storage = project_storage(status)
    recent = [row for row in status.daily_coverage if row.day >= (moment - timedelta(hours=24)).date()]
    backup_age_seconds = int((moment - found[1]).total_seconds()) if found else None

    human = "\n".join(
        [
            f"last collection        {last_run.isoformat() if last_run else '(never)'}",
            f"last successful        {last_success.isoformat() if last_success else '(never)'}",
            f"providers              {len(provider_last_success)} seen, "
            f"{sum(1 for value in provider_last_success.values() if value)} healthy",
            f"documents last 24h     {sum(row.documents for row in recent)}",
            f"events last 24h        {sum(row.events for row in recent)}",
            f"coverage gap days      {status.collection_gap_days}",
            f"latest corpus          {status.corpus_id or '(none registered)'}",
            f"backup age             {backup_age_seconds if backup_age_seconds is not None else '(no backup)'}",
            f"corpus integrity       {integrity.status.value}",
            f"storage                {storage.total_bytes:,} B, ~{storage.projected_365d_bytes:,} B at 365d",
            f"B4 readiness           {status.readiness.value}",
            f"health                 {watchdog.worst.value}",
        ]
    )

    return {
        "generated_at": moment.isoformat(),
        "paths": paths.as_dict(),
        "storage_ok": checked.ok,
        "last_collection": last_run.isoformat() if last_run else None,
        "last_successful_collection": last_success.isoformat() if last_success else None,
        "providers_seen": len(provider_last_success),
        "providers_healthy": sum(1 for value in provider_last_success.values() if value),
        "documents_last_24h": sum(row.documents for row in recent),
        "events_last_24h": sum(row.events for row in recent),
        "coverage_gap_days": status.collection_gap_days,
        "latest_corpus_id": status.corpus_id,
        "backup_age_seconds": backup_age_seconds,
        "corpus_integrity": integrity.status.value,
        "storage": storage.as_dict(),
        "b4_readiness": status.readiness.value,
        "watchdog": watchdog.as_dict(),
        "exit_code": watchdog.exit_code,
        "human": human,
    }
