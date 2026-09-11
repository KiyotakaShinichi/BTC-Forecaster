"""The answers an operator asks for, assembled once.

Three reports -- corpus status, provider availability and the operations
summary -- are each read through two surfaces: `btc-intel` on a host and the
HTTP API behind it. They used to live in `cli.py` as underscore-prefixed
helpers that `api.py` imported anyway, which meant the API depended on the
argument parser to answer a question that has nothing to do with arguments, and
"the CLI and the API cannot drift" rested on one of them importing the other's
private name.

They are reports. They take a store and return a value; they do not print, they
do not parse and they do not exit. Rendering and exit codes belong to whoever
asked.

`cli.py` re-exports the underscore names, because the workflow, the API and
several tests import them by those names and a refactor that renames a used
symbol is a break dressed as a tidy-up.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from .collection.clustering import EventCluster, cluster_events
from .collection.corpus import CorpusCatalog
from .collection.coverage import collection_coverage, span_coverage, successful_days, utc_day
from .collection.feeds import NEWS_API_DECLARATION, SYNDICATION_DECLARATION
from .collection.lag import lag_summary
from .collection.readiness import FamilyReadiness, assess_family
from .collection.statements import STATEMENT_DECLARATION
from .collection.status import CorpusStatus, build_status
from .collection.whales import WHALE_DECLARATION
from .extractors import CURRENT_RULE_EXTRACTOR_VERSION
from .models import EventType, TransferContext
from .ops.backup import backup_age
from .ops.integrity import verify as ops_verify
from .ops.paths import StoragePaths
from .ops.paths import validate as storage_validate
from .ops.scheduled import last_run_times
from .ops.summary import project_storage
from .ops.watchdog import assess as watchdog_assess
from .ops.watchdog import from_status
from .storage import IntelligenceStore

#: The entities the operations report asks about. Named here rather than at the
#: call site so the CLI and the API cannot ask different questions and compare
#: the answers.
OPERATIONS_ENTITIES = ("Donald Trump", "Elon Musk", "Jerome Powell", "SEC")


def corpus_status(store: IntelligenceStore, extractor_version: str, entities: list[str]) -> CorpusStatus:
    """B4.1.34. Assemble the status report from whatever the store actually holds."""
    horizon = datetime.now(timezone.utc)
    documents = store.documents_as_of(horizon)
    # The research view. An observation invalidated by a correction is still in
    # the store, still auditable and still counted by `corpus-corrections`; it
    # is simply not something a readiness gate or a study may count.
    events = [
        event
        for event in store.eligible_signals_as_of(horizon)
        if event.extractor_version == extractor_version
    ]
    clusters = cluster_events(events, documents)
    by_context: dict[str, list[EventCluster]] = {}
    for cluster in clusters:
        for event in events:
            if event.event_id in cluster.event_ids and event.transfer_context is not None:
                by_context.setdefault(event.transfer_context.value, []).append(cluster)
                break

    # Coverage per family, over the family's own span, from the one definition in
    # collection/coverage.py -- the same one B5's Gate 1 audit reads. Before B5.1
    # no coverage was passed here at all, and every family read 0%.
    success = successful_days(store.connection, as_of=horizon)

    def family(name: str, members: Sequence[EventCluster]) -> FamilyReadiness:
        coverage = None
        if members:
            starts = sorted(cluster.first_available_at for cluster in members)
            coverage = span_coverage(success, utc_day(starts[0]), utc_day(starts[-1]))
        return assess_family(name, members, coverage_fraction=coverage)

    named = [entity.strip() for entity in entities if entity.strip()]
    families = [
        family(f"event_type:{name}", [c for c in clusters if c.event_type == name])
        for name in sorted({cluster.event_type for cluster in clusters}) or ["REGULATION"]
    ]
    families.extend(family(f"entity:{name}", [c for c in clusters if (c.entity or "") == name]) for name in named)
    families.extend(family(f"whale:{context.value}", by_context.get(context.value, [])) for context in TransferContext)

    latest = CorpusCatalog(store.connection).latest()
    return build_status(
        documents,
        events,
        clusters,
        families,
        generated_at=horizon,
        corpus_id=latest.corpus_id if latest else None,
        expected_event_types=[member.value for member in EventType],
        expected_entities=named,
        providers_enabled=len({document.provider for document in documents}),
        successful_run_days=[datetime.combine(day, time(), tzinfo=timezone.utc) for day in success],
        collection=collection_coverage(store.connection, as_of=horizon),
        collection_lag=lag_summary(store.connection, documents),
    )


def provider_report() -> list[dict[str, object]]:
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


def ops_report(store: IntelligenceStore, database: Path, state_root: str | None) -> dict[str, Any]:
    """O11. Everything an operator asks, answered from one place.

    Shared by the CLI and the API so the two cannot drift, exactly as the corpus
    status report is.
    """
    moment = datetime.now(timezone.utc)
    paths = StoragePaths.from_environment(state_root or database.parent)
    checked = storage_validate(paths)
    status = corpus_status(store, CURRENT_RULE_EXTRACTOR_VERSION, list(OPERATIONS_ENTITIES))
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
            # Previously left at its default of zero, which meant the watchdog's
            # quarantine check could never fire however bad things got.
            quarantined_last_24h=store.quarantine_count(since=moment - timedelta(hours=24)),
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
            f"collection coverage    "
            f"{status.collection.describe() if status.collection is not None else 'not reported'}",
            f"collection lag         "
            f"{status.collection_lag.describe() if status.collection_lag is not None else 'not reported'}",
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
        # B5.1: the collection record and the lag, as corpus-status and Gate 1
        # compute them. coverage_gap_days counts only days between the first and
        # last document, so a collector that stopped last week shows no gap there.
        "collection": status.collection.model_dump(mode="json") if status.collection is not None else None,
        "collection_lag": (
            status.collection_lag.model_dump(mode="json") if status.collection_lag is not None else None
        ),
        "latest_corpus_id": status.corpus_id,
        "backup_age_seconds": backup_age_seconds,
        "corpus_integrity": integrity.status.value,
        "storage": storage.as_dict(),
        "b4_readiness": status.readiness.value,
        "watchdog": watchdog.as_dict(),
        "exit_code": watchdog.exit_code,
        "human": human,
    }

__all__ = ["OPERATIONS_ENTITIES", "corpus_status", "ops_report", "provider_report"]
