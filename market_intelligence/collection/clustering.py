"""B4.1.14 — one real-world event, however many articles describe it.

B4 already clusters at study time. This clusters at *collection* time and
persists the identity, which matters for a different reason: the corpus needs to
know how many distinct occurrences it holds before anyone asks a research
question. "We have 4,000 documents" and "we have 4,000 events" are very
different readiness claims, and only the second one is about the world.

The clustering rule is deterministic and explicable rather than clever: same
event type, same entity, within a declared window, chained so a multi-day news
cycle stays one cluster. A similarity model would cluster better and would also
make the corpus's shape depend on a model version — which is a thing to avoid in
the layer whose whole job is stable provenance.

Two properties the study layer depends on:

**A primary source is never merged away.** When an announcement and coverage of
it fall in one cluster, the cluster records that a primary source is present and
which document it was. A study separating announcement from interpretation
(B4.1.7) needs that, and a cluster that averaged them would destroy it.

**Corroboration is metadata, not sample size.** Document, publisher and provider
counts are carried so a reader can see how broadly an event was reported without
any of those numbers inflating N.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..models import Document, EventSignal

DEFAULT_CLUSTER_WINDOW_HOURS = 24


class EventCluster(BaseModel):
    """A group of events judged to describe one underlying occurrence."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    event_type: str
    entity: str | None
    #: Earliest availability in the cluster: when the market could first know.
    first_available_at: datetime
    last_available_at: datetime
    event_ids: tuple[str, ...] = Field(min_length=1)
    document_ids: tuple[str, ...] = ()
    document_count: int = 0
    publisher_count: int = 0
    provider_count: int = 0
    #: B4.1.7. Whether the party the news is about is among the sources.
    primary_source_present: bool = False
    primary_document_ids: tuple[str, ...] = ()
    extractor_versions: tuple[str, ...] = ()

    @property
    def span(self) -> timedelta:
        return self.last_available_at - self.first_available_at


def cluster_events(
    events: Sequence[EventSignal],
    documents: Sequence[Document],
    *,
    window_hours: int = DEFAULT_CLUSTER_WINDOW_HOURS,
) -> list[EventCluster]:
    """Group events into underlying occurrences, deterministically.

    Ordering by (availability, id) then chaining means the result does not
    depend on the order events arrive in -- a property a test asserts by feeding
    the same events reversed.
    """
    by_id = {document.document_id: document for document in documents}
    window = timedelta(hours=window_hours)
    ordered = sorted(events, key=lambda event: (event.available_time, event.event_id))

    open_groups: dict[tuple[str, str], list[EventSignal]] = {}
    finished: list[list[EventSignal]] = []
    for event in ordered:
        key = (event.event_type.value, event.entity or "")
        current = open_groups.get(key)
        if current is not None and event.available_time - current[-1].available_time <= window:
            current.append(event)
            continue
        if current is not None:
            finished.append(current)
        open_groups[key] = [event]
    finished.extend(open_groups.values())
    finished.sort(key=lambda group: (group[0].available_time, group[0].event_id))

    clusters: list[EventCluster] = []
    for group in finished:
        document_ids = sorted({source for event in group for source in event.source_ids})
        members = [by_id[document_id] for document_id in document_ids if document_id in by_id]
        primary = [
            document.document_id for document in members if document.source_metadata.primary_source
        ]
        clusters.append(
            EventCluster(
                cluster_id=hashlib.sha256(
                    "|".join(event.event_id for event in group).encode()
                ).hexdigest()[:16],
                event_type=group[0].event_type.value,
                entity=group[0].entity,
                first_available_at=group[0].available_time,
                last_available_at=group[-1].available_time,
                event_ids=tuple(event.event_id for event in group),
                document_ids=tuple(document_ids),
                document_count=len(document_ids),
                publisher_count=len({document.publisher for document in members}),
                provider_count=len({document.provider for document in members}),
                primary_source_present=bool(primary),
                primary_document_ids=tuple(sorted(primary)),
                extractor_versions=tuple(sorted({event.extractor_version for event in group})),
            )
        )
    return clusters


class ClusterStore:
    """Persisted cluster identity, so corpus size is a stable fact.

    Recomputed and rewritten per cycle rather than accumulated: clustering is a
    pure function of the events in the corpus, and a cluster that merged with a
    later one must be allowed to. Nothing about a *document* or an *event*
    changes here, so no history is rewritten -- only the derived grouping.
    """

    def __init__(self, connection: object) -> None:
        self.connection = connection
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS event_clusters (
              cluster_id VARCHAR PRIMARY KEY,
              event_type VARCHAR NOT NULL,
              entity VARCHAR,
              first_available_at TIMESTAMPTZ NOT NULL,
              payload JSON NOT NULL
            )
            """
        )

    def _execute(self, sql: str, parameters: list[object] | None = None) -> object:
        return self.connection.execute(sql, parameters) if parameters else self.connection.execute(sql)  # type: ignore[attr-defined]

    def replace_all(self, clusters: Sequence[EventCluster]) -> int:
        self._execute("DELETE FROM event_clusters")
        for cluster in clusters:
            self._execute(
                "INSERT INTO event_clusters VALUES (?, ?, ?, ?, ?)",
                [
                    cluster.cluster_id,
                    cluster.event_type,
                    cluster.entity,
                    cluster.first_available_at,
                    cluster.model_dump_json(),
                ],
            )
        return len(clusters)

    def all(self) -> list[EventCluster]:
        rows = self._execute(
            "SELECT payload FROM event_clusters ORDER BY first_available_at, cluster_id"
        ).fetchall()  # type: ignore[attr-defined]
        return [EventCluster.model_validate(json.loads(row[0])) for row in rows]

    def count(self) -> int:
        row = self._execute("SELECT count(*) FROM event_clusters").fetchone()  # type: ignore[attr-defined]
        return int(row[0]) if row else 0

    def counts_by_type(self) -> dict[str, int]:
        rows = self._execute(
            "SELECT event_type, count(*) FROM event_clusters GROUP BY 1 ORDER BY 1"
        ).fetchall()  # type: ignore[attr-defined]
        return {str(row[0]): int(row[1]) for row in rows}

    def counts_by_entity(self) -> dict[str, int]:
        rows = self._execute(
            "SELECT coalesce(entity, '(none)'), count(*) FROM event_clusters GROUP BY 1 ORDER BY 1"
        ).fetchall()  # type: ignore[attr-defined]
        return {str(row[0]): int(row[1]) for row in rows}


def effective_non_overlapping(clusters: Sequence[EventCluster], horizon_hours: int) -> int:
    """B4.1.31. How many clusters are independent at a given outcome horizon.

    Two clusters six hours apart share most of a 24-hour outcome window, so the
    study is powered on far fewer observations than the raw count suggests. A
    greedy sweep gives a lower bound, which is the honest direction for a
    readiness gate.
    """
    span = timedelta(hours=horizon_hours)
    count = 0
    frontier: datetime | None = None
    for cluster in sorted(clusters, key=lambda item: item.first_available_at):
        if frontier is None or cluster.first_available_at >= frontier:
            count += 1
            frontier = cluster.first_available_at + span
    return count


def write_clusters(clusters: Sequence[EventCluster], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps([cluster.model_dump(mode="json") for cluster in clusters], indent=2), encoding="utf-8"
    )
    temporary.replace(target)
    return target


__all__ = [
    "DEFAULT_CLUSTER_WINDOW_HOURS",
    "ClusterStore",
    "EventCluster",
    "cluster_events",
    "effective_non_overlapping",
    "write_clusters",
]
