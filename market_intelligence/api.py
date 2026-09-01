from __future__ import annotations

import html
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Generic, TypeVar

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from .catalog import DatasetCatalog, DatasetCatalogEntry
from .collection.corpus import CorpusCatalog, CorpusSnapshot
from .collection.status import CorpusStatus
from .context import request_id_context
from .errors import FeatureContractError, IntelligenceError, ReplayIntegrityError, SnapshotMismatchError, StorageError
from .feature_matrix import DEFAULT_CHUNK_SIZE
from .features import FEATURE_CONTRACT_VERSION, FEATURE_DEFINITIONS, FeatureDefinition
from .historical import HistoricalDatasetService
from .models import Direction, Document, EventSignal, EventType
from .operations import (
    IntelligenceSnapshot,
    ProviderHealth,
    QualityScoreboard,
    QuarantineRecord,
    RunManifest,
    RunStatus,
)
from .origins import OriginFrequency, generate_origins
from .replay_dataset import ReplayMode
from .services import SnapshotService
from .storage import SCHEMA_VERSION, IntelligenceStore

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    limit: int
    offset: int


class HistoricalDatasetRequest(BaseModel):
    """A historical feature-matrix build. Origins are generated, not supplied,
    so the schedule is guaranteed sorted, unique and UTC by construction."""

    start: datetime
    end: datetime
    frequency: str = Field(default="HOURLY", pattern="^(HOURLY|4H|DAILY)$")
    output_path: str = Field(min_length=1, max_length=1024)
    manifest_path: str = Field(min_length=1, max_length=1024)
    configuration_fingerprint: str = Field(min_length=1, max_length=128)
    provider_versions: dict[str, str] = Field(default_factory=dict)
    export_format: str = Field(default="parquet", pattern="^(parquet|csv)$")
    mode: str = Field(default="OPTIMIZED", pattern="^(OPTIMIZED|REFERENCE)$")
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=1, le=10_000)


class ReplayRequest(BaseModel):
    forecast_origin: datetime
    provider_versions: dict[str, str] = Field(default_factory=dict)
    configuration_fingerprint: str = Field(min_length=1, max_length=128)

    @field_validator("forecast_origin")
    @classmethod
    def aware_origin(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("forecast_origin must include a timezone")
        return value.astimezone(timezone.utc)


class EventTrace(BaseModel):
    event: EventSignal
    sources: list[Document]


class TimelineItem(BaseModel):
    timestamp: datetime
    entity: str | None
    event_type: EventType
    direction: Direction
    relevance: float
    confidence: float
    source_count: int


class ServiceStatus(BaseModel):
    status: str
    schema_version: int | None = None


def _bounded_window(start: datetime | None, end: datetime | None) -> None:
    for value in (start, end):
        if value is not None and value.tzinfo is None:
            raise HTTPException(422, "timestamps must include a timezone")
    if start and end and (end < start or end - start > timedelta(days=366)):
        raise HTTPException(422, "time range must be ordered and no longer than 366 days")


def create_app(db_path: str | Path | None = None, store: IntelligenceStore | None = None) -> FastAPI:
    """Explicit factory: importing this module performs no I/O or state mutation."""
    owned = store is None
    if store is None:
        if db_path is None:
            raise ValueError("db_path or store is required")
        store = IntelligenceStore(db_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owned:
            store.close()

    app = FastAPI(title="BTC Market Intelligence Service", version="3.0.0", lifespan=lifespan)
    app.state.store = store
    logger = logging.getLogger("btc_intelligence.api")

    @app.middleware("http")
    async def correlation(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9-]{1,64}", supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_context.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_context.reset(token)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_complete request_id=%s method=%s path=%s status=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    @app.exception_handler(IntelligenceError)
    async def intelligence_error(request: Request, exc: IntelligenceError) -> JSONResponse:
        status = 409 if isinstance(exc, (SnapshotMismatchError, FeatureContractError, ReplayIntegrityError)) else 503
        return JSONResponse(
            status_code=status,
            content={"error": type(exc).__name__, "detail": str(exc), "request_id": request.state.request_id},
        )

    @app.get("/health", response_model=ServiceStatus)
    def health() -> ServiceStatus:
        return ServiceStatus(status="alive")

    @app.get("/ready", response_model=ServiceStatus)
    def ready() -> ServiceStatus:
        if not store.ready():
            raise StorageError("storage schema is not ready")
        return ServiceStatus(status="ready", schema_version=SCHEMA_VERSION)

    @app.get("/runs", response_model=Page[RunManifest])
    def runs(
        status: RunStatus | None = None,
        provider: Annotated[str | None, Query(pattern=r"^[A-Za-z0-9_.-]{1,100}$")] = None,
        started_from: datetime | None = None,
        started_to: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[RunManifest]:
        _bounded_window(started_from, started_to)
        items = store.query_runs(
            status=status.value if status else None,
            provider=provider,
            started_from=started_from,
            started_to=started_to,
            limit=limit,
            offset=offset,
        )
        return Page(items=items, limit=limit, offset=offset)

    @app.get("/runs/{run_id}", response_model=RunManifest)
    def run(run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9-]{1,128}$")]) -> RunManifest:
        value = store.get_run(run_id)
        if value is None:
            raise HTTPException(404, "run not found")
        return value

    @app.get("/documents", response_model=Page[Document])
    def documents(
        provider: Annotated[str | None, Query(max_length=100)] = None,
        publisher: Annotated[str | None, Query(max_length=200)] = None,
        topic: Annotated[str | None, Query(max_length=500)] = None,
        published_from: datetime | None = None,
        published_to: datetime | None = None,
        available_from: datetime | None = None,
        available_to: datetime | None = None,
        forecast_origin: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[Document]:
        _bounded_window(published_from, published_to)
        _bounded_window(available_from, available_to)
        _bounded_window(forecast_origin, forecast_origin)
        items = store.query_documents(
            provider=provider,
            publisher=publisher,
            query=topic,
            published_from=published_from,
            published_to=published_to,
            available_from=available_from,
            available_to=available_to,
            forecast_origin=forecast_origin,
            limit=limit,
            offset=offset,
        )
        return Page(items=items, limit=limit, offset=offset)

    @app.get("/documents/{document_id}", response_model=Document)
    def document(
        document_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")], forecast_origin: datetime | None = None
    ) -> Document:
        _bounded_window(forecast_origin, forecast_origin)
        value = store.get_document(document_id)
        if value is None or (forecast_origin and value.available_at > forecast_origin):
            raise HTTPException(404, "document not found")
        return value

    @app.get("/events", response_model=Page[EventSignal])
    def events(
        entity: Annotated[str | None, Query(max_length=200)] = None,
        event_type: EventType | None = None,
        direction: Direction | None = None,
        minimum_relevance: Annotated[float | None, Query(ge=0, le=1)] = None,
        minimum_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
        available_from: datetime | None = None,
        available_to: datetime | None = None,
        forecast_origin: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[EventSignal]:
        _bounded_window(available_from, available_to)
        _bounded_window(forecast_origin, forecast_origin)
        items = store.query_events(
            entity=entity,
            event_type=event_type.value if event_type else None,
            direction=direction.value if direction else None,
            minimum_relevance=minimum_relevance,
            minimum_confidence=minimum_confidence,
            available_from=available_from,
            available_to=available_to,
            forecast_origin=forecast_origin,
            limit=limit,
            offset=offset,
        )
        return Page(items=items, limit=limit, offset=offset)

    @app.get("/events/{event_id}", response_model=EventTrace)
    def event(
        event_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")], forecast_origin: datetime | None = None
    ) -> EventTrace:
        _bounded_window(forecast_origin, forecast_origin)
        value = store.get_event(event_id)
        if value is None or (forecast_origin and value.available_time > forecast_origin):
            raise HTTPException(404, "event not found")
        sources = store.event_sources(event_id)
        if forecast_origin and any(source.available_at > forecast_origin for source in sources):
            raise HTTPException(404, "event not found")
        return EventTrace(event=value, sources=sources)

    @app.get("/timeline", response_model=Page[TimelineItem])
    def timeline(
        forecast_origin: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[TimelineItem]:
        _bounded_window(forecast_origin, forecast_origin)
        items = store.query_events(forecast_origin=forecast_origin, limit=limit, offset=offset)
        return Page(
            items=[
                TimelineItem(
                    timestamp=e.available_time,
                    entity=e.entity,
                    event_type=e.event_type,
                    direction=e.direction,
                    relevance=e.btc_relevance,
                    confidence=e.confidence,
                    source_count=len(e.source_ids),
                )
                for e in items
            ],
            limit=limit,
            offset=offset,
        )

    @app.get("/providers/health", response_model=list[ProviderHealth])
    def provider_health() -> list[ProviderHealth]:
        return store.all_health()

    @app.get("/providers/observability")
    def provider_observability() -> list[dict[str, object]]:
        return store.provider_observability()

    @app.get("/extraction/observability")
    def extraction_observability() -> dict[str, object]:
        return store.extraction_observability()

    @app.get("/quality/latest", response_model=QualityScoreboard | None)
    def quality_latest() -> QualityScoreboard | None:
        return store.latest_quality_scoreboard()

    @app.get("/snapshots", response_model=Page[IntelligenceSnapshot])
    def snapshots(
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[IntelligenceSnapshot]:
        return Page(items=store.list_snapshots(limit, offset), limit=limit, offset=offset)

    @app.get("/snapshots/{snapshot_id}", response_model=IntelligenceSnapshot)
    def snapshot(
        snapshot_id: str, configuration_fingerprint: str, feature_contract_version: str = FEATURE_CONTRACT_VERSION
    ) -> IntelligenceSnapshot:
        return SnapshotService(store).fetch_verified(snapshot_id, configuration_fingerprint, feature_contract_version)

    @app.get("/features", response_model=list[FeatureDefinition])
    def feature_contract() -> list[FeatureDefinition]:
        return list(FEATURE_DEFINITIONS)

    @app.get("/features/{snapshot_id}")
    def snapshot_features(snapshot_id: str, configuration_fingerprint: str) -> dict[str, object]:
        value = SnapshotService(store).fetch_verified(snapshot_id, configuration_fingerprint)
        return {
            "snapshot_id": value.snapshot_id,
            "feature_contract_version": value.feature_contract_version,
            "forecast_origin": value.forecast_origin,
            "features": value.features,
        }

    @app.post("/replay", response_model=IntelligenceSnapshot)
    def replay(body: ReplayRequest) -> IntelligenceSnapshot:
        return SnapshotService(store).build_snapshot(
            body.forecast_origin, body.provider_versions, body.configuration_fingerprint
        )

    @app.post("/replay/dataset")
    def replay_dataset(body: HistoricalDatasetRequest) -> dict[str, object]:
        """Build a historical feature matrix over a generated origin schedule.

        Calls the same HistoricalDatasetService the CLI does, so the two paths
        cannot produce different datasets from the same request.
        """
        service = HistoricalDatasetService(store, chunk_size=body.chunk_size)
        origins = generate_origins(body.start, body.end, OriginFrequency(body.frequency))
        result = service.build(
            origins,
            body.output_path,
            body.manifest_path,
            body.provider_versions,
            body.configuration_fingerprint,
            export_format=body.export_format,
            mode=ReplayMode(body.mode),
        )
        return {
            "dataset_id": result.manifest.dataset_id,
            "row_count": result.manifest.row_count,
            "chunk_count": result.chunk_count,
            "mode": result.manifest.mode,
            "origin_frequency": result.catalog_entry.origin_frequency,
            "file_hash": result.manifest.file_hash,
            "output": str(result.output_path),
            "manifest": str(result.manifest_path),
        }

    @app.get("/datasets", response_model=list[DatasetCatalogEntry])
    def datasets(limit: int = 50, offset: int = 0) -> list[DatasetCatalogEntry]:
        return DatasetCatalog(store.connection).list_datasets(limit=limit, offset=offset)

    @app.get("/datasets/{dataset_id}", response_model=DatasetCatalogEntry)
    def dataset_entry(dataset_id: str) -> DatasetCatalogEntry:
        entry = DatasetCatalog(store.connection).get(dataset_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
        return entry

    @app.get("/corpus/status", response_model=CorpusStatus)
    def corpus_status(
        extractor_version: str = "rules-v1",
        entities: str = "Donald Trump,Elon Musk,Jerome Powell,Michael Saylor,SEC,CFTC",
    ) -> CorpusStatus:
        """B4.1.35. What the corpus holds and whether B4 can be re-run.

        Counts and readiness only. No raw document text crosses this boundary,
        so a non-redistributable provider's content cannot leak through it.
        """
        from .cli import _corpus_status  # noqa: PLC0415 -- one implementation, shared with the CLI

        return _corpus_status(store, extractor_version, entities.split(","))

    @app.get("/corpora", response_model=list[CorpusSnapshot])
    def corpora(limit: int = 50) -> list[CorpusSnapshot]:
        return CorpusCatalog(store.connection).list_snapshots(limit=limit)

    @app.get("/corpora/{corpus_id}", response_model=CorpusSnapshot)
    def corpus_entry(corpus_id: str) -> CorpusSnapshot:
        snapshot = CorpusCatalog(store.connection).get(corpus_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"unknown corpus {corpus_id}")
        return snapshot

    @app.get("/collection/providers")
    def collection_providers() -> list[dict[str, object]]:
        """Declared providers and whether each can run. Credentials never appear."""
        from .cli import _provider_report  # noqa: PLC0415 -- shared with the CLI

        return _provider_report()

    @app.get("/quarantine", response_model=Page[QuarantineRecord])
    def quarantine(
        provider: str | None = None,
        record_type: str | None = None,
        failure_category: str | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[QuarantineRecord]:
        _bounded_window(time_from, time_to)
        items = store.query_quarantine(
            provider=provider,
            record_type=record_type,
            failure_category=failure_category,
            time_from=time_from,
            time_to=time_to,
            limit=limit,
            offset=offset,
        )
        return Page(items=items, limit=limit, offset=offset)

    @app.get("/metrics")
    def metrics() -> dict[str, int]:
        return store.metrics()

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        metrics = store.metrics()
        quality = store.latest_quality_scoreboard()
        health_items = store.all_health()
        summary = store.dashboard_summary()
        rows = "".join(
            f"<tr><td>{html.escape(item.provider_id)}</td><td>{item.state.value}</td><td>{item.documents_returned}</td></tr>"
            for item in health_items
        )
        return HTMLResponse(f"""<!doctype html><title>Market Intelligence / Research</title>
        <h1>MARKET INTELLIGENCE / RESEARCH</h1><p><strong>Not investment advice.</strong></p>
        <h2>Operations</h2><pre>{html.escape(str(metrics))}</pre>
        <h2>Provider health</h2><table><tr><th>Provider</th><th>State</th><th>Documents</th></tr>{rows}</table>
        <h2>Latest quality</h2><pre>{html.escape(str(quality.model_dump() if quality else {}))}</pre>
        <h2>Events, entities, runs, snapshots and quarantine</h2><pre>{html.escape(str(summary))}</pre>
        <p>This dashboard reports collection evidence and operational state. It does not attribute BTC price causality.</p>""")

    return app
