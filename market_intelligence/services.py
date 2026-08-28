from __future__ import annotations

import logging
from datetime import datetime

from .aggregation import FeatureAggregator
from .context import request_id_context
from .cycle import ReplayService
from .features import FEATURE_CONTRACT_VERSION
from .operations import IntelligenceSnapshot, ProviderHealth, QualityReport
from .quality import evaluate_quality
from .storage import IntelligenceStore


class IntelligenceReadService:
    """Shared read operations used by both CLI and API."""

    def __init__(self, store: IntelligenceStore):
        self.store = store

    def health(self) -> list[ProviderHealth]:
        return self.store.all_health()

    def quality(self, forecast_origin: datetime) -> QualityReport:
        return evaluate_quality(
            self.store.documents_as_of(forecast_origin), self.store.signals_as_of(forecast_origin), forecast_origin
        )

    def aggregate(self, forecast_origin: datetime) -> dict[str, float]:
        from .aggregation import FeatureAggregator

        return FeatureAggregator().aggregate(self.store.signals_as_of(forecast_origin), forecast_origin)


class SnapshotService:
    def __init__(self, store: IntelligenceStore):
        self.store = store

    def build_snapshot(
        self, forecast_origin: datetime, provider_versions: dict[str, str], configuration_fingerprint: str
    ) -> IntelligenceSnapshot:
        logging.getLogger("btc_intelligence.replay").info(
            "snapshot_build request_id=%s forecast_origin=%s",
            request_id_context.get(),
            forecast_origin.isoformat(),
        )
        return ReplayService(self.store).replay(forecast_origin, provider_versions, configuration_fingerprint)

    def fetch_verified(
        self, snapshot_id: str, configuration_fingerprint: str, feature_contract_version: str = FEATURE_CONTRACT_VERSION
    ) -> IntelligenceSnapshot:
        return self.store.verify_snapshot(snapshot_id, configuration_fingerprint, feature_contract_version)

    def build_many(
        self,
        forecast_origins: list[datetime],
        provider_versions: dict[str, str],
        configuration_fingerprint: str,
    ) -> list[IntelligenceSnapshot]:
        """Build ordered snapshots from one bounded evidence read."""
        if not forecast_origins:
            return []
        logging.getLogger("btc_intelligence.replay").info(
            "snapshot_batch request_id=%s origin_count=%s",
            request_id_context.get(),
            len(forecast_origins),
        )
        documents = self.store.documents_as_of(max(forecast_origins))
        events = self.store.signals_as_of(max(forecast_origins))
        snapshots = []
        for origin in forecast_origins:
            eligible_documents = [document for document in documents if document.available_at <= origin]
            document_ids = {document.document_id for document in eligible_documents}
            eligible_events = [
                event for event in events if event.available_time <= origin and set(event.source_ids) <= document_ids
            ]
            snapshot = IntelligenceSnapshot.create(
                origin,
                list(document_ids),
                [event.event_id for event in eligible_events],
                FeatureAggregator().aggregate(eligible_events, origin),
                provider_versions,
                [event.extractor_version for event in eligible_events],
                configuration_fingerprint,
                [document.text_hash for document in eligible_documents],
                FEATURE_CONTRACT_VERSION,
            )
            snapshots.append(snapshot)
        self.store.put_snapshots(snapshots)
        return snapshots
