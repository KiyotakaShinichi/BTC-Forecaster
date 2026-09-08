from __future__ import annotations

from datetime import datetime

from .aggregation import FeatureAggregator
from .context import request_id_context
from .cycle import ReplayService
from .features import FEATURE_CONTRACT_VERSION
from .logs import get_logger
from .operations import IntelligenceSnapshot, ProviderHealth, QualityReport
from .quality import evaluate_quality
from .replay_engine import BulkReplayEngine
from .storage import IntelligenceStore

_log = get_logger("replay")


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

        return FeatureAggregator().aggregate(
            self.store.eligible_signals_as_of(forecast_origin), forecast_origin
        )


class SnapshotService:
    def __init__(self, store: IntelligenceStore):
        self.store = store

    def build_snapshot(
        self, forecast_origin: datetime, provider_versions: dict[str, str], configuration_fingerprint: str
    ) -> IntelligenceSnapshot:
        _log.emit(
            "snapshot_build",
            request_id=request_id_context.get(),
            origin=forecast_origin,
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
        """Reference implementation. The correctness oracle for the bulk path.

        Retained deliberately (B3.1.1): the optimised engine proves equivalence
        against this, so it must not be deleted or quietly replaced. It is also
        the simplest readable statement of the semantics.
        """
        if not forecast_origins:
            return []
        _log.emit(
            "snapshot_batch",
            request_id=request_id_context.get(),
            origin_count=len(forecast_origins),
            mode="REFERENCE",
        )
        documents = self.store.documents_as_of(max(forecast_origins))
        events = self.store.eligible_signals_as_of(max(forecast_origins))
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

    def build_many_bulk(
        self,
        forecast_origins: list[datetime],
        provider_versions: dict[str, str],
        configuration_fingerprint: str,
    ) -> list[IntelligenceSnapshot]:
        """Optimised path (B3.1.2). Semantically identical to build_many.

        Same bounded reads; the difference is entirely in the inner loop, which
        no longer re-scans the whole history per origin. Snapshot ids are
        byte-identical, which is the strongest available statement that nothing
        about membership, features or missingness changed.
        """
        if not forecast_origins:
            return []
        _log.emit(
            "snapshot_batch",
            request_id=request_id_context.get(),
            origin_count=len(forecast_origins),
            mode="OPTIMIZED",
        )
        horizon = max(forecast_origins)
        engine = BulkReplayEngine(
            self.store.documents_as_of(horizon), self.store.eligible_signals_as_of(horizon)
        )

        snapshots = []
        cached_prefix = -1
        document_ids: tuple[str, ...] = ()
        source_hashes: tuple[str, ...] = ()
        for origin in forecast_origins:
            evidence = engine.evidence_for(origin)
            if evidence.document_prefix != cached_prefix:
                # The document set only changes when new evidence becomes
                # available, which is far rarer than once per origin.
                document_ids, source_hashes = engine.document_membership(evidence.document_prefix)
                cached_prefix = evidence.document_prefix
            snapshots.append(
                IntelligenceSnapshot.create(
                    origin,
                    list(document_ids),
                    list(evidence.eligible_event_ids),
                    evidence.features,
                    provider_versions,
                    list(evidence.eligible_extractor_versions),
                    configuration_fingerprint,
                    list(source_hashes),
                    FEATURE_CONTRACT_VERSION,
                )
            )
        self.store.put_snapshots(snapshots)
        return snapshots
