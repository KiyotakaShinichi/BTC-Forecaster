from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable

from pydantic import ValidationError

from .models import (Direction, Document, EventSignal, EventType, ExtractionMethod,
                     SignalCategory, TransferContext)


class ExtractionError(ValueError):
    pass


def _validate_provenance(signals: list[EventSignal], documents: list[Document]) -> list[EventSignal]:
    sources = {document.document_id: document for document in documents}
    for signal in signals:
        if not set(signal.source_ids) <= sources.keys():
            raise ExtractionError("signal references an unknown source document")
        first_usable = max(sources[source_id].available_at for source_id in signal.source_ids)
        if signal.available_time < first_usable:
            raise ExtractionError("signal cannot be available before its source documents")
    return signals


class EventExtractor(ABC):
    version: str

    @abstractmethod
    def extract(self, documents: list[Document]) -> list[EventSignal]: ...


class FixtureExtractor(EventExtractor):
    def __init__(self, signals: list[EventSignal], version: str = "fixture-v1"):
        self._signals, self.version = signals, version

    def extract(self, documents: list[Document]) -> list[EventSignal]:
        source_ids = {d.document_id for d in documents}
        return _validate_provenance(
            [s for s in self._signals if set(s.source_ids) <= source_ids], documents
        )


class StructuredLlmExtractor(EventExtractor):
    """Provider-independent validated-JSON boundary around an injected LLM call."""

    def __init__(self, completion: Callable[[str], str], version: str,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._completion, self.version, self._now = completion, version, now

    def extract(self, documents: list[Document]) -> list[EventSignal]:
        prompt = json.dumps({
            "instruction": "Extract events as JSON array only. Never forecast or recommend trades.",
            "schema": EventSignal.model_json_schema(),
            "documents": [d.model_dump(mode="json") for d in documents],
        }, sort_keys=True)
        try:
            raw = json.loads(self._completion(prompt))
            if not isinstance(raw, list):
                raise ExtractionError("LLM response must be a JSON array")
            # The trusted adapter, not model-authored JSON, records the method.
            signals = [EventSignal.model_validate({**item, "extraction_method": ExtractionMethod.LLM}) for item in raw]
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ExtractionError("invalid structured LLM response") from exc
        for signal in signals:
            if signal.extractor_version != self.version:
                raise ExtractionError("extractor version mismatch")
            if signal.available_time > self._now():
                raise ExtractionError("LLM output has a future availability timestamp")
        return _validate_provenance(signals, documents)


class RuleBasedExtractor(EventExtractor):
    """Conservative routing fallback; hints are lower-confidence than semantic extraction."""
    version = "rules-v1"
    _rules = (
        (("sec", "regulation", "lawsuit"), EventType.REGULATION),
        (("federal reserve", "fed ", "powell", "interest rate"), EventType.MONETARY_POLICY),
        (("etf",), EventType.ETF_FLOW),
        (("hack", "breach", "exploit"), EventType.SECURITY_INCIDENT),
        (("exchange outage", "exchange incident"), EventType.EXCHANGE_INCIDENT),
        (("whale", "large transfer"), EventType.WHALE_TRANSFER),
    )

    def __init__(self, entities: dict[str, tuple[str, ...]] | None = None):
        self.entities = entities or {}

    def extract(self, documents: list[Document]) -> list[EventSignal]:
        output = []
        for doc in documents:
            text = doc.title.casefold()
            event_type = next((event for terms, event in self._rules if any(term in text for term in terms)), None)
            if event_type is None:
                continue
            entity = next((name for name, aliases in self.entities.items()
                           if any(alias.casefold() in text for alias in (name, *aliases))), None)
            context = TransferContext.UNKNOWN if event_type == EventType.WHALE_TRANSFER else None
            output.append(EventSignal(event_id=EventSignal.stable_id([doc.document_id], event_type, doc.available_at),
                event_time=doc.published_at or doc.available_at, available_time=doc.available_at,
                source_ids=(doc.document_id,), category=SignalCategory.ONCHAIN if event_type == EventType.WHALE_TRANSFER else SignalCategory.WEB_EVENT,
                entity=entity, event_type=event_type, direction=Direction.UNKNOWN, sentiment=0.0,
                btc_relevance=0.55, novelty=0.5, confidence=0.35, expected_horizon_hours=24,
                summary=f"Rule-based hint: {doc.title}", transfer_context=context,
                extractor_version=self.version, extraction_method=ExtractionMethod.RULE_BASED))
        return _validate_provenance(output, documents)
