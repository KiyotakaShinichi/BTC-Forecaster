from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Callable

from pydantic import ValidationError

from .models import Document, EventSignal


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

    def __init__(self, completion: Callable[[str], str], version: str):
        self._completion, self.version = completion, version

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
            signals = [EventSignal.model_validate(item) for item in raw]
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ExtractionError("invalid structured LLM response") from exc
        for signal in signals:
            if signal.extractor_version != self.version:
                raise ExtractionError("extractor version mismatch")
        return _validate_provenance(signals, documents)
