from __future__ import annotations

from datetime import datetime

from .cache import DeterministicCache
from .extractors import EventExtractor
from .models import Document, EventSignal
from .providers import SearchProvider


class IntelligencePipeline:
    def __init__(self, provider: SearchProvider, extractor: EventExtractor, cache: DeterministicCache):
        self.provider, self.extractor, self.cache = provider, extractor, cache

    def run(self, query: str, start: datetime, end: datetime) -> tuple[list[Document], list[EventSignal]]:
        key = self.cache.key(query, start, end, self.provider.name, self.extractor.version)
        cached = self.cache.get(key)
        if cached is not None:
            return (
                [Document.model_validate(d) for d in cached["documents"]],
                [EventSignal.model_validate(s) for s in cached["signals"]],
            )
        documents = self.provider.search(query, start, end)
        signals = self.extractor.extract(documents)
        self.cache.put(
            key,
            {
                "documents": [d.model_dump(mode="json") for d in documents],
                "signals": [s.model_dump(mode="json") for s in signals],
            },
        )
        return documents, signals
