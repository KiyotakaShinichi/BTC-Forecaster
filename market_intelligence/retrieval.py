from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

from .collection.backoff import RETRYABLE, classify_exception
from .configuration import ProviderConfig, QuerySpec
from .models import Document, RetrievalProvenance
from .providers import SearchProvider


class ProviderAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider_id: str
    query_id: str
    success: bool
    attempts: int
    latency_ms: float
    documents_received: int = 0
    error: str | None = None
    rate_limited: bool = False


class RetrievalResult(BaseModel):
    documents: list[Document]
    attempts: list[ProviderAttempt]

    @property
    def queries_successful(self) -> int:
        return sum(a.success for a in self.attempts)

    @property
    def queries_failed(self) -> int:
        return sum(not a.success for a in self.attempts)


class MultiProviderRetriever:
    def __init__(
        self,
        providers: dict[str, SearchProvider],
        configs: dict[str, ProviderConfig],
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.providers, self.configs, self.max_retries = providers, configs, max_retries
        self.sleep, self.jitter, self.clock = sleep, jitter, clock
        self._last_call: dict[str, float] = {}

    def retrieve(self, queries: list[QuerySpec]) -> RetrievalResult:
        documents, attempts = [], []
        for query in queries:
            for provider_id in sorted(self.providers, key=lambda p: (self.configs[p].priority, p)):
                config, provider = self.configs[provider_id], self.providers[provider_id]
                start = query.generated_at - __import__("datetime").timedelta(hours=query.lookback_hours)
                began, error, received = self.clock(), None, []
                used = 0
                for used in range(1, self.max_retries + 2):
                    try:
                        if config.rate_limit:
                            minimum_interval = 1.0 / config.rate_limit
                            elapsed = self.clock() - self._last_call.get(provider_id, float("-inf"))
                            if elapsed < minimum_interval:
                                self.sleep(minimum_interval - elapsed)
                        pool = ThreadPoolExecutor(max_workers=1)
                        try:
                            self._last_call[provider_id] = self.clock()
                            future = pool.submit(provider.search, query.query, start, query.generated_at)
                            received = future.result(timeout=config.timeout)
                        finally:
                            pool.shutdown(wait=False, cancel_futures=True)
                        error = None
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        credential = config.credential()
                        if credential:
                            error = error.replace(credential, "[REDACTED]")
                        # B4.1.21. Only retry what retrying can fix. This loop
                        # previously retried everything, so a rejected credential
                        # was re-sent twice more per query -- which is how a key
                        # gets suspended, and it could never have succeeded.
                        failure = classify_exception(exc)
                        # Nor what the provider already retried under its own
                        # policy: two layers of retries multiply the load on a
                        # source that is already failing.
                        if failure not in RETRYABLE or getattr(exc, "retries_exhausted", False):
                            break
                        if used <= self.max_retries:
                            self.sleep((2 ** (used - 1)) + self.jitter())
                success = error is None
                documents.extend(received if success else [])
                attempts.append(
                    ProviderAttempt(
                        provider_id=provider_id,
                        query_id=query.query_id,
                        success=success,
                        attempts=used,
                        latency_ms=(self.clock() - began) * 1000,
                        documents_received=len(received),
                        error=error,
                        rate_limited=bool(error and "429" in error),
                    )
                )
        return RetrievalResult(documents=documents, attempts=attempts)


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(query), ""))


def _normalized_title(title: str) -> str:
    return " ".join("".join(c.casefold() if c.isalnum() else " " for c in title).split())


def deduplicate_across_providers(documents: Iterable[Document]) -> list[Document]:
    groups: list[list[Document]] = []
    for document in sorted(documents, key=lambda d: (d.available_at, d.document_id)):
        match = None
        for group in groups:
            representative = group[0]
            exact = canonical_url(str(document.url)) == canonical_url(str(representative.url))
            # Same publisher, same disclosure stream. Every SEC feed carries the
            # identical publisher string, so the corroboration clause below --
            # same title, same publisher, published within five minutes -- would
            # merge an administrative proceeding with the press release
            # announcing it. They are different instruments, issued through
            # different channels, and one of the two classifications would be
            # silently lost in whichever document happened to survive.
            #
            # An identical URL is still identity and still merges: that is the
            # same document however it was reached. Identical text and identical
            # titles are only *evidence* of identity, and that evidence is
            # overridden when the two came out of different official streams.
            same_stream = (
                document.source_metadata.disclosure_stream
                == representative.source_metadata.disclosure_stream
            )
            same_hash = document.text_hash == representative.text_hash and same_stream
            corroborated = (
                same_stream
                and _normalized_title(document.title) == _normalized_title(representative.title)
                and document.publisher.casefold() == representative.publisher.casefold()
                and document.published_at is not None
                and representative.published_at is not None
                and abs((document.published_at - representative.published_at).total_seconds()) <= 300
            )
            if exact or same_hash or corroborated:
                match = group
                break
        if match is None:
            groups.append([document])
        else:
            match.append(document)
    output = []
    for group in groups:
        first = min(group, key=lambda d: d.retrieved_at)
        provenance = {
            (p.provider, p.retrieved_at, p.provider_document_id): p for d in group for p in d.retrieval_provenance
        }
        for d in group:
            p = RetrievalProvenance(
                provider=d.provider, retrieved_at=d.retrieved_at, provider_document_id=d.document_id, query=d.query
            )
            provenance[(p.provider, p.retrieved_at, p.provider_document_id)] = p
        output.append(
            first.model_copy(
                update={
                    "retrieval_provenance": tuple(
                        sorted(provenance.values(), key=lambda p: (p.retrieved_at, p.provider))
                    )
                }
            )
        )
    return sorted(output, key=lambda d: (d.available_at, d.document_id))
