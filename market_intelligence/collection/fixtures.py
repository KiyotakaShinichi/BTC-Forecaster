"""B4.1.37 — deterministic fixture providers for every collection condition.

CI never touches a network, so the only way to know the pipeline handles a rate
limit, a schema change or a cross-provider duplicate is to be able to produce
one on demand. These are those producers.

The list is chosen from the failures that actually corrupt a corpus rather than
the ones that are easy to simulate: a source arriving late, an article
rediscovered a week later, a document stamped in the future, the same story from
two providers, a primary announcement beside its secondary coverage. Each is a
case where doing the obvious thing silently produces wrong evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from ..models import Document, RetrievalProvenance, SourceMetadata, SourceType
from ..providers import SearchProvider
from .backoff import FailureClass, ProviderFailure

FIXTURE_BASE = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def fixture_document(
    *,
    url: str,
    title: str,
    retrieved_at: datetime,
    publisher: str = "example.com",
    provider: str = "fixture",
    published_at: datetime | None = None,
    query: str = "bitcoin",
    primary_source: bool = False,
    official_source: bool = False,
    body: str | None = None,
) -> Document:
    """Build a Document with availability pinned to retrieval, as production does."""
    text = body if body is not None else title
    text_hash = Document.content_hash(text)
    return Document(
        document_id=Document.stable_id(url, text_hash),
        url=url,
        publisher=publisher,
        title=title,
        published_at=published_at,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        text_hash=text_hash,
        query=query,
        provider=provider,
        retrieval_provenance=(
            RetrievalProvenance(provider=provider, retrieved_at=retrieved_at, query=query),
        ),
        source_metadata=SourceMetadata(
            source_type=SourceType.PRIMARY_OFFICIAL if primary_source else SourceType.REPUTABLE_NEWS,
            official_source=official_source,
            primary_source=primary_source,
            known_publisher=True,
            timestamp_quality=0.9 if primary_source else 0.5,
            content_completeness=0.4,
        ),
    )


class StaticFixtureProvider(SearchProvider):
    """Returns a fixed set of documents, re-stamped at each retrieval.

    Re-stamping is the point: a real feed re-serves yesterday's items with
    today's retrieval time, and a fixture that returned frozen timestamps would
    never exercise the rediscovery path that first-write-wins exists to handle.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        name: str = "fixture",
        now: Callable[[], datetime] = lambda: FIXTURE_BASE,
        restamp: bool = True,
    ) -> None:
        self._documents = list(documents)
        self.name = name
        self._now = now
        self._restamp = restamp

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        moment = self._now()
        output: list[Document] = []
        for document in self._documents:
            if document.query != query:
                continue
            candidate = (
                document.model_copy(update={"retrieved_at": moment, "available_at": moment})
                if self._restamp
                else document
            )
            if start <= candidate.available_at <= end:
                output.append(candidate)
        return output


class EmptyProvider(SearchProvider):
    """Succeeds and returns nothing. Distinct from a failure, deliberately."""

    def __init__(self, name: str = "empty") -> None:
        self.name = name

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        return []


class RateLimitedProvider(SearchProvider):
    """Fails with a classified rate limit until `succeed_after` calls."""

    def __init__(self, name: str = "rate-limited", succeed_after: int = 99) -> None:
        self.name = name
        self.calls = 0
        self.succeed_after = succeed_after

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        self.calls += 1
        if self.calls <= self.succeed_after:
            raise ProviderFailure(FailureClass.RATE_LIMIT, "429 Too Many Requests")
        return []


class AuthFailingProvider(SearchProvider):
    """Fails with a credential error. Must never be retried."""

    def __init__(self, name: str = "auth-failing") -> None:
        self.name = name
        self.calls = 0

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        self.calls += 1
        raise ProviderFailure(FailureClass.AUTH, "401 Unauthorized")


class SchemaInvalidProvider(SearchProvider):
    """Returns a response the contract cannot parse."""

    def __init__(self, name: str = "schema-invalid") -> None:
        self.name = name

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        raise KeyError("published_at")


class OutageProvider(SearchProvider):
    """Down entirely. The failure a coverage report must not read as zero events."""

    def __init__(self, name: str = "outage") -> None:
        self.name = name

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        raise ProviderFailure(FailureClass.TRANSIENT, "connection refused")


def cross_provider_duplicate(retrieved_at: datetime) -> list[Document]:
    """One story, two providers, identical URL and content.

    Dedup must collapse these to one document. It must *not* collapse two
    genuinely distinct primary sources that happen to share a headline, which
    `distinct_primary_sources` covers.
    """
    return [
        fixture_document(
            url="https://newswire.example.com/sec-decision",
            title="SEC issues bitcoin ETF decision",
            retrieved_at=retrieved_at,
            provider="provider-a",
            body="The SEC issued a decision today.",
        ),
        fixture_document(
            url="https://newswire.example.com/sec-decision",
            title="SEC issues bitcoin ETF decision",
            retrieved_at=retrieved_at + timedelta(minutes=3),
            provider="provider-b",
            body="The SEC issued a decision today.",
        ),
    ]


def distinct_primary_sources(retrieved_at: datetime) -> list[Document]:
    """Two different regulators announcing on the same day. Never merge these."""
    return [
        fixture_document(
            url="https://sec.example.gov/2026/decision",
            title="SEC announces bitcoin regulation decision",
            retrieved_at=retrieved_at,
            publisher="sec.example.gov",
            provider="syndication",
            primary_source=True,
            official_source=True,
            body="SEC text",
        ),
        fixture_document(
            url="https://cftc.example.gov/2026/decision",
            title="CFTC announces bitcoin regulation decision",
            retrieved_at=retrieved_at,
            publisher="cftc.example.gov",
            provider="syndication",
            primary_source=True,
            official_source=True,
            body="CFTC text",
        ),
    ]


def primary_and_secondary(retrieved_at: datetime) -> list[Document]:
    """B4.1.7. An announcement and a report about it, kept distinguishable."""
    return [
        fixture_document(
            url="https://sec.example.gov/2026/enforcement",
            title="SEC charges firm over bitcoin custody failures",
            retrieved_at=retrieved_at,
            publisher="sec.example.gov",
            provider="syndication",
            primary_source=True,
            official_source=True,
            body="Official announcement text.",
        ),
        fixture_document(
            url="https://news.example.com/sec-charges-firm",
            title="SEC regulation crackdown on bitcoin custody, analysts say",
            retrieved_at=retrieved_at + timedelta(minutes=40),
            publisher="news.example.com",
            provider="news-api",
            body="Media interpretation of the announcement.",
        ),
    ]


def late_source(retrieved_at: datetime) -> Document:
    """Published weeks ago, first seen now. Availability is now."""
    return fixture_document(
        url="https://archive.example.com/old-analysis",
        title="Retrospective bitcoin analysis",
        retrieved_at=retrieved_at,
        published_at=retrieved_at - timedelta(days=21),
        body="An old piece surfaced by a feed today.",
    )


def future_timestamped_document(retrieved_at: datetime) -> Document:
    """A publisher-stamped future date. Availability must still be retrieval."""
    return fixture_document(
        url="https://example.com/embargoed",
        title="Embargoed bitcoin release",
        retrieved_at=retrieved_at,
        published_at=retrieved_at + timedelta(days=3),
        body="A feed carrying a future publication stamp.",
    )


__all__ = [
    "FIXTURE_BASE",
    "AuthFailingProvider",
    "EmptyProvider",
    "OutageProvider",
    "RateLimitedProvider",
    "SchemaInvalidProvider",
    "StaticFixtureProvider",
    "cross_provider_duplicate",
    "distinct_primary_sources",
    "fixture_document",
    "future_timestamped_document",
    "late_source",
    "primary_and_secondary",
]
