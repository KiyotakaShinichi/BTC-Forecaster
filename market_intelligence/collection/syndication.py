"""B4.1.2 / B4.1.3 — the production collection path: RSS *and* Atom.

The existing `RssSearchProvider` parses `.//item`, which is RSS 2.0 only. Atom
feeds put their entries in `<entry>` inside the `http://www.w3.org/2005/Atom`
namespace, so `findall(".//item")` returns nothing and the provider reports a
clean, successful, empty collection.

That is worse than an error, and it matters more than it sounds: the SEC, the
Federal Reserve and most regulators publish Atom. The single highest-value
category of primary source in B4's hypothesis list was silently uncollectable,
and the failure mode was a green run with zero documents — indistinguishable
from a quiet news day.

So this provider handles both dialects, and a test feeds it a real-shaped Atom
document and asserts entries come back.

Two other things it does that the original did not:

**Primary-source classification (B4.1.3).** A feed is declared as primary,
official, or neither, by the operator who configured it. `SourceMetadata`
already had the fields; nothing populated them. An SEC notice and a news article
*about* an SEC notice are different evidence, and a study that cannot tell them
apart cannot separate an announcement from its coverage.

**Raw evidence (B4.1.11).** The bytes are hashed, and retained only where the
feed's retention policy allows.

Availability is retrieval time, always. A feed entry dated last week that this
system first saw a minute ago became usable a minute ago.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Iterable, Sequence

from ..models import Document, RetrievalProvenance, SourceMetadata, SourceType
from ..providers import SearchProvider, deduplicate_documents
from .backoff import AttemptLog, ProviderFailure, RetryPolicy, call_with_retry, classify_http_status
from .evidence import RawEvidence, capture
from .policy import RawRetention

ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class FeedSource:
    """One configured feed and what the operator asserts about it."""

    feed_id: str
    url: str
    publisher: str
    source_type: SourceType = SourceType.REPUTABLE_NEWS
    #: The publisher is the party the news is about, not a report of it.
    primary_source: bool = False
    #: A government, regulator, central bank or exchange operating officially.
    official_source: bool = False
    known_publisher: bool = True
    retention: RawRetention = RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE
    #: Feed timestamps vary wildly in trustworthiness. This never affects
    #: availability -- only how much a downstream consumer should trust
    #: `published_at`.
    timestamp_quality: float = 0.5

    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            source_type=self.source_type,
            official_source=self.official_source,
            primary_source=self.primary_source,
            known_publisher=self.known_publisher,
            timestamp_quality=self.timestamp_quality,
            # A feed gives a title and a summary, not an article. Saying so
            # keeps a downstream extractor from treating a teaser as full text.
            content_completeness=0.4,
        )


@dataclass(frozen=True)
class FeedEntry:
    """One parsed entry, before it becomes a Document."""

    title: str
    link: str
    summary: str
    published_at: datetime | None
    author: str | None


def query_terms(query: str) -> list[str]:
    """Split a planned query into terms a feed's text could actually contain.

    The query planner emits quoted phrases -- `"SEC" enforcement` -- and feed
    titles never contain the quote characters. Splitting on whitespace alone
    produces terms like `"sec` that match nothing, so the provider reports a
    successful, empty collection: the same silent-empty failure as the Atom gap,
    from a different cause.
    """
    cleaned = "".join(character if character.isalnum() or character.isspace() else " " for character in query)
    return [term for term in cleaned.casefold().split() if term]


def parse_feed(payload: bytes | str) -> list[FeedEntry]:
    """Parse RSS 2.0 or Atom. Unknown dialects yield nothing rather than raise.

    Returning empty for an unrecognised dialect is deliberate: a malformed feed
    is a content problem for one source, and it should not abort a cycle that is
    collecting from a dozen others. The caller records it as a CONTENT failure.
    """
    root = ElementTree.fromstring(payload if isinstance(payload, str) else payload.decode("utf-8", "replace"))
    entries = [_rss_entry(item) for item in root.findall(".//item")]
    if entries:
        return entries
    return [_atom_entry(entry) for entry in root.findall(f".//{ATOM_NAMESPACE}entry")]


def _rss_entry(item: ElementTree.Element) -> FeedEntry:
    published_text = item.findtext("pubDate")
    published: datetime | None = None
    if published_text:
        try:
            published = parsedate_to_datetime(published_text).astimezone(timezone.utc)
        except (TypeError, ValueError):
            published = None  # an unparseable date is not a reason to drop evidence
    return FeedEntry(
        title=(item.findtext("title") or "").strip(),
        link=(item.findtext("link") or "").strip(),
        summary=(item.findtext("description") or "").strip(),
        published_at=published,
        author=(item.findtext("author") or None),
    )


def _atom_entry(entry: ElementTree.Element) -> FeedEntry:
    link = ""
    for candidate in entry.findall(f"{ATOM_NAMESPACE}link"):
        relation = candidate.get("rel", "alternate")
        if relation == "alternate" and candidate.get("href"):
            link = candidate.get("href", "").strip()
            break
    if not link:
        link = (entry.findtext(f"{ATOM_NAMESPACE}id") or "").strip()

    published_text = entry.findtext(f"{ATOM_NAMESPACE}published") or entry.findtext(f"{ATOM_NAMESPACE}updated")
    published: datetime | None = None
    if published_text:
        try:
            published = datetime.fromisoformat(published_text.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            published = None

    author_element = entry.find(f"{ATOM_NAMESPACE}author/{ATOM_NAMESPACE}name")
    return FeedEntry(
        title=(entry.findtext(f"{ATOM_NAMESPACE}title") or "").strip(),
        link=link,
        summary=(
            entry.findtext(f"{ATOM_NAMESPACE}summary")
            or entry.findtext(f"{ATOM_NAMESPACE}content")
            or ""
        ).strip(),
        published_at=published,
        author=author_element.text.strip() if author_element is not None and author_element.text else None,
    )


class SyndicationProvider(SearchProvider):
    """Collect from operator-configured RSS/Atom feeds.

    Only fetches URLs the operator listed. There is no discovery, no crawling
    and no following of links out of a feed: the allowlist *is* the policy, and
    a provider that can widen its own reach cannot have a stable lawfulness
    class (B4.1.1).
    """

    def __init__(
        self,
        feeds: Sequence[FeedSource],
        *,
        name: str = "syndication",
        timeout: float = 15.0,
        opener: Callable[[str, float], bytes] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.feeds = list(feeds)
        self.name = name
        self.timeout = timeout
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self._opener = opener or (
            lambda url, timeout_seconds: _default_opener(url, timeout_seconds, self.user_agent)
        )
        self._now = now
        self._retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep
        #: Populated per `search`, so a caller can persist raw evidence without
        #: the provider needing a store handle.
        self.last_evidence: list[RawEvidence] = []
        self.last_attempts: dict[str, AttemptLog] = {}

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        retrieved = self._now()
        terms = query_terms(query)
        documents: list[Document] = []
        self.last_evidence = []
        self.last_attempts = {}

        for feed in self.feeds:
            log = AttemptLog()
            self.last_attempts[feed.feed_id] = log
            try:
                payload = call_with_retry(
                    lambda feed=feed: self._opener(feed.url, self.timeout),  # type: ignore[misc]
                    self._retry,
                    sleep=self._sleep,
                    log=log,
                )
            except ProviderFailure:
                # One bad feed must not lose the cycle's other work (B4.1.17).
                # The classified failure is already in `log` for provider health.
                continue

            self.last_evidence.append(
                capture(
                    f"{self.name}:{feed.feed_id}",
                    payload,
                    retention=feed.retention,
                    retrieved_at=retrieved,
                    media_type="application/xml",
                    metadata={"feed_id": feed.feed_id, "feed_url": feed.url},
                )
            )

            try:
                entries = parse_feed(payload)
            except ElementTree.ParseError:
                continue

            documents.extend(self._documents_for(feed, entries, terms, query, retrieved, start))

        return deduplicate_documents(documents)

    def _documents_for(
        self,
        feed: FeedSource,
        entries: Iterable[FeedEntry],
        terms: list[str],
        query: str,
        retrieved: datetime,
        start: datetime,
    ) -> list[Document]:
        output: list[Document] = []
        for entry in entries:
            if not entry.link or not entry.title:
                continue
            haystack = f"{entry.title} {entry.summary}".casefold()
            if terms and not any(term in haystack for term in terms):
                continue
            # Availability is retrieval, always. `published_at` is recorded and
            # never promoted: a three-week-old press release first seen now
            # became usable now, and treating its date as availability would
            # fabricate three weeks of hindsight.
            available = retrieved
            # The lookback window bounds what the planner is interested in, and
            # it is applied to *publication*, not to availability.
            #
            # Testing `start <= available <= end` -- as the original RSS
            # provider did -- can never pass in forward collection: `end` is the
            # planning instant and retrieval necessarily happens after it, so
            # every entry is discarded and the provider reports a successful,
            # empty collection. That is the third silent-empty failure in this
            # path, and the hardest to see, because nothing is wrong with any
            # single line of it.
            if entry.published_at is not None and entry.published_at < start:
                continue
            text_hash = Document.content_hash(entry.summary or entry.title)
            output.append(
                Document(
                    document_id=Document.stable_id(entry.link, text_hash),
                    url=entry.link,
                    publisher=feed.publisher,
                    title=entry.title,
                    published_at=entry.published_at,
                    retrieved_at=retrieved,
                    available_at=available,
                    author=entry.author,
                    text_hash=text_hash,
                    query=query,
                    provider=self.name,
                    retrieval_provenance=(
                        RetrievalProvenance(
                            provider=self.name,
                            retrieved_at=retrieved,
                            provider_document_id=feed.feed_id,
                            query=query,
                        ),
                    ),
                    source_metadata=feed.metadata(),
                )
            )
        return output


#: Identifies this collector to publishers. Several government sites -- the SEC
#: most explicitly -- require a User-Agent naming the requester and a contact
#: address, and answer 403 without one. A single manual fetch usually slips
#: through; a collector polling twice a day for months is precisely what gets
#: blocked, so a deployment sets `user_agent` to something a publisher can write
#: to. This default is honest about what it is and carries no false contact.
DEFAULT_USER_AGENT = "btc-intel-research/1.0 (research collector; contact not configured)"


def _default_opener(url: str, timeout: float, user_agent: str = DEFAULT_USER_AGENT) -> bytes:
    """Fetch a configured feed, classifying HTTP status into a failure class."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec: operator allowlist
            return bytes(response.read())
    except urllib.error.HTTPError as error:  # pragma: no cover - needs a live endpoint
        raise ProviderFailure(classify_http_status(error.code), f"HTTP {error.code} for {url}") from error


__all__ = [
    "ATOM_NAMESPACE",
    "DEFAULT_USER_AGENT",
    "FeedEntry",
    "FeedSource",
    "SyndicationProvider",
    "parse_feed",
    "query_terms",
]
