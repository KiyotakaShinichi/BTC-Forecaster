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

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Iterable, Sequence

from ..models import DisclosureStream, Document, RetrievalProvenance, SourceMetadata, SourceType
from ..providers import SearchProvider, deduplicate_documents
from .backoff import AttemptLog, FailureClass, ProviderFailure, RetryPolicy, call_with_retry, classify_http_status
from .evidence import RawEvidence, capture
from .matching import term_pattern
from .policy import RawRetention
from .telemetry import FeedDiagnosis, diagnose

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
    #: Which official stream this feed carries. The operator's assertion, not
    #: something derived from the URL, and recorded on every document the feed
    #: produces so a study can separate an administrative proceeding from a
    #: press release about one.
    disclosure_stream: DisclosureStream = DisclosureStream.UNCLASSIFIED
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
            disclosure_stream=self.disclosure_stream,
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


@dataclass(frozen=True)
class StreamMatchPolicy:
    """Which entry fields topical candidate matching is allowed to read.

    This was implicit -- every stream matched against title plus description --
    and implicit was the problem. Auditing the SEC's administrative-proceedings
    feed for a richer field to match on turned up the opposite result, and it is
    worth stating in the type system rather than rediscovering:

        title        Comscore, Inc., Serge Matta
        description  Comscore, Inc.; Serge Matta
        link         .../admin/2026/34-106258.pdf
        dc:creator   34-106258

    The description is the respondent name again -- byte-identical to the title
    in 23 of 25 entries, differing only in punctuation in the other two. There
    is no field in this feed that says what a proceeding is *about*. Subject
    matter lives in the linked PDF, which is a different retention and
    lawfulness question and is not fetched.

    So `subject_bearing` is the honest part of this record. Where it is False, a
    zero match count is a property of the source, not evidence of a quiet week,
    and reading it as the latter is how a collector gets "fixed" by loosening a
    filter until unrelated documents come through.
    """

    #: Entry attributes consulted, in order. Never the link, never metadata the
    #: publisher did not intend as content.
    match_fields: tuple[str, ...] = ("title", "summary")
    #: Whether those fields can express what the item is about at all.
    subject_bearing: bool = True
    note: str = ""

    def haystack(self, entry: "FeedEntry") -> str:
        return " ".join(str(getattr(entry, field, "") or "") for field in self.match_fields).casefold()


#: Matching is title+description everywhere. The SEC's administrative feed is
#: recorded as non-subject-bearing, which changes no filter and prevents a
#: misreading: nothing here widens what any stream admits.
STREAM_MATCH_POLICIES: dict[DisclosureStream, StreamMatchPolicy] = {
    DisclosureStream.ADMINISTRATIVE_PROCEEDINGS: StreamMatchPolicy(
        match_fields=("title", "summary"),
        subject_bearing=False,
        note=(
            "Title and description are both the respondent's name; the SEC puts "
            "the subject matter only in the linked order PDF. Topical terms will "
            "match a proceeding only when the respondent is itself a watched "
            "entity, so zero is the expected result for a topical query and is "
            "not a collection fault."
        ),
    ),
}

DEFAULT_MATCH_POLICY = StreamMatchPolicy()


def match_policy(stream: DisclosureStream) -> StreamMatchPolicy:
    return STREAM_MATCH_POLICIES.get(stream, DEFAULT_MATCH_POLICY)


@dataclass(frozen=True)
class FeedYield:
    """What one feed produced in one search, before anything downstream.

    A matched document is a *candidate*, not an event: it still has to survive
    normalisation, extraction, schema validation, relevance scoring, dedup and
    clustering. Keeping the counts separate is what makes it possible to say
    which of those stages a stream is actually losing entries at.
    """

    entries: int = 0
    #: Passed the topical candidate filter.
    matched: int = 0
    #: ...and then also fell inside the planner's publication lookback. The two
    #: are separate because they fail for opposite reasons: `matched` low means
    #: the query does not describe this source, while `matched` high and
    #: `admitted` low means the source is simply older than the window, and
    #: widening the query would fix nothing.
    admitted: int = 0
    subject_bearing: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "entries": self.entries,
            "matched": self.matched,
            "admitted": self.admitted,
            "subject_bearing": self.subject_bearing,
        }


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


#: An ampersand that is not already the start of a character or entity
#: reference. XML requires these escaped; publishers routinely do not.
_BARE_AMPERSAND = re.compile(r"&(?!(?:[A-Za-z][A-Za-z0-9]*|#[0-9]+|#[xX][0-9A-Fa-f]+);)")


def repair_xml(text: str) -> tuple[str, tuple[str, ...]]:
    """Fix escaping errors a publisher made, and say which ones were fixed.

    The SEC's administrative-proceedings feed emits company names verbatim, so a
    respondent called "TIAA-CREF Individual & Institutional Services" produces a
    bare ampersand and the whole document stops being XML. The same feed escapes
    the same name correctly two lines later, so this is inconsistency at the
    source rather than a dialect this parser does not know.

    Refusing the feed would be defensible if it were rare. It is not: an
    ampersand in a company name is ordinary, so this would drop an unpredictable
    subset of enforcement actions -- the ones against firms whose names contain
    "&" -- and nothing downstream would look wrong. That is a selection effect,
    not a gap, and it is far more dangerous than a repaired character.

    Deliberately narrow. It escapes ampersands that cannot begin a reference and
    changes nothing else; it does not close tags, guess encodings, or strip
    content. Every repair is returned so the caller can record that the feed
    needed one -- a publisher whose feed is degrading should become visible, not
    be quietly compensated for forever.
    """
    repairs: list[str] = []
    fixed, count = _BARE_AMPERSAND.subn("&amp;", text)
    if count:
        repairs.append(f"escaped {count} bare ampersand{'s' if count > 1 else ''}")
    return fixed, tuple(repairs)


def parse_feed(payload: bytes | str) -> list[FeedEntry]:
    """Parse RSS 2.0 or Atom. Unknown dialects yield nothing rather than raise.

    Returning empty for an unrecognised dialect is deliberate: a malformed feed
    is a content problem for one source, and it should not abort a cycle that is
    collecting from a dozen others. The caller records it as a CONTENT failure.
    """
    return parse_feed_with_repairs(payload)[0]


def parse_feed_with_repairs(payload: bytes | str) -> tuple[list[FeedEntry], tuple[str, ...]]:
    """`parse_feed`, plus what had to be repaired to get there."""
    text = payload if isinstance(payload, str) else payload.decode("utf-8", "replace")
    repairs: tuple[str, ...] = ()
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        # Only now, and only after a strict parse has actually failed: a feed
        # that is already valid is never rewritten.
        repaired, repairs = repair_xml(text)
        if not repairs:
            raise
        root = ElementTree.fromstring(repaired)
    return _entries(root), repairs


def _entries(root: ElementTree.Element) -> list[FeedEntry]:
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
        #: Which feeds needed their XML repaired, and how, on the last search.
        self.last_repairs: dict[str, tuple[str, ...]] = {}
        #: Entries seen versus entries admitted, per feed, on the last search.
        self.last_yield: dict[str, FeedYield] = {}

    def diagnostics(self) -> list[FeedDiagnosis]:
        """What each feed did on the last search, classified.

        Operations reads this rather than the bare counts: "0 documents" is the
        same number for a dead endpoint, an empty feed, a query that no longer
        describes the source, and a week with no news, and those need four
        different responses.
        """
        # Iterating the configured feeds rather than the counters: a feed that
        # failed to fetch never reaches the counting loop, and that is exactly
        # the feed an operator most needs a diagnosis for.
        return [
            diagnose(
                feed.feed_id,
                self.last_yield.get(feed.feed_id, FeedYield()),
                failed=bool(self.last_attempts.get(feed.feed_id, AttemptLog()).failures),
            )
            for feed in sorted(self.feeds, key=lambda item: item.feed_id)
        ]

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        retrieved = self._now()
        terms = query_terms(query)
        documents: list[Document] = []
        self.last_evidence = []
        self.last_attempts = {}
        self.last_repairs = {}
        self.last_yield = {}
        read: list[str] = []

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
                entries, repairs = parse_feed_with_repairs(payload)
            except ElementTree.ParseError:
                # Previously a bare `continue`. A feed that fails to parse would
                # then be fetched every cycle forever, contribute nothing, and
                # report success -- the silent-empty failure again, and the
                # docstring's promise that the caller records it was simply not
                # kept.
                log.record(FailureClass.SCHEMA)
                continue
            if repairs:
                # Recorded, not hidden. A publisher whose feed is degrading
                # should be visible in provider health rather than compensated
                # for indefinitely.
                self.last_repairs[feed.feed_id] = repairs

            read.append(feed.feed_id)
            documents.extend(self._documents_for(feed, entries, terms, query, retrieved, start))

        if self.feeds and not read:
            raise nothing_read(self.last_attempts)
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
        policy = match_policy(feed.disclosure_stream)
        pattern = term_pattern(terms)
        seen = matched = admitted = 0
        for entry in entries:
            if not entry.link or not entry.title:
                continue
            seen += 1
            if pattern is not None and not pattern.search(policy.haystack(entry)):
                continue
            matched += 1
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
            admitted += 1
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
        self.last_yield[feed.feed_id] = FeedYield(
            entries=seen,
            matched=matched,
            admitted=admitted,
            subject_bearing=policy.subject_bearing,
        )
        return output


#: Identifies this collector to publishers. Several government sites -- the SEC
#: most explicitly -- require a User-Agent naming the requester and a contact
#: address, and answer 403 without one. A single manual fetch usually slips
#: through; a collector polling twice a day for months is precisely what gets
#: blocked, so a deployment sets `user_agent` to something a publisher can write
#: to. This default is honest about what it is and carries no false contact.
DEFAULT_USER_AGENT = "btc-intel-research/1.0 (research collector; contact not configured)"


def nothing_read(attempts: dict[str, AttemptLog]) -> ProviderFailure:
    """B5.1. A search that could read no feed at all is a failed attempt.

    Until B5.1 it returned an empty list, and the retrieval layer recorded a
    *successful* attempt with nothing in it. A cycle in which every feed was down
    looked like a quiet news day, and collection coverage -- days with a
    successful attempt -- would have counted it as collected. One unreadable feed
    still costs only itself (B4.1.17); the search fails only when none could be
    read, fetched and parsed.

    The class is the feeds' own when they agree, and TRANSIENT when they do not
    (classify_exception's rule: wrongly transient costs a retry, wrongly permanent
    hides a provider). Each feed was already retried under its own policy, so
    the failure says so and the retrieval layer does not retry the search again.
    """
    last = {feed_id: log.failures[-1] for feed_id, log in sorted(attempts.items()) if log.failures}
    classes = set(last.values())
    failure = classes.pop() if len(classes) == 1 else FailureClass.TRANSIENT
    detail = ", ".join(f"{feed_id}={cls.value}" for feed_id, cls in last.items())
    return ProviderFailure(failure, f"no feed could be read on this search ({detail})", retries_exhausted=True)


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
    "DEFAULT_MATCH_POLICY",
    "DEFAULT_USER_AGENT",
    "STREAM_MATCH_POLICIES",
    "FeedYield",
    "StreamMatchPolicy",
    "diagnose",
    "match_policy",
    "term_pattern",
    "parse_feed_with_repairs",
    "repair_xml",
    "FeedEntry",
    "FeedSource",
    "SyndicationProvider",
    "parse_feed",
    "query_terms",
]
