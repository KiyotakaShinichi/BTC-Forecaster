"""Disclosure streams — three SEC feeds that must not become one.

`publisher` says who published. `source_type` says how much to trust them.
Neither says what *kind* of disclosure a document is, and for a regulator that
is the distinction that matters: an administrative proceeding decided in-house,
a civil suit filed in federal court, and a press release are different
instruments with different legal weight.

Every SEC feed carries the identical publisher string, so the machinery that
merges "the same story reached twice" was one step away from merging them.
These tests hold the three apart.

No network. The malformed fixture is a verbatim excerpt of what the SEC actually
served on 2026-09-03.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

import pytest

from market_intelligence.collection.backoff import FailureClass, ProviderFailure, RetryPolicy
from market_intelligence.collection.feeds import OFFICIAL_FEEDS, RETIRED_FEED_IDS, feeds_by_id
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.collection.syndication import (
    SyndicationProvider,
    parse_feed,
    parse_feed_with_repairs,
    repair_xml,
)
from market_intelligence.models import DisclosureStream, SourceType
from market_intelligence.ops.profile import CollectionProfile
from market_intelligence.retrieval import deduplicate_across_providers

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
DEPLOYED_PROFILE = Path(__file__).resolve().parents[1] / "deploy" / "collection-profile.json"

#: Verbatim from https://www.sec.gov/rss/litigation/admin.xml on 2026-09-03.
#: The bare `&` in the title is the SEC's; the `&amp;amp;` in the description
#: two lines later is also theirs, which is what makes this inconsistency at the
#: source rather than a dialect we do not know.
SEC_ADMIN_FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
  <title>Administrative Proceedings</title>
  <item>
    <title>X METAVERSE INC.</title>
    <link>https://www.sec.gov/files/litigation/admin/2026/34-106250.pdf</link>
    <description>X METAVERSE INC.</description>
    <pubDate>Tue, 01 Sep 2026 14:42:32 -0400</pubDate>
    <dc:creator>34-106250</dc:creator>
  </item>
  <item>
    <title>TIAA-CREF Individual &amp; Institutional Services, LLC</title>
    <link>https://www.sec.gov/files/litigation/admin/2026/34-106249.pdf</link>
    <description>TIAA-CREF Individual &amp;amp; Institutional Services, LLC</description>
    <pubDate>Tue, 01 Sep 2026 13:10:54 -0400</pubDate>
    <dc:creator>34-106249</dc:creator>
  </item>
</channel></rss>""".replace(
    b"TIAA-CREF Individual &amp; Institutional", b"TIAA-CREF Individual & Institutional"
)


# ------------------------------------------------------------- the catalogue


class TestTheCatalogueKeepsThreeSecStreamsApart:
    def test_the_new_feed_has_its_own_identity(self) -> None:
        (feed,) = feeds_by_id("sec-admin-proceedings")
        assert feed.url == "https://www.sec.gov/rss/litigation/admin.xml"
        assert feed.disclosure_stream is DisclosureStream.ADMINISTRATIVE_PROCEEDINGS

    def test_it_is_classified_as_asked(self) -> None:
        """PRIMARY_SOURCE, OFFICIAL, SEC, ADMINISTRATIVE_PROCEEDINGS."""
        (feed,) = feeds_by_id("sec-admin-proceedings")
        assert feed.primary_source is True
        assert feed.official_source is True
        assert feed.source_type is SourceType.PRIMARY_OFFICIAL
        assert feed.publisher == "U.S. Securities and Exchange Commission"
        assert feed.disclosure_stream is DisclosureStream.ADMINISTRATIVE_PROCEEDINGS

    def test_it_is_not_classified_as_civil_litigation(self) -> None:
        """The whole point of adding it separately."""
        (feed,) = feeds_by_id("sec-admin-proceedings")
        assert feed.disclosure_stream is not DisclosureStream.CIVIL_LITIGATION

    def test_the_retired_civil_litigation_feed_survives_and_keeps_its_meaning(self) -> None:
        """Preserved so a manifest written before it died stays resolvable, and
        still labelled civil litigation, because that is what it carried."""
        (feed,) = feeds_by_id("sec-litigation")
        assert feed.feed_id in RETIRED_FEED_IDS
        assert feed.disclosure_stream is DisclosureStream.CIVIL_LITIGATION

    def test_every_retired_feed_is_still_resolvable(self) -> None:
        assert len(feeds_by_id(*sorted(RETIRED_FEED_IDS))) == len(RETIRED_FEED_IDS)

    def test_the_three_sec_streams_are_distinguishable(self) -> None:
        """A future study asking 'what did the SEC do' must be able to ask which
        kind, without knowing which URLs were configured at the time."""
        sec = [f for f in OFFICIAL_FEEDS if f.publisher.startswith("U.S. Securities")]
        streams = {feed.feed_id: feed.disclosure_stream for feed in sec}
        assert streams == {
            "sec-press": DisclosureStream.REGULATORY_ANNOUNCEMENT,
            "sec-admin-proceedings": DisclosureStream.ADMINISTRATIVE_PROCEEDINGS,
            "sec-litigation": DisclosureStream.CIVIL_LITIGATION,
        }
        assert len(set(streams.values())) == 3, "two SEC streams share a classification"

    def test_no_feed_is_left_unclassified(self) -> None:
        unclassified = [
            feed.feed_id
            for feed in OFFICIAL_FEEDS
            if feed.disclosure_stream is DisclosureStream.UNCLASSIFIED
        ]
        assert not unclassified, unclassified


# ---------------------------------------------------------- the deployed profile


class TestTheProfileEnablesIt:
    ENVIRONMENT = {"BTC_INTEL_CONTACT": "tests@example.org"}

    def deployed(self) -> CollectionProfile:
        return CollectionProfile.load(DEPLOYED_PROFILE, environment=self.ENVIRONMENT)

    def test_six_feeds_are_active(self) -> None:
        assert len(self.deployed().feeds) == 6

    def test_the_new_feed_is_among_them(self) -> None:
        assert "sec-admin-proceedings" in {feed.feed_id for feed in self.deployed().feeds}

    def test_no_retired_feed_came_back(self) -> None:
        enabled = {feed.feed_id for feed in self.deployed().feeds}
        assert not (enabled & RETIRED_FEED_IDS)

    def test_the_active_set_spans_four_distinct_streams(self) -> None:
        streams = {feed.disclosure_stream for feed in self.deployed().feeds}
        assert DisclosureStream.ADMINISTRATIVE_PROCEEDINGS in streams
        assert len(streams) >= 3


# ------------------------------------------------------------------ parsing


class TestTheFeedParses:
    def test_the_sec_feed_as_served_is_not_valid_xml(self) -> None:
        """Establishes the premise. If the SEC fixes their escaping this fails,
        and the repair below can be reconsidered rather than kept forever."""
        with pytest.raises(ElementTree.ParseError):
            ElementTree.fromstring(SEC_ADMIN_FEED.decode())

    def test_it_parses_anyway_and_says_what_it_repaired(self) -> None:
        entries, repairs = parse_feed_with_repairs(SEC_ADMIN_FEED)
        assert len(entries) == 2
        assert repairs == ("escaped 1 bare ampersand",)

    def test_the_repair_recovers_the_entry_rather_than_dropping_it(self) -> None:
        """Without it the whole feed is lost, not just the offending item -- and
        the items lost would be exactly the enforcement actions against firms
        with an ampersand in their name. That is a selection effect, not a gap."""
        entries = parse_feed(SEC_ADMIN_FEED)
        titles = [entry.title for entry in entries]
        assert "TIAA-CREF Individual & Institutional Services, LLC" in titles
        assert "X METAVERSE INC." in titles

    def test_a_valid_feed_is_never_rewritten(self) -> None:
        """The repair runs only after a strict parse has actually failed."""
        valid = SEC_ADMIN_FEED.replace(
            b"TIAA-CREF Individual & Institutional", b"TIAA-CREF Individual &amp; Institutional"
        )
        entries, repairs = parse_feed_with_repairs(valid)
        assert len(entries) == 2
        assert repairs == ()

    def test_the_repair_is_narrow(self) -> None:
        """It escapes ampersands that cannot begin a reference, and nothing
        else. A repair that rewrote content would be inventing evidence."""
        text = "a & b &amp; c &#38; d &#x26; e &lt;"
        fixed, repairs = repair_xml(text)
        assert fixed == "a &amp; b &amp; c &#38; d &#x26; e &lt;"
        assert repairs == ("escaped 1 bare ampersand",)

    def test_the_repair_leaves_unfixable_xml_unfixed(self) -> None:
        """It does not close tags or guess structure."""
        with pytest.raises(ElementTree.ParseError):
            parse_feed_with_repairs(b"<rss><channel><item><title>unclosed</channel></rss>")

    def test_published_at_is_read_but_never_promoted_to_availability(self) -> None:
        """The PIT contract: published_at is the feed's claim, available_at is
        our retrieval. No backdating."""
        provider = SyndicationProvider(
            feeds_by_id("sec-admin-proceedings"),
            now=lambda: NOW,
            retry=RetryPolicy(max_attempts=1),
            opener=lambda url, timeout: SEC_ADMIN_FEED,
        )
        documents = provider.search("metaverse services", NOW - timedelta(days=30), NOW)
        assert documents
        for document in documents:
            assert document.available_at == NOW
            assert document.retrieved_at == NOW
            assert document.published_at is not None
            assert document.published_at < document.available_at


# ------------------------------------------------------ metadata on documents


class TestDocumentsCarryTheStream:
    def _documents(self):
        provider = SyndicationProvider(
            feeds_by_id("sec-admin-proceedings"),
            now=lambda: NOW,
            retry=RetryPolicy(max_attempts=1),
            opener=lambda url, timeout: SEC_ADMIN_FEED,
        )
        return provider.search("metaverse services", NOW - timedelta(days=30), NOW)

    def test_every_document_records_the_stream(self) -> None:
        """So a study years later does not have to know which feed URL was
        configured at the time."""
        documents = self._documents()
        assert documents
        for document in documents:
            meta = document.source_metadata
            assert meta.disclosure_stream is DisclosureStream.ADMINISTRATIVE_PROCEEDINGS
            assert meta.primary_source and meta.official_source
            assert document.publisher == "U.S. Securities and Exchange Commission"

    def test_the_feed_is_named_in_provenance(self) -> None:
        for document in self._documents():
            assert document.retrieval_provenance[0].provider_document_id == "sec-admin-proceedings"

    def test_a_document_persisted_before_the_field_existed_stays_unclassified(self) -> None:
        """Relabelling old documents would assert a classification nobody made
        at the time. They read back exactly as written."""
        legacy = fixture_document(
            url="https://sec.example.gov/old", title="older release", retrieved_at=NOW, body="b"
        )
        assert legacy.source_metadata.disclosure_stream is DisclosureStream.UNCLASSIFIED


# ------------------------------------------------------------ deduplication


class TestDedupDoesNotCollapseSecStreams:
    """Every SEC feed carries the identical publisher string, and the
    corroboration clause merges on title + publisher + a five-minute window. An
    administrative proceeding and the press release announcing it would have
    become one document, losing one classification."""

    def _sec(self, url: str, title: str, stream: DisclosureStream, *, body: str):
        document = fixture_document(
            url=url,
            title=title,
            retrieved_at=NOW,
            published_at=NOW - timedelta(hours=1),
            publisher="U.S. Securities and Exchange Commission",
            primary_source=True,
            official_source=True,
            body=body,
        )
        return document.model_copy(
            update={
                "source_metadata": document.source_metadata.model_copy(
                    update={"disclosure_stream": stream}
                )
            }
        )

    def test_same_title_and_publisher_across_streams_stays_two_documents(self) -> None:
        proceeding = self._sec(
            "https://www.sec.gov/files/litigation/admin/2026/34-1.pdf",
            "In the Matter of Example Capital LLC",
            DisclosureStream.ADMINISTRATIVE_PROCEEDINGS,
            body="order instituting proceedings",
        )
        announcement = self._sec(
            "https://www.sec.gov/news/press-release/2026-1",
            "In the Matter of Example Capital LLC",
            DisclosureStream.REGULATORY_ANNOUNCEMENT,
            body="press release about the order",
        )
        merged = deduplicate_across_providers([proceeding, announcement])
        assert len(merged) == 2
        assert {d.source_metadata.disclosure_stream for d in merged} == {
            DisclosureStream.ADMINISTRATIVE_PROCEEDINGS,
            DisclosureStream.REGULATORY_ANNOUNCEMENT,
        }

    def test_identical_summary_text_across_streams_stays_two_documents(self) -> None:
        """Feed summaries are often just the respondent's name, so an identical
        text hash across two SEC streams is ordinary, not evidence of identity."""
        shared = "Example Capital LLC"
        proceeding = self._sec(
            "https://www.sec.gov/files/litigation/admin/2026/34-2.pdf",
            "Example Capital LLC",
            DisclosureStream.ADMINISTRATIVE_PROCEEDINGS,
            body=shared,
        )
        announcement = self._sec(
            "https://www.sec.gov/news/press-release/2026-2",
            "Example Capital LLC",
            DisclosureStream.REGULATORY_ANNOUNCEMENT,
            body=shared,
        )
        assert proceeding.text_hash == announcement.text_hash
        assert len(deduplicate_across_providers([proceeding, announcement])) == 2

    def test_the_same_document_within_one_stream_still_merges(self) -> None:
        """The guard must not disable deduplication, only stop it crossing
        streams."""
        first = self._sec(
            "https://www.sec.gov/files/litigation/admin/2026/34-3.pdf",
            "Example Capital LLC",
            DisclosureStream.ADMINISTRATIVE_PROCEEDINGS,
            body="same body",
        )
        second = self._sec(
            "https://www.sec.gov/files/litigation/admin/2026/34-4.pdf",
            "Example Capital LLC",
            DisclosureStream.ADMINISTRATIVE_PROCEEDINGS,
            body="same body",
        )
        assert len(deduplicate_across_providers([first, second])) == 1

    def test_the_same_url_always_merges_whatever_the_stream(self) -> None:
        """An identical URL is identity, not evidence of it."""
        url = "https://www.sec.gov/files/litigation/admin/2026/34-5.pdf"
        one = self._sec(url, "Example", DisclosureStream.ADMINISTRATIVE_PROCEEDINGS, body="x")
        two = self._sec(url, "Example", DisclosureStream.REGULATORY_ANNOUNCEMENT, body="x")
        assert len(deduplicate_across_providers([one, two])) == 1

    def test_streams_are_not_merged_to_raise_an_event_count(self) -> None:
        """Guarding the reason this matters: three SEC streams collected
        separately are three sources, and collapsing them into one publisher
        would inflate nothing while destroying the distinction. Kept explicit so
        a later change that merges them for convenience fails here."""
        streams = {feed.disclosure_stream for feed in OFFICIAL_FEEDS}
        assert DisclosureStream.ADMINISTRATIVE_PROCEEDINGS in streams
        assert DisclosureStream.CIVIL_LITIGATION in streams
        assert DisclosureStream.REGULATORY_ANNOUNCEMENT in streams


# ------------------------------------------------------------ silent failures


class TestAParseFailureIsNoLongerSilent:
    def test_an_unparseable_feed_records_a_schema_failure(self) -> None:
        """It used to `continue`. The feed would then be fetched every cycle
        forever, contribute nothing, and report success -- the silent-empty
        failure that has bitten this collector four times."""
        provider = SyndicationProvider(
            feeds_by_id("sec-admin-proceedings"),
            now=lambda: NOW,
            retry=RetryPolicy(max_attempts=1),
            opener=lambda url, timeout: b"<rss><channel><item><title>x</channel></rss>",
        )
        # B5.1: nor does it return an empty list that the retrieval layer would
        # record as a successful attempt. A search that could read no feed fails.
        with pytest.raises(ProviderFailure) as raised:
            provider.search("SEC", NOW - timedelta(days=1), NOW)
        assert raised.value.failure_class is FailureClass.SCHEMA
        assert provider.last_attempts["sec-admin-proceedings"].failures == (FailureClass.SCHEMA,)

    def test_a_repaired_feed_is_reported_as_repaired(self) -> None:
        """Compensating for a publisher forever without saying so hides a feed
        that is degrading."""
        provider = SyndicationProvider(
            feeds_by_id("sec-admin-proceedings"),
            now=lambda: NOW,
            retry=RetryPolicy(max_attempts=1),
            opener=lambda url, timeout: SEC_ADMIN_FEED,
        )
        provider.search("metaverse services", NOW - timedelta(days=30), NOW)
        assert provider.last_repairs["sec-admin-proceedings"] == ("escaped 1 bare ampersand",)

    def test_a_clean_feed_reports_no_repairs(self) -> None:
        valid = SEC_ADMIN_FEED.replace(
            b"TIAA-CREF Individual & Institutional", b"TIAA-CREF Individual &amp; Institutional"
        )
        provider = SyndicationProvider(
            feeds_by_id("sec-admin-proceedings"),
            now=lambda: NOW,
            retry=RetryPolicy(max_attempts=1),
            opener=lambda url, timeout: valid,
        )
        provider.search("metaverse services", NOW - timedelta(days=30), NOW)
        assert provider.last_repairs == {}


# --------------------------------------------------------------- no secrets


class TestNoContactLeak:
    def test_the_new_feed_carries_no_address_in_the_catalogue(self) -> None:
        (feed,) = feeds_by_id("sec-admin-proceedings")
        assert "@" not in json.dumps({"url": feed.url, "publisher": feed.publisher})

    def test_the_profile_fingerprint_names_the_feed_and_no_address(self) -> None:
        profile = CollectionProfile.load(
            DEPLOYED_PROFILE, environment={"BTC_INTEL_CONTACT": "secret@example.org"}
        )
        rendered = json.dumps(profile.fingerprint())
        assert "sec-admin-proceedings" in rendered
        assert "secret@example.org" not in rendered


class TestWhatThisFeedActuallyYields:
    """A property of the source worth recording rather than discovering in six
    months: administrative-proceeding titles are respondent names, not topical
    prose. The query planner filters on terms, so this feed contributes a
    document only when a respondent's name happens to contain a watch term.

    That is the filter working correctly and it is deliberately not loosened
    here -- widening the terms to make this feed productive would be admitting
    documents nobody asked for, which is the same mistake as merging streams to
    raise an event count."""

    def _search(self, query: str):
        provider = SyndicationProvider(
            feeds_by_id("sec-admin-proceedings"),
            now=lambda: NOW,
            retry=RetryPolicy(max_attempts=1),
            opener=lambda url, timeout: SEC_ADMIN_FEED,
        )
        return provider.search(query, NOW - timedelta(days=30), NOW)

    def test_a_topical_query_matches_nothing_in_this_feed(self) -> None:
        assert self._search("bitcoin regulation") == []

    def test_a_query_naming_a_respondent_matches(self) -> None:
        documents = self._search("metaverse")
        assert [d.title for d in documents] == ["X METAVERSE INC."]

    def test_the_repaired_entry_is_reachable_like_any_other(self) -> None:
        """The entry recovered by the ampersand repair is a normal document, not
        a second-class one."""
        documents = self._search("institutional")
        assert [d.title for d in documents] == [
            "TIAA-CREF Individual & Institutional Services, LLC"
        ]
        assert documents[0].source_metadata.disclosure_stream is (
            DisclosureStream.ADMINISTRATIVE_PROCEEDINGS
        )
