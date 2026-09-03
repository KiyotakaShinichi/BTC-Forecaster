"""Stream-aware candidate matching, and what it deliberately does not do.

The SEC's administrative-proceedings feed titles its entries with respondent
names, so topical filtering on the title looked structurally wrong. Auditing the
feed for a better field to match on produced the opposite of the expected
answer, and these tests pin it down: the description is the respondent name too,
byte-identical to the title in 23 of 25 live entries. There is no field in this
feed that says what a proceeding is about.

So nothing here widens what any stream admits. What changes is that the fields
consulted are now declared per stream instead of hardcoded, and a stream whose
fields cannot carry subject matter says so -- because a zero match count from
such a feed is a property of the source, and reading it as a collection fault is
how a filter gets loosened until unrelated documents come through.

No network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree

import pytest

from market_intelligence.collection.backoff import RetryPolicy
from market_intelligence.collection.feeds import feeds_by_id
from market_intelligence.collection.matching import term_pattern
from market_intelligence.collection.syndication import (
    DEFAULT_MATCH_POLICY,
    STREAM_MATCH_POLICIES,
    FeedEntry,
    StreamMatchPolicy,
    SyndicationProvider,
    match_policy,
    parse_feed,
    parse_feed_with_repairs,
    repair_xml,
)
from market_intelligence.models import DisclosureStream
from market_intelligence.retrieval import deduplicate_across_providers

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
WINDOW = (NOW - timedelta(days=30), NOW)


def rss(*items: str) -> bytes:
    body = "".join(items)
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<channel><title>Feed</title>{body}</channel></rss>"
    ).encode()


def item(title: str, link: str, description: str, *, published: str = "Tue, 01 Sep 2026 14:42:32 -0400") -> str:
    return (
        f"<item><title>{title}</title><link>{link}</link>"
        f"<description>{description}</description><pubDate>{published}</pubDate></item>"
    )


#: Shaped exactly like the live feed: respondent name in both fields.
ADMIN_FEED = rss(
    item(
        "Bitwise Digital Holdings, LLC",
        "https://www.sec.gov/files/litigation/admin/2026/34-106260.pdf",
        "Bitwise Digital Holdings, LLC",
    ),
    item(
        "Hancock Whitney Investment Services, Inc.",
        "https://www.sec.gov/files/litigation/admin/2026/34-106253.pdf",
        "Hancock Whitney Investment Services, Inc.",
    ),
)

#: A press release, where the description genuinely carries subject matter.
PRESS_FEED = rss(
    item(
        "SEC Charges Bitwise Digital Holdings With Unregistered Bitcoin Offering",
        "https://www.sec.gov/news/press-release/2026-140",
        "The Securities and Exchange Commission today announced settled charges "
        "concerning an unregistered bitcoin offering.",
    )
)


def provider(feed_id: str, payload: bytes) -> SyndicationProvider:
    return SyndicationProvider(
        feeds_by_id(feed_id),
        now=lambda: NOW,
        retry=RetryPolicy(max_attempts=1),
        opener=lambda url, timeout: payload,
    )


# ------------------------------------------------------- the policy is explicit


class TestTheMatchPolicyIsDeclaredNotHardcoded:
    def test_the_admin_stream_declares_its_match_fields(self) -> None:
        policy = match_policy(DisclosureStream.ADMINISTRATIVE_PROCEEDINGS)
        assert policy.match_fields == ("title", "summary")

    def test_the_admin_stream_is_recorded_as_carrying_no_subject_matter(self) -> None:
        """The load-bearing part. A zero match count here is the source's
        shape, not a collection fault, and the type system now says so."""
        policy = match_policy(DisclosureStream.ADMINISTRATIVE_PROCEEDINGS)
        assert policy.subject_bearing is False
        assert "respondent" in policy.note

    def test_other_streams_keep_the_default_and_are_subject_bearing(self) -> None:
        """Section 2: do not change matching for press releases or anything
        else without a separate structural reason."""
        for stream in (
            DisclosureStream.REGULATORY_ANNOUNCEMENT,
            DisclosureStream.MONETARY_POLICY,
            DisclosureStream.ECONOMIC_STATISTICS,
            DisclosureStream.CIVIL_LITIGATION,
        ):
            assert match_policy(stream) is DEFAULT_MATCH_POLICY
            assert match_policy(stream).subject_bearing is True

    def test_only_the_admin_stream_deviates(self) -> None:
        assert set(STREAM_MATCH_POLICIES) == {DisclosureStream.ADMINISTRATIVE_PROCEEDINGS}

    def test_the_matched_text_is_title_and_description(self) -> None:
        entry = FeedEntry(
            title="Respondent Ltd", link="x", summary="about bitcoin", published_at=None, author=None
        )
        assert match_policy(DisclosureStream.ADMINISTRATIVE_PROCEEDINGS).haystack(entry) == (
            "respondent ltd about bitcoin"
        )

    def test_the_policy_never_reads_the_link(self) -> None:
        """A URL path is not content the publisher wrote as content, and
        matching on it would admit documents on the strength of a filename."""
        for policy in (DEFAULT_MATCH_POLICY, *STREAM_MATCH_POLICIES.values()):
            assert "link" not in policy.match_fields

    def test_an_unknown_field_in_a_policy_contributes_nothing(self) -> None:
        entry = FeedEntry(title="a", link="x", summary="b", published_at=None, author=None)
        assert StreamMatchPolicy(match_fields=("title", "nope")).haystack(entry) == "a "


# ---------------------------------------------------------------- title kept


class TestTheTitleIsNeverRewritten:
    def test_the_respondent_name_remains_the_canonical_title(self) -> None:
        """Section 3. The respondent name is genuine source data."""
        documents = provider("sec-admin-proceedings", ADMIN_FEED).search("bitwise", *WINDOW)
        assert [d.title for d in documents] == ["Bitwise Digital Holdings, LLC"]

    def test_matching_does_not_alter_the_stored_text(self) -> None:
        documents = provider("sec-admin-proceedings", ADMIN_FEED).search("bitwise", *WINDOW)
        assert documents[0].title == "Bitwise Digital Holdings, LLC"
        assert "bitcoin" not in documents[0].title.casefold()


# ------------------------------------------------- the contract is not widened


class TestRelevanceIsNotLoosened:
    def test_a_topical_query_still_matches_nothing_in_this_feed(self) -> None:
        """Section 1. The whole point: recall is not bought by admitting
        proceedings that say nothing about crypto."""
        assert provider("sec-admin-proceedings", ADMIN_FEED).search("bitcoin regulation", *WINDOW) == []

    def test_a_proceeding_matches_only_when_the_respondent_is_watched(self) -> None:
        """The one honest recall path, and it depends on the watchlist naming
        the firm -- not on the filter being relaxed."""
        documents = provider("sec-admin-proceedings", ADMIN_FEED).search("Bitwise", *WINDOW)
        assert [d.title for d in documents] == ["Bitwise Digital Holdings, LLC"]

    def test_an_unrelated_proceeding_is_still_excluded(self) -> None:
        documents = provider("sec-admin-proceedings", ADMIN_FEED).search("Bitwise", *WINDOW)
        assert all("Hancock" not in d.title for d in documents)

    def test_a_query_whose_terms_are_present_admits_them(self) -> None:
        """The filter is unchanged in the other direction too: what genuinely
        matches still comes through."""
        documents = provider("sec-admin-proceedings", ADMIN_FEED).search("LLC Inc", *WINDOW)
        assert len(documents) == 2

    def test_press_release_matching_is_unchanged(self) -> None:
        """Its description carries subject matter, so a topical term matches --
        which is exactly the asymmetry the policy documents."""
        documents = provider("sec-press", PRESS_FEED).search("bitcoin", *WINDOW)
        assert len(documents) == 1


# ------------------------------------------------------------- yield telemetry


class TestYieldIsMeasuredNotInferred:
    def test_entries_and_matches_are_counted_separately(self) -> None:
        """Section 8 needs these apart: 'nothing retrieved' and 'nothing
        relevant' are different failures with different remedies."""
        collector = provider("sec-admin-proceedings", ADMIN_FEED)
        collector.search("bitcoin regulation", *WINDOW)
        measured = collector.last_yield["sec-admin-proceedings"]
        assert measured.entries == 2
        assert measured.matched == 0
        assert measured.subject_bearing is False

    def test_a_match_is_counted(self) -> None:
        collector = provider("sec-admin-proceedings", ADMIN_FEED)
        collector.search("Bitwise", *WINDOW)
        assert collector.last_yield["sec-admin-proceedings"].matched == 1

    def test_a_subject_bearing_feed_reports_itself_as_such(self) -> None:
        collector = provider("sec-press", PRESS_FEED)
        collector.search("bitcoin", *WINDOW)
        measured = collector.last_yield["sec-press"]
        assert measured.subject_bearing is True
        assert (measured.entries, measured.matched) == (1, 1)

    def test_telemetry_resets_between_searches(self) -> None:
        collector = provider("sec-admin-proceedings", ADMIN_FEED)
        collector.search("Bitwise", *WINDOW)
        collector.search("nothing-matches-this", *WINDOW)
        assert collector.last_yield["sec-admin-proceedings"].matched == 0


# ----------------------------------------------- a candidate is not an event


class TestAMatchedDocumentIsOnlyACandidate:
    def test_matching_produces_a_document_not_an_event(self) -> None:
        """Section 5. Everything downstream -- extraction, validation,
        relevance, dedup, clustering -- still has to happen."""
        from market_intelligence.extractors import RuleBasedExtractor

        documents = provider("sec-admin-proceedings", ADMIN_FEED).search("Bitwise", *WINDOW)
        assert len(documents) == 1
        events = RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}).extract(documents)
        assert events == [], "a respondent-named document produced an event by matching alone"

    def test_a_document_that_does_carry_an_entity_yields_an_event(self) -> None:
        """The contrast: extraction is doing its own work, not rubber-stamping
        whatever the candidate filter admitted."""
        from market_intelligence.extractors import RuleBasedExtractor

        documents = provider("sec-press", PRESS_FEED).search("bitcoin", *WINDOW)
        events = RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}).extract(documents)
        assert len(events) >= 1


# --------------------------------------------------------- cross-stream dedup


class TestAnnouncementAndProceedingStayDistinct:
    """Section 6. The realistic case: the SEC announces a proceeding in a press
    release on the same day the administrative feed carries the order."""

    def _both(self) -> list:
        proceeding = provider("sec-admin-proceedings", ADMIN_FEED).search("Bitwise", *WINDOW)
        announcement = provider("sec-press", PRESS_FEED).search("Bitwise", *WINDOW)
        assert proceeding and announcement, "the fixture must produce one of each"
        return proceeding + announcement

    def test_both_survive_deduplication(self) -> None:
        merged = deduplicate_across_providers(self._both())
        assert len(merged) == 2

    def test_each_keeps_its_own_stream(self) -> None:
        merged = deduplicate_across_providers(self._both())
        assert {d.source_metadata.disclosure_stream for d in merged} == {
            DisclosureStream.ADMINISTRATIVE_PROCEEDINGS,
            DisclosureStream.REGULATORY_ANNOUNCEMENT,
        }

    def test_each_keeps_its_own_source_url(self) -> None:
        """Provenance survives: a study can go and read either artefact."""
        merged = deduplicate_across_providers(self._both())
        urls = sorted(str(d.url) for d in merged)
        assert any("/litigation/admin/" in url for url in urls)
        assert any("/news/press-release/" in url for url in urls)

    def test_both_name_their_feed_in_provenance(self) -> None:
        merged = deduplicate_across_providers(self._both())
        feeds = {d.retrieval_provenance[0].provider_document_id for d in merged}
        assert feeds == {"sec-admin-proceedings", "sec-press"}

    def test_they_share_a_publisher_which_is_why_this_needed_a_guard(self) -> None:
        merged = deduplicate_across_providers(self._both())
        assert len({d.publisher for d in merged}) == 1


# ------------------------------------------------------------- the XML repair


class TestRepairContract:
    """Section 7. Strict first; repair only bare ampersands; never touch valid
    XML; still fail on anything else."""

    def test_valid_entity_references_are_left_alone(self) -> None:
        fixed, repairs = repair_xml("a &amp; b &lt; c &gt; d &quot; e &apos;")
        assert fixed == "a &amp; b &lt; c &gt; d &quot; e &apos;"
        assert repairs == ()

    def test_numeric_references_are_left_alone(self) -> None:
        fixed, repairs = repair_xml("a &#38; b &#169;")
        assert fixed == "a &#38; b &#169;"
        assert repairs == ()

    def test_hex_references_are_left_alone(self) -> None:
        fixed, repairs = repair_xml("a &#x26; b &#XA9; c")
        assert fixed == "a &#x26; b &#XA9; c"
        assert repairs == ()

    def test_a_bare_ampersand_is_escaped(self) -> None:
        fixed, repairs = repair_xml("Smith & Jones")
        assert fixed == "Smith &amp; Jones"
        assert repairs == ("escaped 1 bare ampersand",)

    def test_several_bare_ampersands_are_counted(self) -> None:
        _, repairs = repair_xml("A & B & C")
        assert repairs == ("escaped 2 bare ampersands",)

    def test_a_mixture_repairs_only_what_is_broken(self) -> None:
        fixed, repairs = repair_xml("&amp; & &#38; & &#x26;")
        assert fixed == "&amp; &amp; &#38; &amp; &#x26;"
        assert repairs == ("escaped 2 bare ampersands",)

    def test_valid_xml_is_never_rewritten(self) -> None:
        payload = rss(item("Clean &amp; Tidy", "https://x.test/1", "no repair needed"))
        entries, repairs = parse_feed_with_repairs(payload)
        assert repairs == ()
        assert entries[0].title == "Clean & Tidy"

    def test_a_bare_ampersand_feed_parses_after_repair(self) -> None:
        payload = rss(item("Smith & Jones LLC", "https://x.test/2", "Smith & Jones LLC"))
        entries, repairs = parse_feed_with_repairs(payload)
        assert repairs == ("escaped 2 bare ampersands",)
        assert entries[0].title == "Smith & Jones LLC"

    def test_unfixable_xml_still_raises(self) -> None:
        """The repair does not close tags or guess structure."""
        with pytest.raises(ElementTree.ParseError):
            parse_feed_with_repairs(b"<rss><channel><item><title>x</channel></rss>")

    def test_an_ampersand_problem_that_is_not_the_only_problem_still_raises(self) -> None:
        with pytest.raises(ElementTree.ParseError):
            parse_feed_with_repairs(b"<rss><channel><item><title>A & B</title></channel></rss>")

    def test_parse_feed_keeps_its_simple_signature(self) -> None:
        payload = rss(item("Smith & Jones", "https://x.test/3", "d"))
        assert [entry.title for entry in parse_feed(payload)] == ["Smith & Jones"]


class TestTermsMatchAtWordStarts:
    """Plain containment let a two-letter term match inside unrelated words.

    The planner emits `"US Treasury" sanctions`, so `us` was a live term, and
    it matched "SEC Anno(us)nces", "B(us)iness", "Foc(us)" -- 18 of 382
    candidate matches in one measured cycle. Every one was a genuine SEC
    release about something else entirely, which is why nothing downstream
    looked wrong. This narrows relevance; it never widens it."""

    def test_a_short_term_no_longer_matches_inside_a_word(self) -> None:
        pattern = term_pattern(["us"])
        for text in ("sec announces a roundtable", "business advisory", "focus group"):
            assert pattern.search(text) is None, text

    def test_the_same_term_still_matches_as_a_word(self) -> None:
        assert term_pattern(["us"]).search("the us treasury") is not None

    def test_suffixes_are_still_matched(self) -> None:
        """A whole-word rule would have cost real recall."""
        pattern = term_pattern(["regulation", "etf", "bitcoin"])
        for text in ("new regulations issued", "etfs approved", "bitcoin's price"):
            assert pattern.search(text) is not None, text

    def test_a_prefix_of_a_longer_word_does_not_match_from_the_middle(self) -> None:
        assert term_pattern(["regulation"]).search("deregulation") is None

    def test_no_terms_means_no_filtering(self) -> None:
        assert term_pattern([]) is None

    def test_regex_metacharacters_in_a_term_are_literal(self) -> None:
        """Terms come from a planned query, and a stray character must not
        become a wildcard that admits everything."""
        assert term_pattern(["c++"]).search("we use c++ here") is not None
        assert term_pattern(["a.c"]).search("abc") is None

    def test_the_provider_uses_it(self) -> None:
        """The false positive that started this: a respondent named Rebus."""
        payload = rss(item("Rebus Holdings, Inc.", "https://x.test/9", "Rebus Holdings, Inc."))
        collector = provider("sec-admin-proceedings", payload)
        assert collector.search("US Treasury sanctions", *WINDOW) == []
        assert collector.last_yield["sec-admin-proceedings"].matched == 0
