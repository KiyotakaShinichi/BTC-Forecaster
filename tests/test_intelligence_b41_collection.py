"""B4.1.1 – B4.1.21 — provider policy, syndication, evidence, backoff, immutability.

The two tests that matter most here are the ones that fail on the code as B4
left it: rediscovery rewriting a document's availability, and an improved
extractor silently overwriting the event a previous one produced. Both would
have destroyed a forward-accumulated corpus quietly.

No network anywhere. Every provider is driven through an injected opener.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.collection.backoff import (
    AttemptLog,
    FailureClass,
    ProviderFailure,
    RetryPolicy,
    call_with_retry,
    classify_exception,
    classify_http_status,
)
from market_intelligence.collection.evidence import EvidenceStore, capture, hash_payload
from market_intelligence.collection.feeds import (
    OFFICIAL_FEEDS,
    SYNDICATION_DECLARATION,
    default_feed_catalogue,
    feeds_by_id,
)
from market_intelligence.collection.policy import (
    ProviderCatalogue,
    ProviderDeclaration,
    ProviderPolicy,
    RawRetention,
    redact,
    redact_mapping,
)
from market_intelligence.collection.statements import (
    STATEMENT_DECLARATION,
    DisabledStatementProvider,
    build_statement_provider,
)
from market_intelligence.collection.syndication import (
    FeedSource,
    SyndicationProvider,
    parse_feed,
)
from market_intelligence.collection.whales import (
    WHALE_DECLARATION,
    build_whale_provider,
    classify_transfer,
    context_counts,
    parse_observations,
)
from market_intelligence.errors import ConfigurationError, ReplayIntegrityError
from market_intelligence.extractors import RuleBasedExtractor
from market_intelligence.models import Document, EventType, SourceType, TransferContext
from market_intelligence.storage import IntelligenceStore

NOW = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ fixtures


RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Example Press</title>
  <item>
    <title>Bitcoin ETF decision announced</title>
    <link>https://example.gov/news/2026/etf-decision</link>
    <description>The commission announced a decision on a bitcoin ETF.</description>
    <pubDate>Mon, 04 May 2026 09:30:00 GMT</pubDate>
    <author>press@example.gov</author>
  </item>
  <item>
    <title>Unrelated infrastructure notice</title>
    <link>https://example.gov/news/2026/roads</link>
    <description>Road maintenance schedule.</description>
    <pubDate>Mon, 04 May 2026 08:00:00 GMT</pubDate>
  </item>
</channel></rss>"""

#: Shaped like a regulator's real Atom feed, namespace and all. The provider as
#: B4 left it returns nothing for this, silently.
ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Regulator Press Releases</title>
  <entry>
    <title>Commission charges firm over bitcoin custody failures</title>
    <link rel="alternate" href="https://regulator.example.gov/litigation/2026-77"/>
    <id>urn:regulator:2026-77</id>
    <summary>The Commission today charged a firm relating to bitcoin custody.</summary>
    <published>2026-05-04T10:15:00Z</published>
    <author><name>Office of Public Affairs</name></author>
  </entry>
  <entry>
    <title>Commission announces quarterly bitcoin roundtable</title>
    <link rel="alternate" href="https://regulator.example.gov/news/2026-78"/>
    <id>urn:regulator:2026-78</id>
    <content>A roundtable on bitcoin market structure.</content>
    <updated>2026-05-04T11:00:00Z</updated>
  </entry>
</feed>"""


def feed(feed_id: str = "example", **overrides: object) -> FeedSource:
    payload: dict[str, object] = {
        "feed_id": feed_id,
        "url": f"https://example.gov/{feed_id}.xml",
        "publisher": "example.gov",
        "source_type": SourceType.PRIMARY_OFFICIAL,
        "primary_source": True,
        "official_source": True,
        "retention": RawRetention.FULL,
    }
    payload.update(overrides)
    return FeedSource(**payload)  # type: ignore[arg-type]


def provider(payloads: dict[str, str], *, now: datetime = NOW, **options: object) -> SyndicationProvider:
    def opener(url: str, timeout: float) -> bytes:
        for key, payload in payloads.items():
            if key in url:
                return payload.encode("utf-8")
        raise ProviderFailure(FailureClass.PERMANENT, f"no fixture for {url}")

    return SyndicationProvider(
        [feed(key) for key in payloads],
        opener=opener,
        now=lambda: now,
        sleep=lambda _seconds: None,
        **options,  # type: ignore[arg-type]
    )


# ----------------------------------------------------- the two integrity bugs


class TestDocumentImmutability:
    """B4.1.12. The defect that would have emptied the corpus."""

    def _document(self, retrieved: datetime, provider_name: str = "syndication") -> Document:
        text_hash = Document.content_hash("Commission announces decision")
        return Document(
            document_id=Document.stable_id("https://regulator.example.gov/a", text_hash),
            url="https://regulator.example.gov/a",
            publisher="regulator.example.gov",
            title="Decision",
            published_at=NOW - timedelta(days=2),
            retrieved_at=retrieved,
            available_at=retrieved,
            text_hash=text_hash,
            query="bitcoin",
            provider=provider_name,
        )

    def test_rediscovery_does_not_rewrite_original_availability(self, tmp_path: Path) -> None:
        """Every cycle re-reads the same feed and rediscovers the same items.

        Under `INSERT OR REPLACE` each rediscovery pushed the document's
        availability forward, so a replay at any past origin saw nothing and the
        accumulated corpus was permanently empty.
        """
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            store.put_documents([self._document(NOW)])
            store.put_documents([self._document(NOW + timedelta(days=4), "news-api")])

            stored = store.documents_as_of(NOW + timedelta(days=10))
            assert len(stored) == 1
            assert stored[0].available_at == NOW, "the first sighting is the availability"
            assert len(store.documents_as_of(NOW + timedelta(days=1))) == 1, (
                "a replay one day later must still see day-one evidence"
            )
        finally:
            store.close()

    def test_rediscovery_is_recorded_rather_than_discarded(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            store.put_documents([self._document(NOW)])
            store.put_documents([self._document(NOW + timedelta(days=4), "news-api")])
            sighting = store.sighting(self._document(NOW).document_id)
            assert sighting is not None
            assert sighting.first_seen_at == NOW
            assert sighting.latest_seen_at == NOW + timedelta(days=4)
            assert sighting.sighting_count == 2
            assert sighting.providers == ("news-api", "syndication")
        finally:
            store.close()

    def test_sighting_times_are_canonical_utc_whatever_the_session_zone(self, tmp_path: Path) -> None:
        """B4.1.10. DuckDB renders TIMESTAMPTZ in the session zone."""
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            store.put_documents([self._document(NOW)])
            for zone in ("UTC", "Asia/Manila", "America/New_York"):
                store.connection.execute(f"SET TimeZone='{zone}'")
                sighting = store.sighting(self._document(NOW).document_id)
                assert sighting is not None
                assert sighting.first_seen_at.utcoffset() == timedelta(0), zone
                assert sighting.first_seen_at == NOW
            store.connection.execute("SET TimeZone='UTC'")
        finally:
            store.close()

    def test_an_unseen_document_has_no_sighting(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            assert store.sighting("nope") is None
        finally:
            store.close()


class TestExtractionVersioning:
    """B4.1.16. An improved extractor must not overwrite history."""

    def _document(self) -> Document:
        text = "The SEC announced an enforcement action concerning bitcoin."
        text_hash = Document.content_hash(text)
        return Document(
            document_id=Document.stable_id("https://sec.example.gov/x", text_hash),
            url="https://sec.example.gov/x",
            publisher="sec.example.gov",
            title="SEC announces enforcement action on bitcoin",
            published_at=NOW - timedelta(hours=2),
            retrieved_at=NOW,
            available_at=NOW,
            text_hash=text_hash,
            query="bitcoin regulation",
            provider="syndication",
        )

    def test_two_extractor_versions_produce_distinct_events(self, tmp_path: Path) -> None:
        document = self._document()
        first = RuleBasedExtractor()
        second = RuleBasedExtractor()
        second.version = "rules-v2"  # type: ignore[misc]

        events_v1 = first.extract([document])
        events_v2 = second.extract([document])
        assert events_v1 and events_v2
        assert {event.event_id for event in events_v1}.isdisjoint({event.event_id for event in events_v2}), (
            "identical ids would let v2 overwrite v1 on persistence"
        )

        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            store.put_documents([document])
            store.put_signals(events_v1)
            store.put_signals(events_v2)
            stored = store.signals_as_of(NOW + timedelta(days=1))
            assert {event.extractor_version for event in stored} == {"rules-v1", "rules-v2"}
        finally:
            store.close()

    def test_the_same_extractor_re_run_is_idempotent(self, tmp_path: Path) -> None:
        document = self._document()
        events = RuleBasedExtractor().extract([document])
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            store.put_documents([document])
            store.put_signals(events)
            store.put_signals(RuleBasedExtractor().extract([document]))
            assert len(store.signals_as_of(NOW + timedelta(days=1))) == len(events)
        finally:
            store.close()


# ----------------------------------------------------------------- providers


class TestFeedParsing:
    def test_rss_entries_are_parsed(self) -> None:
        entries = parse_feed(RSS_FEED)
        assert [entry.title for entry in entries] == [
            "Bitcoin ETF decision announced",
            "Unrelated infrastructure notice",
        ]
        assert entries[0].published_at == datetime(2026, 5, 4, 9, 30, tzinfo=timezone.utc)

    def test_atom_entries_are_parsed(self) -> None:
        """The gap. `findall('.//item')` returns nothing for this document, and
        regulators publish Atom."""
        entries = parse_feed(ATOM_FEED)
        assert len(entries) == 2
        assert entries[0].link == "https://regulator.example.gov/litigation/2026-77"
        assert entries[0].published_at == datetime(2026, 5, 4, 10, 15, tzinfo=timezone.utc)
        assert entries[0].author == "Office of Public Affairs"

    def test_an_atom_entry_falls_back_to_updated_and_content(self) -> None:
        entries = parse_feed(ATOM_FEED)
        assert entries[1].published_at == datetime(2026, 5, 4, 11, 0, tzinfo=timezone.utc)
        assert "market structure" in entries[1].summary

    def test_an_unparseable_date_does_not_drop_the_entry(self) -> None:
        broken = RSS_FEED.replace("Mon, 04 May 2026 09:30:00 GMT", "not a date")
        entries = parse_feed(broken)
        assert len(entries) == 2
        assert entries[0].published_at is None

    def test_an_unknown_dialect_yields_nothing_rather_than_raising(self) -> None:
        assert parse_feed("<html><body><p>not a feed</p></body></html>") == []


class TestSyndicationProvider:
    def test_availability_is_retrieval_not_publication(self) -> None:
        """The absolute rule of this track, asserted directly."""
        documents = provider({"example": RSS_FEED}).search("bitcoin", NOW - timedelta(hours=6), NOW)
        assert documents
        for document in documents:
            assert document.available_at == NOW
            assert document.published_at is not None and document.published_at < NOW

    def test_atom_feeds_produce_documents(self) -> None:
        documents = provider({"regulator": ATOM_FEED}).search("bitcoin", NOW - timedelta(hours=6), NOW)
        assert len(documents) == 2

    def test_primary_and_official_classification_is_recorded(self) -> None:
        """B4.1.3. An announcement and coverage of it are different evidence."""
        documents = provider({"regulator": ATOM_FEED}).search("bitcoin", NOW - timedelta(hours=6), NOW)
        assert all(document.source_metadata.primary_source for document in documents)
        assert all(document.source_metadata.official_source for document in documents)
        assert all(
            document.source_metadata.source_type is SourceType.PRIMARY_OFFICIAL for document in documents
        )

    def test_query_terms_filter_entries(self) -> None:
        documents = provider({"example": RSS_FEED}).search("bitcoin", NOW - timedelta(hours=6), NOW)
        assert [document.title for document in documents] == ["Bitcoin ETF decision announced"]

    def test_retrieval_provenance_names_the_feed(self) -> None:
        documents = provider({"regulator": ATOM_FEED}).search("bitcoin", NOW - timedelta(hours=6), NOW)
        provenance = documents[0].retrieval_provenance
        assert provenance and provenance[0].provider_document_id == "regulator"

    def test_raw_evidence_is_captured_and_hashed(self) -> None:
        engine = provider({"regulator": ATOM_FEED})
        engine.search("bitcoin", NOW - timedelta(hours=6), NOW)
        assert len(engine.last_evidence) == 1
        evidence = engine.last_evidence[0]
        assert evidence.content_hash == hash_payload(ATOM_FEED)
        assert evidence.payload is not None, "this feed's retention is FULL"

    def test_one_failing_feed_does_not_lose_the_others(self) -> None:
        """B4.1.17. Partial provider failure must not corrupt successful work."""

        def opener(url: str, timeout: float) -> bytes:
            if "broken" in url:
                raise ProviderFailure(FailureClass.PERMANENT, "404")
            return ATOM_FEED.encode("utf-8")

        engine = SyndicationProvider(
            [feed("broken"), feed("regulator")],
            opener=opener,
            now=lambda: NOW,
            sleep=lambda _seconds: None,
        )
        documents = engine.search("bitcoin", NOW - timedelta(hours=6), NOW)
        assert len(documents) == 2
        assert engine.last_attempts["broken"].failures == (FailureClass.PERMANENT,)

    def test_the_lookback_window_bounds_publication_not_availability(self) -> None:
        """The third silent-empty failure, pinned.

        Testing the window against *availability* can never pass in forward
        collection -- availability is retrieval, which is always after the
        planning instant the window ends at -- so every entry is discarded and
        the provider reports a successful, empty collection. The window belongs
        on publication.
        """
        engine = provider({"example": RSS_FEED})
        # A window ending at the planning instant, as the retriever supplies.
        current = engine.search("bitcoin", NOW - timedelta(hours=6), NOW)
        assert current, "entries published inside the lookback must be collected"
        assert all(document.available_at == NOW for document in current)

        # A lookback whose start is after the entries were published collects
        # nothing. Only the lower bound is applied: an entry carrying a future
        # publication stamp -- publishers do embargo -- is real evidence that was
        # really retrieved, and an upper bound would silently discard it.
        narrow = engine.search("bitcoin", NOW - timedelta(minutes=5), NOW)
        assert narrow == []

    def test_an_entry_without_a_publication_date_is_still_collected(self) -> None:
        """A missing date is not evidence the entry is old, so it is kept."""
        undated = RSS_FEED.replace("<pubDate>Mon, 04 May 2026 09:30:00 GMT</pubDate>", "")
        collected = provider({"example": undated}).search("bitcoin", NOW - timedelta(minutes=5), NOW)
        assert [document.title for document in collected] == ["Bitcoin ETF decision announced"]

    def test_entries_without_a_link_or_title_are_skipped(self) -> None:
        partial = """<?xml version="1.0"?><rss version="2.0"><channel>
          <item><title>Bitcoin note</title><description>no link</description></item>
          <item><link>https://example.gov/x</link><description>bitcoin, no title</description></item>
        </channel></rss>"""
        assert provider({"example": partial}).search("bitcoin", NOW - timedelta(hours=6), NOW) == []


# ------------------------------------------------------------------- policy


class TestProviderPolicy:
    def test_a_declaration_without_its_credential_is_not_operable(self) -> None:
        assert not STATEMENT_DECLARATION.operable(credential_present=False)
        assert STATEMENT_DECLARATION.operable(credential_present=True)
        assert "BTC_INTEL_STATEMENTS_API_KEY" in (
            STATEMENT_DECLARATION.disabled_reason(credential_present=False) or ""
        )

    def test_a_public_documented_provider_needs_no_credential(self) -> None:
        assert SYNDICATION_DECLARATION.policy is ProviderPolicy.PUBLIC_DOCUMENTED
        assert SYNDICATION_DECLARATION.operable(credential_present=False)
        assert SYNDICATION_DECLARATION.disabled_reason(credential_present=False) is None

    def test_the_catalogue_refuses_a_duplicate_declaration(self) -> None:
        catalogue = ProviderCatalogue([SYNDICATION_DECLARATION])
        with pytest.raises(ConfigurationError, match="already declared"):
            catalogue.register(SYNDICATION_DECLARATION)

    def test_an_undeclared_provider_cannot_be_required(self) -> None:
        with pytest.raises(ConfigurationError, match="no policy declaration"):
            ProviderCatalogue().require("mystery")

    def test_declarations_are_grouped_by_policy(self) -> None:
        catalogue = ProviderCatalogue([SYNDICATION_DECLARATION, STATEMENT_DECLARATION, WHALE_DECLARATION])
        assert [item.provider_id for item in catalogue.by_policy(ProviderPolicy.PUBLIC_DOCUMENTED)] == [
            "syndication"
        ]
        assert len(catalogue.by_policy(ProviderPolicy.AUTHENTICATED_LICENSED)) == 2

    def test_every_official_feed_is_declared_primary_and_official(self) -> None:
        assert len(OFFICIAL_FEEDS) >= 5
        assert all(item.primary_source and item.official_source for item in OFFICIAL_FEEDS)
        assert default_feed_catalogue() == OFFICIAL_FEEDS

    def test_selecting_an_unknown_feed_is_an_error(self) -> None:
        assert len(feeds_by_id("sec-press", "cftc-press")) == 2
        with pytest.raises(KeyError, match="unknown feed ids"):
            feeds_by_id("sec-press", "not-a-feed")


class TestCredentialRedaction:
    """B4.1.40. Secrets never reach a log, a manifest or a response."""

    def test_a_secret_becomes_a_marker_not_a_prefix(self) -> None:
        assert redact("sk-live-abcdef123456") == "<redacted>"
        assert redact(None) == "<unset>"
        assert "abcdef" not in redact("sk-live-abcdef123456")

    def test_secret_shaped_keys_are_redacted_even_when_not_listed(self) -> None:
        redacted = redact_mapping(
            {
                "endpoint": "https://api.example.com",
                "api_key": "sk-live-1",
                "Authorization": "Bearer xyz",
                "refresh_token": "rt-9",
                "timeout": "15",
            }
        )
        assert redacted["endpoint"] == "https://api.example.com"
        assert redacted["timeout"] == "15"
        assert redacted["api_key"] == redacted["Authorization"] == redacted["refresh_token"] == "<redacted>"

    def test_an_explicitly_named_secret_is_redacted(self) -> None:
        assert redact_mapping({"opaque": "value"}, secret_keys=("opaque",))["opaque"] == "<redacted>"


# ----------------------------------------------------------------- evidence


class TestRawEvidence:
    def test_full_retention_keeps_the_payload(self) -> None:
        record = capture("p", ATOM_FEED, retention=RawRetention.FULL, retrieved_at=NOW)
        assert record.payload is not None
        assert record.redistributable
        assert not record.payload_withheld

    def test_restricted_retention_keeps_only_the_hash(self) -> None:
        record = capture(
            "p", ATOM_FEED, retention=RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE, retrieved_at=NOW
        )
        assert record.payload is None
        assert record.payload_withheld
        assert record.content_hash == hash_payload(ATOM_FEED)
        assert record.content_bytes == len(ATOM_FEED.encode("utf-8"))

    def test_local_only_retention_keeps_the_payload_but_forbids_export(self) -> None:
        record = capture("p", ATOM_FEED, retention=RawRetention.LOCAL_ONLY, retrieved_at=NOW)
        assert record.payload is not None
        assert not record.redistributable

    def test_identical_bytes_address_the_same_record(self) -> None:
        first = capture("p", ATOM_FEED, retention=RawRetention.FULL, retrieved_at=NOW)
        again = capture("p", ATOM_FEED, retention=RawRetention.FULL, retrieved_at=NOW + timedelta(days=1))
        assert first.evidence_id == again.evidence_id

    def test_a_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ReplayIntegrityError, match="timezone-aware"):
            capture("p", "x", retention=RawRetention.FULL, retrieved_at=datetime(2026, 5, 4))

    def test_storing_identical_evidence_twice_is_a_no_op(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            evidence = EvidenceStore(store.connection)
            record = capture("p", ATOM_FEED, retention=RawRetention.FULL, retrieved_at=NOW)
            assert evidence.put([record]) == 1
            assert evidence.put([record]) == 0
        finally:
            store.close()

    def test_changing_the_bytes_under_an_existing_id_is_refused(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            evidence = EvidenceStore(store.connection)
            record = capture("p", ATOM_FEED, retention=RawRetention.FULL, retrieved_at=NOW)
            evidence.put([record])
            tampered = record.model_copy(update={"content_hash": "0" * 64})
            with pytest.raises(ReplayIntegrityError, match="different content hash"):
                evidence.put([tampered])
        finally:
            store.close()

    def test_export_omits_restricted_payloads_and_says_so(self, tmp_path: Path) -> None:
        """B4.1.44. Never export restricted raw content by default."""
        import json

        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            evidence = EvidenceStore(store.connection)
            evidence.put(
                [
                    capture("open", "public bytes", retention=RawRetention.FULL, retrieved_at=NOW),
                    capture(
                        "closed",
                        "licensed bytes",
                        retention=RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE,
                        retrieved_at=NOW,
                    ),
                ]
            )
            summary = evidence.export_redistributable(tmp_path / "export.json")
            assert summary == {"records": 2, "omitted_license": 1}
            exported = json.loads((tmp_path / "export.json").read_text(encoding="utf-8"))
            withheld = [row for row in exported if row.get("omission_reason") == "OMITTED_LICENSE"]
            assert len(withheld) == 1
            assert withheld[0]["payload"] is None
            assert withheld[0]["content_hash"], "the hash still travels, so a copy can be checked"
            assert "licensed bytes" not in (tmp_path / "export.json").read_text(encoding="utf-8")
        finally:
            store.close()

    def test_stored_bytes_are_measurable_for_growth_estimates(self, tmp_path: Path) -> None:
        store = IntelligenceStore(tmp_path / "s.duckdb")
        try:
            evidence = EvidenceStore(store.connection)
            evidence.put([capture("open", "x" * 500, retention=RawRetention.FULL, retrieved_at=NOW)])
            evidence.put(
                [
                    capture(
                        "closed",
                        "y" * 5000,
                        retention=RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE,
                        retrieved_at=NOW,
                    )
                ]
            )
            assert evidence.total_bytes() == 500, "withheld payloads occupy no bytes"
            assert evidence.counts_by_retention() == {"FULL": 1, "NON_REDISTRIBUTABLE_RAW_SOURCE": 1}
        finally:
            store.close()


# ------------------------------------------------------------------ backoff


class TestFailureClassification:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, FailureClass.AUTH),
            (403, FailureClass.AUTH),
            (429, FailureClass.RATE_LIMIT),
            (404, FailureClass.PERMANENT),
            (410, FailureClass.PERMANENT),
            (400, FailureClass.PERMANENT),
            (500, FailureClass.TRANSIENT),
            (503, FailureClass.TRANSIENT),
        ],
    )
    def test_http_statuses_map_to_classes(self, status: int, expected: FailureClass) -> None:
        assert classify_http_status(status) is expected

    def test_a_parse_error_is_schema_not_transient(self) -> None:
        assert classify_exception(KeyError("published_at")) is FailureClass.SCHEMA
        assert classify_exception(ValueError("bad iso")) is FailureClass.SCHEMA

    def test_an_unknown_error_falls_back_to_transient(self) -> None:
        assert classify_exception(RuntimeError("?")) is FailureClass.TRANSIENT


class TestRetry:
    def test_auth_failures_are_never_retried(self) -> None:
        """The expensive mistake: hammering an auth endpoint with a dead key."""
        calls = {"n": 0}

        def operation() -> str:
            calls["n"] += 1
            raise ProviderFailure(FailureClass.AUTH, "401")

        log = AttemptLog()
        with pytest.raises(ProviderFailure) as caught:
            call_with_retry(operation, RetryPolicy(max_attempts=5), sleep=lambda _s: None, log=log)
        assert calls["n"] == 1
        assert caught.value.failure_class is FailureClass.AUTH
        assert log.slept_seconds == 0.0

    def test_transient_failures_are_retried_up_to_the_bound(self) -> None:
        calls = {"n": 0}

        def operation() -> str:
            calls["n"] += 1
            raise ProviderFailure(FailureClass.TRANSIENT, "503")

        with pytest.raises(ProviderFailure):
            call_with_retry(operation, RetryPolicy(max_attempts=3, seed=1), sleep=lambda _s: None)
        assert calls["n"] == 3

    def test_a_transient_failure_that_clears_returns_the_value(self) -> None:
        calls = {"n": 0}

        def operation() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderFailure(FailureClass.TRANSIENT, "503")
            return "ok"

        assert call_with_retry(operation, RetryPolicy(seed=2), sleep=lambda _s: None) == "ok"

    def test_rate_limits_wait_longer_than_transient_failures(self) -> None:
        policy = RetryPolicy(jitter=False, base_delay_seconds=2.0, rate_limit_multiplier=4.0)
        import random

        generator = random.Random(0)
        transient = policy.delay_for(1, FailureClass.TRANSIENT, generator)
        limited = policy.delay_for(1, FailureClass.RATE_LIMIT, generator)
        assert limited == 4 * transient

    def test_delays_grow_exponentially_and_are_capped(self) -> None:
        import random

        policy = RetryPolicy(jitter=False, base_delay_seconds=2.0, max_delay_seconds=10.0)
        generator = random.Random(0)
        delays = [policy.delay_for(attempt, FailureClass.TRANSIENT, generator) for attempt in range(1, 6)]
        assert delays == [2.0, 4.0, 8.0, 10.0, 10.0]

    def test_jitter_spreads_retries_that_would_otherwise_be_synchronised(self) -> None:
        import random

        policy = RetryPolicy(jitter=True, base_delay_seconds=8.0)
        delays = {policy.delay_for(2, FailureClass.TRANSIENT, random.Random(seed)) for seed in range(20)}
        assert len(delays) > 15, "identical delays would keep cron-launched pollers in lockstep"

    def test_schema_failures_are_not_retried(self) -> None:
        calls = {"n": 0}

        def operation() -> str:
            calls["n"] += 1
            raise KeyError("published_at")

        with pytest.raises(ProviderFailure) as caught:
            call_with_retry(operation, RetryPolicy(max_attempts=4), sleep=lambda _s: None)
        assert calls["n"] == 1
        assert caught.value.failure_class is FailureClass.SCHEMA


# ----------------------------------------------------- optional providers


class TestStatementProvider:
    def test_no_endpoint_yields_an_honest_disabled_provider(self) -> None:
        built = build_statement_provider(None, None)
        assert isinstance(built, DisabledStatementProvider)
        assert built.statements(["Elon Musk"], NOW - timedelta(days=1), NOW) == []
        assert built.status()["state"] == "DISABLED"
        assert "endpoint" in built.status()["reason"]

    def test_a_missing_credential_names_the_variable(self) -> None:
        built = build_statement_provider("https://api.example.com/statements", None)
        assert isinstance(built, DisabledStatementProvider)
        assert "BTC_INTEL_STATEMENTS_API_KEY" in built.status()["reason"]

    def test_a_configured_provider_normalises_statements_with_retrieval_availability(self) -> None:
        import json

        payload = json.dumps(
            {
                "statements": [
                    {
                        "entity": "Elon Musk",
                        "statement_hash": "a" * 64,
                        "published_at": "2026-05-04T09:00:00Z",
                        "source_url": "https://platform.example.com/1",
                    },
                    {"entity": "broken record"},
                ]
            }
        )
        built = build_statement_provider(
            "https://api.example.com/statements",
            "sk-test",
            opener=lambda url, key, timeout: payload.encode("utf-8"),
            now=lambda: NOW,
            sleep=lambda _s: None,
        )
        statements = built.statements(["Elon Musk"], NOW - timedelta(days=1), NOW)
        assert len(statements) == 1, "a malformed record is skipped, not fatal"
        assert statements[0].available_at == NOW
        assert statements[0].published_at < NOW


class TestWhaleProvider:
    def test_no_contract_yields_an_honest_disabled_provider(self) -> None:
        built = build_whale_provider(None, None)
        assert built.observations(NOW - timedelta(days=1), NOW) == []
        assert built.status()["state"] == "DISABLED"  # type: ignore[attr-defined]

    def test_unattributed_transfers_stay_unknown(self) -> None:
        """B4.1.6. The inference this module refuses to make."""
        assert (
            classify_transfer(None, from_attribution="binance-hot-wallet", to_attribution="unknown")
            is TransferContext.UNKNOWN
        )
        assert (
            classify_transfer("", from_attribution="coinbase", to_attribution="coinbase")
            is TransferContext.UNKNOWN
        )

    def test_an_unrecognised_provider_label_becomes_unknown_not_a_guess(self) -> None:
        assert classify_transfer("PROBABLY_EXCHANGE", from_attribution=None, to_attribution=None) is (
            TransferContext.UNKNOWN
        )

    def test_provider_labels_map_to_contexts(self) -> None:
        assert classify_transfer("EXCHANGE_INFLOW", from_attribution=None, to_attribution=None) is (
            TransferContext.EXCHANGE_INFLOW
        )
        assert classify_transfer("custody", from_attribution=None, to_attribution=None) is (
            TransferContext.CUSTODY_TRANSFER
        )

    def test_observations_use_retrieval_availability_not_block_time(self) -> None:
        observations = parse_observations(
            {
                "transfers": [
                    {
                        "observation_id": "t1",
                        "amount_btc": 1200.0,
                        "observed_at": "2026-05-04T11:40:00Z",
                        "classification": "EXCHANGE_INFLOW",
                        "source_url": "https://chain.example.com/t1",
                    }
                ]
            },
            NOW,
            "whales",
        )
        assert len(observations) == 1
        assert observations[0].available_at == NOW
        assert observations[0].observed_at < NOW

    def test_every_context_is_counted_even_at_zero(self) -> None:
        """B4.1.32. A missing category must be visible, not absent."""
        observations = parse_observations(
            {
                "transfers": [
                    {
                        "observation_id": "t1",
                        "amount_btc": 900.0,
                        "observed_at": "2026-05-04T11:00:00Z",
                        "classification": "EXCHANGE_INFLOW",
                        "source_url": "https://chain.example.com/t1",
                    }
                ]
            },
            NOW,
            "whales",
        )
        counts = context_counts(observations)
        assert counts[TransferContext.EXCHANGE_INFLOW.value] == 1
        assert set(counts) == {context.value for context in TransferContext}
        assert counts[TransferContext.UNKNOWN.value] == 0

    def test_a_malformed_transfer_is_skipped(self) -> None:
        assert parse_observations({"transfers": [{"observation_id": "x"}]}, NOW, "whales") == []


def test_the_declared_event_types_still_cover_the_b4_hypotheses() -> None:
    """A guard: B4's studies are keyed on these names."""
    required = {"REGULATION", "MONETARY_POLICY", "ETF_FLOW", "WHALE_TRANSFER", "ENTITY_STATEMENT"}
    assert required <= {member.value for member in EventType}


def test_the_provider_declaration_is_explicit_that_terms_may_change() -> None:
    declaration: ProviderDeclaration = SYNDICATION_DECLARATION
    assert "change" in declaration.rate_limit_note.casefold()
