"""Every silent-empty defect this collector has had, in one place.

They all share a shape. A cycle finishes, reports success, and writes nothing --
and nothing in the output distinguishes that from a quiet news day. It is the
hardest class of bug in this system because the failure mode is *plausible
output*: a green run with zero documents looks exactly like a Tuesday when the
SEC published nothing.

Fourteen of them have been found. Each one below is named after the defect it
prevents and asserts the property that would have caught it, so a regression
cannot come back wearing a green tick. Several overlap with tests elsewhere;
that duplication is the point -- this file is the index of what has actually
gone wrong, and it should be read before anyone changes collection semantics.

No network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

import pytest

from market_intelligence.collection.backoff import FailureClass, ProviderFailure, RetryPolicy
from market_intelligence.collection.clustering import cluster_events
from market_intelligence.collection.feeds import feeds_by_id
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.collection.matching import term_pattern
from market_intelligence.collection.readiness import AdequacyPolicy, assess_family
from market_intelligence.collection.syndication import (
    SyndicationProvider,
    parse_feed,
    query_terms,
)
from market_intelligence.collection.telemetry import CollectionState
from market_intelligence.corrections import (
    REDISCOVERY_DUPLICATE_PRE_FIX,
    CorrectionStatus,
    EventCorrection,
)
from market_intelligence.models import EventSignal, EventType, SignalCategory
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import CollectionProfile, collect_once
from market_intelligence.storage import IntelligenceStore

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
WINDOW = (NOW - timedelta(days=30), NOW)


def rss(*items: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?><rss version="2.0"><channel><title>f</title>'
        + "".join(items)
        + "</channel></rss>"
    ).encode()


def entry(title: str, link: str, description: str, published: str = "Thu, 03 Sep 2026 09:30:00 GMT") -> str:
    return (
        f"<item><title>{title}</title><link>{link}</link>"
        f"<description>{description}</description><pubDate>{published}</pubDate></item>"
    )


SEC_FEED = rss(
    entry(
        "Securities and Exchange Commission announces bitcoin regulation decision",
        "https://sec.example.gov/news/2026/decision",
        "The Securities and Exchange Commission announced a bitcoin regulation decision.",
    )
)


def minimal(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "name": "silent-empty",
        "feeds": ["sec-press"],
        "watchlist": [
            {
                "canonical_name": "SEC",
                "aliases": ["Securities and Exchange Commission"],
                "entity_type": "REGULATOR",
                "topics": ["bitcoin regulation"],
                "expected_event_types": ["REGULATION"],
            }
        ],
    }
    raw.update(overrides)
    return raw


def offline(monkeypatch: pytest.MonkeyPatch, payload: bytes = SEC_FEED) -> None:
    from market_intelligence.collection import syndication

    monkeypatch.setattr(syndication, "_default_opener", lambda url, timeout, agent=None: payload)


# ---------------------------------------------------------------------- 1


def test_1_rediscovery_never_moves_first_availability(tmp_path: Path, monkeypatch) -> None:
    """`INSERT OR REPLACE` rewrote availability to the latest collection time,
    so a replay at any past origin saw nothing: the corpus reported that nothing
    had ever been available before now."""
    offline(monkeypatch)
    paths = StoragePaths.from_environment(tmp_path / "s").ensure()
    profile = CollectionProfile.from_mapping(minimal())

    collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)
    store = IntelligenceStore(paths.database)
    before = {d.document_id: d.available_at for d in store.documents_as_of(NOW)}
    store.close()
    assert before, "nothing collected; the test proves nothing"

    later = NOW + timedelta(days=3)
    collect_once(paths, profile, now=lambda: later, require_free_bytes=1)
    store = IntelligenceStore(paths.database)
    try:
        after = {d.document_id: d.available_at for d in store.documents_as_of(later)}
    finally:
        store.close()
    assert {k: after[k] for k in before} == before


# ---------------------------------------------------------------------- 2


def test_2_a_new_extractor_version_cannot_overwrite_an_old_event() -> None:
    """Identity omitted the extractor version, so v2 silently replaced v1 and
    the corpus lost the events a published result was computed on."""
    args = (["doc-1"], EventType.REGULATION, NOW)
    assert EventSignal.stable_id(*args, "rules-v1") != EventSignal.stable_id(*args, "rules-v2")


# ---------------------------------------------------------------------- 3


def test_3_atom_feeds_are_not_silently_uncollectable() -> None:
    """The parser looked only for RSS `<item>`, so a regulator publishing Atom
    produced a successful, empty collection forever."""
    atom = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry><title>Commission charges firm</title>"
        "<link rel='alternate' href='https://regulator.example.gov/1'/>"
        "<summary>bitcoin custody failures</summary></entry></feed>"
    ).encode()
    assert len(parse_feed(atom)) == 1


# ---------------------------------------------------------------------- 4


def test_4_the_lookback_window_is_tested_against_publication_not_availability(monkeypatch) -> None:
    """Availability is retrieval, which is necessarily after the planning
    instant, so `start <= available <= end` could never pass. Every entry was
    discarded and the provider reported success."""
    provider = SyndicationProvider(
        feeds_by_id("sec-press"),
        now=lambda: NOW,
        retry=RetryPolicy(max_attempts=1),
        opener=lambda url, timeout: SEC_FEED,
    )
    documents = provider.search("bitcoin regulation", NOW - timedelta(days=1), NOW)
    assert documents, "the availability comparison has regressed"
    assert documents[0].available_at == NOW
    assert documents[0].available_at > NOW - timedelta(days=1)


# ---------------------------------------------------------------------- 5


def test_5_planner_quoting_does_not_kill_every_match() -> None:
    """The planner emits `"SEC" enforcement`; feed text never contains quote
    characters, so every term failed to match."""
    assert query_terms('"SEC" enforcement') == ["sec", "enforcement"]
    assert '"' not in "".join(query_terms('"Federal Reserve" monetary policy'))


# ---------------------------------------------------------------------- 6


def test_6_the_corpus_view_includes_the_cycle_that_just_ran(tmp_path: Path, monkeypatch) -> None:
    """The snapshot was taken at cycle start, so this run's own documents --
    which become available during the run -- were correctly excluded, leaving
    every snapshot one cycle behind its evidence."""
    offline(monkeypatch)
    paths = StoragePaths.from_environment(tmp_path / "s").ensure()
    outcome = collect_once(
        paths, CollectionProfile.from_mapping(minimal()), now=lambda: NOW, require_free_bytes=1
    )
    assert outcome.result is not None
    assert outcome.result.snapshot is not None, "the cycle registered no snapshot"
    assert outcome.result.snapshot.document_count >= 1


# ---------------------------------------------------------------------- 7


def test_7_repeated_retrieval_does_not_manufacture_events(tmp_path: Path, monkeypatch) -> None:
    """Event identity read the re-retrieved document's availability rather than
    the stored one, so an unchanged document produced a new event every cycle --
    56 a week at the deployed cadence."""
    offline(monkeypatch)
    paths = StoragePaths.from_environment(tmp_path / "s").ensure()
    profile = CollectionProfile.from_mapping(minimal())
    for cycle in range(6):
        collect_once(
            paths, profile, now=lambda c=cycle: NOW + timedelta(hours=3 * c), require_free_bytes=1
        )
    store = IntelligenceStore(paths.database)
    try:
        horizon = NOW + timedelta(days=2)
        documents = store.documents_as_of(horizon)
        events = store.signals_as_of(horizon)
    finally:
        store.close()
    assert len(documents) == 1
    assert len(events) <= len(documents), f"{len(events)} events from {len(documents)} documents"


# ---------------------------------------------------------------------- 8


def test_8_deduplication_is_not_reported_as_quarantine(tmp_path: Path, monkeypatch) -> None:
    """A healthy run reported twelve quarantined records against an empty
    quarantine table. The field carried the duplicate count."""
    offline(monkeypatch)
    paths = StoragePaths.from_environment(tmp_path / "s").ensure()
    watchlist = [
        dict(minimal()["watchlist"][0], topics=["bitcoin regulation", "regulation decision"])  # type: ignore[index]
    ]
    outcome = collect_once(
        paths,
        CollectionProfile.from_mapping(minimal(watchlist=watchlist)),
        now=lambda: NOW,
        require_free_bytes=1,
    )
    assert outcome.result is not None
    manifest = outcome.result.manifest
    assert manifest.documents_deduplicated >= 1, "no duplication occurred; test is vacuous"
    assert manifest.quarantined == 0


# ---------------------------------------------------------------------- 9


def test_9_the_quarantine_watchdog_is_not_fed_a_constant_zero(tmp_path: Path, monkeypatch) -> None:
    """`quarantined_last_24h` defaulted to zero on the deployed path, so the
    alert could not fire however bad collection became."""
    from market_intelligence.cli import _ops_report
    from market_intelligence.operations import QuarantineRecord

    offline(monkeypatch)
    paths = StoragePaths.from_environment(tmp_path / "s").ensure()
    collect_once(
        paths, CollectionProfile.from_mapping(minimal()), now=lambda: NOW, require_free_bytes=1
    )
    store = IntelligenceStore(paths.database)
    try:
        store.put_quarantine(
            [
                QuarantineRecord.from_raw(
                    f"f{i}", "syndication", datetime.now(timezone.utc), "boom", "PROVIDER_FAILURE"
                )
                for i in range(40)
            ]
        )
        report = _ops_report(store, paths.database, str(paths.root))
    finally:
        store.close()
    assert any(a["code"] == "QUARANTINE_SPIKE" for a in report["watchdog"]["alerts"])


# --------------------------------------------------------------------- 10


def test_10_an_http_failure_is_not_swallowed() -> None:
    """A classified failure must reach provider health, not vanish into a
    `continue` that leaves the cycle looking successful."""
    def dead(url: str, timeout: float) -> bytes:
        raise ProviderFailure(FailureClass.PERMANENT, f"HTTP 404 for {url}")

    provider = SyndicationProvider(
        feeds_by_id("sec-press"), now=lambda: NOW, retry=RetryPolicy(max_attempts=1), opener=dead
    )
    # B5.1: nor into an empty list the retrieval layer would record as a
    # successful attempt. A search that could read no feed fails.
    with pytest.raises(ProviderFailure, match="no feed could be read"):
        provider.search("bitcoin", *WINDOW)
    assert provider.last_attempts["sec-press"].failures, "the failure was swallowed"


# --------------------------------------------------------------------- 11


def test_11_a_narrowly_repairable_feed_is_not_dropped_whole() -> None:
    """One bare ampersand in a respondent's name cost the entire 25-entry SEC
    feed, every cycle, while the run reported success."""
    payload = rss(
        entry("X METAVERSE INC.", "https://sec.example.gov/a", "X METAVERSE INC."),
        entry("TIAA-CREF Individual & Institutional", "https://sec.example.gov/b", "d"),
    )
    with pytest.raises(ElementTree.ParseError):
        ElementTree.fromstring(payload.decode())
    assert len(parse_feed(payload)) == 2


def test_11b_an_unrepairable_feed_records_a_schema_failure() -> None:
    """And when the repair genuinely cannot help, that is reported rather than
    silently continued."""
    provider = SyndicationProvider(
        feeds_by_id("sec-press"),
        now=lambda: NOW,
        retry=RetryPolicy(max_attempts=1),
        opener=lambda url, timeout: b"<rss><channel><item><title>x</channel></rss>",
    )
    with pytest.raises(ProviderFailure) as raised:
        provider.search("bitcoin", *WINDOW)
    assert raised.value.failure_class is FailureClass.SCHEMA
    assert provider.last_attempts["sec-press"].failures == (FailureClass.SCHEMA,)


# --------------------------------------------------------------------- 12


def test_12_substring_false_positives_do_not_pollute_matching() -> None:
    """`us`, from `"US Treasury"`, matched inside "Anno-us-nces" and
    "B-us-iness" -- 4.7% of live candidate matches were unrelated releases."""
    pattern = term_pattern(["us", "regulation"])
    assert pattern.search("sec announces a roundtable") is None
    assert pattern.search("small business advisory") is None
    assert pattern.search("the us treasury") is not None
    assert pattern.search("new regulations issued") is not None


# --------------------------------------------------------------------- 13


def test_13_publisher_diversity_can_actually_reach_its_threshold() -> None:
    """Counting `max(per-cluster publishers)` measured per-event corroboration.
    Official feeds have one publisher per announcement, so the clause sat at 1
    forever and the gate could never open however long collection ran."""
    documents = [
        fixture_document(
            url=f"https://p{i}.example.gov/{i}",
            title=f"SEC bitcoin regulation notice {i}",
            retrieved_at=NOW + timedelta(days=i),
            publisher=f"publisher-{i}.example.gov",
            body=f"b{i}",
        )
        for i in range(3)
    ]
    events = [
        EventSignal(
            event_id=EventSignal.stable_id([d.document_id], EventType.REGULATION, d.available_at, "rules-v1"),
            event_time=d.available_at,
            available_time=d.available_at,
            source_ids=(d.document_id,),
            category=SignalCategory.WEB_EVENT,
            entity="SEC",
            event_type=EventType.REGULATION,
            sentiment=0.0,
            btc_relevance=0.55,
            novelty=0.5,
            confidence=0.5,
            expected_horizon_hours=24,
            summary=f"notice {i}",
            extractor_version="rules-v1",
        )
        for i, d in enumerate(documents)
    ]
    clusters = cluster_events(events, documents, window_hours=1)
    policy = AdequacyPolicy(
        minimum_events=1,
        minimum_effective_events=1,
        minimum_publishers=3,
        minimum_providers=1,
        minimum_span_days=1,
        horizon_hours=1,
        minimum_coverage_fraction=0.0,
    )
    outcome = assess_family("regulation", clusters, policy=policy, coverage_fraction=1.0)
    assert outcome.ready, f"the publisher clause is unsatisfiable again: {outcome.unmet}"


# --------------------------------------------------------------------- 14


def test_14_corrected_events_stop_counting_as_eligible(tmp_path: Path, monkeypatch) -> None:
    """An invalidated observation must leave the research view while staying
    physically present -- otherwise a correction is decoration."""
    offline(monkeypatch)
    paths = StoragePaths.from_environment(tmp_path / "s").ensure()
    collect_once(
        paths, CollectionProfile.from_mapping(minimal()), now=lambda: NOW, require_free_bytes=1
    )
    store = IntelligenceStore(paths.database)
    try:
        horizon = NOW + timedelta(days=1)
        events = store.signals_as_of(horizon)
        assert events, "nothing extracted; the test proves nothing"
        store.put_corrections(
            [
                EventCorrection(
                    event_id=events[0].event_id,
                    status=CorrectionStatus.INVALIDATED,
                    reason=REDISCOVERY_DUPLICATE_PRE_FIX,
                    invalidated_at=horizon,
                    invalidated_by_version="test",
                    source_bug="test",
                )
            ]
        )
        assert len(store.signals_as_of(horizon)) == len(events)
        assert len(store.eligible_signals_as_of(horizon)) == len(events) - 1
    finally:
        store.close()


# ------------------------------------------------------- F12: quiet vs broken


class TestQuietIsDistinguishableFromBroken:
    """The distinction that took longest to see. "0 documents" is the same
    number for a dead endpoint, an empty feed, a query that no longer describes
    the source, and a week with no news."""

    def _diagnose(self, payload_or_error, query: str = "bitcoin regulation"):
        if isinstance(payload_or_error, bytes):
            opener = lambda url, timeout: payload_or_error  # noqa: E731
        else:
            def opener(url: str, timeout: float) -> bytes:
                raise payload_or_error

        provider = SyndicationProvider(
            feeds_by_id("sec-press"),
            now=lambda: NOW,
            retry=RetryPolicy(max_attempts=1),
            opener=opener,
        )
        if isinstance(payload_or_error, bytes):
            provider.search(query, NOW - timedelta(hours=24), NOW)
        else:
            # B5.1: a search that could read no feed fails rather than returning
            # an empty list; the feed's diagnosis is recorded either way.
            with pytest.raises(ProviderFailure):
                provider.search(query, NOW - timedelta(hours=24), NOW)
        return provider.diagnostics()[0]

    def test_a_dead_endpoint_reads_as_broken(self) -> None:
        diagnosis = self._diagnose(ProviderFailure(FailureClass.PERMANENT, "HTTP 404"))
        assert diagnosis.state is CollectionState.BROKEN
        assert not diagnosis.healthy

    def test_an_empty_feed_is_not_broken_but_is_not_expected_either(self) -> None:
        diagnosis = self._diagnose(rss())
        assert diagnosis.state is CollectionState.EMPTY_FEED
        assert diagnosis.expected is False

    def test_a_genuinely_quiet_feed_reads_as_no_topical_match(self) -> None:
        payload = rss(entry("Road maintenance schedule", "https://x.test/1", "roads"))
        diagnosis = self._diagnose(payload)
        assert diagnosis.state is CollectionState.NO_TOPICAL_MATCH
        assert diagnosis.entries == 1 and diagnosis.matched == 0

    def test_old_matches_read_as_historical_match_only(self) -> None:
        """The feed works and the query works; there is simply nothing new.
        Widening the query would fix nothing, and is the tempting wrong move."""
        payload = rss(
            entry(
                "SEC bitcoin regulation decision",
                "https://x.test/2",
                "bitcoin regulation",
                published="Mon, 01 Jan 2024 09:00:00 GMT",
            )
        )
        diagnosis = self._diagnose(payload)
        assert diagnosis.state is CollectionState.HISTORICAL_MATCH_ONLY
        assert diagnosis.matched == 1 and diagnosis.admitted == 0
        assert "widening the query" in diagnosis.detail

    def test_a_working_feed_reads_as_collecting(self) -> None:
        diagnosis = self._diagnose(SEC_FEED)
        assert diagnosis.state is CollectionState.COLLECTING
        assert diagnosis.admitted >= 1

    def test_a_non_subject_bearing_stream_reports_zero_as_expected(self) -> None:
        """Zero from the administrative feed is a property of the source. It
        must not read as a fault, or someone will loosen the filter."""
        payload = rss(entry("Rebus Holdings, Inc.", "https://x.test/3", "Rebus Holdings, Inc."))
        provider = SyndicationProvider(
            feeds_by_id("sec-admin-proceedings"),
            now=lambda: NOW,
            retry=RetryPolicy(max_attempts=1),
            opener=lambda url, timeout: payload,
        )
        provider.search("bitcoin regulation", NOW - timedelta(hours=24), NOW)
        diagnosis = provider.diagnostics()[0]
        assert diagnosis.state is CollectionState.NO_TOPICAL_MATCH
        assert diagnosis.expected is True
        assert "respondent identity" in diagnosis.detail

    def test_every_state_is_distinct(self) -> None:
        """Five outcomes, five codes. If any two collapsed, an operator would
        be back to reading one number for four situations."""
        assert len({state.value for state in CollectionState}) == 5
