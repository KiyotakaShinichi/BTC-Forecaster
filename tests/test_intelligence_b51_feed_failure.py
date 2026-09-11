"""B5.1 — a syndication search that could read no feed is a failed attempt.

The syndication provider catches each feed's failure so that one bad feed does
not lose the cycle's other work (B4.1.17). But when *every* feed failed it
returned an empty list, and the retrieval layer recorded a successful attempt
with nothing in it. An outage looked like a quiet news day -- to the run status,
to the watchdog, and to collection coverage, which counts days with a
successful attempt and would have counted that day as collected.

Pinned here:

* one unreadable feed still costs only itself;
* a search that could read no feed -- fetched and parsed -- raises a classified
  failure, and says which feeds failed and how;
* a feed that was read and matched nothing is still a success: a quiet day is
  not an outage;
* the retrieval layer records the failed attempt and does not retry a search
  whose feeds were each already retried;
* on the deployed path the cycle's run is FAILED, not DEGRADED.

No network.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.collection.backoff import FailureClass, ProviderFailure, RetryPolicy
from market_intelligence.collection.feeds import feeds_by_id
from market_intelligence.collection.syndication import SyndicationProvider
from market_intelligence.configuration import ProviderCategory, ProviderConfig, QuerySpec
from market_intelligence.models import EventType
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import CollectionProfile, collect_once
from market_intelligence.retrieval import MultiProviderRetriever
from market_intelligence.storage import IntelligenceStore
from tests.test_intelligence_silent_empty import SEC_FEED, minimal, rss

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
WINDOW = (NOW - timedelta(days=30), NOW)
FEEDS = feeds_by_id("sec-press", "cftc-press")
UNPARSEABLE = b"<rss><channel><item><title>x</channel></rss>"


def opener(failures: dict[str, FailureClass], payload: bytes = SEC_FEED) -> tuple[Callable[[str, float], bytes], list[str]]:
    """Serves `payload` for every feed except those named, which fail with the given class."""
    calls: list[str] = []
    by_url = {feed.url: feed.feed_id for feed in FEEDS}

    def fetch(url: str, timeout: float) -> bytes:
        calls.append(url)
        failure = failures.get(by_url.get(url, ""))
        if failure is not None:
            raise ProviderFailure(failure, f"{failure.value} for {url}")
        return payload

    return fetch, calls


def provider(fetch: Callable[[str, float], bytes]) -> SyndicationProvider:
    return SyndicationProvider(FEEDS, now=lambda: NOW, retry=RetryPolicy(max_attempts=1), opener=fetch)


class TestTheProvider:
    def test_one_unreadable_feed_costs_only_itself(self) -> None:
        fetch, _ = opener({"cftc-press": FailureClass.PERMANENT})
        source = provider(fetch)
        assert source.search("bitcoin", *WINDOW), "the readable feed's work was lost"
        assert source.last_attempts["cftc-press"].failures == (FailureClass.PERMANENT,)

    def test_a_search_that_read_no_feed_fails(self) -> None:
        fetch, _ = opener({"sec-press": FailureClass.PERMANENT, "cftc-press": FailureClass.PERMANENT})
        with pytest.raises(ProviderFailure, match="no feed could be read") as raised:
            provider(fetch).search("bitcoin", *WINDOW)
        assert raised.value.failure_class is FailureClass.PERMANENT
        assert raised.value.retries_exhausted
        assert "cftc-press=PERMANENT" in str(raised.value) and "sec-press=PERMANENT" in str(raised.value)

    def test_feeds_that_disagree_fail_as_transient(self) -> None:
        fetch, _ = opener({"sec-press": FailureClass.PERMANENT, "cftc-press": FailureClass.TRANSIENT})
        with pytest.raises(ProviderFailure) as raised:
            provider(fetch).search("bitcoin", *WINDOW)
        assert raised.value.failure_class is FailureClass.TRANSIENT

    def test_a_feed_fetched_but_unparseable_was_not_read(self) -> None:
        fetch, _ = opener({}, payload=UNPARSEABLE)
        with pytest.raises(ProviderFailure) as raised:
            provider(fetch).search("bitcoin", *WINDOW)
        assert raised.value.failure_class is FailureClass.SCHEMA

    def test_a_feed_read_that_matched_nothing_is_a_quiet_day_not_an_outage(self) -> None:
        fetch, _ = opener({"cftc-press": FailureClass.PERMANENT}, payload=rss())
        assert provider(fetch).search("bitcoin", *WINDOW) == []


CONFIG = ProviderConfig(id="syndication", type="syndication", source_category=ProviderCategory.OFFICIAL_GOVERNMENT)
QUERY = QuerySpec(
    query_id="q1",
    query="bitcoin",
    topic="bitcoin regulation",
    entities=("SEC",),
    event_types=(EventType.REGULATION,),
    lookback_hours=24 * 30,
    priority=50,
    generated_at=NOW,
)


class TestTheRetrievalLayer:
    def test_the_attempt_is_recorded_as_failed_and_not_retried_again(self) -> None:
        fetch, calls = opener({"sec-press": FailureClass.TRANSIENT, "cftc-press": FailureClass.TRANSIENT})
        slept: list[float] = []
        retriever = MultiProviderRetriever(
            {"syndication": provider(fetch)}, {"syndication": CONFIG}, sleep=slept.append, jitter=lambda: 0.0
        )
        (attempt,) = retriever.retrieve([QUERY]).attempts
        assert not attempt.success and attempt.attempts == 1
        assert "no feed could be read" in (attempt.error or "")
        assert slept == [] and len(calls) == len(FEEDS)

    def test_an_attempt_that_read_a_feed_is_still_a_success(self) -> None:
        fetch, _ = opener({"cftc-press": FailureClass.TRANSIENT})
        retriever = MultiProviderRetriever({"syndication": provider(fetch)}, {"syndication": CONFIG}, sleep=lambda _: None)
        (attempt,) = retriever.retrieve([QUERY]).attempts
        assert attempt.success and attempt.documents_received >= 1


def test_a_scheduled_cycle_that_read_no_feed_is_a_failed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_intelligence.collection import syndication

    def retired(url: str, timeout: float, agent: str | None = None) -> bytes:
        raise ProviderFailure(FailureClass.PERMANENT, f"HTTP 404 for {url}")

    monkeypatch.setattr(syndication, "_default_opener", retired)
    paths = StoragePaths.from_environment(tmp_path / "s").ensure()
    collect_once(paths, CollectionProfile.from_mapping(minimal()), now=lambda: NOW, require_free_bytes=1)
    store = IntelligenceStore(paths.database)
    try:
        statuses = [row[0] for row in store.connection.execute("SELECT status FROM runs").fetchall()]
        outcomes = {bool(row[0]) for row in store.connection.execute("SELECT success FROM provider_attempts").fetchall()}
    finally:
        store.close()
    assert statuses == ["FAILED"]
    assert outcomes == {False}
