"""B4.1.3 — a catalogue of official and primary feeds, and their declarations.

These are the sources B4's hypotheses actually need: the regulator that issues
the decision, the central bank that sets the policy, the company that announces
the purchase. All are documented public syndication endpoints their publishers
operate for exactly this purpose.

Two disclaimers that belong in the code rather than in a README nobody reads:

**These URLs are not guaranteed stable.** Publishers move and retire feeds. A
feed that 404s is a `PERMANENT` failure the health report surfaces; it is not a
silent gap. Every entry here is a *default suggestion* the operator confirms,
and nothing is enabled without them saying so.

**Classification is an assertion, not a fact derived from the URL.** Marking the
SEC's litigation feed `primary_source` says the SEC is the party the news is
about. That is the operator's claim, recorded so a study can separate an
announcement from its coverage — and so a wrong claim is visible and arguable
rather than buried in a heuristic.

Nothing here is enabled by default. `default_feed_catalogue()` returns
descriptions; turning any of them on is a deliberate configuration step.
"""

from __future__ import annotations

from ..models import SourceType
from .policy import ProviderDeclaration, ProviderPolicy, RawRetention
from .syndication import FeedSource

#: Regulators and central banks. The highest-value primary sources for B4's
#: regulation, monetary-policy and enforcement hypotheses.
OFFICIAL_FEEDS: tuple[FeedSource, ...] = (
    FeedSource(
        feed_id="sec-press",
        url="https://www.sec.gov/news/pressreleases.rss",
        publisher="U.S. Securities and Exchange Commission",
        source_type=SourceType.PRIMARY_OFFICIAL,
        primary_source=True,
        official_source=True,
        retention=RawRetention.FULL,
        timestamp_quality=0.9,
    ),
    FeedSource(
        # DEAD as of 2026-09-03: HTTP 404. The SEC retired this path. Its
        # administrative-proceedings feed at /rss/litigation/admin.xml is alive
        # and is *not* a drop-in replacement -- administrative proceedings and
        # civil litigation releases are different things, and quietly swapping
        # one for the other would change what every study built on this feed
        # was measuring. Enabling it is a decision, not a repair.
        # The entry stays so a manifest naming this feed can still be read.
        feed_id="sec-litigation",
        url="https://www.sec.gov/rss/litigation/litreleases.xml",
        publisher="U.S. Securities and Exchange Commission",
        source_type=SourceType.PRIMARY_OFFICIAL,
        primary_source=True,
        official_source=True,
        retention=RawRetention.FULL,
        timestamp_quality=0.9,
    ),
    FeedSource(
        feed_id="federalreserve-press",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        publisher="Board of Governors of the Federal Reserve System",
        source_type=SourceType.PRIMARY_OFFICIAL,
        primary_source=True,
        official_source=True,
        retention=RawRetention.FULL,
        timestamp_quality=0.9,
    ),
    FeedSource(
        feed_id="federalreserve-monetary",
        url="https://www.federalreserve.gov/feeds/press_monetary.xml",
        publisher="Board of Governors of the Federal Reserve System",
        source_type=SourceType.PRIMARY_OFFICIAL,
        primary_source=True,
        official_source=True,
        retention=RawRetention.FULL,
        timestamp_quality=0.9,
    ),
    FeedSource(
        feed_id="cftc-press",
        url="https://www.cftc.gov/RSS/RSSGP/rssgp.xml",
        publisher="U.S. Commodity Futures Trading Commission",
        source_type=SourceType.PRIMARY_OFFICIAL,
        primary_source=True,
        official_source=True,
        retention=RawRetention.FULL,
        timestamp_quality=0.9,
    ),
    FeedSource(
        # DEAD as of 2026-09-03: HTTP 404, served slowly enough (35.7s) that a
        # 20s client reads it as a timeout. Treasury moved its newsroom and no
        # replacement feed responded within a sane bound.
        feed_id="treasury-press",
        url="https://home.treasury.gov/rss/press.xml",
        publisher="U.S. Department of the Treasury",
        source_type=SourceType.PRIMARY_OFFICIAL,
        primary_source=True,
        official_source=True,
        retention=RawRetention.FULL,
        timestamp_quality=0.9,
    ),
    FeedSource(
        feed_id="bls-news",
        url="https://www.bls.gov/feed/bls_latest.rss",
        publisher="U.S. Bureau of Labor Statistics",
        source_type=SourceType.PRIMARY_OFFICIAL,
        primary_source=True,
        official_source=True,
        retention=RawRetention.FULL,
        timestamp_quality=0.9,
    ),
)


#: Confirmed 404 by live probe on 2026-09-03. Kept in the catalogue so an old
#: manifest naming one is still resolvable, and kept out of the deployed profile
#: so the collector does not spend every cycle failing at them -- a PERMANENT
#: failure that recurs forever is noise, and noise is how a real failure gets
#: ignored.
RETIRED_FEED_IDS: frozenset[str] = frozenset({"sec-litigation", "treasury-press"})


def default_feed_catalogue() -> tuple[FeedSource, ...]:
    """Suggested official feeds. Returned as data; enabled only by an operator."""
    return OFFICIAL_FEEDS


def feeds_by_id(*feed_ids: str) -> tuple[FeedSource, ...]:
    """Select named feeds. Unknown ids are an error, not a silent omission."""
    index = {feed.feed_id: feed for feed in OFFICIAL_FEEDS}
    missing = [feed_id for feed_id in feed_ids if feed_id not in index]
    if missing:
        raise KeyError(f"unknown feed ids: {', '.join(sorted(missing))}")
    return tuple(index[feed_id] for feed_id in feed_ids)


SYNDICATION_DECLARATION = ProviderDeclaration(
    provider_id="syndication",
    policy=ProviderPolicy.PUBLIC_DOCUMENTED,
    purpose=(
        "Collect announcements from RSS/Atom feeds the publisher operates for public "
        "consumption. Primary and official sources are classified explicitly so an "
        "announcement can be separated from coverage of it."
    ),
    credentials_env=None,
    requires_paid_contract=False,
    rate_limit_note=(
        "Public feeds carry no published quota, but government sites do rate-limit and "
        "some require a descriptive User-Agent. Poll no faster than research needs. "
        "Third-party terms change; confirm before enabling."
    ),
    minimum_interval_seconds=900,
    data_returned="entry title, link, summary, publication timestamp, author when present",
    raw_retention=RawRetention.FULL,
    primary_source=True,
    official_source=True,
    terms_note=(
        "Only URLs the operator configures are fetched. No crawling, no link-following, "
        "no discovery. The allowlist is the policy."
    ),
)

NEWS_API_DECLARATION = ProviderDeclaration(
    provider_id="news-api",
    policy=ProviderPolicy.AUTHENTICATED_LICENSED,
    purpose="Broad news coverage through a contracted JSON search API.",
    credentials_env="BTC_INTEL_SEARCH_API_KEY",
    requires_paid_contract=True,
    rate_limit_note="Set by the vendor contract; configure minimum_interval_seconds to match.",
    minimum_interval_seconds=3600,
    data_returned="article title, url, publisher, snippet, publication timestamp",
    raw_retention=RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE,
    primary_source=False,
    official_source=False,
    terms_note=(
        "Most news APIs forbid redistributing article text. Only a hash and permitted "
        "metadata leave this machine."
    ),
)


__all__ = [
    "NEWS_API_DECLARATION",
    "RETIRED_FEED_IDS",
    "OFFICIAL_FEEDS",
    "SYNDICATION_DECLARATION",
    "default_feed_catalogue",
    "feeds_by_id",
]
