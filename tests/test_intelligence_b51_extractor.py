"""B5.1 — rules-v2's extraction confidence is derived from evidence.

B5 found that rules-v1 gives every event confidence 0.35, below B4's floor of
0.5, so the deployed collector could never produce a countable event. The fix is
not a bigger constant. Confidence has to express how well the document supports
the extracted event -- its type, its entity, the context that corroborates or
contests the type, the source, the timestamp -- and nothing about whether the
event will move BTC.

Pinned here:

* confidence varies with the evidence, and ambiguous, contested or malformed
  extractions fall below the floor while well-supported ones clear it;
* identical input under one version gives identical output, whatever else is in
  the batch;
* nothing in the extractor can reach market data;
* rules-v1 is unchanged, so the events already in the corpus keep their meaning;
* the CFTC release rules-v1 labelled monetary policy is a regulation under
  rules-v2, and "sec" inside "second" is not the SEC.

Entities and expected event types come from the committed collection profile, so
these tests follow the deployed watchlist. No network.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence import extractors
from market_intelligence.b5.contracts import DEFAULT_SUFFICIENCY
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.extractors import (
    CONFIDENCE_WEIGHTS,
    CURRENT_RULE_EXTRACTOR_VERSION,
    ROUTING_NOVELTY,
    ROUTING_RELEVANCE,
    ROUTING_SENTIMENT,
    EvidenceRuleExtractor,
    RuleBasedExtractor,
)
from market_intelligence.models import DisclosureStream, Document, EventType, SourceMetadata, SourceType

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / "deploy" / "collection-profile.json").read_text(encoding="utf-8"))
ENTITIES = {entry["canonical_name"]: tuple(entry.get("aliases", ())) for entry in PROFILE["watchlist"]}
EXPECTED = {
    entry["canonical_name"]: tuple(EventType(value) for value in entry.get("expected_event_types", ()))
    for entry in PROFILE["watchlist"]
}
FLOOR = DEFAULT_SUFFICIENCY.minimum_confidence
RETRIEVED = datetime(2026, 9, 3, 9, 15, 13, tzinfo=timezone.utc)

SEC = "U.S. Securities and Exchange Commission"
CFTC = "U.S. Commodity Futures Trading Commission"


def official(title: str, publisher: str, *, published: datetime | None = RETRIEVED - timedelta(hours=17), stream: DisclosureStream = DisclosureStream.REGULATORY_ANNOUNCEMENT) -> Document:
    doc = fixture_document(
        url=f"https://{publisher.split()[-1].lower()}.example.gov/{abs(hash(title)) % 10**8}",
        title=title,
        retrieved_at=RETRIEVED,
        publisher=publisher,
        published_at=published,
        primary_source=True,
        official_source=True,
    )
    return doc.model_copy(update={"source_metadata": doc.source_metadata.model_copy(update={"disclosure_stream": stream})})


def unknown_source(title: str, *, published: datetime | None = None) -> Document:
    doc = fixture_document(url=f"https://blog.example.com/{abs(hash(title)) % 10**8}", title=title, retrieved_at=RETRIEVED, publisher="some blog", published_at=published)
    return doc.model_copy(update={"source_metadata": SourceMetadata(source_type=SourceType.UNKNOWN, known_publisher=False, timestamp_quality=0.3)})


@pytest.fixture(scope="module")
def v2() -> EvidenceRuleExtractor:
    return EvidenceRuleExtractor(ENTITIES, EXPECTED)


def confidence(extractor: EvidenceRuleExtractor, document: Document) -> float:
    evidence = extractor.explain(document)
    assert evidence is not None, document.title
    return evidence.confidence


class TestTheContract:
    def test_the_weights_are_declared_and_sum_to_one(self) -> None:
        assert set(CONFIDENCE_WEIGHTS) == {"type_evidence", "context_agreement", "entity_certainty", "source_reliability", "timestamp_certainty"}
        assert sum(CONFIDENCE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_the_collector_version_is_rules_v2_and_v1_is_untouched(self) -> None:
        assert CURRENT_RULE_EXTRACTOR_VERSION == EvidenceRuleExtractor.version == "rules-v2"
        assert RuleBasedExtractor.version == "rules-v1"
        doc = official("SEC Charges Fund Executives with Fraud", SEC)
        (event,) = RuleBasedExtractor(ENTITIES).extract([doc])
        assert (event.confidence, event.btc_relevance, event.novelty, event.sentiment) == (0.35, 0.55, 0.5, 0.0)


class TestConfidenceFollowsTheEvidence:
    def test_it_is_not_a_constant(self, v2: EvidenceRuleExtractor) -> None:
        documents = [
            official("SEC Charges Fund Executives with Ponzi-Like Scheme", SEC),
            official("CFTC Issues Final Rule to Modify Clearing Requirement for Interest Rate Swaps", CFTC),
            official("CFTC Comments on Interest Rate Benchmarks", CFTC),
            unknown_source("Crypto rule and interest rate chatter"),
            official("SEC Enforcement: Charges and Rule Changes", SEC, published=None),
        ]
        values = {confidence(v2, doc) for doc in documents}
        assert len(values) == len(documents), values

    def test_a_well_supported_extraction_clears_the_floor(self, v2: EvidenceRuleExtractor) -> None:
        evidence = v2.explain(official("SEC Charges San Francisco Bay Area Private Fund Executives with Ponzi-Like Scheme", SEC))
        assert evidence is not None
        assert (evidence.event_type, evidence.entity, evidence.entity_source) == (EventType.REGULATION, "SEC", "title")
        assert evidence.confidence >= 0.85 > FLOOR

    def test_repeated_type_evidence_is_worth_more_than_one_term(self, v2: EvidenceRuleExtractor) -> None:
        one = confidence(v2, official("SEC Charges Fund Executives", SEC))
        two = confidence(v2, official("SEC Charges Fund Executives Under New Rule", SEC))
        assert two > one

    def test_a_contested_type_falls_below_the_floor(self, v2: EvidenceRuleExtractor) -> None:
        """The CFTC declares only REGULATION; a title that only says 'interest rate' contests it."""
        evidence = v2.explain(official("CFTC Comments on Interest Rate Benchmarks", CFTC))
        assert evidence is not None and evidence.event_type is EventType.MONETARY_POLICY
        assert evidence.contested and evidence.confidence < FLOOR

    def test_unresolved_ambiguity_falls_below_the_floor(self, v2: EvidenceRuleExtractor) -> None:
        evidence = v2.explain(unknown_source("Crypto rule and interest rate chatter"))
        assert evidence is not None and evidence.factors["type_evidence"] == 0.0
        assert evidence.entity is None and evidence.confidence < FLOOR

    def test_a_missing_timestamp_lowers_it(self, v2: EvidenceRuleExtractor) -> None:
        with_time = confidence(v2, official("SEC Charges Fund Executives", SEC))
        without = confidence(v2, official("SEC Charges Fund Executives", SEC, published=None))
        assert without < with_time

    def test_a_timestamp_after_retrieval_counts_as_no_timestamp(self, v2: EvidenceRuleExtractor) -> None:
        evidence = v2.explain(official("SEC Charges Fund Executives", SEC, published=RETRIEVED + timedelta(hours=2)))
        assert evidence is not None and evidence.factors["timestamp_certainty"] == 0.0

    def test_an_unknown_source_lowers_it(self, v2: EvidenceRuleExtractor) -> None:
        """Same title, same absence of context; only the source differs. (Published by the
        SEC, 'hack' would be contested: the SEC is declared never to produce one.)"""
        trusted = confidence(v2, official("Exchange hack drains hot wallet", "Example Exchange Operator"))
        unknown = confidence(v2, unknown_source("Exchange hack drains hot wallet", published=RETRIEVED - timedelta(hours=3)))
        assert unknown < trusted

    def test_a_malformed_extraction_falls_below_the_floor(self, v2: EvidenceRuleExtractor) -> None:
        """No timestamp, unknown source, no entity, and two types with nothing to choose between them."""
        assert confidence(v2, unknown_source("etf hack")) < FLOOR

    def test_a_stream_that_contradicts_the_type_lowers_it(self, v2: EvidenceRuleExtractor) -> None:
        agrees = confidence(v2, official("Federal Reserve Statement on Monetary Policy", "Board of Governors of the Federal Reserve System", stream=DisclosureStream.MONETARY_POLICY))
        disagrees = confidence(v2, official("Federal Reserve Statement on Monetary Policy", "Board of Governors of the Federal Reserve System", stream=DisclosureStream.ECONOMIC_STATISTICS))
        assert disagrees < agrees

    def test_confidence_is_bounded(self, v2: EvidenceRuleExtractor) -> None:
        for title in ("SEC Charges Charges Rule Rule Enforcement", "etf", "whale"):
            value = confidence(v2, official(title, SEC))
            assert 0.0 <= value <= 1.0


class TestDeterminism:
    def test_identical_input_gives_identical_output(self, v2: EvidenceRuleExtractor) -> None:
        doc = official("CFTC Issues Final Rule to Modify Clearing Requirement for Interest Rate Swaps", CFTC)
        assert v2.extract([doc]) == v2.extract([doc])
        assert EvidenceRuleExtractor(ENTITIES, EXPECTED).explain(doc) == v2.explain(doc)

    def test_the_rest_of_the_batch_changes_nothing(self, v2: EvidenceRuleExtractor) -> None:
        target = official("SEC Charges Fund Executives", SEC)
        others = [official("CFTC Issues Final Rule", CFTC), unknown_source("etf hack"), official("Whale moves coins", SEC)]
        alone = v2.extract([target])
        together = [e for e in v2.extract([*others, target]) if e.source_ids == (target.document_id,)]
        assert alone == together

    def test_the_version_is_part_of_the_identity(self, v2: EvidenceRuleExtractor) -> None:
        doc = official("SEC Charges Fund Executives", SEC)
        (v1_event,) = RuleBasedExtractor(ENTITIES).extract([doc])
        (v2_event,) = v2.extract([doc])
        assert v1_event.event_id != v2_event.event_id
        assert (v1_event.extractor_version, v2_event.extractor_version) == ("rules-v1", "rules-v2")


class TestNoMarketData:
    ALLOWED_IMPORTS = {"__future__", "json", "re", "abc", "collections.abc", "datetime", "typing", "pydantic", "models"}

    def test_the_extractor_module_imports_nothing_that_could_see_a_price(self) -> None:
        tree = ast.parse(Path(extractors.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert imported <= self.ALLOWED_IMPORTS, imported - self.ALLOWED_IMPORTS

    def test_extraction_takes_documents_and_nothing_else(self) -> None:
        assert list(inspect.signature(EvidenceRuleExtractor.extract).parameters) == ["self", "documents"]
        assert list(inspect.signature(EvidenceRuleExtractor.explain).parameters) == ["self", "document"]


class TestTheCftcRegression:
    """The release B5 found labelled MONETARY_POLICY, and its sibling rules-v1 missed."""

    SWAPS = "CFTC Issues Final Rule to Modify Clearing Requirement for Canadian Dollar- and Mexican Peso-Denominated Interest Rate Swaps"
    NO_ACTION = "CFTC Staff Issues No-Action Position on Large Trader Reporting for Direct Participants"

    def test_rules_v1_still_says_monetary_policy(self) -> None:
        """Pinned, so the event already in the corpus keeps the interpretation it was written with."""
        (event,) = RuleBasedExtractor(ENTITIES).extract([official(self.SWAPS, CFTC)])
        assert event.event_type is EventType.MONETARY_POLICY and event.entity == "CFTC"

    def test_rules_v2_says_regulation_because_the_context_singles_it_out(self, v2: EvidenceRuleExtractor) -> None:
        evidence = v2.explain(official(self.SWAPS, CFTC))
        assert evidence is not None
        assert evidence.candidates == {"REGULATION": 1, "MONETARY_POLICY": 1}
        assert evidence.event_type is EventType.REGULATION and evidence.entity == "CFTC" and not evidence.contested
        assert evidence.confidence >= FLOOR

    def test_the_no_action_position_rules_v1_missed_is_an_event(self, v2: EvidenceRuleExtractor) -> None:
        assert RuleBasedExtractor(ENTITIES).extract([official(self.NO_ACTION, CFTC)]) == []
        (event,) = v2.extract([official(self.NO_ACTION, CFTC)])
        assert event.event_type is EventType.REGULATION and event.entity == "CFTC"

    def test_an_entity_name_is_not_type_evidence(self, v2: EvidenceRuleExtractor) -> None:
        """rules-v1 made an SEC roundtable a regulation event because the title said 'SEC'."""
        roundtable = official("SEC Announces Agenda and Panelists for Roundtable on Preparations for 24-Hour Trading", SEC)
        assert [e.event_type for e in RuleBasedExtractor(ENTITIES).extract([roundtable])] == [EventType.REGULATION]
        assert v2.extract([roundtable]) == []

    def test_sec_inside_another_word_is_not_the_sec(self, v2: EvidenceRuleExtractor) -> None:
        evidence = v2.explain(unknown_source("Second-quarter rule update from a secure-messaging vendor"))
        assert evidence is not None and evidence.entity is None


class TestFieldsStaySeparate:
    def test_relevance_novelty_and_sentiment_are_declared_routing_defaults(self, v2: EvidenceRuleExtractor) -> None:
        strong = v2.extract([official("SEC Charges Fund Executives Under New Rule", SEC)])[0]
        weak = v2.extract([unknown_source("etf hack")])[0]
        assert strong.confidence != weak.confidence
        for event in (strong, weak):
            assert (event.btc_relevance, event.novelty, event.sentiment) == (ROUTING_RELEVANCE, ROUTING_NOVELTY, ROUTING_SENTIMENT)

    def test_nothing_it_emits_is_a_trading_instruction(self, v2: EvidenceRuleExtractor) -> None:
        (event,) = v2.extract([official("SEC Charges Fund Executives", SEC)])
        assert event.direction.value == "UNKNOWN"
        text = event.model_dump_json().upper()
        assert not any(word in text for word in ('"BUY"', '"SELL"', '"LONG"', '"SHORT"'))


class TestPointInTimeUnchanged:
    def test_event_time_is_the_publication_time_and_availability_is_retrieval(self, v2: EvidenceRuleExtractor) -> None:
        doc = official("SEC Charges Fund Executives", SEC)
        (event,) = v2.extract([doc])
        assert event.event_time == doc.published_at and event.available_time == doc.available_at == RETRIEVED

    def test_an_old_article_retrieved_today_is_available_today(self, v2: EvidenceRuleExtractor) -> None:
        doc = official("SEC Charges Fund Executives", SEC, published=RETRIEVED - timedelta(days=400))
        (event,) = v2.extract([doc])
        assert event.available_time == RETRIEVED and event.event_time < event.available_time
