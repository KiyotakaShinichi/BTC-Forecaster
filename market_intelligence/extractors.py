from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from .models import (
    Direction,
    DisclosureStream,
    Document,
    EventSignal,
    EventType,
    ExtractionMethod,
    SignalCategory,
    TransferContext,
)


class ExtractionError(ValueError):
    pass


def _validate_provenance(signals: list[EventSignal], documents: list[Document]) -> list[EventSignal]:
    sources = {document.document_id: document for document in documents}
    for signal in signals:
        if not set(signal.source_ids) <= sources.keys():
            raise ExtractionError("signal references an unknown source document")
        first_usable = max(sources[source_id].available_at for source_id in signal.source_ids)
        if signal.available_time < first_usable:
            raise ExtractionError("signal cannot be available before its source documents")
    return signals


class EventExtractor(ABC):
    version: str

    @abstractmethod
    def extract(self, documents: list[Document]) -> list[EventSignal]: ...


class FixtureExtractor(EventExtractor):
    def __init__(self, signals: list[EventSignal], version: str = "fixture-v1"):
        self._signals, self.version = signals, version

    def extract(self, documents: list[Document]) -> list[EventSignal]:
        source_ids = {d.document_id for d in documents}
        return _validate_provenance([s for s in self._signals if set(s.source_ids) <= source_ids], documents)


class StructuredLlmExtractor(EventExtractor):
    """Provider-independent validated-JSON boundary around an injected LLM call."""

    def __init__(
        self,
        completion: Callable[[str], str],
        version: str,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._completion, self.version, self._now = completion, version, now

    def extract(self, documents: list[Document]) -> list[EventSignal]:
        prompt = json.dumps(
            {
                "instruction": "Extract events as JSON array only. Never forecast or recommend trades.",
                "schema": EventSignal.model_json_schema(),
                "documents": [d.model_dump(mode="json") for d in documents],
            },
            sort_keys=True,
        )
        try:
            raw = json.loads(self._completion(prompt))
            if not isinstance(raw, list):
                raise ExtractionError("LLM response must be a JSON array")
            # The trusted adapter, not model-authored JSON, records the method.
            signals = [EventSignal.model_validate({**item, "extraction_method": ExtractionMethod.LLM}) for item in raw]
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ExtractionError("invalid structured LLM response") from exc
        for signal in signals:
            if signal.extractor_version != self.version:
                raise ExtractionError("extractor version mismatch")
            if signal.available_time > self._now():
                raise ExtractionError("LLM output has a future availability timestamp")
        return _validate_provenance(signals, documents)


class RuleBasedExtractor(EventExtractor):
    """Conservative routing fallback; hints are lower-confidence than semantic extraction.

    This is `rules-v1`, kept exactly as it was. Its events are in the corpus, and
    event identity carries the extractor version precisely so that old events
    keep their old interpretation. Every event it produces carries the same
    scores -- relevance 0.55, confidence 0.35, novelty 0.5, sentiment 0.0 -- and
    it matches raw substrings, so "interest rate" in "Interest Rate Swaps" made a
    CFTC rule monetary policy and "sec" matched inside "second". The collector
    uses `EvidenceRuleExtractor` (`rules-v2`) instead.
    """

    version = "rules-v1"
    _rules = (
        (("sec", "regulation", "lawsuit"), EventType.REGULATION),
        (("federal reserve", "fed ", "powell", "interest rate"), EventType.MONETARY_POLICY),
        (("etf",), EventType.ETF_FLOW),
        (("hack", "breach", "exploit"), EventType.SECURITY_INCIDENT),
        (("exchange outage", "exchange incident"), EventType.EXCHANGE_INCIDENT),
        (("whale", "large transfer"), EventType.WHALE_TRANSFER),
    )

    def __init__(self, entities: dict[str, tuple[str, ...]] | None = None):
        self.entities = entities or {}

    def extract(self, documents: list[Document]) -> list[EventSignal]:
        output = []
        for doc in documents:
            text = doc.title.casefold()
            event_type = next((event for terms, event in self._rules if any(term in text for term in terms)), None)
            if event_type is None:
                continue
            entity = next(
                (
                    name
                    for name, aliases in self.entities.items()
                    if any(alias.casefold() in text for alias in (name, *aliases))
                ),
                None,
            )
            context = TransferContext.UNKNOWN if event_type == EventType.WHALE_TRANSFER else None
            output.append(
                EventSignal(
                    event_id=EventSignal.stable_id([doc.document_id], event_type, doc.available_at, self.version),
                    event_time=doc.published_at or doc.available_at,
                    available_time=doc.available_at,
                    source_ids=(doc.document_id,),
                    category=SignalCategory.ONCHAIN
                    if event_type == EventType.WHALE_TRANSFER
                    else SignalCategory.WEB_EVENT,
                    entity=entity,
                    event_type=event_type,
                    direction=Direction.UNKNOWN,
                    sentiment=0.0,
                    btc_relevance=0.55,
                    novelty=0.5,
                    confidence=0.35,
                    expected_horizon_hours=24,
                    summary=f"Rule-based hint: {doc.title}",
                    transfer_context=context,
                    extractor_version=self.version,
                    extraction_method=ExtractionMethod.RULE_BASED,
                )
            )
        return _validate_provenance(output, documents)


# -- rules-v2 -------------------------------------------------------------------

#: The rule extractor the collector uses. `rules-v1` stays importable, unchanged,
#: because its events are in the corpus and keep their meaning.
CURRENT_RULE_EXTRACTOR_VERSION = "rules-v2"

#: How much each kind of evidence contributes to rules-v2's extraction confidence.
#: Declared before any rules-v2 event existed and fitted to nothing -- in
#: particular to nothing about BTC, which this module cannot see. Every factor is
#: in [0, 1] and the weights sum to 1, so confidence is too.
#:
#: Type evidence carries the most because it is what the event *is*; without it,
#: an official source with a clean timestamp would clear any floor on reputation
#: alone. Context agreement is independent corroboration of the type. Entity
#: certainty is who the event is about. Source reliability and timestamp
#: certainty describe the document rather than the extraction, so they count least.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "type_evidence": 0.40,
    "context_agreement": 0.20,
    "entity_certainty": 0.20,
    "source_reliability": 0.10,
    "timestamp_certainty": 0.10,
}

#: Type evidence, by how the type was determined.
TYPE_EVIDENCE_REPEATED = 1.0  #: one type, supported by two or more terms
TYPE_EVIDENCE_SINGLE = 0.8  #: one type, one term
TYPE_EVIDENCE_BY_CONTEXT = 0.6  #: several types matched; context singled one out
TYPE_EVIDENCE_GUESS = 0.0  #: several types matched; nothing singled one out
#: A type the resolved entity is declared never to produce is contested, however
#: many terms support it. The CFTC declares only REGULATION; "Interest Rate
#: Swaps" in its title does not make it a central bank.
CONTESTED_TYPE_EVIDENCE = 0.2

#: Entity certainty, by where the entity was found.
ENTITY_IN_TITLE = 1.0
ENTITY_FROM_PUBLISHER = 0.7
ENTITY_AMBIGUOUS = 0.4
ENTITY_NONE = 0.0

#: Context agreement when no context speaks to the type at all: neither
#: corroboration nor contradiction.
CONTEXT_UNKNOWN = 0.5

#: Which event types a disclosure stream implies. Only streams that are specific
#: instruments say anything; REGULATORY_ANNOUNCEMENT is "the newsroom generally"
#: and UNCLASSIFIED is the absence of a claim, so neither constrains the type.
STREAM_EVENT_TYPES: dict[DisclosureStream, frozenset[EventType]] = {
    DisclosureStream.ADMINISTRATIVE_PROCEEDINGS: frozenset({EventType.REGULATION}),
    DisclosureStream.CIVIL_LITIGATION: frozenset({EventType.REGULATION}),
    DisclosureStream.MONETARY_POLICY: frozenset({EventType.MONETARY_POLICY}),
    DisclosureStream.ECONOMIC_STATISTICS: frozenset({EventType.MACRO_SHOCK}),
}

#: Type terms, matched as whole words or phrases with an optional plural, in the
#: order ties are broken. Terms added over rules-v1 come from the corpus actually
#: collected ("Final Rule", "Proposes ... Rules", "Charges", "No-Action
#: Position") or from the deployed watchlist's topics ("enforcement"). An entity's
#: name is not type evidence: rules-v1's "sec" made anything mentioning the SEC a
#: regulation event.
RULES_V2: tuple[tuple[EventType, tuple[str, ...]], ...] = (
    (EventType.REGULATION, ("regulation", "regulatory", "rule", "rulemaking", "lawsuit", "charge", "enforcement", "no-action")),
    (EventType.MONETARY_POLICY, ("federal reserve", "fed", "fomc", "powell", "monetary policy", "interest rate")),
    (EventType.ETF_FLOW, ("etf", "exchange-traded fund")),
    (EventType.SECURITY_INCIDENT, ("hack", "breach", "exploit")),
    (EventType.EXCHANGE_INCIDENT, ("exchange outage", "exchange incident")),
    (EventType.WHALE_TRANSFER, ("whale", "large transfer")),
)

#: rules-v2 assesses the type, the entity and how certain it is of both. It does
#: not assess BTC relevance, novelty or sentiment: those stay at rules-v1's routing
#: defaults, declared here so their constancy is a documented fact rather than a
#: hidden one. No study may stratify on them while they come from this extractor.
ROUTING_RELEVANCE = 0.55
ROUTING_NOVELTY = 0.5
ROUTING_SENTIMENT = 0.0


def phrase_pattern(terms: Sequence[str]) -> re.Pattern[str]:
    """Whole words or phrases, case-folded, allowing a plural.

    `rule` matches "Rules" but not "ruler"; `sec` matches "SEC" but not "second".
    Stricter than candidate matching's word-start rule on purpose: admitting a
    document can afford recall, but assigning it a type cannot afford a guess.
    """
    alternation = "|".join(re.escape(term.casefold()) for term in sorted(terms, key=len, reverse=True))
    return re.compile(r"\b(?:" + alternation + r")(?:s|es)?\b")


class ExtractionEvidence(BaseModel):
    """Why rules-v2 extracted what it did, and how confident it is in that."""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    entity: str | None
    #: "title", "publisher", "ambiguous" or "none".
    entity_source: str
    #: Every type whose terms matched, with the number of matches.
    candidates: dict[str, int]
    #: The types the entity's declaration and the disclosure stream allow; empty
    #: when neither says anything.
    context_types: tuple[str, ...]
    contested: bool
    factors: dict[str, float]
    confidence: float


def _source_reliability(document: Document) -> float:
    metadata = document.source_metadata
    if metadata.official_source and metadata.primary_source:
        return 1.0
    if metadata.official_source or metadata.primary_source:
        return 0.75
    if metadata.known_publisher:
        return 0.5
    return 0.25


def _timestamp_certainty(document: Document) -> float:
    """The source's own timestamp quality, or nothing when there is no usable timestamp."""
    if document.published_at is None or document.published_at > document.retrieved_at:
        return 0.0
    return float(document.source_metadata.timestamp_quality)


class EvidenceRuleExtractor(EventExtractor):
    """rules-v2: rule-based routing with a confidence derived from its evidence.

    Confidence is confidence in the *extracted event* -- that the document
    announces an event of this type, about this entity -- and never confidence
    that the event matters to BTC. It is a weighted sum of five factors
    (`CONFIDENCE_WEIGHTS`), each computed from the document and the declared
    watchlist alone: how the type was determined, whether the entity's declared
    event types and the feed's disclosure stream agree with it, where the entity
    was found, how reliable the source is, and how certain its timestamp is.
    Nothing here reads a price, a return, a label or a later correction, and the
    same document under the same version always yields the same confidence.
    """

    version = CURRENT_RULE_EXTRACTOR_VERSION

    def __init__(
        self,
        entities: Mapping[str, Sequence[str]] | None = None,
        expected_types: Mapping[str, Sequence[EventType]] | None = None,
    ) -> None:
        self.entities = {name: tuple(aliases) for name, aliases in (entities or {}).items()}
        self.expected_types = {name: frozenset(types) for name, types in (expected_types or {}).items() if types}
        self._entity_patterns = {name: phrase_pattern((name, *aliases)) for name, aliases in self.entities.items()}
        self._type_patterns = [(event_type, phrase_pattern(terms)) for event_type, terms in RULES_V2]
        self._order = {event_type: index for index, (event_type, _) in enumerate(RULES_V2)}

    def _resolve_entity(self, document: Document, text: str) -> tuple[str | None, str, float]:
        in_title = [name for name, pattern in self._entity_patterns.items() if pattern.search(text)]
        if len(in_title) == 1:
            return in_title[0], "title", ENTITY_IN_TITLE
        if len(in_title) > 1:
            return in_title[0], "ambiguous", ENTITY_AMBIGUOUS
        publisher = document.publisher.casefold()
        by_publisher = [name for name, pattern in self._entity_patterns.items() if pattern.search(publisher)]
        if len(by_publisher) == 1:
            return by_publisher[0], "publisher", ENTITY_FROM_PUBLISHER
        return None, "none", ENTITY_NONE

    def _choose_type(
        self, candidates: dict[EventType, int], allowed: frozenset[EventType] | None
    ) -> tuple[EventType, float]:
        if len(candidates) == 1:
            (event_type, hits), = candidates.items()
            return event_type, TYPE_EVIDENCE_REPEATED if hits >= 2 else TYPE_EVIDENCE_SINGLE
        consistent = [event_type for event_type in candidates if allowed and event_type in allowed]
        if len(consistent) == 1:
            return consistent[0], TYPE_EVIDENCE_BY_CONTEXT
        best = min(candidates, key=lambda event_type: (-candidates[event_type], self._order[event_type]))
        return best, TYPE_EVIDENCE_GUESS

    def explain(self, document: Document) -> ExtractionEvidence | None:
        """The evidence for the one event this document yields, or None if it yields none."""
        text = document.title.casefold()
        candidates = {
            event_type: hits
            for event_type, pattern in self._type_patterns
            if (hits := len(pattern.findall(text)))
        }
        if not candidates:
            return None
        entity, entity_source, entity_certainty = self._resolve_entity(document, text)
        declared = self.expected_types.get(entity) if entity is not None else None
        implied = STREAM_EVENT_TYPES.get(document.source_metadata.disclosure_stream)
        contexts = [types for types in (declared, implied) if types is not None]
        allowed = frozenset.intersection(*contexts) if contexts else None
        event_type, type_evidence = self._choose_type(candidates, allowed or None)
        contested = declared is not None and event_type not in declared
        if contested:
            type_evidence = min(type_evidence, CONTESTED_TYPE_EVIDENCE)
        agreement = [1.0 if event_type in types else 0.0 for types in contexts]
        factors = {
            "type_evidence": type_evidence,
            "context_agreement": sum(agreement) / len(agreement) if agreement else CONTEXT_UNKNOWN,
            "entity_certainty": entity_certainty,
            "source_reliability": _source_reliability(document),
            "timestamp_certainty": _timestamp_certainty(document),
        }
        confidence = sum(CONFIDENCE_WEIGHTS[name] * value for name, value in factors.items())
        return ExtractionEvidence(
            event_type=event_type,
            entity=entity,
            entity_source=entity_source,
            candidates={event_type.value: hits for event_type, hits in sorted(candidates.items(), key=lambda item: self._order[item[0]])},
            context_types=tuple(sorted(member.value for member in allowed)) if allowed else (),
            contested=contested,
            factors={name: round(value, 4) for name, value in factors.items()},
            confidence=round(min(1.0, max(0.0, confidence)), 4),
        )

    def extract(self, documents: list[Document]) -> list[EventSignal]:
        output = []
        for doc in documents:
            evidence = self.explain(doc)
            if evidence is None:
                continue
            event_type = evidence.event_type
            output.append(
                EventSignal(
                    event_id=EventSignal.stable_id([doc.document_id], event_type, doc.available_at, self.version),
                    event_time=doc.published_at or doc.available_at,
                    available_time=doc.available_at,
                    source_ids=(doc.document_id,),
                    category=SignalCategory.ONCHAIN if event_type == EventType.WHALE_TRANSFER else SignalCategory.WEB_EVENT,
                    entity=evidence.entity,
                    event_type=event_type,
                    direction=Direction.UNKNOWN,
                    sentiment=ROUTING_SENTIMENT,
                    btc_relevance=ROUTING_RELEVANCE,
                    novelty=ROUTING_NOVELTY,
                    confidence=evidence.confidence,
                    expected_horizon_hours=24,
                    summary=f"Rule-based extraction ({self.version}): {doc.title}"[:1000],
                    transfer_context=TransferContext.UNKNOWN if event_type == EventType.WHALE_TRANSFER else None,
                    extractor_version=self.version,
                    extraction_method=ExtractionMethod.RULE_BASED,
                )
            )
        return _validate_provenance(output, documents)
