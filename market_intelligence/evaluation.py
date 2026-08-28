from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .extractors import EventExtractor
from .models import Direction, Document, EventType


class GoldLabel(BaseModel):
    model_config = ConfigDict(frozen=True)
    case_id: str
    document_ids: tuple[str, ...]
    expected_event_type: EventType | None
    expected_entity: str | None = None
    expected_direction: Direction | None = None
    relevance_min: float = Field(ge=0, le=1)
    relevance_max: float = Field(ge=0, le=1)


class EvaluationResult(BaseModel):
    extractor_version: str
    cases: int
    event_type_accuracy: float
    entity_precision: float
    entity_recall: float
    source_provenance_accuracy: float
    relevance_in_range_rate: float
    invalid_output_rate: float


def evaluate_extractor(extractor: EventExtractor, documents: list[Document], labels: list[GoldLabel]) -> EvaluationResult:
    by_id = {d.document_id: d for d in documents}
    type_correct = provenance_correct = relevance_correct = invalid = 0
    true_entities = predicted_entities = correct_entities = 0
    for label in labels:
        subset = [by_id[source_id] for source_id in label.document_ids]
        try:
            predictions = extractor.extract(subset)
        except Exception:
            predictions, invalid = [], invalid + 1
        prediction = predictions[0] if predictions else None
        type_correct += bool((prediction.event_type if prediction else None) == label.expected_event_type)
        provenance_correct += bool(prediction and set(prediction.source_ids) == set(label.document_ids))
        relevance_correct += bool(prediction and label.relevance_min <= prediction.btc_relevance <= label.relevance_max)
        true_entities += label.expected_entity is not None
        predicted_entities += bool(prediction and prediction.entity is not None)
        correct_entities += bool(prediction and prediction.entity == label.expected_entity and label.expected_entity is not None)
    total = len(labels) or 1
    return EvaluationResult(extractor_version=extractor.version, cases=len(labels),
        event_type_accuracy=type_correct / total, entity_precision=correct_entities / predicted_entities if predicted_entities else 0.0,
        entity_recall=correct_entities / true_entities if true_entities else 0.0,
        source_provenance_accuracy=provenance_correct / total, relevance_in_range_rate=relevance_correct / total,
        invalid_output_rate=invalid / total)


class ExtractorComparison(BaseModel):
    results: list[EvaluationResult]


def compare_extractors(extractors: list[EventExtractor], documents: list[Document], labels: list[GoldLabel]) -> ExtractorComparison:
    """Extraction calibration only; this never assigns forecast or fusion weights."""
    return ExtractorComparison(results=[evaluate_extractor(extractor, documents, labels) for extractor in extractors])
