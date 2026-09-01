"""B4.41 / B4.42 — split the history, then freeze the plan before opening it.

The failure this prevents is not fraud. It is the ordinary, well-intentioned
process of looking at results, noticing that a threshold of 0.3 would have made
a finding cleaner, and adjusting it -- at which point the reported p-value no
longer means what it says, and nobody involved has done anything they would
describe as wrong.

So: hypotheses, filters, horizons, minimum sample sizes, bootstrap
configuration, practical thresholds, testing families and the carry-forward
policy are written down and hashed **before** the final-validation period is
touched. The hash goes in the run manifest, and a final-period analysis that
does not reference a preregistration is refused rather than warned about.

The split itself (B4.41) is honest about its own limits: if the history is too
short to hold out a useful final period, that is stated and the results are
labelled exploratory. Carving a 30-observation "holdout" out of a short sample
produces a validation that cannot reject anything -- worse than no split at all,
because it looks like one.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from .contracts import B4DataError, require_utc
from .stats import BootstrapConfig, PracticalThreshold

PREREGISTRATION_VERSION = "b4-preregistration-v1"


class PeriodSplit(BaseModel):
    """B4.41. Development and final-validation windows."""

    model_config = ConfigDict(frozen=True)

    development_start: datetime
    development_end: datetime
    validation_start: datetime
    validation_end: datetime
    #: False when the history could not support a useful holdout. Results are
    #: then labelled exploratory rather than validated.
    validation_is_useful: bool
    rationale: str

    def contains_development(self, moment: datetime) -> bool:
        return self.development_start <= moment < self.development_end

    def contains_validation(self, moment: datetime) -> bool:
        return self.validation_start <= moment < self.validation_end


class CarryForwardPolicy(BaseModel):
    """B4.38. The bar a candidate must clear, frozen before results."""

    model_config = ConfigDict(frozen=True)

    minimum_observations: int = 30
    minimum_effective_observations: int = 20
    require_pit_validated: bool = True
    maximum_q_value: float = 0.10
    require_interval_excludes_zero: bool = True
    require_stability: bool = True
    require_not_fragile: bool = True
    maximum_single_source_share: float = 0.8
    #: A candidate may be carried forward on exploratory grounds without formal
    #: significance, but only with an explicit reason and only up to this many
    #: per run. Track C should receive a short list, not a dozen weak leads.
    allow_exploratory: bool = True
    maximum_exploratory_candidates: int = 3


class Preregistration(BaseModel):
    """The frozen analysis plan. Hash it, then open the validation period."""

    model_config = ConfigDict(frozen=True)

    version: str = PREREGISTRATION_VERSION
    created_at: datetime
    git_sha: str
    research_questions: tuple[str, ...]
    event_types: tuple[str, ...]
    entities: tuple[str, ...]
    horizons: tuple[str, ...]
    extraction_quality_filters: dict[str, float]
    minimum_event_count: int
    bootstrap: BootstrapConfig
    practical_thresholds: tuple[PracticalThreshold, ...]
    multiple_testing_families: tuple[str, ...]
    carry_forward_policy: CarryForwardPolicy
    period_split: PeriodSplit
    #: Composites must be named before results are seen (B4.8). Adding one
    #: afterwards is an undeclared extra hypothesis.
    predefined_composites: tuple[str, ...] = ()
    notes: str = ""

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def write(self, path: str | Path) -> Path:
        """Write atomically, and refuse to change an existing plan.

        A preregistration that can be overwritten is not a preregistration. If
        the plan genuinely needs to change, the new one belongs in a new file
        with its own hash, so both are on the record.
        """
        target = Path(path)
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("content_hash") != self.content_hash():
                raise B4DataError(
                    f"{target} already holds a different preregistration; "
                    "write the revised plan to a new file so both remain on the record"
                )
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        payload["content_hash"] = self.content_hash()
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def read(cls, path: str | Path) -> "Preregistration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        declared = payload.pop("content_hash", None)
        plan = cls.model_validate(payload)
        if declared is not None and declared != plan.content_hash():
            raise B4DataError(
                f"{path}: the stored hash does not match its contents; the plan was edited after it was frozen"
            )
        return plan

    def threshold_for(self, outcome: str) -> PracticalThreshold:
        for threshold in self.practical_thresholds:
            if threshold.outcome == outcome:
                return threshold
        raise B4DataError(f"no practical threshold was preregistered for {outcome!r}")


def split_history(
    start: datetime,
    end: datetime,
    *,
    validation_fraction: float = 0.3,
    minimum_validation_days: int = 180,
) -> PeriodSplit:
    """B4.41. Carve a final-validation window off the end of the history.

    The end of the history, not a random sample: a random holdout from a time
    series leaks, because neighbouring observations are nearly the same
    observation. The last block is the only split that resembles how the result
    would be used.
    """
    start = require_utc(start, "start")
    end = require_utc(end, "end")
    if end <= start:
        raise B4DataError("history end must be after its start")
    if not 0.0 < validation_fraction < 1.0:
        raise B4DataError("validation_fraction must be in (0, 1)")

    span = end - start
    boundary = end - span * validation_fraction
    validation_days = (end - boundary).days
    useful = validation_days >= minimum_validation_days

    rationale = (
        f"the last {validation_fraction:.0%} of the history spans {validation_days} days, "
        f"at or above the {minimum_validation_days}-day minimum for a useful holdout"
        if useful
        else (
            f"the last {validation_fraction:.0%} of the history spans only {validation_days} days, "
            f"below the {minimum_validation_days}-day minimum; results are exploratory and are "
            "labelled as such rather than presented as validated"
        )
    )
    return PeriodSplit(
        development_start=start,
        development_end=boundary,
        validation_start=boundary,
        validation_end=end,
        validation_is_useful=useful,
        rationale=rationale,
    )


def require_preregistration(plan: Preregistration | None, hash_reference: str | None) -> Preregistration:
    """Refuse a final-period analysis that does not reference a frozen plan."""
    if plan is None:
        raise B4DataError(
            "a final-validation analysis requires a preregistration; "
            "freeze the plan before opening the validation period"
        )
    if hash_reference is not None and hash_reference != plan.content_hash():
        raise B4DataError(
            "the analysis references a different preregistration hash than the plan supplied"
        )
    return plan


def filter_to_period(
    rows: Sequence[dict[str, Any]],
    split: PeriodSplit,
    *,
    period: str = "development",
    origin_field: str = "forecast_origin",
) -> list[dict[str, Any]]:
    """Restrict rows to one side of the split."""
    if period not in {"development", "validation"}:
        raise B4DataError(f"unknown period {period!r}")
    inside = split.contains_development if period == "development" else split.contains_validation
    kept: list[dict[str, Any]] = []
    for row in rows:
        moment = row.get(origin_field)
        if not isinstance(moment, datetime):
            raise B4DataError(f"row has no usable {origin_field}")
        if inside(require_utc(moment, origin_field)):
            kept.append(row)
    return kept


__all__ = [
    "PREREGISTRATION_VERSION",
    "CarryForwardPolicy",
    "PeriodSplit",
    "Preregistration",
    "filter_to_period",
    "require_preregistration",
    "split_history",
]
