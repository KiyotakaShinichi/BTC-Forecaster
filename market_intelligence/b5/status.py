"""B5.2 -- how far the collected corpus is from Gate 1, asked at any instant.

A collector running on a host for months has one question worth asking about
it, and it is not "is it running": it is "how far is this corpus from the gate
it is being collected for". This answers it from the audit Gate 1 itself runs,
under the committed preregistration, so the figures here are the gate's own and
not a parallel calculation that could drift from it.

Deterministic: the instant is an argument, never the clock, so two readings of
one store at one instant are identical. Read-only: the store is opened the way
the audit opens it. And it restates no threshold: every requirement it shows is
read from the preregistered policy object.

It never says "ready" unless Gate 1 passed, and it never implies that elapsed
time is progress on its own -- a span can be 180 days long and still hold a
family of three events.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ..collection.coverage import CollectionCoverage
from .audit import AuditResult
from .contracts import Preregistration


class ReadinessStatus(BaseModel):
    """Collection progress against Gate 1, at one instant."""

    model_config = ConfigDict(frozen=True)

    as_of: datetime
    extractor_version: str | None
    preregistration_hash: str
    content_fingerprint: str
    #: What collection did: elapsed, expected and successful days and cycles.
    collection: CollectionCoverage | None
    documents: int
    raw_events: int
    eligible_events: int
    quality_events: int
    independent_events: int
    effective_events: int
    publishers: int
    providers: int
    longest_family_span_days: int
    families: int
    ready_families: tuple[str, ...]
    # Requirements, read from the preregistered policy object -- never restated.
    required_events: int
    required_effective_events: int
    required_publishers: int
    required_span_days: int
    required_coverage: float
    required_ready_families: int
    gate1_passed: bool
    decision: str | None
    unmet_clauses: tuple[str, ...]

    def statement(self) -> str:
        if self.gate1_passed:
            return (
                f"Gate 1 passed at {self.as_of.isoformat()}: {len(self.ready_families)} family(ies) meet every "
                "preregistered clause. The next gate decides what may be studied; this is not a result."
            )
        return (
            f"Gate 1 not passed ({self.decision}). {len(self.ready_families)} of {self.families} families ready; "
            f"the longest family spans {self.longest_family_span_days} of the {self.required_span_days} days "
            "required. The corpus is not sufficient."
        )

    def human_readable(self) -> str:
        coverage = self.collection.describe() if self.collection is not None else "not recorded"
        lines = [
            f"as of        {self.as_of.isoformat()}  (extractor {self.extractor_version or '-'})",
            f"collection   {coverage}",
            f"corpus       {self.documents} documents, {self.raw_events} events "
            f"({self.eligible_events} eligible, {self.quality_events} above the quality floor, "
            f"{self.independent_events} independent, {self.effective_events} effective)",
            f"sources      {self.publishers} publisher(s), {self.providers} provider(s)",
            f"families     {len(self.ready_families)} of {self.families} ready; "
            f"need {self.required_ready_families}, each with {self.required_events} events, "
            f"{self.required_effective_events} effective, {self.required_publishers} publishers, "
            f"{self.required_span_days} days, {self.required_coverage:.0%} coverage",
            f"gate 1       {'PASSED' if self.gate1_passed else self.decision}",
        ]
        lines.extend(f"  unmet      {clause}" for clause in self.unmet_clauses)
        lines.append("")
        lines.append(self.statement())
        return "\n".join(lines)


def readiness_status(result: AuditResult, plan: Preregistration) -> ReadinessStatus:
    """The gate's own measurements, arranged as progress toward it."""
    audit, gate, policy = result.audit, result.gate, plan.policy
    adequacy = policy.adequacy
    return ReadinessStatus(
        as_of=audit.as_of,
        extractor_version=audit.extractor_version,
        preregistration_hash=plan.content_hash(),
        content_fingerprint=audit.content_fingerprint,
        collection=audit.collection_coverage,
        documents=audit.documents,
        raw_events=audit.raw_events,
        eligible_events=audit.funnel["2_not_invalidated"],
        quality_events=audit.funnel["5_quality"],
        independent_events=audit.funnel["7_independent_events"],
        effective_events=audit.funnel["8_effective_events"],
        publishers=len(audit.publishers),
        providers=len(audit.providers),
        longest_family_span_days=max((family.span_days for family in audit.families), default=0),
        families=len(audit.families),
        ready_families=gate.ready_families,
        required_events=adequacy.minimum_events,
        required_effective_events=adequacy.minimum_effective_events,
        required_publishers=adequacy.minimum_publishers,
        required_span_days=adequacy.minimum_span_days,
        required_coverage=adequacy.minimum_coverage_fraction,
        required_ready_families=policy.minimum_ready_families,
        gate1_passed=gate.passed,
        decision=gate.decision.value if gate.decision is not None else None,
        unmet_clauses=tuple(f"{clause.name}: {clause.observed}" for clause in gate.clauses if not clause.met),
    )


__all__ = ["ReadinessStatus", "readiness_status"]
