"""B5 Gate 1 contracts: decision states and the preregistered sufficiency policy.

The load-bearing property is that B5 does not own its thresholds. Every number
it inherits is read back from B4's committed preregistration, so a B5 bar that
drifted from B4's -- in either direction -- fails here rather than in a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_intelligence.b5.contracts import (
    B4_PRACTICAL_FLOORS,
    B4_PREREGISTRATION_HASH,
    DEFAULT_SUFFICIENCY,
    Decision,
    Preregistration,
    SufficiencyPolicy,
    default_preregistration,
    minimum_detectable_effects,
    study_families,
)
from market_intelligence.collection.readiness import DEFAULT_POLICY
from market_intelligence.models import EventType, TransferContext

ROOT = Path(__file__).resolve().parents[1]
B4_PREREGISTRATION = ROOT / "research" / "market_intelligence" / "b4" / "runs" / "b4-3980bac0b72aeaa0" / "preregistration.json"
B5_PREREGISTRATION = ROOT / "research" / "market_intelligence" / "b5" / "preregistration.json"


@pytest.fixture(scope="module")
def b4() -> dict:
    return json.loads(B4_PREREGISTRATION.read_text(encoding="utf-8"))


class TestTheDecisionStates:
    def test_there_are_exactly_four_and_none_trades(self) -> None:
        assert {member.value for member in Decision} == {
            "INTELLIGENCE_CORPUS_INSUFFICIENT",
            "EVENT_EFFECTS_NOT_DETECTED",
            "EVENT_EFFECTS_FRAGILE",
            "EVENT_EFFECTS_ROBUST_CANDIDATE",
        }
        for member in Decision:
            assert not any(word in member.value for word in ("TRADE", "BUY", "SELL", "LONG", "SHORT", "PROMOT", "LIVE"))


class TestThresholdsAreB4s:
    def test_the_adequacy_policy_is_the_readiness_gates_own_object(self) -> None:
        assert DEFAULT_SUFFICIENCY.adequacy is DEFAULT_POLICY

    def test_the_inherited_numbers_match_b4s_committed_preregistration(self, b4: dict) -> None:
        adequacy = DEFAULT_SUFFICIENCY.adequacy
        assert b4["content_hash"] == B4_PREREGISTRATION_HASH
        assert adequacy.minimum_events == b4["minimum_event_count"] == b4["carry_forward_policy"]["minimum_observations"]
        assert adequacy.minimum_effective_events == b4["carry_forward_policy"]["minimum_effective_observations"]
        assert DEFAULT_SUFFICIENCY.minimum_relevance == b4["extraction_quality_filters"]["minimum_relevance"]
        assert DEFAULT_SUFFICIENCY.minimum_confidence == b4["extraction_quality_filters"]["minimum_confidence"]
        floors = {row["outcome"].removeprefix("forward_return_"): row["minimum_absolute_effect"] for row in b4["practical_thresholds"]}
        assert floors == B4_PRACTICAL_FLOORS

    def test_the_readiness_clauses_are_the_frozen_ones(self) -> None:
        """deploy/COLLECTION_FREEZE.md section 10: 30, 20, 3, 1, 180 days, 80%."""
        adequacy = DEFAULT_SUFFICIENCY.adequacy
        assert (
            adequacy.minimum_events,
            adequacy.minimum_effective_events,
            adequacy.minimum_publishers,
            adequacy.minimum_providers,
            adequacy.minimum_span_days,
            adequacy.minimum_coverage_fraction,
            adequacy.horizon_hours,
        ) == (30, 20, 3, 1, 180, 0.8, 168)


class TestB5ClausesOnlyTighten:
    def test_no_b5_clause_is_weaker_than_b4(self, b4: dict) -> None:
        filters = b4["extraction_quality_filters"]
        assert DEFAULT_SUFFICIENCY.minimum_relevance >= filters["minimum_relevance"]
        assert DEFAULT_SUFFICIENCY.minimum_confidence >= filters["minimum_confidence"]
        assert DEFAULT_SUFFICIENCY.minimum_ready_families >= 1

    def test_the_thresholds_are_a_floor_on_power_not_a_comfortable_level(self) -> None:
        """At the threshold a study can only see effects well above the practical floor."""
        effects = minimum_detectable_effects()
        assert set(effects) == set(B4_PRACTICAL_FLOORS)
        for horizon, floor in B4_PRACTICAL_FLOORS.items():
            assert 2.0 < effects[horizon] / floor < 3.0, horizon

    def test_fewer_effective_events_would_make_the_study_blinder(self) -> None:
        weaker = SufficiencyPolicy(adequacy=DEFAULT_POLICY.model_copy(update={"minimum_effective_events": 10}))
        assert all(minimum_detectable_effects(weaker)[h] > minimum_detectable_effects()[h] for h in B4_PRACTICAL_FLOORS)


class TestFamiliesAreNeverPooled:
    def test_every_event_type_is_its_own_family_and_whales_split_by_context(self) -> None:
        families = study_families()
        assert len(families) == len(set(families))
        for member in EventType:
            if member is EventType.WHALE_TRANSFER:
                assert f"event_type:{member.value}" not in families
            else:
                assert f"event_type:{member.value}" in families
        assert {f"whale:{context.value}" for context in TransferContext} <= set(families)


class TestThePreregistration:
    def test_its_hash_is_stable(self) -> None:
        assert default_preregistration().content_hash() == default_preregistration().content_hash()

    def test_the_committed_plan_is_exactly_what_the_code_declares(self) -> None:
        committed = Preregistration.read(B5_PREREGISTRATION)
        assert committed == default_preregistration()
        assert committed.content_hash() == default_preregistration().content_hash()

    def test_an_edited_plan_is_refused(self, tmp_path: Path) -> None:
        path = default_preregistration().write(tmp_path / "plan.json")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["plan"]["policy"]["minimum_confidence"] = 0.3
        path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="edited after it was frozen"):
            Preregistration.read(path)

    def test_a_different_plan_cannot_overwrite_a_frozen_one(self, tmp_path: Path) -> None:
        path = default_preregistration().write(tmp_path / "plan.json")
        laxer = default_preregistration(SufficiencyPolicy(minimum_event_time_coverage=0.5))
        with pytest.raises(ValueError, match="different frozen plan"):
            laxer.write(path)
        assert default_preregistration().write(path) == path  # re-writing the same plan is a no-op

    def test_it_names_every_decision_state_and_stop_rule(self) -> None:
        plan = default_preregistration()
        assert set(plan.decision_states) == {member.value for member in Decision}
        assert any("INTELLIGENCE_CORPUS_INSUFFICIENT" in rule for rule in plan.stop_rules)
