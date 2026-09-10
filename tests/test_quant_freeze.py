"""The quantitative research freeze, made load-bearing.

``docs/quant-research-status.md`` declares ``QUANT_RESEARCH_FROZEN``. A
declaration nothing checks decays into an aspiration, so the facts it rests on
are pinned here:

* live trading is off, and the paper engine decides from A2's evidence alone;
* nothing in the research packages can reach the paper engine;
* A7 has no decision state that promotes or trades, and the committed A6 and A7
  runs promote nothing;
* the committed A6 and A7 negative results are exactly as recorded, so a retune
  that improved a published number would have to change a file this suite
  reads back.

None of this is forbidden to change. Changing it takes its own commit and its own
reasoning, and -- per the status document -- a study that meets
``NO_MODEL_PROMOTION_UNTIL`` first.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

from btc_forecaster.paper.a2 import current_live_permission
from btc_forecaster.paper.decision import LIVE_TRADING_ELIGIBLE
from btc_forecaster.research.runner import TIMING_COLUMNS
from btc_forecaster.research.walk_forward.stability import (
    DECISIONS,
    FRAGILE_SIGNAL,
    ROBUST_RESEARCH_CANDIDATE,
    ROBUSTLY_UNINTERESTING,
)

ROOT = Path(__file__).resolve().parents[1]
A2_PROMOTION = ROOT / "research" / "runs" / "2026-08-29-a2-benchmark" / "promotion.csv"
A6_RUN = ROOT / "research" / "runs" / "a6-model-zoo"
A7_RUN = ROOT / "research" / "runs" / "a7-walk-forward"
STATUS = ROOT / "docs" / "quant-research-status.md"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def imported_modules(path: Path) -> set[str]:
    """Every module a source file imports, relative imports resolved to absolute names."""
    package = list(path.relative_to(ROOT).with_suffix("").parts[:-1])
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *(node.module.split(".") if node.module else [])])
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


class TestLiveTradingStaysOff:
    def test_the_paper_engine_is_not_live_eligible(self) -> None:
        assert LIVE_TRADING_ELIGIBLE is False

    def test_the_live_permission_is_zero(self) -> None:
        permission = current_live_permission(A2_PROMOTION)
        assert permission["live_trading_eligible"] is False
        assert permission["live_candidates"] == 0 and permission["allowed_live_leverage"] == 0.0
        assert permission["promoted_models"] == ()

    def test_the_import_check_sees_relative_imports(self) -> None:
        """A check that resolves nothing passes vacuously; this is its control."""
        runner = imported_modules(ROOT / "btc_forecaster" / "research" / "walk_forward" / "runner.py")
        assert "btc_forecaster.research.walk_forward.manifest" in runner
        shadow = imported_modules(ROOT / "btc_forecaster" / "shadow" / "evaluation.py")
        assert "btc_forecaster.paper.calibration" in shadow

    def test_no_research_module_can_reach_the_paper_engine(self) -> None:
        offenders = sorted(
            f"{path.relative_to(ROOT).as_posix()}: {module}"
            for path in (ROOT / "btc_forecaster" / "research").rglob("*.py")
            for module in imported_modules(path)
            if module == "btc_forecaster.paper" or module.startswith("btc_forecaster.paper.")
        )
        assert offenders == []


class TestNothingCanBePromoted:
    def test_a7_has_no_decision_that_promotes_or_trades(self) -> None:
        assert set(DECISIONS) == {ROBUSTLY_UNINTERESTING, FRAGILE_SIGNAL, ROBUST_RESEARCH_CANDIDATE}
        assert not any(word in state for state in DECISIONS for word in ("PROMOT", "TRAD", "LIVE"))

    def test_the_committed_runs_promote_nothing(self) -> None:
        for record in (A6_RUN / "manifest.json", A7_RUN / "manifest.json", A7_RUN / "decision.json"):
            content = _json(record)
            assert content["promoted_models"] == [], record
            assert content["live_trading_enabled"] is False, record


class TestTheNegativeResultsAreAsRecorded:
    def test_the_a6_results_table_hashes_to_its_recorded_digest(self) -> None:
        """Parsed at full precision: pandas' default float parser is not exact, and
        re-serialising what it reads changes the bytes the digest was taken over."""
        manifest = _json(A6_RUN / "manifest.json")
        frame = pd.read_csv(A6_RUN / "results.csv", float_precision="round_trip")
        scientific = frame.drop(columns=list(TIMING_COLUMNS), errors="ignore")
        assert list(scientific.columns) == manifest["results_table_hashed_columns"]
        digest = hashlib.sha256(scientific.to_csv(index=False).encode()).hexdigest()
        assert digest == manifest["results_table_sha256"]

    def test_nothing_in_a6_was_significantly_better_than_naive(self) -> None:
        results = _json(A6_RUN / "manifest.json")["analysis"]["comparison"]["results"]
        significant = [r for r in results if r["significant_after_bh"]]
        assert len(results) == 39 and len(significant) == 28
        assert all(r["mean_loss_difference"] > 0 for r in significant), "an A6 model beat naive"

    def test_nothing_in_a7_was_significantly_better_than_naive(self) -> None:
        decision = _json(A7_RUN / "decision.json")
        assert decision["decision"] == ROBUSTLY_UNINTERESTING
        assert decision["candidates"] == [] and decision["fragile_signals"] == []
        family = _json(A7_RUN / "comparisons.json")["family"]
        assert family["family_size"] == 200
        assert family["significantly_better_after_bh"] == 0
        assert family["significantly_worse_after_bh"] == 103


class TestTheFreezeIsDeclared:
    def test_the_status_document_declares_the_freeze_and_names_its_evidence(self) -> None:
        text = STATUS.read_text(encoding="utf-8")
        assert "**`QUANT_RESEARCH_FROZEN`**" in text and "`NO_MODEL_PROMOTION_UNTIL`" in text
        assert _json(A7_RUN / "manifest.json")["result_digest"] in text
        assert _json(A6_RUN / "manifest.json")["results_table_sha256"] in text
