"""B5 — write one canonical, verifiable Gate 1 result.

The canonical files carry nothing about the run itself: no time it ran, no
timing, no path, no machine. Only what was audited and what was found, so two
audits of the same store at the same instant produce the same bytes. Each is
compact JSON with sorted keys and no newline, so a checkout's line-ending
conversion cannot change a byte of it.

    preregistration.json   the frozen plan the audit applied, with its own hash
    corpus_audit.json      the measurements
    event_catalog.json     every event, with the reasons it was or was not counted
    gate1.json             the clauses, the ready families, and the decision
    manifest.json          written last: the input's fingerprints, every file's
                           sha256 and the result digest
    README.md              for people; outside the digest

The input is named by a label the caller chooses and identified by hashes -- the
store file's sha256 and a content fingerprint of what the audit read -- never by
a path, which would describe the machine rather than the evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .audit import AuditResult
from .contracts import B5_TRACK, Preregistration, canonical_json

CANONICAL_FILES: tuple[str, ...] = (
    "preregistration.json",
    "corpus_audit.json",
    "event_catalog.json",
    "gate1.json",
)

GATE = "GATE_1_CORPUS_SUFFICIENCY"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def result_digest(parts: dict[str, bytes]) -> str:
    """One hash over named contents, independent of the order they were written in."""
    lines = [f"{name}\0{sha256_hex(parts[name])}" for name in sorted(parts)]
    return sha256_hex("\n".join(lines).encode("utf-8"))


def canonical_contents(result: AuditResult, plan: Preregistration) -> dict[str, bytes]:
    return {
        "preregistration.json": canonical_json(plan.as_record()).encode("utf-8"),
        "corpus_audit.json": canonical_json(result.audit.model_dump(mode="json")).encode("utf-8"),
        "event_catalog.json": canonical_json([record.model_dump(mode="json") for record in result.catalog]).encode("utf-8"),
        "gate1.json": canonical_json(result.gate.model_dump(mode="json")).encode("utf-8"),
    }


def build_manifest(
    result: AuditResult,
    plan: Preregistration,
    parts: dict[str, bytes],
    *,
    input_label: str,
    input_sha256: str,
) -> dict[str, Any]:
    return {
        "track": B5_TRACK,
        "gate": GATE,
        "gate1_passed": result.gate.passed,
        "decision": result.gate.decision.value if result.gate.decision is not None else None,
        "as_of": result.audit.as_of.isoformat(),
        "input": {
            "label": input_label,
            "file_sha256": input_sha256,
            "content_fingerprint": result.audit.content_fingerprint,
            "schema_version": result.audit.schema_version,
            "extractor_version": result.audit.extractor_version,
        },
        "preregistration_hash": plan.content_hash(),
        "files": {name: sha256_hex(data) for name, data in sorted(parts.items())},
        "digest_covers": sorted(parts),
        "result_digest": result_digest(parts),
        "produces_trading_signal": False,
        "live_trading_enabled": False,
    }


def readme(manifest: dict[str, Any], result: AuditResult) -> str:
    audit, gate = result.audit, result.gate
    lines = [
        "# B5 Gate 1 — corpus sufficiency",
        "",
        f"**Decision: `{manifest['decision'] or 'GATE_1_PASSED'}`.** Research evidence only; nothing here is a "
        "trading signal, and live trading is disabled.",
        "",
        f"- Audited as of `{manifest['as_of']}`, input `{manifest['input']['label']}`",
        f"- Store sha256 `{manifest['input']['file_sha256']}`",
        f"- Content fingerprint `{manifest['input']['content_fingerprint']}`",
        f"- Extractor version counted `{manifest['input']['extractor_version']}`",
        f"- Preregistration `{manifest['preregistration_hash']}`",
        f"- Result digest `{manifest['result_digest']}`",
        "",
        "## Clauses",
        "",
        "| clause | required | observed | met |",
        "|---|---|---|---|",
        *(f"| `{c.name}` | {c.required} | {c.observed} | {'yes' if c.met else '**no**'} |" for c in gate.clauses),
        "",
        "## Funnel",
        "",
        "| stage | events |",
        "|---|---:|",
        *(f"| {stage} | {count} |" for stage, count in audit.funnel.items()),
        "",
        "## Checking it",
        "",
        "`python -m market_intelligence.b5 verify <this directory>` recomputes every file's sha256 and the result "
        "digest. The store itself is not redistributed; its sha256 and the content fingerprint say whether a "
        "store someone else holds is the one audited here.",
        "",
    ]
    return "\n".join(lines)


def write_run(
    result: AuditResult,
    plan: Preregistration,
    output: str | Path,
    *,
    input_label: str,
    input_sha256: str,
) -> dict[str, Any]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    parts = canonical_contents(result, plan)
    for name, data in parts.items():
        (out / name).write_bytes(data)
    manifest = build_manifest(result, plan, parts, input_label=input_label, input_sha256=input_sha256)
    (out / "README.md").write_text(readme(manifest, result), encoding="utf-8", newline="\n")
    # Last. Its presence is the claim that the run finished.
    (out / "manifest.json").write_bytes(canonical_json(manifest).encode("utf-8"))
    return manifest


def verify_run(directory: str | Path) -> dict[str, Any]:
    """Recompute every hash from disk and name what does not match or is missing."""
    out = Path(directory)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    recorded: dict[str, str] = manifest["files"]
    parts = {name: (out / name).read_bytes() for name in CANONICAL_FILES if (out / name).is_file()}
    absent = [name for name in CANONICAL_FILES if name not in parts]
    mismatched = [name for name in sorted(parts) if sha256_hex(parts[name]) != recorded.get(name)]
    plan_ok = False
    if "preregistration.json" in parts and "preregistration.json" not in mismatched:
        try:
            plan_ok = Preregistration.read(out / "preregistration.json").content_hash() == manifest["preregistration_hash"]
        except (ValueError, KeyError):
            plan_ok = False
    digest = None if absent else result_digest(parts)
    return {
        "absent_files": absent,
        "mismatched_files": mismatched,
        "preregistration_intact": plan_ok,
        "result_digest_recorded": manifest["result_digest"],
        "result_digest_recomputed": digest,
        "verified": not absent and not mismatched and plan_ok and digest == manifest["result_digest"],
    }


__all__ = [
    "CANONICAL_FILES",
    "GATE",
    "build_manifest",
    "canonical_contents",
    "readme",
    "result_digest",
    "sha256_hex",
    "verify_run",
    "write_run",
]
