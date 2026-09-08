"""Commands that append to, and read back, the correction ledger.

The ledger's rule is that nothing is deleted. An observation later found to be
an artefact of a defect is invalidated by a new row saying so; the observation
stays exactly where it was written. Two views follow, and both commands here
exist to keep them visible: `corpus-correct` writes the ledger, and
`corpus-corrections` shows what it hides -- because an observation excluded
from research is deliberately not hidden from anyone who goes looking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from . import CommandContext


def apply_corrections(ctx: CommandContext) -> int:
    """Record a correction file. Additive, idempotent, and it checks its aim.

    A correction naming an event the corpus does not hold is refused rather than
    recorded. The value of a ledger keyed on event ids is that every row points
    at something; a row pointing at nothing is indistinguishable from a typo,
    and it would sit there looking authoritative.
    """
    from ..corrections import load_corrections

    args = ctx.args
    store = ctx.store
    source = Path(args.file)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"correction file not found: {source}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"correction file {source} is not valid JSON: {error}") from error

    try:
        corrections = load_corrections(raw)
    except ValueError as error:
        raise ConfigurationError(f"correction file {source}: {error}") from error

    known = {event.event_id for event in store.signals_as_of(datetime.now(timezone.utc))}
    missing = sorted({item.event_id for item in corrections} - known)
    if missing:
        raise ConfigurationError(
            f"correction file {source} names {len(missing)} event(s) this corpus does not "
            f"hold, starting with {missing[0]}. A correction must point at something."
        )

    already = {item.correction_id for item in store.corrections()}
    fresh = [item for item in corrections if item.correction_id not in already]
    payload: dict[str, Any] = {
        "file": str(source),
        "reason": raw.get("reason"),
        "corrections_in_file": len(corrections),
        "already_recorded": len(corrections) - len(fresh),
        "newly_recorded": 0,
        "dry_run": bool(args.dry_run),
        "events": sorted(item.event_id for item in corrections),
    }
    if not args.dry_run:
        payload["newly_recorded"] = store.put_corrections(corrections)
    print(json.dumps(payload, indent=2))
    return 0


def report_corrections(ctx: CommandContext) -> int:
    """The audit surface. An invalidated observation is hidden from research and
    deliberately not hidden from anybody who goes looking for it."""
    from ..corrections import ELIGIBILITY_CONTRACT_VERSION, resolve

    store = ctx.store
    horizon = datetime.now(timezone.utc)
    corrections = store.corrections()
    standing = resolve(corrections)
    excluded = store.ineligible_event_ids()
    raw_events = store.signals_as_of(horizon)
    hidden = [event for event in raw_events if event.event_id in excluded]

    payload: dict[str, Any] = {
        "eligibility_contract": ELIGIBILITY_CONTRACT_VERSION,
        "corrections_recorded": len(corrections),
        "events_raw": len(raw_events),
        "events_eligible": len(raw_events) - len(hidden),
        "events_excluded": len(hidden),
        "corrections": [correction.as_dict() for correction in corrections],
    }
    if ctx.args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"eligibility contract   {payload['eligibility_contract']}")
    print(f"corrections recorded   {payload['corrections_recorded']}")
    print(f"events raw             {payload['events_raw']}  (physically present, always)")
    print(f"events eligible        {payload['events_eligible']}  (research may count these)")
    print(f"events excluded        {payload['events_excluded']}")
    for event_id, correction in sorted(standing.items()):
        print(f"  {event_id[:16]}  {correction.status.value}  {correction.reason}")
        print(
            f"      recorded {correction.invalidated_at.isoformat()} "
            f"by {correction.invalidated_by_version[:12]}"
        )
    return 0


__all__ = ["apply_corrections", "report_corrections"]
