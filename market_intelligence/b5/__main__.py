"""`python -m market_intelligence.b5` -- the B5 Gate 1 command.

    python -m market_intelligence.b5 audit --db <store.duckdb> --as-of 2026-09-11T00:00:00+00:00 \\
        --input-label <label> --output research/market_intelligence/b5/gate1
    python -m market_intelligence.b5 verify research/market_intelligence/b5/gate1

`audit` opens the store read-only, applies the policy in the committed
preregistration (checked against its own hash first), and writes one canonical
result. The audit instant is required and must carry a timezone: "now" is not a
reproducible input.

Exit codes: 0 the audit was written or the run verified -- whatever the decision,
because an insufficient corpus is a finding, not a failure; 2 an integrity
failure: a refused store, an altered preregistration, or a run that does not
verify.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ..errors import StorageError
from . import B5_RESEARCH_NAMESPACE
from .audit import audit_corpus, file_sha256, open_read_only
from .contracts import Preregistration
from .runner import verify_run, write_run

EXIT_OK = 0
EXIT_INTEGRITY = 2

DEFAULT_PREREGISTRATION = Path(B5_RESEARCH_NAMESPACE) / "preregistration.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m market_intelligence.b5",
        description="B5 Gate 1: is there a point-in-time corpus sufficient for an event study?",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="audit a store read-only and write the canonical Gate 1 result")
    audit.add_argument("--db", required=True, help="intelligence store (DuckDB) to audit; opened read-only")
    audit.add_argument("--as-of", required=True, help="audit instant, ISO 8601 with a timezone")
    audit.add_argument("--input-label", required=True, help="a name for the input; recorded instead of its path")
    audit.add_argument("--output", required=True)
    audit.add_argument("--preregistration", default=str(DEFAULT_PREREGISTRATION))

    verify = commands.add_parser("verify", help="recompute every hash of a written Gate 1 result")
    verify.add_argument("path")
    return parser


def _as_of(text: str) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError(f"--as-of {text!r} has no timezone; an audit instant must be unambiguous")
    return moment


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "verify":
        try:
            verification = verify_run(args.path)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_INTEGRITY
        print(json.dumps(verification, indent=2, sort_keys=True))
        return EXIT_OK if verification["verified"] else EXIT_INTEGRITY

    try:
        as_of = _as_of(args.as_of)
        plan = Preregistration.read(args.preregistration)
        connection = open_read_only(args.db)
    except (ValueError, FileNotFoundError, StorageError) as exc:
        print(f"INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return EXIT_INTEGRITY
    try:
        result = audit_corpus(connection, as_of=as_of, policy=plan.policy)
    finally:
        connection.close()

    manifest = write_run(result, plan, args.output, input_label=args.input_label, input_sha256=file_sha256(args.db))
    print("B5 Gate 1 -- corpus sufficiency (research evidence; no trading signal)")
    print(f"  as of     {manifest['as_of']}")
    print(f"  input     {args.input_label}, sha256 {manifest['input']['file_sha256'][:16]}")
    print(f"  funnel    {result.audit.funnel}")
    for clause in result.gate.clauses:
        print(f"  {'met  ' if clause.met else 'UNMET'}     {clause.name}: {clause.observed}")
    print(f"  decision  {manifest['decision'] or 'GATE_1_PASSED'}")
    print(f"  digest    {manifest['result_digest']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
