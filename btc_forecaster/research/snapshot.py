"""`python -m btc_forecaster.research.snapshot` -- fingerprint and verify an input.

    python -m btc_forecaster.research.snapshot fingerprint data/snapshots/BTC-USD
    python -m btc_forecaster.research.snapshot verify data/snapshots/BTC-USD --expect <sha256>

``fingerprint`` prints the input manifest as canonical JSON: source, retrieval
time, range, rows, columns, the canonical-frame digest and the raw file digest.
``verify`` compares the frame digest against a recorded one.

Exit codes are part of the contract, so a script can branch on them:

    0  the input matches the record
    2  the snapshot does not match its own manifest (corrupted or hand-edited)
    3  the snapshot is intact but is not the recorded input -- a rerun on it is
       a different experiment, not a reproduction

Fetching a snapshot is not this command's job; ``btc-forecast snapshot`` does
that. This one only says what an existing snapshot is.
"""

from __future__ import annotations

import argparse
import sys

from ..data.snapshot import SnapshotIntegrityError
from ..features.spec import default_specs
from .walk_forward.config import PREPROCESSING_VERSION
from .walk_forward.manifest import (
    INPUT_MATCHES_RECORD,
    input_manifest,
    input_status,
    load_snapshot,
)
from .walk_forward.targets import TARGET_DEFINITION

EXIT_MATCH = 0
EXIT_INTEGRITY = 2
EXIT_CHANGED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m btc_forecaster.research.snapshot",
        description="Fingerprint or verify a hash-manifested market snapshot.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprint = commands.add_parser("fingerprint", help="print the input manifest")
    fingerprint.add_argument("path", help="snapshot directory holding data.csv and manifest.json")
    verify = commands.add_parser("verify", help="compare against a recorded frame digest")
    verify.add_argument("path")
    verify.add_argument("--expect", required=True, help="the recorded frame sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot, file_bytes = load_snapshot(args.path)
    except SnapshotIntegrityError as exc:
        print(f"INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return EXIT_INTEGRITY
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INTEGRITY

    manifest = input_manifest(
        snapshot,
        file_bytes=file_bytes,
        preprocessing_version=PREPROCESSING_VERSION,
        feature_specs=tuple(spec.name for spec in default_specs()),
        target=TARGET_DEFINITION,
    )

    if args.command == "fingerprint":
        print(manifest.canonical_json())
        return EXIT_MATCH

    status = input_status(args.expect, manifest.frame_sha256)
    print(status)
    if status != INPUT_MATCHES_RECORD:
        print(
            f"  recorded: {args.expect}\n  actual:   {manifest.frame_sha256}\n"
            "  a rerun on this input is a different experiment, not a reproduction",
            file=sys.stderr,
        )
        return EXIT_CHANGED
    return EXIT_MATCH


if __name__ == "__main__":
    raise SystemExit(main())
