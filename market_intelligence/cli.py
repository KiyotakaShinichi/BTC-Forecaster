"""`btc-intel` -- the command-line surface, and nothing else.

This module composes the parser and runs what the parser produced. The commands
themselves live in `commands/`, registered by name; before that split, `_main`
was a 230-line `if args.command == ...` chain covering 27 of them.

Parser composition stays here on purpose. A parser assembled from fragments
each command contributes is a parser whose `--help` nobody can read in one
place, and the flags are the interface an operator actually sees.

Exit codes are part of that interface: 0 ran, 2 failed, 3 another collector
holds the lock, 4 nothing was due.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from .commands import command_names, dispatch
from .errors import IntelligenceError
from .extractors import CURRENT_RULE_EXTRACTOR_VERSION
from .ops.scheduled import EXIT_FAILED
from .origins import parse_origin

# The three report builders moved to `reports.py`, where the API can reach them
# without importing an argument parser to answer a question about the corpus.
# Re-exported under their original names because a CI workflow, the API and
# several tests import them that way, and renaming a symbol in use is a break
# dressed as a tidy-up.
from .reports import corpus_status as _corpus_status
from .reports import ops_report as _ops_report
from .reports import provider_report as _provider_report


def _time(value: str) -> datetime:
    """argparse type for a timezone-aware instant.

    Wraps `origins.parse_origin` so a bad `--origin` produces argparse's usage
    message rather than a traceback.
    """
    try:
        return parse_origin(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc-intel", description="Point-in-time BTC market-intelligence operations")
    parser.add_argument("--db", default="btc-intelligence.duckdb")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--origin", type=_time, required=True)
    collect.add_argument("--manifest", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--origin", type=_time, required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--origin", type=_time, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--origin", type=_time, required=True)
    replay.add_argument("--config-fingerprint", default="manual")
    backfill = sub.add_parser("backfill")
    backfill.add_argument("--from", dest="from_time", type=_time, required=True)
    backfill.add_argument("--to", dest="to_time", type=_time, required=True)
    backfill.add_argument("--window-hours", type=int, default=24)
    backfill.add_argument("--max-windows", type=int, default=31)
    backfill.add_argument("--config", required=True)
    backfill.add_argument("--progress", default="backfill-progress.json")
    backfill.add_argument("--manifests-dir", default="backfill-manifests")
    sub.add_parser("health")
    quality = sub.add_parser("quality")
    quality.add_argument("--origin", type=_time, default=datetime.now(timezone.utc))
    dataset = sub.add_parser("dataset")
    dataset.add_argument("--origins", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--manifest", required=True)
    dataset.add_argument("--config-fingerprint", required=True)
    dataset.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    dataset.add_argument(
        "--mode",
        choices=("OPTIMIZED", "REFERENCE"),
        default="OPTIMIZED",
        help="REFERENCE is the correctness oracle; both produce identical output",
    )

    replay_dataset = sub.add_parser(
        "replay-dataset",
        help="historical intelligence feature matrix over a generated origin schedule",
    )
    replay_dataset.add_argument("--start", type=_time, required=True)
    replay_dataset.add_argument("--end", type=_time, required=True)
    replay_dataset.add_argument(
        "--frequency", choices=("HOURLY", "4H", "DAILY"), default="HOURLY"
    )
    replay_dataset.add_argument("--output", required=True)
    replay_dataset.add_argument("--manifest", required=True)
    replay_dataset.add_argument("--config-fingerprint", required=True)
    replay_dataset.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    replay_dataset.add_argument("--mode", choices=("OPTIMIZED", "REFERENCE"), default="OPTIMIZED")
    replay_dataset.add_argument("--chunk-size", type=int, default=None)
    replay_dataset.add_argument(
        "--extend",
        default=None,
        metavar="DATASET_ID",
        help="append to an existing dataset instead of building a new one",
    )
    replay_dataset.add_argument("--no-resume", action="store_true")

    catalog = sub.add_parser("catalog", help="list registered historical datasets")
    catalog.add_argument("--limit", type=int, default=20)

    corpus_status = sub.add_parser(
        "corpus-status",
        help="what the accumulated intelligence corpus holds, and whether B4 can be re-run",
    )
    corpus_status.add_argument("--json", action="store_true", help="machine-readable output")
    corpus_status.add_argument("--output", default=None, help="also write the JSON report here")
    corpus_status.add_argument("--extractor-version", default=CURRENT_RULE_EXTRACTOR_VERSION)
    corpus_status.add_argument(
        "--entities",
        default="Donald Trump,Elon Musk,Jerome Powell,Michael Saylor,SEC,CFTC",
        help="comma-separated entities to assess for readiness",
    )

    corpora = sub.add_parser("corpora", help="list registered corpus snapshots")
    corpora.add_argument("--limit", type=int, default=20)

    sub.add_parser("providers", help="declared providers and whether they can run")

    verify_parser = sub.add_parser("corpus-verify", help="check corpus integrity; fails closed")
    verify_parser.add_argument("--json", action="store_true")

    backup_parser = sub.add_parser("corpus-backup", help="write a verifiable corpus archive")
    backup_parser.add_argument("--output", required=True)
    backup_parser.add_argument("--manifest-dir", default=None)

    restore_parser = sub.add_parser(
        "corpus-restore", help="restore an archive into a NEW location and verify it"
    )
    restore_parser.add_argument("--archive", required=True)
    restore_parser.add_argument("--destination", required=True)

    correct = sub.add_parser(
        "corpus-correct",
        help="record an append-only correction; never edits or deletes an observation",
    )
    correct.add_argument("--file", required=True, help="a committed correction file")
    correct.add_argument(
        "--dry-run", action="store_true", help="report what would be recorded, write nothing"
    )

    corrections_parser = sub.add_parser(
        "corpus-corrections",
        help="every correction on record, and what each one excludes from research",
    )
    corrections_parser.add_argument("--json", action="store_true")

    ops_status = sub.add_parser(
        "ops-status", help="collection health, storage, backup age and B4 readiness"
    )
    ops_status.add_argument("--json", action="store_true")
    ops_status.add_argument("--state-root", default=None)
    ops_status.add_argument(
        "--profile", default=None, help="also check the deployed collection profile loads"
    )

    watch = sub.add_parser("ops-watch", help="watchdog assessment; exit 0 ok, 1 warning, 2 critical")
    watch.add_argument("--json", action="store_true")
    watch.add_argument("--state-root", default=None)
    watch.add_argument(
        "--profile", default=None, help="also check the deployed collection profile loads"
    )

    scheduled = sub.add_parser(
        "collect-scheduled",
        help="one scheduled collection cycle; exit 0 ran, 2 failed, 3 lock held, 4 nothing due",
    )
    scheduled.add_argument("--profile", required=True, help="path to a collection profile")
    scheduled.add_argument("--state-root", default=None)
    scheduled.add_argument("--json", action="store_true")
    scheduled.add_argument(
        "--require-free-mb",
        type=int,
        default=64,
        help="refuse to start below this much free space",
    )

    paths_parser = sub.add_parser(
        "ops-paths", help="resolved persistent paths and whether they are usable"
    )
    paths_parser.add_argument("--state-root", default=None)

    config_check = sub.add_parser(
        "ops-config-check",
        help="is this host's configuration deployable; exit 0 yes, 2 no. Prints no secret",
    )
    config_check.add_argument("--profile", required=True, help="path to a collection profile")
    config_check.add_argument("--state-root", default=None)
    config_check.add_argument("--json", action="store_true")

    probe = sub.add_parser(
        "ops-probe",
        help="fetch each configured feed once, storing nothing; exit 0 all readable, 1 some, 2 none",
    )
    probe.add_argument("--profile", required=True, help="path to a collection profile")
    probe.add_argument("--json", action="store_true")

    scheduled_backup = sub.add_parser(
        "ops-backup",
        help="scheduled backup: lock, archive, verify by restoring, prune; exit 0 ok, 2 failed, 3 lock held",
    )
    scheduled_backup.add_argument("--state-root", default=None)
    scheduled_backup.add_argument(
        "--retain", type=int, default=30, help="scheduled archives to keep, newest first (default 30)"
    )
    scheduled_backup.add_argument(
        "--wait-seconds", type=int, default=600, help="how long to wait for a running cycle's lock (default 600)"
    )

    backup_verify = sub.add_parser(
        "corpus-backup-verify",
        help="restore an archive into a scratch location and verify it; exit 0 sound, 2 not",
    )
    which = backup_verify.add_mutually_exclusive_group(required=True)
    which.add_argument("--archive", default=None)
    which.add_argument("--latest", action="store_true", help="the newest archive in the backup directory")
    backup_verify.add_argument("--state-root", default=None)

    demo = sub.add_parser("demo")
    demo.add_argument("--output-dir", default="market-intelligence-demo")
    gold = sub.add_parser("gold-report")
    gold.add_argument("--output", default="gold-evaluation-report.json")
    return parser

def main(argv: list[str] | None = None) -> int:
    """Entry point. Operational failures exit 2 with a sentence, not a traceback.

    The exit code is a documented interface -- 0 ran, 2 failed, 3 locked, 4
    nothing due -- and a stack trace escaping to the shell breaks it twice over:
    it exits 1, which the contract does not mention, and it hands an operator
    fifteen lines of Python where one line would do. A programming error still
    raises, because a traceback is exactly what that needs.
    """
    try:
        return dispatch(build_parser().parse_args(argv))
    except IntelligenceError as error:
        # BackupError and ConfigurationError both live under this.
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_corpus_status",
    "_ops_report",
    "_provider_report",
    "build_parser",
    "command_names",
    "main",
]
