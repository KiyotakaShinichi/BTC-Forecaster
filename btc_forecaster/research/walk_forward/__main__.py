"""`python -m btc_forecaster.research.walk_forward` -- the A7 benchmark command.

    python -m btc_forecaster.research.walk_forward config
    python -m btc_forecaster.research.walk_forward run \\
        --data data/snapshots/BTC-USD --output research/runs/a7-walk-forward --workers 4
    python -m btc_forecaster.research.walk_forward run --world NOISE --smoke --output out/a7-smoke
    python -m btc_forecaster.research.walk_forward verify research/runs/a7-walk-forward

``--data`` takes a hash-manifested snapshot directory and verifies it on load.
``--world`` builds one of the synthetic worlds instead, so the whole pipeline can
be exercised where the market data may not be redistributed -- CI, for one.
``--expect-input`` refuses to run as a reproduction on any other input.

Exit codes: 0 success, 2 integrity failure or verification mismatch, 3 the input
is not the recorded one.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from ...data.snapshot import MarketSnapshot, SnapshotIntegrityError
from ...features.spec import default_specs
from .config import PREPROCESSING_VERSION, WalkForwardConfig, WindowSpec
from .manifest import InputChangedError, input_manifest, load_snapshot, require_recorded_input
from .report import readme_builder
from .runner import (
    assert_nothing_promoted,
    clean_statuses,
    run_benchmark,
    stored_name,
    verify_run,
    write_run,
)
from .targets import TARGET_DEFINITION
from .worlds import WORLDS
from .worlds import build as build_world

EXIT_OK = 0
EXIT_INTEGRITY = 2
EXIT_INPUT_CHANGED = 3

#: The benchmark CI runs: small enough for every push, large enough to exercise
#: both forecasting paths, a network, both window kinds and two horizons.
SMOKE_CONFIG = WalkForwardConfig(
    models=("naive_last_value", "random_walk_drift", "ar_p", "mlp"),
    horizons=(1, 3),
    windows=(WindowSpec("rolling", 150), WindowSpec("expanding")),
    n_refits=3,
    bootstrap_resamples=200,
    minimum_paired_origins=30,
)

#: A synthetic world has no retrieval time. A fixed one keeps its input manifest,
#: and therefore the run's digest, the same on every run.
SYNTHETIC_RETRIEVED_AT = pd.Timestamp("2000-01-01", tz="UTC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m btc_forecaster.research.walk_forward",
        description="A7: walk-forward robustness of the A6 negative result.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="print the configuration, its gates and its digest")
    config.add_argument("--smoke", action="store_true", help="the small CI configuration")

    run = commands.add_parser("run", help="run the benchmark and write a canonical result")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="snapshot directory (data.csv + manifest.json)")
    source.add_argument("--world", choices=sorted(WORLDS), help="a synthetic world instead of market data")
    run.add_argument("--output", required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--smoke", action="store_true", help="the small CI configuration")
    run.add_argument("--expect-input", help="refuse unless the input frame has this sha256")

    verify = commands.add_parser("verify", help="recompute every hash of a written run")
    verify.add_argument("path")
    return parser


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, object]:
    specs = tuple(spec.name for spec in default_specs())
    if args.data:
        snapshot, file_bytes = load_snapshot(args.data)
    else:
        world = build_world(args.world)
        snapshot = MarketSnapshot.build(
            world.frame,
            ticker=f"SYNTHETIC-{world.name}",
            provider="synthetic",
            retrieved_at=SYNTHETIC_RETRIEVED_AT,
            note=world.expectation,
        )
        file_bytes = None
    manifest = input_manifest(
        snapshot,
        file_bytes=file_bytes,
        preprocessing_version=PREPROCESSING_VERSION,
        feature_specs=specs,
        target=TARGET_DEFINITION,
    )
    return snapshot.frame, manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "config":
        config = SMOKE_CONFIG if args.smoke else WalkForwardConfig()
        print(json.dumps(config.as_dict(), indent=2, sort_keys=True))
        print(f"config digest: {config.digest()}")
        return EXIT_OK

    if args.command == "verify":
        try:
            result = verify_run(args.path)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_INTEGRITY
        print(json.dumps(result, indent=2, sort_keys=True))
        for name in result["absent_files"]:
            print(
                f"not verified: {stored_name(name)} is not in {args.path}. Regenerate the run "
                "from the pinned snapshot with `run`, then verify it.",
                file=sys.stderr,
            )
        return EXIT_OK if result["verified"] else EXIT_INTEGRITY

    try:
        frame, manifest = _load(args)
    except (SnapshotIntegrityError, InputChangedError, FileNotFoundError) as exc:
        print(f"INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return EXIT_INTEGRITY
    if args.expect_input:
        try:
            require_recorded_input(args.expect_input, manifest.frame_sha256)  # type: ignore[attr-defined]
        except InputChangedError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_INPUT_CHANGED

    config = SMOKE_CONFIG if args.smoke else WalkForwardConfig()
    print("A7 walk-forward robustness -- EXPLORATORY, never a promotion")
    print(f"  input   {manifest.ticker} {manifest.rows} bars, sha256 {manifest.frame_sha256[:16]}")  # type: ignore[attr-defined]
    print(f"  config  {config.digest()[:16]}  models {len(config.models)}  horizons {config.horizons}")
    print(f"  workers {args.workers}")

    outcome = run_benchmark(frame, config, workers=args.workers)
    assert_nothing_promoted(outcome)
    written = write_run(
        outcome,
        args.output,
        input_manifest=manifest,  # type: ignore[arg-type]
        readme=readme_builder(outcome, manifest),  # type: ignore[arg-type]
    )

    print(f"  folds   {clean_statuses(outcome)}")
    print(
        f"  timing  {len(outcome.timings['over_budget_folds'])} fold(s) over budget, "
        f"{len(outcome.timings['training_time_capped_folds'])} training-time-capped "
        "(run_info.json; not part of the result)"
    )
    print(f"  leakage {outcome.leakage}")
    print(f"  decision {outcome.decision['decision']}")
    print(f"  digest  {written['result_digest']}")
    print(f"  wall clock {outcome.timings['wall_clock_seconds']} s")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
