"""`python -m btc_forecaster.research.model_zoo` -- the A6 benchmark command.

Deliberately an explicit research command rather than something CI runs on every
push. The full zoo fits forty models; the deep family alone trains seven
networks. CI runs the registry checks, the synthetic behaviour suite, the
leakage adversaries and a small smoke benchmark. This is what produces the
evidence under `research/runs/a6-model-zoo/`.

    python -m btc_forecaster.research.model_zoo --list
    python -m btc_forecaster.research.model_zoo --train-rows 1000 --output research/runs/a6-model-zoo
    python -m btc_forecaster.research.model_zoo --sample-efficiency

Every list it prints comes from the registry. There is no second copy of the
model set here, in the README, or in the tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ..data.snapshot import read_canonical_csv
from . import registry
from .contracts import EXPLORATORY, ModelStatus
from .partition import PartitionSpec
from .runner import assert_nothing_promoted, build_manifest, run_benchmark, write_run

#: The curated subset for the sample-efficiency mini-study. Phase 14 is explicit
#: that the whole zoo must not be re-run at every budget; seven models across
#: five families is enough to see whether the ranking is stable in n.
SAMPLE_EFFICIENCY_MODELS = (
    "random_walk_drift",
    "arima",
    "ridge",
    "xgboost",
    "mlp",
    "lstm",
    "transformer",
)
SAMPLE_EFFICIENCY_BUDGETS = (250, 500, 1000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m btc_forecaster.research.model_zoo",
        description=(
            "Run the A6 resource-constrained model zoo. Exploratory evidence "
            "only: this does not supersede A2 and cannot promote a model."
        ),
    )
    parser.add_argument(
        "--data",
        default="data/snapshots/BTC-USD/data.csv",
        help="hash-verified market snapshot to run on",
    )
    parser.add_argument("--train-rows", type=int, default=1000)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--output", default=None, help="directory to write the run into")
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated subset; default is every registered model",
    )
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument(
        "--sample-efficiency",
        action="store_true",
        help=f"also run {len(SAMPLE_EFFICIENCY_MODELS)} models at {SAMPLE_EFFICIENCY_BUDGETS}",
    )
    return parser


def print_registry() -> int:
    """Everything registered, generated from the registry and nowhere else."""
    summary = registry.summary()
    print(f"{summary['total_registered']} models registered, {summary['runnable']} runnable\n")
    print(f"  {'model_id':26s} {'family':16s} {'status':32s} requires")
    print(f"  {'-' * 26} {'-' * 16} {'-' * 32} {'-' * 20}")
    for registration in registry.all_registrations():
        status = registration.effective_status()
        requires = ",".join(registration.requires) or "-"
        print(
            f"  {registration.model_id:26s} {registration.family.value:16s} "
            f"{status.value:32s} {requires}"
        )
        if status is ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB:
            print(f"      reason: {registration.unsuitable_reason}")
    print()
    for status_name, ids in registry_counts().items():
        print(f"  {status_name:34s} {len(ids)}")
    return 0


def registry_counts() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for registration in registry.all_registrations():
        grouped.setdefault(registration.effective_status().value, []).append(
            registration.model_id
        )
    return {k: sorted(v) for k, v in sorted(grouped.items())}


def run_sample_efficiency(frame: pd.DataFrame, spec: PartitionSpec) -> list[dict]:
    """A curated subset at three budgets. Not a promotion study either.

    The question is whether the ranking is stable in n, which is a statement
    about how much data these architectures need before they stop being noise --
    not about which one to use.
    """
    rows: list[dict] = []
    for budget in SAMPLE_EFFICIENCY_BUDGETS:
        narrowed = PartitionSpec(
            train_fraction=spec.train_fraction,
            dev_fraction=spec.dev_fraction,
            train_rows=budget,
            step=spec.step,
        )
        result = run_benchmark(frame, spec=narrowed, model_ids=list(SAMPLE_EFFICIENCY_MODELS))
        for outcome in result.outcomes:
            row = {
                "train_rows": budget,
                "model_id": outcome.model_id,
                "status": outcome.status.value,
                "scientific_status": EXPLORATORY,
            }
            if outcome.scores is not None:
                row["mae"] = outcome.scores.point.mae
                row["skill_vs_naive"] = outcome.scores.point.skill_vs_naive
                row["directional_accuracy"] = outcome.scores.direction.accuracy
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return print_registry()

    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"error: no snapshot at {data_path}.\n"
            "  The market snapshot is not committed -- Yahoo's terms do not grant\n"
            "  redistribution -- so run `btc-forecast snapshot` first, or pass\n"
            "  --data to an existing one.",
            file=sys.stderr,
        )
        return 2

    frame = read_canonical_csv(data_path)
    spec = PartitionSpec(
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        train_rows=args.train_rows,
    )
    model_ids = args.models.split(",") if args.models else None

    print(f"A6 model zoo -- {EXPLORATORY} evidence, not a promotion study")
    print(f"  data           {data_path} ({len(frame)} bars)")
    print(f"  training rows  {args.train_rows} (deterministic tail of TRAIN)")
    print(f"  models         {len(model_ids) if model_ids else len(registry.model_ids())}")
    print()

    result = run_benchmark(frame, spec=spec, model_ids=model_ids)

    extra: dict = {}
    if args.sample_efficiency:
        print("  running the sample-efficiency subset...")
        extra["sample_efficiency"] = {
            "models": list(SAMPLE_EFFICIENCY_MODELS),
            "budgets": list(SAMPLE_EFFICIENCY_BUDGETS),
            "purpose": (
                "whether the ranking is stable in n; explicitly not a basis for "
                "choosing a model"
            ),
            "rows": run_sample_efficiency(frame, spec),
        }

    manifest = build_manifest(result, extra=extra)
    assert_nothing_promoted(manifest)

    frame_out = result.results_frame()
    if len(frame_out):
        print(frame_out.head(15).to_string(index=False))
    print()
    for status, ids in result.by_status().items():
        print(f"  {status:34s} {len(ids):3d}  {', '.join(ids[:5])}{' ...' if len(ids) > 5 else ''}")
    print(f"\n  wall clock {result.seconds:.1f}s")
    print("  best by family (never a global winner, never a promotion):")
    for family, best in sorted(result.best_by_family().items()):
        print(f"    {family:16s} {best['model_id']:24s} skill {best['skill_vs_naive']:+.5f}")

    if args.output:
        path = write_run(result, args.output, extra=extra)
        print(f"\n  wrote {path}")
    else:
        print("\n  (no --output given; nothing written)")
        print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SAMPLE_EFFICIENCY_BUDGETS",
    "SAMPLE_EFFICIENCY_MODELS",
    "build_parser",
    "main",
    "print_registry",
    "run_sample_efficiency",
]
