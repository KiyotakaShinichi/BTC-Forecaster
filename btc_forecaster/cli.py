"""Command-line entry point: ``btc-forecast``.

Subcommands:

    benchmark    the A2 study: many folds, nested tuning, promotion verdicts
    run          fetch/reuse data, backtest every model, forecast forward
    backtest     backtest only -- no forward forecast, no plots
    diagnose     stationarity/autocorrelation/heteroskedasticity report
    snapshot     fetch and pin a market data snapshot
    models       list registered models and whether they can be built

Every flag has an environment-variable equivalent (see
:class:`btc_forecaster.config.settings.RunConfig`), so the Docker images and
shell invocations that predate this CLI keep working unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config.settings import RunConfig, WalkForwardConfig
from .models import registry


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start", default=None, help="earliest bar to load (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--snapshot-dir", default=None, type=Path)
    parser.add_argument("--refresh-data", action="store_true", help="re-fetch instead of reusing the snapshot")


def _add_backtest_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default=None, help="comma-separated model names")
    parser.add_argument("--baseline", default=None, help="model to measure skill against")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--wf-horizon", type=int, default=None, help="bars scored per fold")
    parser.add_argument("--min-train-bars", type=int, default=None)
    parser.add_argument("--mode", choices=["expanding", "rolling"], default=None)
    parser.add_argument("--window-bars", type=int, default=None, help="training length in rolling mode")
    parser.add_argument("--embargo-bars", type=int, default=None)


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig.from_env()

    walk = config.walk_forward
    walk_overrides = {
        "n_folds": args.folds,
        "horizon": getattr(args, "wf_horizon", None),
        "min_train_bars": args.min_train_bars,
        "mode": args.mode,
        "window_bars": args.window_bars,
        "embargo_bars": args.embargo_bars,
    }
    walk = replace(walk, **{k: v for k, v in walk_overrides.items() if v is not None})

    overrides: dict = {"walk_forward": walk}
    if args.ticker:
        overrides["ticker"] = args.ticker
    if args.start:
        overrides["start"] = args.start
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.snapshot_dir:
        overrides["snapshot_dir"] = args.snapshot_dir
    if args.refresh_data:
        overrides["refresh_data"] = True
    if getattr(args, "models", None):
        overrides["models"] = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if getattr(args, "baseline", None):
        overrides["baseline_model"] = args.baseline
    if getattr(args, "horizon", None):
        overrides["horizon_days"] = args.horizon
    if getattr(args, "primary_model", None):
        overrides["primary_model"] = args.primary_model
    if getattr(args, "no_plots", False):
        overrides["make_plots"] = False

    # The baseline must be among the scored models, or skill is unmeasurable.
    models = overrides.get("models", config.models)
    baseline = overrides.get("baseline_model", config.baseline_model)
    if baseline not in models:
        overrides["models"] = tuple(models) + (baseline,)

    primary = overrides.get("primary_model", config.primary_model)
    if primary not in overrides.get("models", config.models):
        overrides["primary_model"] = overrides["models"][0]

    return replace(config, **overrides)


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import format_report, run_forecast

    config = _config_from_args(args)
    outcome = run_forecast(config)
    print(format_report(outcome))
    print(f"\nartifacts written to {config.output_dir.resolve()}:")
    for name in outcome.artifacts:
        print(f"  - {name}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtesting.engine import rank_models, run_walk_forward
    from .backtesting.splits import WalkForwardSplitter
    from .pipeline import build_models, load_market_data

    config = _config_from_args(args)
    snapshot = load_market_data(config)
    models, skipped = build_models(config)
    if not models:
        print(f"no models could be built; skipped: {skipped}", file=sys.stderr)
        return 1

    walk: WalkForwardConfig = config.walk_forward
    splitter = WalkForwardSplitter(
        horizon=walk.horizon,
        n_folds=walk.n_folds,
        min_train_bars=walk.min_train_bars,
        mode=walk.mode,  # type: ignore[arg-type]
        window_bars=walk.window_bars,
        embargo_bars=walk.embargo_bars,
    )

    result = run_walk_forward(snapshot.frame, models, splitter, baseline=config.baseline_model)

    print(f"{snapshot.describe()}\n")
    print(rank_models(result, metric=args.metric).to_string(float_format=lambda v: f"{v:,.4f}"))

    skill = result.skill_table(config.baseline_model)
    if not skill.empty:
        print(f"\nskill vs {config.baseline_model} (positive = better):")
        print(skill.to_string(float_format=lambda v: f"{v:,.4f}"))

    for name, reason in skipped.items():
        print(f"\nskipped {name}: {reason}", file=sys.stderr)
    for failure in result.failures():
        print(f"failed  {failure.model} fold {failure.fold}: {failure.error}", file=sys.stderr)
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .pipeline import load_market_data, run_diagnostics

    config = _config_from_args(args)
    snapshot = load_market_data(config)
    report = run_diagnostics(snapshot.frame, max_lag=args.max_lag)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    from .diagnostics.suite import diagnose_prices, diagnose_returns
    from .features.spec import simple_returns

    print(snapshot.describe())
    print()
    print(diagnose_returns(simple_returns(snapshot.frame).dropna(), max_lag=args.max_lag).summary())
    print()
    print(diagnose_prices(snapshot.frame["close"]).summary())
    print()
    corrected = report["returns_multiple_testing"]
    print(f"after {corrected['method']} correction over {corrected['n_tests']} tests:")
    for row in corrected["results"]:
        mark = "significant" if row["significant_after_correction"] else "not significant"
        print(f"  {row['name']:<16} p={row['p_value']:.4g} -> {row['p_adjusted']:.4g} ({mark})")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    from .pipeline import load_market_data

    config = replace(_config_from_args(args), refresh_data=True)
    snapshot = load_market_data(config)
    snapshot.save(config.snapshot_path)
    print(snapshot.describe())
    print(f"saved to {config.snapshot_path.resolve()}")
    return 0


#: The A2 benchmark model set (A2.6). Deliberately short: naive baselines, a
#: representative statistical family, the preserved leakage-corrected reference,
#: and one causal challenger. Model-count inflation buys nothing.
BENCHMARK_MODELS: tuple[str, ...] = (
    "random_walk",
    "random_walk_drift",
    "historical_mean_return",
    "arima",
    "ets",
    "prophet",
    "prophet_xgb_hybrid",
    "xgboost_causal_retuned",
)


def cmd_benchmark(args: argparse.Namespace) -> int:
    from dataclasses import replace as dc_replace

    from .backtesting.splits import WalkForwardSplitter
    from .benchmark import format_benchmark_report, run_benchmark, write_benchmark
    from .evaluation.promotion import PromotionPolicy
    from .pipeline import build_models, load_market_data

    config = _config_from_args(args)
    if not getattr(args, "models", None):
        config = dc_replace(config, models=BENCHMARK_MODELS)

    snapshot = load_market_data(config)
    models, skipped = build_models(config)
    if not models:
        print(f"no models could be built; skipped: {skipped}", file=sys.stderr)
        return 1

    walk = config.walk_forward
    splitter = WalkForwardSplitter(
        horizon=walk.horizon,
        n_folds=walk.n_folds,
        min_train_bars=walk.min_train_bars,
        mode=walk.mode,  # type: ignore[arg-type]
        window_bars=walk.window_bars,
        embargo_bars=walk.embargo_bars,
    )

    result = run_benchmark(
        snapshot.frame,
        models,
        snapshot=snapshot,
        splitter=splitter,
        baseline=config.baseline_model,
        policy=PromotionPolicy(),
        seed=config.random_state,
        skipped_models=skipped,
    )

    print(format_benchmark_report(result))

    target = Path(args.run_dir) if args.run_dir else config.output_dir / f"benchmark-{result.run_id}"
    result.artifacts = write_benchmark(result, target)
    print(f"\nartifacts written to {target.resolve()}:")
    for name in result.artifacts:
        print(f"  - {name}")

    for name, reason in skipped.items():
        print(f"skipped {name}: {reason}", file=sys.stderr)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    report = registry.available()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    width = max(len(name) for name in report)
    for name, info in report.items():
        status = "ok" if info["available"] else f"MISSING {info['missing_dependency']}"
        print(f"{name:<{width}}  {info['family']:<12} {status:<18} {info['description']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btc-forecast",
        description="Point-in-time correct forecasting research platform for BTC-USD.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark", help="the A2 study: many folds, nested tuning, promotion verdicts"
    )
    _add_common(benchmark)
    _add_backtest_options(benchmark)
    benchmark.add_argument(
        "--run-dir", default=None, help="where to write the run (must not already exist)"
    )
    benchmark.set_defaults(func=cmd_benchmark)

    run = subparsers.add_parser("run", help="backtest every model, then forecast forward")
    _add_common(run)
    _add_backtest_options(run)
    run.add_argument("--horizon", type=int, default=None, help="forward forecast length in days")
    run.add_argument("--primary-model", default=None, help="model whose forecast is the headline")
    run.add_argument("--no-plots", action="store_true")
    run.set_defaults(func=cmd_run)

    backtest = subparsers.add_parser("backtest", help="walk-forward comparison only")
    _add_common(backtest)
    _add_backtest_options(backtest)
    backtest.add_argument("--metric", default="mae", help="metric to rank by")
    backtest.set_defaults(func=cmd_backtest)

    diagnose = subparsers.add_parser("diagnose", help="statistical diagnostics on the series")
    _add_common(diagnose)
    diagnose.add_argument("--max-lag", type=int, default=20)
    diagnose.add_argument("--json", action="store_true")
    diagnose.set_defaults(func=cmd_diagnose)

    snapshot = subparsers.add_parser("snapshot", help="fetch and pin a data snapshot")
    _add_common(snapshot)
    snapshot.set_defaults(func=cmd_snapshot)

    models = subparsers.add_parser("models", help="list registered models and availability")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
