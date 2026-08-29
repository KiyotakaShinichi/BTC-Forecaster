"""The A2 benchmark: one canonical run producing one canonical table.

Composes the A2 pieces into a single reproducible study:

    snapshot -> outer folds -> every model, identical folds
             -> per-step metrics
             -> dependence-aware intervals and pairwise comparison
             -> regime stratification
             -> stability, failures, residuals, cost
             -> promotion verdicts under a pre-declared policy
             -> manifest, written last

Two ordering rules are load-bearing:

* **The promotion policy is fixed before the aggregates are inspected.** It is a
  constructor argument, recorded in the manifest, not something chosen after
  seeing the table.
* **The manifest is written last** (A2.22), after every artifact exists, so a
  manifest's presence means the run completed rather than that it started.

Nothing here overwrites a previous run: :func:`run_benchmark` writes into a
directory named by its run id and refuses to clobber an existing one.
"""

from __future__ import annotations

import platform
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts.writer import environment_fingerprint
from .backtesting.engine import BacktestResult, run_walk_forward
from .backtesting.splits import WalkForwardSplitter
from .data.snapshot import MarketSnapshot
from .evaluation.inference import (
    INFERENCE_NOTES,
    compare_models,
    directional_accuracy_ci,
    loss_series,
    stationary_bootstrap,
)
from .evaluation.promotion import PromotionPolicy, decide_all
from .evaluation.regimes import RegimeConfig, label_fold_origins, performance_by_regime
from .evaluation.stability import (
    failure_summary,
    largest_failures,
    residual_diagnostics,
    resource_costs,
    stability_by_period,
    stability_profile,
)
from .evaluation.targets import (
    TARGET_AUDIT,
    evaluate_by_step,
    first_scored_step,
    one_step_direction_sample,
)
from .evidence import PRESERVED, verify_all
from .models.base import ForecastModel
from .timebase import UTC

#: Files a completed benchmark writes. `manifest.json` is written last.
BENCHMARK_TABLE_CSV = "benchmark_table.csv"
PER_FOLD_CSV = "per_fold.csv"
PREDICTIONS_CSV = "predictions.csv"
PER_STEP_CSV = "per_step.csv"
REGIME_CSV = "by_regime.csv"
STABILITY_CSV = "stability.csv"
FAILURES_CSV = "largest_failures.csv"
COMPARISONS_CSV = "pairwise_comparisons.csv"
PROMOTION_CSV = "promotion.csv"
SELECTION_JSON = "fold_selection.json"
DIAGNOSTICS_JSON = "residual_diagnostics.json"
BENCHMARK_MANIFEST = "manifest.json"


@dataclass
class BenchmarkResult:
    """Everything one benchmark run produced."""

    run_id: str
    snapshot: MarketSnapshot
    backtest: BacktestResult
    splitter: WalkForwardSplitter
    policy: PromotionPolicy
    baseline: str
    #: Shortest forecast distance scored; equals embargo_bars + 1.
    scored_step: int = 1
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_step: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_period: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)
    promotion: pd.DataFrame = field(default_factory=pd.DataFrame)
    costs: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)
    direction_intervals: dict = field(default_factory=dict)
    skill_intervals: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    fold_selection: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.backtest.folds)


def _skill_interval(
    records: pd.DataFrame,
    model: str,
    baseline: str,
    *,
    step: int,
    seed: int,
) -> tuple[float, float] | None:
    """Bootstrap interval for MAE skill vs the baseline, at one horizon step.

    Skill is a ratio of means, so the interval is built by resampling the paired
    per-origin losses in blocks and recomputing the ratio on each resample --
    not by propagating two independent intervals, which would ignore the fact
    that both models saw the same origins.
    """
    step_rows = records[records["step"] == step]
    challenger = step_rows[step_rows["model"] == model].sort_values("origin")
    base = step_rows[step_rows["model"] == baseline].sort_values("origin")

    if len(challenger) < 8 or len(challenger) != len(base):
        return None

    challenger_loss = loss_series(
        challenger["actual"].to_numpy(float), challenger["predicted"].to_numpy(float)
    )
    base_loss = loss_series(base["actual"].to_numpy(float), base["predicted"].to_numpy(float))

    paired = np.stack([challenger_loss, base_loss], axis=1)
    index = np.arange(len(paired))

    def skill_of(resampled_positions: np.ndarray) -> float:
        rows = paired[resampled_positions.astype(int)]
        base_mean = rows[:, 1].mean()
        if abs(base_mean) < 1e-12:
            return float("nan")
        return float(1.0 - rows[:, 0].mean() / base_mean)

    interval = stationary_bootstrap(index, skill_of, seed=seed, null_value=0.0)
    return (interval.lower, interval.upper)


def build_benchmark_table(
    result: BacktestResult,
    *,
    baseline: str,
    stability: pd.DataFrame,
    costs: pd.DataFrame,
    direction_intervals: dict,
) -> pd.DataFrame:
    """The one canonical table (A2.20).

    Deliberately not sorted by a single metric, and not reduced to a rank. A
    model is a point in several dimensions -- error, direction, calibration,
    stability, cost -- and collapsing that to one number is how a benchmark
    starts telling people what they wanted to hear.
    """
    summary = result.summary()
    skill = result.skill_table(baseline)
    if summary.empty:
        return pd.DataFrame()

    table = pd.DataFrame(index=summary.index)
    table["n_folds"] = summary.get("n_folds_ok")
    table["n_failed"] = summary.get("n_folds_failed")

    for column in ("mae", "rmse", "mase", "directional_accuracy", "return_correlation"):
        if column in summary.columns:
            table[column] = summary[column]

    for column in (f"mae_skill_vs_{baseline}", f"rmse_skill_vs_{baseline}"):
        if column in skill.columns:
            table[column] = skill[column]

    for model in table.index:
        interval = direction_intervals.get(str(model))
        if interval:
            table.loc[model, "dir_acc_first_step"] = interval["statistic"]
            table.loc[model, "dir_ci_lower"] = interval["ci_lower"]
            table.loc[model, "dir_ci_upper"] = interval["ci_upper"]
            table.loc[model, "dir_beats_coin"] = interval["excludes_null"]

    for column in ("interval_coverage", "coverage_error", "relative_interval_width", "winkler_score"):
        if column in summary.columns:
            table[column] = summary[column]

    if not stability.empty:
        table["mae_std"] = stability["std"]
        table["mae_worst_to_median"] = stability["worst_to_median"]

    if not costs.empty:
        for column in (
            "fit_seconds_median",
            "predict_seconds_median",
            "cost_multiple_vs_cheapest",
        ):
            if column in costs.columns:
                table[column] = costs[column]

    return table


def run_benchmark(
    frame: pd.DataFrame,
    models: list[ForecastModel],
    *,
    snapshot: MarketSnapshot,
    splitter: WalkForwardSplitter,
    baseline: str = "random_walk",
    policy: PromotionPolicy | None = None,
    regime_config: RegimeConfig | None = None,
    seed: int = 42,
    run_id: str | None = None,
    skipped_models: dict[str, str] | None = None,
) -> BenchmarkResult:
    """Execute the full A2 study. Pure computation -- writing is separate."""
    policy = policy or PromotionPolicy()
    regime_config = regime_config or RegimeConfig()
    run_id = run_id or f"{pd.Timestamp.now(tz=UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"

    backtest = run_walk_forward(frame, models, splitter, baseline=baseline)
    backtest.skipped_models = skipped_models or {}

    records = backtest.prediction_records()
    per_fold = backtest.to_frame()
    model_names = [m.name for m in models]

    stability = stability_profile(per_fold)
    costs = resource_costs(per_fold)

    # Under an embargo the shortest scored forecast distance is embargo+1, not 1.
    scored_step = first_scored_step(records) if not records.empty else 1

    direction_intervals: dict = {}
    skill_intervals: dict = {}
    for name in model_names:
        model_records = records[records["model"] == name]
        if model_records.empty:
            continue
        try:
            hits = one_step_direction_sample(model_records, step=scored_step)
            direction_intervals[name] = directional_accuracy_ci(hits, seed=seed).to_dict()
        except ValueError:
            pass
        if name != baseline:
            interval = _skill_interval(records, name, baseline, step=scored_step, seed=seed)
            if interval is not None:
                skill_intervals[name] = interval

    per_step = pd.DataFrame()
    if not records.empty:
        rows = []
        for name in model_names:
            model_records = records[records["model"] == name]
            if model_records.empty:
                continue
            for metrics in evaluate_by_step(model_records):
                rows.append({"model": name, **metrics.to_dict()})
        per_step = pd.DataFrame(rows)

    origins = [fold.origin.last_observed_bar for fold in backtest.folds]
    origin_regimes = label_fold_origins(frame, origins, regime_config)
    by_regime = (
        performance_by_regime(records, origin_regimes) if not records.empty else pd.DataFrame()
    )

    comparisons = (
        compare_models(records, baseline=baseline, horizon=splitter.horizon, step=scored_step)
        if not records.empty
        else pd.DataFrame()
    )

    failures = (
        largest_failures(records, top_n=25, origin_regimes=origin_regimes)
        if not records.empty
        else pd.DataFrame()
    )

    residuals = {}
    for name in model_names:
        if not records[records["model"] == name].empty:
            residuals[name] = {
                "diagnostics": residual_diagnostics(records, model=name),
                "failure_concentration": failure_summary(records, model=name),
            }

    table = build_benchmark_table(
        backtest,
        baseline=baseline,
        stability=stability,
        costs=costs,
        direction_intervals=direction_intervals,
    )

    promotion = decide_all(
        backtest.summary(),
        backtest.skill_table(baseline),
        stability,
        costs,
        baseline=baseline,
        n_folds=len(backtest.folds),
        skill_intervals=skill_intervals,
        policy=policy,
    )

    fold_selection = {
        m.name: m.tuning.to_dict() for m in models if hasattr(m, "tuning") and m.is_fitted
    }

    return BenchmarkResult(
        run_id=run_id,
        scored_step=scored_step,
        snapshot=snapshot,
        backtest=backtest,
        splitter=splitter,
        policy=policy,
        baseline=baseline,
        table=table,
        per_step=per_step,
        by_regime=by_regime,
        stability=stability,
        by_period=stability_by_period(per_fold) if not per_fold.empty else pd.DataFrame(),
        comparisons=comparisons,
        promotion=promotion,
        costs=costs,
        failures=failures,
        direction_intervals=direction_intervals,
        skill_intervals=skill_intervals,
        residuals=residuals,
        fold_selection=fold_selection,
    )


def write_benchmark(result: BenchmarkResult, output_dir: Path | str) -> list[str]:
    """Write every artifact, then the manifest last (A2.22).

    Refuses to write into a directory that already holds a manifest: completed
    research runs are never overwritten.
    """
    from .artifacts.writer import ArtifactWriter

    path = Path(output_dir)
    if (path / BENCHMARK_MANIFEST).exists():
        raise FileExistsError(
            f"{path} already holds a completed benchmark run. Research runs are "
            "immutable -- write to a new directory."
        )

    writer = ArtifactWriter(path)

    frames = {
        BENCHMARK_TABLE_CSV: (result.table, True),
        PER_FOLD_CSV: (result.backtest.to_frame(), False),
        PREDICTIONS_CSV: (result.backtest.prediction_records(), False),
        PER_STEP_CSV: (result.per_step, False),
        REGIME_CSV: (result.by_regime, True),
        STABILITY_CSV: (result.stability, True),
        FAILURES_CSV: (result.failures, False),
        COMPARISONS_CSV: (result.comparisons, True),
        PROMOTION_CSV: (result.promotion, True),
    }
    for filename, (frame, with_index) in frames.items():
        if frame is not None and not frame.empty:
            writer.write_frame(filename, frame, index=with_index)

    writer.write_json(SELECTION_JSON, result.fold_selection)
    writer.write_json(DIAGNOSTICS_JSON, result.residuals)

    # Written last: a manifest means the run completed.
    writer.write_json(BENCHMARK_MANIFEST, benchmark_manifest(result))
    return [entry["name"] for entry in writer.listing()]


def benchmark_manifest(result: BenchmarkResult) -> dict:
    """Everything needed to reproduce or audit the run (A2.22)."""
    return {
        "run_id": result.run_id,
        "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "dataset": result.snapshot.manifest.to_dict(),
        "outer_folds": {
            "splitter": result.splitter.describe(),
            "n_folds": result.n_folds,
            "folds": [fold.to_dict() for fold in result.backtest.folds],
        },
        "models": result.backtest.model_descriptions,
        "skipped_models": result.backtest.skipped_models,
        "baseline": result.baseline,
        "fold_selection": result.fold_selection,
        "target_audit": TARGET_AUDIT,
        "inference_notes": INFERENCE_NOTES,
        "promotion_policy": result.policy.to_dict(),
        "promotion_decisions": (
            [] if result.promotion.empty
            else result.promotion.reset_index().to_dict(orient="records")
        ),
        "benchmark_table": (
            [] if result.table.empty else result.table.reset_index().to_dict(orient="records")
        ),
        "scored_step": result.scored_step,
        "direction_intervals": result.direction_intervals,
        "skill_intervals": {k: list(v) for k, v in result.skill_intervals.items()},
        "residual_diagnostics": result.residuals,
        "preserved_evidence": {
            "verified": dict(verify_all()),
            "records": [r.to_dict() for r in PRESERVED],
        },
        "environment": {
            **environment_fingerprint(),
            "python_full": sys.version,
            "platform_machine": platform.machine(),
        },
        "n_backtest_failures": len(result.backtest.failures()),
    }


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Console report for the run log."""
    lines: list[str] = []
    add = lines.append

    add("=" * 100)
    add(f"A2 BENCHMARK  run_id={result.run_id}")
    add("=" * 100)

    manifest = result.snapshot.manifest
    add(f"data     : {manifest.rows} bars {manifest.start[:10]}..{manifest.end[:10]}")
    add(f"           sha256={manifest.sha256[:16]} provider={manifest.provider}")
    walk = result.splitter.describe()
    add(
        f"folds    : {result.n_folds} {walk['mode']}, horizon={walk['horizon']}, "
        f"embargo={walk['embargo_bars']}, min_train={walk['min_train_bars']}"
    )
    add(f"baseline : {result.baseline}")
    embargo = walk["embargo_bars"]
    add(
        f"direction: step {result.scored_step} (embargo {embargo} + 1), "
        "one observation per origin"
    )

    if not result.table.empty:
        add("")
        add("BENCHMARK TABLE")
        columns = [
            c
            for c in (
                "mae", f"mae_skill_vs_{result.baseline}", "rmse", "mase",
                "dir_acc_first_step", "dir_ci_lower", "dir_ci_upper",
                "interval_coverage", "mae_worst_to_median", "cost_multiple_vs_cheapest",
            )
            if c in result.table.columns
        ]
        add(result.table[columns].sort_values("mae").to_string(float_format=lambda v: f"{v:,.4f}"))

    if not result.promotion.empty:
        add("")
        add("PROMOTION DECISIONS")
        for model, row in result.promotion.iterrows():
            add(f"  {model:<26} {row['decision']}")
            add(f"      {row['rationale']}")

    if not result.comparisons.empty:
        usable = result.comparisons[result.comparisons["usable"]]
        add("")
        add(
            f"PAIRWISE vs {result.baseline} "
            f"(step {result.scored_step}, Diebold-Mariano)"
        )
        if usable.empty:
            add("  every comparison is nested; see the bootstrap intervals instead")
        else:
            add(
                usable[["dm_statistic", "p_value", "p_adjusted", "significant_after_correction"]]
                .to_string(float_format=lambda v: f"{v:,.4f}")
            )
        nested = result.comparisons[~result.comparisons["usable"]]
        if not nested.empty:
            add(f"  nested, not tested: {', '.join(nested.index)}")

    add("=" * 100)
    return "\n".join(lines)


__all__ = [
    "BENCHMARK_MANIFEST",
    "BENCHMARK_TABLE_CSV",
    "BenchmarkResult",
    "benchmark_manifest",
    "build_benchmark_table",
    "format_benchmark_report",
    "run_benchmark",
    "write_benchmark",
]
