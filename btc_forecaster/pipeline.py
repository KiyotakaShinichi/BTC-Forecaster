"""End-to-end run orchestration.

Replaces the 700-line module-level script in ``bayesianCutoff.py``, which
downloaded data, fitted models, wrote files and printed a report as a side
effect of being imported -- so it could not be tested, called twice, or used as
a library.

The order matters and is fixed:

1. Load a hash-verified data snapshot (fetching only if absent or refreshed).
2. Diagnose the series, and record how many hypotheses that involved.
3. Backtest every requested model through **identical** walk-forward folds.
4. Only then fit on the full history and produce the forward forecast.

Step 3 before step 4 is deliberate. The forward forecast is the thing people
look at, and the backtest is the thing that says whether to believe it. Running
the backtest first means the summary cannot be written without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts.writer import ArtifactWriter, environment_fingerprint
from .backtesting.engine import BacktestResult, run_walk_forward
from .backtesting.splits import WalkForwardSplitter
from .config.settings import RunConfig
from .data.providers import MarketDataProvider, load_or_fetch
from .data.snapshot import MarketSnapshot
from .diagnostics.suite import diagnose_prices, diagnose_returns
from .evaluation.metrics import skill_score
from .features.spec import simple_returns
from .models import registry
from .models.base import ForecastModel, ForecastResult, TrainingWindow
from .timebase import UTC


@dataclass
class RunOutcome:
    """Everything a run produced."""

    config: RunConfig
    snapshot: MarketSnapshot
    backtest: BacktestResult
    forecasts: dict[str, ForecastResult] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    skipped_models: dict[str, str] = field(default_factory=dict)
    forecast_errors: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    @property
    def primary(self) -> ForecastResult | None:
        return self.forecasts.get(self.config.primary_model)

    def summary(self) -> dict:
        """The headline numbers, with the honest caveats attached."""
        backtest_summary = self.backtest.summary()
        skill = self.backtest.skill_table(self.config.baseline_model)
        primary = self.primary
        name = self.config.primary_model
        baseline = self.config.baseline_model

        beat_baseline = None
        skill_column = f"mae_skill_vs_{baseline}"
        if not skill.empty and skill_column in skill.columns and name in skill.index:
            value = skill.loc[name, skill_column]
            beat_baseline = bool(np.isfinite(value) and value > 0)

        payload: dict = {
            "ticker": self.config.ticker,
            "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
            "data": {
                "provider": self.snapshot.manifest.provider,
                "sha256": self.snapshot.manifest.sha256,
                "rows": self.snapshot.manifest.rows,
                "start": self.snapshot.manifest.start,
                "end": self.snapshot.manifest.end,
                "retrieved_at": self.snapshot.manifest.retrieved_at,
            },
            "primary_model": name,
            "baseline_model": baseline,
            "horizon_days": self.config.horizon_days,
            "interval_level": self.config.interval_level,
            "walk_forward": self.backtest.splitter.describe(),
            "n_folds": len(self.backtest.folds),
            "model_comparison": (
                [] if backtest_summary.empty
                else backtest_summary.reset_index().to_dict(orient="records")
            ),
            "skill_vs_baseline": (
                [] if skill.empty else skill.reset_index().to_dict(orient="records")
            ),
            "primary_beats_baseline": beat_baseline,
            "direction_test": self.backtest.direction_test(name),
            "skipped_models": self.skipped_models,
            "forecast_errors": self.forecast_errors,
            "backtest_failures": len(self.backtest.failures()),
            "interpretation": _interpretation(beat_baseline, name, baseline),
        }

        if primary is not None:
            index = primary.index
            payload["forecast"] = {
                "origin_bar": str(primary.origin.last_observed_bar.date()),
                "issued_at": primary.origin.timestamp.isoformat(),
                "first_bar": str(index[0].date()),
                "last_bar": str(index[-1].date()),
                "current_price": float(self.snapshot.frame["close"].iloc[-1]),
                "point_30d": _at(primary.point, 29),
                "point_90d": _at(primary.point, 89),
                "point_final": float(primary.point.iloc[-1]),
                "interval_low_final": None if primary.lower is None else float(primary.lower.iloc[-1]),
                "interval_high_final": None if primary.upper is None else float(primary.upper.iloc[-1]),
            }

        return payload


def _at(series: pd.Series, position: int) -> float:
    return float(series.iloc[min(position, len(series) - 1)])


def _interpretation(beat_baseline: bool | None, model: str, baseline: str) -> str:
    if beat_baseline is None:
        return (
            f"{model} could not be scored against {baseline}. Treat the forward forecast as "
            "unvalidated."
        )
    if beat_baseline:
        return (
            f"{model} reduced walk-forward MAE relative to {baseline} across identical folds. "
            "That is evidence of skill, not proof: check the fold-level dispersion and the "
            "interval coverage before acting on it."
        )
    return (
        f"{model} did NOT beat {baseline} on walk-forward MAE across identical folds. The "
        "forward forecast should be read as a scenario, not a prediction, and the added "
        "complexity is not currently earning anything."
    )


def load_market_data(
    config: RunConfig, *, provider: MarketDataProvider | None = None
) -> MarketSnapshot:
    """Fetch or reuse a hash-verified snapshot for this run."""
    return load_or_fetch(
        config.ticker,
        cache_dir=config.snapshot_path,
        start=config.start,
        provider=provider,
        refresh=config.refresh_data,
    )


def build_models(config: RunConfig) -> tuple[list[ForecastModel], dict[str, str]]:
    """Construct requested models, reporting any whose dependencies are missing."""
    kwargs_by_model = {
        "prophet_xgb_hybrid": {
            "monte_carlo_runs": config.monte_carlo_runs,
            "random_state": config.random_state,
            "interval_level": config.interval_level,
        },
    }

    built: list[ForecastModel] = []
    skipped: dict[str, str] = {}
    for name in config.models:
        try:
            built.append(registry.build(name, **kwargs_by_model.get(name, {})))
        except registry.MissingDependencyError as exc:  # type: ignore[attr-defined]
            skipped[name] = str(exc).splitlines()[0]
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
    return built, skipped


def run_diagnostics(frame: pd.DataFrame, *, max_lag: int = 20) -> dict:
    returns = simple_returns(frame).dropna()
    returns_report = diagnose_returns(returns, max_lag=max_lag)
    price_report = diagnose_prices(frame["close"])

    return {
        "returns": returns_report.to_dict(),
        "returns_multiple_testing": returns_report.adjusted("benjamini-hochberg"),
        "log_price": price_report.to_dict(),
    }


def run_forecast(
    config: RunConfig | None = None,
    *,
    provider: MarketDataProvider | None = None,
    frame: pd.DataFrame | None = None,
    write: bool = True,
) -> RunOutcome:
    """Execute a full run.

    ``frame`` bypasses data loading entirely, which is how the tests exercise the
    whole pipeline without a network call or a snapshot on disk.
    """
    config = config or RunConfig.from_env()

    if frame is not None:
        snapshot = MarketSnapshot.build(
            frame, ticker=config.ticker, provider="supplied", normalise=False
        )
    else:
        snapshot = load_market_data(config, provider=provider)

    market = snapshot.frame
    diagnostics = run_diagnostics(market)

    models, skipped = build_models(config)
    if not models:
        raise RuntimeError(
            f"no models could be built from {list(config.models)}; skipped: {skipped}"
        )

    splitter = WalkForwardSplitter(
        horizon=config.walk_forward.horizon,
        n_folds=config.walk_forward.n_folds,
        min_train_bars=config.walk_forward.min_train_bars,
        mode=config.walk_forward.mode,  # type: ignore[arg-type]
        window_bars=config.walk_forward.window_bars,
        embargo_bars=config.walk_forward.embargo_bars,
    )

    backtest = run_walk_forward(market, models, splitter, baseline=config.baseline_model)
    backtest.skipped_models = skipped

    # Only now: refit on everything and look forward.
    full_window = TrainingWindow(market)
    forecasts: dict[str, ForecastResult] = {}
    forecast_errors: dict[str, str] = {}
    for model in models:
        try:
            forecasts[model.name] = model.fit_predict(full_window, config.horizon_days)
        except Exception as exc:
            forecast_errors[model.name] = f"{type(exc).__name__}: {exc}"

    outcome = RunOutcome(
        config=config,
        snapshot=snapshot,
        backtest=backtest,
        forecasts=forecasts,
        diagnostics=diagnostics,
        skipped_models=skipped,
        forecast_errors=forecast_errors,
    )

    if write:
        outcome.artifacts = write_artifacts(outcome)
    return outcome


def write_artifacts(outcome: RunOutcome) -> list[str]:
    """Write every artifact for a completed run and return their names."""
    config = outcome.config
    writer = ArtifactWriter(config.output_dir)
    market = outcome.snapshot.frame

    primary = outcome.primary
    if primary is None and outcome.forecasts:
        primary = next(iter(outcome.forecasts.values()))

    if primary is not None:
        forecast_frame = primary.to_frame()
        for name, result in outcome.forecasts.items():
            forecast_frame[f"{name}__point"] = result.point
        writer.write_forecast(forecast_frame)

    writer.write_history(market["close"])
    writer.write_backtest(outcome.backtest.to_frame(), outcome.backtest.summary())
    writer.write_diagnostics(outcome.diagnostics)
    writer.write_summary(outcome.summary())
    writer.write_manifest(
        {
            "config": config.to_dict(),
            "data_snapshot": outcome.snapshot.manifest.to_dict(),
            "backtest": outcome.backtest.to_manifest(),
            "environment": environment_fingerprint(),
            "forecast_errors": outcome.forecast_errors,
        }
    )

    if config.make_plots:
        _write_plots(outcome, writer, primary)

    return [entry["name"] for entry in writer.listing()]


def _write_plots(outcome: RunOutcome, writer: ArtifactWriter, primary) -> None:
    from .artifacts.plots import (
        plot_forecast,
        plot_interval_calibration,
        plot_model_comparison,
        plot_pacf_diagnostic,
    )

    config = outcome.config
    market = outcome.snapshot.frame

    if primary is not None:
        plot_forecast(
            market["close"],
            primary.to_frame(),
            writer.path("hybrid_forecast_montecarlo.png"),
            title=f"{config.ticker} {config.horizon_days}-day forecast ({primary.model})",
            show=config.show_plots,
        )

    plot_pacf_diagnostic(
        simple_returns(market),
        writer.path("pacf_diagnostic.png"),
        max_lag=config.max_lag,
        title=f"PACF of {config.ticker} returns",
        show=config.show_plots,
    )
    plot_model_comparison(
        outcome.backtest.summary(),
        writer.path("model_comparison.png"),
        baseline=config.baseline_model,
        show=config.show_plots,
    )
    plot_interval_calibration(
        outcome.backtest.to_frame(),
        writer.path("interval_calibration.png"),
        show=config.show_plots,
    )


def format_report(outcome: RunOutcome) -> str:
    """A plain-text run report for the console and the run log."""
    summary = outcome.summary()
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"BTC-Forecaster run - {summary['ticker']}")
    add("=" * 78)

    data = summary["data"]
    add(f"data      : {data['rows']} bars {data['start'][:10]}..{data['end'][:10]}")
    add(f"            provider={data['provider']} sha256={data['sha256'][:12]}")

    walk = summary["walk_forward"]
    add(
        f"backtest  : {summary['n_folds']} {walk['mode']} folds, horizon={walk['horizon']}, "
        f"embargo={walk['embargo_bars']}, min_train={walk['min_train_bars']}"
    )

    table = outcome.backtest.summary()
    if not table.empty:
        add("")
        add("model comparison (walk-forward means, lower MAE is better):")
        columns = [c for c in ("mae", "rmse", "mase", "directional_accuracy", "interval_coverage") if c in table.columns]
        add(table[columns].sort_values("mae").to_string(float_format=lambda v: f"{v:,.4f}"))

    if summary["skipped_models"]:
        add("")
        add("skipped models (missing dependencies):")
        for name, reason in summary["skipped_models"].items():
            add(f"  - {name}: {reason}")

    if summary["backtest_failures"]:
        add("")
        add(f"warning: {summary['backtest_failures']} model/fold combination(s) failed")

    forecast = summary.get("forecast")
    if forecast:
        add("")
        add(f"forward forecast ({summary['primary_model']}, {summary['horizon_days']} days):")
        add(f"  origin bar   : {forecast['origin_bar']} (issued {forecast['issued_at']})")
        add(f"  current      : ${forecast['current_price']:,.2f}")
        add(f"  +30 days     : ${forecast['point_30d']:,.2f}")
        add(f"  +90 days     : ${forecast['point_90d']:,.2f}")
        add(f"  final        : ${forecast['point_final']:,.2f}")
        if forecast["interval_low_final"] is not None:
            add(
                f"  final {summary['interval_level']:.0%} PI: "
                f"${forecast['interval_low_final']:,.2f} .. ${forecast['interval_high_final']:,.2f}"
            )

    add("")
    add("verdict:")
    add(f"  {summary['interpretation']}")

    direction = summary.get("direction_test") or {}
    if direction:
        add("")
        add(
            f"direction test ({summary['primary_model']}): "
            f"{direction['n_correct']}/{direction['n_total']} = {direction['accuracy']:.4f}, "
            f"p={direction['p_value']:.4g}"
        )
        add(f"  caveat: {direction['caveat']}")

    add("=" * 78)
    return "\n".join(lines)


__all__ = [
    "RunOutcome",
    "build_models",
    "format_report",
    "load_market_data",
    "run_diagnostics",
    "run_forecast",
    "write_artifacts",
]
