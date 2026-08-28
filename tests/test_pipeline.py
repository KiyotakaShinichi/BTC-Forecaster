"""End-to-end pipeline, configuration, artifacts and CLI.

Runs the whole pipeline on a synthetic frame with fast models only, so it
exercises orchestration and artifact writing without the network or a Prophet
fit.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from btc_forecaster.artifacts.writer import (
    BACKTEST_CSV,
    FORECAST_CSV,
    HISTORY_CSV,
    LEGACY_FORECAST_CSV,
    MANIFEST_JSON,
    SUMMARY_JSON,
    ArtifactWriter,
    environment_fingerprint,
)
from btc_forecaster.cli import build_parser, main
from btc_forecaster.config.settings import RunConfig, WalkForwardConfig
from btc_forecaster.pipeline import (
    build_models,
    format_report,
    run_diagnostics,
    run_forecast,
)
from btc_forecaster.testing import constant_growth_frame, synthetic_market_frame

FAST_MODELS = ("random_walk", "random_walk_drift", "historical_mean_return", "arima")


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_market_frame(periods=1100, seed=21)


@pytest.fixture
def config(tmp_path) -> RunConfig:
    return RunConfig(
        ticker="TEST-USD",
        horizon_days=30,
        models=FAST_MODELS,
        primary_model="arima",
        baseline_model="random_walk",
        walk_forward=WalkForwardConfig(n_folds=3, horizon=20, min_train_bars=500),
        output_dir=tmp_path / "out",
        snapshot_dir=tmp_path / "snapshots",
        make_plots=False,
    )


class TestRunConfig:
    def test_defaults_are_sensible(self):
        config = RunConfig()
        assert config.ticker == "BTC-USD"
        assert config.interval_level == 0.95
        assert config.baseline_model in config.models

    def test_baseline_must_be_among_the_scored_models(self):
        """Skill cannot be measured against a model that was never run."""
        with pytest.raises(ValueError, match="must be one of the models"):
            RunConfig(models=("arima",), baseline_model="random_walk")

    def test_invalid_horizon_is_rejected(self):
        with pytest.raises(ValueError, match="horizon_days"):
            RunConfig(horizon_days=0)

    def test_invalid_interval_level_is_rejected(self):
        with pytest.raises(ValueError, match="interval_level"):
            RunConfig(interval_level=1.5)

    def test_environment_variables_are_honoured(self, monkeypatch):
        monkeypatch.setenv("TICKER", "ETH-USD")
        monkeypatch.setenv("HORIZON_DAYS", "90")
        monkeypatch.setenv("MONTE_CARLO_RUNS", "250")
        monkeypatch.setenv("WF_FOLDS", "8")

        config = RunConfig.from_env()
        assert config.ticker == "ETH-USD"
        assert config.horizon_days == 90
        assert config.monte_carlo_runs == 250
        assert config.walk_forward.n_folds == 8

    def test_original_env_var_names_still_work(self, monkeypatch):
        """Existing Dockerfiles and shell scripts must not break."""
        monkeypatch.setenv("OUTPUT_DIR", "/tmp/somewhere")
        monkeypatch.setenv("RANDOM_STATE", "7")
        monkeypatch.setenv("MAX_LAG", "45")

        config = RunConfig.from_env()
        assert str(config.output_dir) in ("/tmp/somewhere", "\\tmp\\somewhere")
        assert config.random_state == 7
        assert config.max_lag == 45

    def test_config_is_serialisable(self, config):
        json.dumps(config.to_dict())

    def test_config_is_frozen(self, config):
        with pytest.raises(Exception):
            config.ticker = "OTHER"  # type: ignore[misc]

    def test_snapshot_path_is_per_ticker(self):
        config = RunConfig(ticker="BTC-USD")
        assert config.snapshot_path.name == "BTC-USD"


class TestBuildModels:
    def test_it_builds_the_requested_models(self, config):
        models, skipped = build_models(config)
        assert [m.name for m in models] == list(FAST_MODELS)
        assert skipped == {}

    def test_unknown_models_are_skipped_not_fatal(self, config):
        from dataclasses import replace

        models, skipped = build_models(replace(config, models=FAST_MODELS + ("nonexistent",)))
        assert len(models) == 4
        assert "nonexistent" in skipped


class TestRunForecast:
    def test_it_produces_a_backtest_and_a_forward_forecast(self, config, frame):
        outcome = run_forecast(config, frame=frame, write=False)

        assert len(outcome.backtest.folds) == 3
        assert set(outcome.forecasts) == set(FAST_MODELS)
        assert outcome.primary is not None
        assert outcome.primary.horizon == 30

    def test_the_forward_forecast_starts_after_the_last_observed_bar(self, config, frame):
        outcome = run_forecast(config, frame=frame, write=False)
        assert outcome.primary.index[0] == frame.index[-1] + pd.Timedelta(days=1)

    def test_every_model_is_backtested_on_identical_folds(self, config, frame):
        outcome = run_forecast(config, frame=frame, write=False)
        table = outcome.backtest.to_frame()
        per_model = table.groupby("model")["train_end"].apply(list)
        assert all(folds == per_model.iloc[0] for folds in per_model)

    def test_summary_reports_skill_against_the_baseline(self, config, frame):
        summary = run_forecast(config, frame=frame, write=False).summary()

        assert summary["baseline_model"] == "random_walk"
        assert summary["primary_beats_baseline"] in (True, False)
        assert summary["skill_vs_baseline"]

    def test_summary_says_plainly_when_the_model_does_not_beat_the_baseline(self, config, frame):
        """The negative result must be stated, not omitted."""
        summary = run_forecast(config, frame=frame, write=False).summary()
        if summary["primary_beats_baseline"] is False:
            assert "did NOT beat" in summary["interpretation"]
            assert "scenario, not a prediction" in summary["interpretation"]
        else:
            assert "evidence of skill, not proof" in summary["interpretation"]

    def test_no_model_beats_the_random_walk_on_a_random_walk(self, config, frame):
        """The honest negative result, end to end."""
        outcome = run_forecast(config, frame=frame, write=False)
        skill = outcome.backtest.skill_table("random_walk")
        column = "mae_skill_vs_random_walk"
        for model in ("random_walk_drift", "historical_mean_return"):
            assert skill.loc[model, column] <= 0.02, f"{model} found skill in pure noise"

    def test_diagnostics_are_included_and_corrected(self, config, frame):
        outcome = run_forecast(config, frame=frame, write=False)
        assert "returns" in outcome.diagnostics
        assert outcome.diagnostics["returns_multiple_testing"]["method"] == "benjamini-hochberg"

    def test_summary_records_the_data_hash(self, config, frame):
        summary = run_forecast(config, frame=frame, write=False).summary()
        assert len(summary["data"]["sha256"]) == 64
        assert summary["data"]["rows"] == len(frame)

    def test_a_failing_model_does_not_abort_the_run(self, config, frame):
        from dataclasses import replace

        outcome = run_forecast(
            replace(config, models=FAST_MODELS + ("sarimax_missing_exog",)),
            frame=frame,
            write=False,
        )
        assert outcome.skipped_models
        assert outcome.primary is not None

    def test_report_is_human_readable(self, config, frame):
        report = format_report(run_forecast(config, frame=frame, write=False))
        assert "model comparison" in report
        assert "verdict:" in report
        assert "caveat:" in report


class TestArtifacts:
    def test_a_run_writes_every_expected_artifact(self, config, frame):
        outcome = run_forecast(config, frame=frame, write=True)
        written = set(outcome.artifacts)

        for expected in (FORECAST_CSV, SUMMARY_JSON, BACKTEST_CSV, HISTORY_CSV, MANIFEST_JSON):
            assert expected in written, f"{expected} was not written"

    def test_the_legacy_forecast_filename_is_still_written(self, config, frame):
        """An existing dashboard deployment must keep working across the change."""
        outcome = run_forecast(config, frame=frame, write=True)
        assert LEGACY_FORECAST_CSV in outcome.artifacts

        primary = pd.read_csv(config.output_dir / FORECAST_CSV, index_col=0)
        legacy = pd.read_csv(config.output_dir / LEGACY_FORECAST_CSV, index_col=0)
        pd.testing.assert_frame_equal(primary, legacy)

    def test_manifest_records_what_is_needed_to_reproduce_the_run(self, config, frame):
        run_forecast(config, frame=frame, write=True)
        manifest = json.loads((config.output_dir / MANIFEST_JSON).read_text(encoding="utf-8"))

        assert manifest["config"]["ticker"] == "TEST-USD"
        assert len(manifest["data_snapshot"]["sha256"]) == 64
        assert manifest["backtest"]["n_folds"] == 3
        assert manifest["environment"]["python"]
        assert "numpy" in manifest["environment"]["packages"]

    def test_backtest_csv_has_one_row_per_model_fold(self, config, frame):
        run_forecast(config, frame=frame, write=True)
        table = pd.read_csv(config.output_dir / BACKTEST_CSV)
        assert len(table) == len(FAST_MODELS) * 3

    def test_forecast_csv_carries_every_models_point_forecast(self, config, frame):
        run_forecast(config, frame=frame, write=True)
        forecast = pd.read_csv(config.output_dir / FORECAST_CSV, index_col=0)
        for name in FAST_MODELS:
            assert f"{name}__point" in forecast.columns

    def test_summary_json_is_valid_and_complete(self, config, frame):
        run_forecast(config, frame=frame, write=True)
        summary = json.loads((config.output_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
        assert summary["ticker"] == "TEST-USD"
        assert "interpretation" in summary
        assert "model_comparison" in summary

    def test_writer_creates_its_directory(self, tmp_path):
        writer = ArtifactWriter(tmp_path / "deep" / "nested")
        assert writer.output_dir.exists()

    def test_environment_fingerprint_names_versions(self):
        fingerprint = environment_fingerprint()
        assert fingerprint["python"]
        assert fingerprint["packages"]["numpy"]
        json.dumps(fingerprint)

    def test_history_csv_is_capped_and_labelled(self, config, frame):
        run_forecast(config, frame=frame, write=True)
        history = pd.read_csv(config.output_dir / HISTORY_CSV, index_col=0)
        assert len(history) == 365
        assert list(history.columns) == ["close"]


class TestDiagnosticsEntryPoint:
    def test_it_reports_both_returns_and_price(self, frame):
        report = run_diagnostics(frame)
        assert set(report) == {"returns", "returns_multiple_testing", "log_price"}
        assert report["returns"]["n_tests"] == 6

    def test_multiple_testing_note_is_present(self, frame):
        assert "uncorrected" in run_diagnostics(frame)["returns"]["multiple_testing_note"]


class TestCli:
    def test_parser_exposes_every_subcommand(self):
        parser = build_parser()
        for command in ("run", "backtest", "diagnose", "snapshot", "models"):
            assert parser.parse_args([command] + ([] if command == "models" else [])).command == command

    def test_models_command_lists_availability(self, capsys):
        assert main(["models"]) == 0
        out = capsys.readouterr().out
        assert "random_walk" in out
        assert "prophet_xgb_hybrid" in out

    def test_models_json_output_is_parseable(self, capsys):
        assert main(["models", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["random_walk"]["available"] is True

    def test_a_baseline_not_in_models_is_added_rather_than_rejected(self):
        from btc_forecaster.cli import _config_from_args

        args = build_parser().parse_args(
            ["backtest", "--models", "arima", "--baseline", "random_walk"]
        )
        config = _config_from_args(args)
        assert "random_walk" in config.models

    def test_walk_forward_flags_reach_the_config(self):
        from btc_forecaster.cli import _config_from_args

        args = build_parser().parse_args(
            ["run", "--folds", "9", "--mode", "rolling", "--window-bars", "300", "--embargo-bars", "5"]
        )
        config = _config_from_args(args)
        assert config.walk_forward.n_folds == 9
        assert config.walk_forward.mode == "rolling"
        assert config.walk_forward.window_bars == 300
        assert config.walk_forward.embargo_bars == 5

    def test_errors_return_a_nonzero_exit_code(self, capsys):
        assert main(["backtest", "--models", "does_not_exist", "--baseline", "does_not_exist"]) != 0

    def test_unknown_subcommand_exits_with_usage(self):
        with pytest.raises(SystemExit):
            main(["nonsense"])


class TestPipelineOnATrendingSeries:
    def test_drift_wins_where_drift_exists(self, tmp_path):
        """The pipeline must be able to report a positive result too."""
        from dataclasses import replace

        config = RunConfig(
            ticker="TREND-USD",
            horizon_days=20,
            models=("random_walk", "random_walk_drift"),
            primary_model="random_walk_drift",
            baseline_model="random_walk",
            walk_forward=WalkForwardConfig(n_folds=3, horizon=20, min_train_bars=400),
            output_dir=tmp_path / "out",
            snapshot_dir=tmp_path / "snap",
            make_plots=False,
        )
        frame = constant_growth_frame(periods=900, daily_growth=0.002)
        summary = run_forecast(config, frame=frame, write=False).summary()

        assert summary["primary_beats_baseline"] is True
        assert "evidence of skill" in summary["interpretation"]
