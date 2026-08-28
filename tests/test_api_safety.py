"""The API must import safely and expose a stable contract.

These are import-and-shape tests, not integration tests: no server is started
and no forecast is run. The point is that `import api_server` cannot download
data, fit a model, or crash on a missing directory -- all of which the previous
arrangement risked, because bayesianCutoff.py executed its whole pipeline at
import time.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

needs_fastapi = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None, reason="fastapi not installed"
)


@needs_fastapi
class TestApiImportSafety:
    def test_importing_the_api_does_not_run_a_forecast(self):
        """Importing must not touch the network or fit anything."""
        for module in ("api_server", "btc_forecaster.pipeline"):
            sys.modules.pop(module, None)

        import api_server  # noqa: F401

        assert "yfinance" not in sys.modules, "importing the API reached for live data"
        assert "prophet" not in sys.modules, "importing the API fitted a model"

    def test_every_documented_endpoint_still_exists(self):
        import api_server

        paths = {route.path for route in api_server.app.routes if hasattr(route, "path")}
        assert {"/health", "/status", "/run", "/latest", "/artifacts", "/"} <= paths

    def test_the_run_command_targets_the_package_cli(self):
        """bayesianCutoff.py is gone; the API must not still shell out to it."""
        import inspect

        import api_server

        source = inspect.getsource(api_server._run_forecast_job)
        assert "btc_forecaster.cli" in source
        assert "bayesianCutoff" not in source

    def test_legacy_result_filename_is_still_served(self):
        import api_server

        assert api_server.LEGACY_RESULT_CSV == "nextgen_hybrid_forecast_results_montecarlo.csv"


@needs_fastapi
class TestRunRequestContract:
    def test_original_fields_are_unchanged(self):
        from api_server import ForecastRunRequest

        request = ForecastRunRequest()
        assert request.ticker == "BTC-USD"
        assert request.horizon_days == 365
        assert request.monte_carlo_runs == 1000
        assert request.random_state == 42

    def test_the_baseline_is_added_when_omitted_from_models(self):
        """Skill cannot be measured against a model that was never scored."""
        from api_server import ForecastRunRequest

        request = ForecastRunRequest(
            models=["random_walk", "prophet_xgb_hybrid"],
            primary_model="prophet_xgb_hybrid",
            baseline_model="arima",
        )
        assert "arima" in request.models

    def test_a_primary_model_outside_the_set_is_rejected(self):
        from pydantic import ValidationError

        from api_server import ForecastRunRequest

        with pytest.raises(ValidationError, match="primary_model"):
            ForecastRunRequest(models=["random_walk"], primary_model="ets")

    def test_walk_forward_settings_are_bounded(self):
        from pydantic import ValidationError

        from api_server import ForecastRunRequest

        with pytest.raises(ValidationError, match="walk_forward_folds"):
            ForecastRunRequest(walk_forward_folds=0)
        with pytest.raises(ValidationError, match="embargo_bars"):
            ForecastRunRequest(embargo_bars=-1)


@needs_fastapi
class TestEndpointBehaviour:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
        monkeypatch.setenv("ALLOWED_OUTPUT_ROOT", str(tmp_path / "out"))
        for module in list(sys.modules):
            if module == "api_server":
                del sys.modules[module]
        import api_server

        return TestClient(api_server.app)

    def test_health_reports_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_status_starts_idle(self, client):
        assert client.get("/status").json()["running"] is False

    def test_latest_explains_how_to_produce_a_forecast(self, client):
        """A fresh clone has an empty out/ by design (ARTIFACTS.md)."""
        response = client.get("/latest")
        assert response.status_code == 404
        assert "POST /run" in response.json()["detail"]

    def test_artifacts_is_empty_not_an_error_on_a_fresh_output_dir(self, client):
        response = client.get("/artifacts")
        assert response.status_code == 200
        assert response.json()["files"] == []
