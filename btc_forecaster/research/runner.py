"""The benchmark: fit every model once, score it, and record what happened.

Three properties matter more than the numbers it produces.

**It is deterministic.** The 1,000 training rows are a contiguous slice with no
seed to get wrong, every stochastic estimator takes the same seed, and the
manifest carries the fingerprint of the exact rows each model saw. Re-running
produces the same table.

**It records failures.** A model that raises is written down as FAILED with its
exception class, message and runtime, and the benchmark carries on. A model that
exceeds its declared resource class is RESOURCE_LIMIT. A model whose dependency
is absent is SKIPPED_DEPENDENCY. The one outcome that cannot happen is a model
quietly missing from the table -- which is the failure mode that makes a
benchmark look better than it is.

**It writes the manifest last.** Everything before it can crash and leave a
directory of partial artifacts; the manifest's existence is the claim that the
run completed. That is the same rule the collector's `ForwardCollector` uses,
for the same reason.

Nothing here promotes anything. Every result carries `EXPLORATORY`, the runner
asserts it, and a `promoted` field does not exist to be set.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import registry
from .contracts import (
    EXPLORATORY,
    RESOURCE_BUDGET_SECONDS,
    EvaluationContext,
    ModelStatus,
    TrainingSet,
    ZooForecast,
    ZooModel,
)
from .metrics import (
    SMAPE_EXCLUSION_REASON,
    ModelScores,
    score_direction,
    score_point,
    score_probabilistic,
    score_variance,
)
from .partition import PartitionSpec, ZooDataset, build_dataset

#: What the benchmark is, stamped on every artifact it writes.
BENCHMARK_KIND = "RESOURCE_CONSTRAINED_EXPLORATORY"


@dataclass
class ModelOutcome:
    """One model's row: what it did, what it cost, or why it did not run."""

    model_id: str
    family: str
    status: ModelStatus
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    parameter_count: int | None = None
    train_fingerprint: str | None = None
    hyperparameters: dict = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    scores: ModelScores | None = None
    failure: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return self.fit_seconds + self.predict_seconds

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "status": self.status.value,
            "scientific_status": EXPLORATORY,
            "fit_seconds": round(self.fit_seconds, 4),
            "predict_seconds": round(self.predict_seconds, 4),
            "total_seconds": round(self.total_seconds, 4),
            "parameter_count": self.parameter_count,
            "train_fingerprint": self.train_fingerprint,
            "capabilities": self.capabilities,
            "hyperparameters": self.hyperparameters,
            "scores": None if self.scores is None else self.scores.as_dict(),
            "failure": self.failure,
            "notes": self.notes,
        }


@dataclass
class BenchmarkResult:
    """The whole run: outcomes, forecasts, and the context they were made in."""

    outcomes: list[ModelOutcome]
    forecasts: dict[str, ZooForecast]
    #: The fitted models, kept so the run can serialize them and prove the
    #: round trip. Forty small models; the largest is about a megabyte.
    models: dict[str, ZooModel]
    dataset: ZooDataset
    actual: np.ndarray
    origins: pd.DatetimeIndex
    target_bars: pd.DatetimeIndex
    naive_mae: float
    seconds: float
    data_fingerprint: str

    def succeeded(self) -> list[ModelOutcome]:
        return [o for o in self.outcomes if o.status is ModelStatus.ACTIVE and o.scores]

    def by_status(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.status.value, []).append(outcome.model_id)
        return {k: sorted(v) for k, v in sorted(grouped.items())}

    def results_frame(self) -> pd.DataFrame:
        """The results table, one row per model that produced a forecast."""
        rows = []
        for outcome in self.succeeded():
            assert outcome.scores is not None
            point = outcome.scores.point
            direction = outcome.scores.direction
            rows.append(
                {
                    "model_id": outcome.model_id,
                    "family": outcome.family,
                    "mae": point.mae,
                    "rmse": point.rmse,
                    "mase": point.mase,
                    "bias": point.bias,
                    "skill_vs_naive": point.skill_vs_naive,
                    "directional_accuracy": direction.accuracy,
                    "balanced_accuracy": direction.balanced_accuracy,
                    "mcc": direction.mcc,
                    "beats_train_constant": direction.accuracy
                    > direction.train_constant_baseline,
                    "parameters": outcome.parameter_count,
                    "fit_seconds": outcome.fit_seconds,
                    "total_seconds": outcome.total_seconds,
                    "scientific_status": EXPLORATORY,
                }
            )
        frame = pd.DataFrame(rows)
        return frame.sort_values("mae").reset_index(drop=True) if len(frame) else frame

    def best_by_family(self) -> dict[str, dict]:
        """Best MAE within each family. Never a global winner.

        Reporting one champion across forty models is the multiple-comparison
        error Phase 21 exists to prevent: the best of forty draws from a
        distribution centred on nothing is still the best of forty.
        """
        frame = self.results_frame()
        if not len(frame):
            return {}
        best: dict[str, dict] = {}
        for family, group in frame.groupby("family"):
            row = group.nsmallest(1, "mae").iloc[0]
            best[str(family)] = {
                "model_id": row["model_id"],
                "mae": float(row["mae"]),
                "skill_vs_naive": float(row["skill_vs_naive"]),
                "scientific_status": EXPLORATORY,
                "note": "best MAE within this family only; not a promotion",
            }
        return best


def _environment() -> dict:
    import numpy as _np

    versions = {"python": platform.python_version(), "numpy": _np.__version__}
    for name in ("pandas", "sklearn", "statsmodels", "arch", "xgboost"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "absent"
    return {"platform": platform.platform(), "versions": versions}


def run_model(
    model_id: str,
    train: TrainingSet,
    development: EvaluationContext,
    evaluation: EvaluationContext,
    *,
    naive_mae: float,
) -> tuple[ModelOutcome, ZooForecast | None, ZooModel | None]:
    """Fit, calibrate if declared, predict and score one model.

    Every failure path lands in the outcome rather than propagating, because one
    model raising must not end a forty-model benchmark. The exception class,
    message and a trimmed traceback are all recorded.
    """
    registration = registry.get(model_id)
    outcome = ModelOutcome(
        model_id=model_id,
        family=registration.family.value,
        status=registration.effective_status(),
        notes=list(registration.notes),
    )

    if outcome.status is ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB:
        outcome.failure = {"reason": registration.unsuitable_reason}
        return outcome, None, None
    if outcome.status is ModelStatus.SKIPPED_DEPENDENCY:
        outcome.failure = {"missing_dependency": registration.missing_dependency()}
        return outcome, None, None

    budget = RESOURCE_BUDGET_SECONDS[registration.resource_class]
    started = time.perf_counter()
    try:
        model = registry.build(model_id)
        model.fit(train)
        if model.needs_calibration:
            model.calibrate(development)
        outcome.fit_seconds = time.perf_counter() - started

        predict_started = time.perf_counter()
        forecast = model.predict(evaluation)
        outcome.predict_seconds = time.perf_counter() - predict_started

        outcome.parameter_count = model.parameter_count()
        outcome.train_fingerprint = model.describe()["train_fingerprint"]
        outcome.hyperparameters = model.hyperparameters()
        outcome.capabilities = sorted(c.value for c in model.capabilities)
    except Exception as exc:  # noqa: BLE001 -- one model must not end the run
        outcome.status = ModelStatus.FAILED
        outcome.fit_seconds = time.perf_counter() - started
        outcome.failure = {
            "exception": type(exc).__name__,
            "message": str(exc)[:500],
            "traceback": traceback.format_exc(limit=4)[-1200:],
            "resource_class": registration.resource_class.value,
            "budget_seconds": budget,
        }
        return outcome, None, None

    if outcome.total_seconds > budget:
        # Recorded, not discarded: the forecast is valid, the model simply cost
        # more than it declared, and both facts belong in the table.
        outcome.status = ModelStatus.RESOURCE_LIMIT
        outcome.failure = {
            "reason": "exceeded the declared resource budget",
            "resource_class": registration.resource_class.value,
            "budget_seconds": budget,
            "actual_seconds": round(outcome.total_seconds, 3),
        }

    actual = evaluation.y.to_numpy(dtype=float)
    outcome.scores = ModelScores(
        model_id=model_id,
        point=score_point(actual, forecast.point, naive_mae=naive_mae),
        direction=score_direction(
            actual, forecast.point, train_actual=train.y.to_numpy(dtype=float)
        ),
        probabilistic=score_probabilistic(
            actual, forecast.quantiles, forecast.direction_probability
        ),
        variance=score_variance(actual, forecast.variance),
    )
    if outcome.status is not ModelStatus.RESOURCE_LIMIT:
        outcome.status = ModelStatus.ACTIVE
    return outcome, forecast, model


def run_benchmark(
    frame: pd.DataFrame,
    *,
    spec: PartitionSpec | None = None,
    model_ids: list[str] | None = None,
) -> BenchmarkResult:
    """Fit and score every runnable model on one deterministic partition."""
    spec = spec or PartitionSpec()
    dataset = build_dataset(frame, spec=spec)
    train = dataset.training_set()
    development = dataset.development_context()
    evaluation = dataset.evaluation_context()

    actual = evaluation.y.to_numpy(dtype=float)
    naive_mae = float(np.mean(np.abs(actual)))

    wanted = model_ids if model_ids is not None else registry.model_ids()
    started = time.perf_counter()
    outcomes: list[ModelOutcome] = []
    forecasts: dict[str, ZooForecast] = {}
    models: dict[str, ZooModel] = {}
    for model_id in wanted:
        outcome, forecast, model = run_model(
            model_id, train, development, evaluation, naive_mae=naive_mae
        )
        outcomes.append(outcome)
        if forecast is not None:
            forecasts[model_id] = forecast
        if model is not None:
            models[model_id] = model

    return BenchmarkResult(
        outcomes=outcomes,
        forecasts=forecasts,
        models=models,
        dataset=dataset,
        actual=actual,
        origins=evaluation.origins,
        target_bars=evaluation.target_bars,
        naive_mae=naive_mae,
        seconds=time.perf_counter() - started,
        data_fingerprint=_frame_fingerprint(frame),
    )


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_numpy(dtype=float).tobytes()
    digest = hashlib.sha256(payload)
    digest.update(str(frame.index[0]).encode())
    digest.update(str(frame.index[-1]).encode())
    return digest.hexdigest()


def build_manifest(result: BenchmarkResult, *, extra: dict | None = None) -> dict:
    """Everything needed to know what ran, on what, and what it cost."""
    frame = result.results_frame()
    manifest = {
        "benchmark": BENCHMARK_KIND,
        "scientific_status": EXPLORATORY,
        "generated_at": datetime.now(UTC).isoformat(),
        "warning": (
            "A6 is exploratory model-zoo evidence at n=1,000 on a single "
            "partition. It is not a promotion study and does not supersede A2's "
            "36-fold historical validation. No model here is PROMOTED, and the "
            "paper-trading engine remains fail-closed."
        ),
        "data": {
            "rows": int(len(result.dataset.X)),
            "fingerprint_sha256": result.data_fingerprint,
            "first_target_bar": str(result.target_bars.min()),
            "last_target_bar": str(result.target_bars.max()),
        },
        "dataset": result.dataset.as_dict(),
        "naive_mae": result.naive_mae,
        "metric_notes": {
            "smape": SMAPE_EXCLUSION_REASON,
            "directional_accuracy": (
                "a forecast of exactly zero is scored as wrong rather than "
                "half-right; three baselines forecast a constant and one "
                "forecasts exactly zero, and crediting the tie would hand it a "
                "free 50%. Balanced accuracy and MCC are reported beside it."
            ),
            "directional_null": (
                "tested against the training-chosen constant direction, not "
                "against 0.5. This series has roughly a 53% up-day base rate, "
                "so always saying up beats a coin while containing no "
                "information."
            ),
        },
        "registry": registry.summary(),
        "registry_entries": [r.as_dict() for r in registry.all_registrations()],
        "capability_matrix": registry.capability_matrix(),
        "counts": {
            "attempted": len(result.outcomes),
            "scored": len(result.succeeded()),
            **{k: len(v) for k, v in result.by_status().items()},
        },
        "by_status": result.by_status(),
        "best_by_family": result.best_by_family(),
        "models": [outcome.as_dict() for outcome in result.outcomes],
        "wall_clock_seconds": round(result.seconds, 2),
        "environment": _environment(),
        "promoted_models": [],
        "live_trading_enabled": False,
    }
    if len(frame):
        manifest["results_table_sha256"] = hashlib.sha256(
            frame.to_csv(index=False).encode()
        ).hexdigest()
    if extra:
        manifest.update(extra)
    return manifest


def analyse(result: BenchmarkResult) -> dict:
    """The four post-hoc analyses, computed from one run's forecasts.

    Kept out of `run_benchmark` because they answer questions *about* the table
    rather than producing it, and because each can be recomputed from a saved
    run without refitting forty models.
    """
    from . import comparison as comparison_module
    from . import diagnostics as diagnostics_module

    points = {model_id: forecast.point for model_id, forecast in result.forecasts.items()}
    if not points:
        return {"status": "no model produced a forecast"}

    comparisons: dict = {"status": "the baseline did not run"}
    if comparison_module.DEFAULT_BASELINE in points:
        raw = comparison_module.compare_against_baseline(result.actual, points)
        comparisons = comparison_module.correct_for_multiplicity(raw)

    stability = comparison_module.stability_by_block(result.actual, points)
    return {
        "comparison": comparisons,
        "stability": stability.to_dict(orient="records"),
        "diversity": comparison_module.error_diversity(result.actual, points),
        "diagnostics": diagnostics_module.diagnose_all(
            {model_id: result.actual - point for model_id, point in points.items()}
        ),
    }


def write_run(
    result: BenchmarkResult,
    directory: Path | str,
    *,
    extra: dict | None = None,
    serialize_models: bool = True,
) -> Path:
    """Write the artifacts, manifest **last**.

    Everything before the manifest can crash and leave a partial directory; the
    manifest's presence is the claim that the run finished. Same rule, same
    reason, as the collector's manifest.
    """
    from .cards import build_run_readme, write_cards
    from .serialization import round_trip_is_exact, write_artifact

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    frame = result.results_frame()
    if len(frame):
        frame.to_csv(out / "results.csv", index=False)

    predictions = pd.DataFrame(
        {"target_bar": result.target_bars, "actual": result.actual}
    )
    for model_id, forecast in sorted(result.forecasts.items()):
        predictions[model_id] = forecast.point
    predictions.to_csv(out / "predictions.csv", index=False)

    (out / "capability_matrix.json").write_text(
        json.dumps(registry.capability_matrix(), indent=2), encoding="utf-8"
    )
    (out / "registry.json").write_text(
        json.dumps([r.as_dict() for r in registry.all_registrations()], indent=2),
        encoding="utf-8",
    )

    analyses = analyse(result)
    (out / "analysis.json").write_text(
        json.dumps(analyses, indent=2, default=str), encoding="utf-8"
    )

    artifacts: dict[str, dict] = {}
    if serialize_models:
        evaluation = result.dataset.evaluation_context()
        for model_id, model in sorted(result.models.items()):
            try:
                artifact = write_artifact(model, out / "artifacts")
                entry = artifact.as_dict()
                entry["round_trip_exact"] = round_trip_is_exact(model, evaluation)
                artifacts[model_id] = entry
            except Exception as exc:  # noqa: BLE001 -- recorded like any other failure
                artifacts[model_id] = {
                    "model_id": model_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        (out / "artifacts.json").write_text(
            json.dumps(artifacts, indent=2), encoding="utf-8"
        )

    manifest = build_manifest(
        result, extra={**(extra or {}), "analysis": analyses, "artifacts": artifacts}
    )
    write_cards(
        manifest,
        out / "cards",
        diagnostics=analyses.get("diagnostics"),
        comparisons=analyses.get("comparison"),
        stability=analyses.get("stability"),
        artifacts=artifacts,
    )

    # The directory's own entry point. A reader who opens it before reading
    # anything else must meet the scientific status before the numbers.
    (out / "README.md").write_text(build_run_readme(manifest), encoding="utf-8")

    # Last. Deliberately.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out / "manifest.json"


def assert_nothing_promoted(manifest: dict) -> None:
    """The invariant the whole track rests on. Cheap to check, so it is checked."""
    if manifest.get("promoted_models"):
        raise AssertionError("A6 produced a promoted model; it must not")
    if manifest.get("live_trading_enabled"):
        raise AssertionError("A6 enabled live trading; it must not")
    if manifest.get("scientific_status") != EXPLORATORY:
        raise AssertionError("A6 results must be EXPLORATORY")
    for model in manifest.get("models", []):
        if model.get("scientific_status") != EXPLORATORY:
            raise AssertionError(f"{model.get('model_id')} is not EXPLORATORY")


__all__ = [
    "BENCHMARK_KIND",
    "analyse",
    "BenchmarkResult",
    "ModelOutcome",
    "assert_nothing_promoted",
    "build_manifest",
    "run_benchmark",
    "run_model",
    "write_run",
]
