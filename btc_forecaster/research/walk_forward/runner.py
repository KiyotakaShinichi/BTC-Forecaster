"""Run a walk-forward benchmark and write one canonical, verifiable result.

**Execution.** Every (model, horizon, window) job runs in a spawned worker
process, always -- a single worker is a pool of one, not a different code path.
Every worker pins its linear-algebra libraries to one thread before importing
numpy, because a multi-threaded BLAS may sum in a different order and change a
result in its last bits; the worker count must change how long a run takes and
nothing else. Results are assembled by key, never by completion order.

**Output.** Two kinds of file:

canonical
    ``config.json``, ``input_manifest.json``, ``schedule.json``,
    ``predictions.csv`` (stored gzipped), ``metrics.json``,
    ``comparisons.json``, ``gates.json``, ``decision.json``, ``leakage.json``,
    ``folds.json``. Sorted keys, sorted rows, fixed float text, no timestamps,
    no run ids, no timings. The result digest is taken over these and nothing
    else, so two identical runs agree on it byte for byte.

non-canonical
    ``run_info.json`` -- when it ran, on what, how long each fold took, how long
    imports took, how many workers. Real, recorded, and outside every digest.

``manifest.json`` is written last: its presence is the claim that the run
finished, and it carries each canonical file's hash and the result digest.
:func:`verify_run` recomputes all of them from the files on disk.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing
import os
import platform
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..contracts import ModelStatus
from ..runner import warm_imports
from .config import WalkForwardConfig, WindowSpec
from .evaluator import InformationCache as _InformationCache
from .evaluator import JobResult, JobSpec, run_job, schedule_for
from .leakage import AdversaryReport, run_adversaries
from .manifest import (
    InputManifest,
    canonical_csv,
    canonical_json,
    deterministic_gzip,
    result_digest,
    sha256_hex,
)
from .origins import OriginSchedule
from .scoring import ConfigScores, score_configs
from .significance import Comparison, compare_all, correct
from .stability import (
    LEAKAGE_FAILED,
    LEAKAGE_NOT_CHECKED,
    LEAKAGE_PASSED,
    GateResult,
    decide,
    evaluate_gates,
)

RUN_KIND = "WALK_FORWARD_ROBUSTNESS"

#: Environment every worker starts with. Set before numpy is imported there.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

CANONICAL_FILES: tuple[str, ...] = (
    "config.json",
    "input_manifest.json",
    "schedule.json",
    "predictions.csv",
    "metrics.json",
    "comparisons.json",
    "gates.json",
    "decision.json",
    "leakage.json",
    "folds.json",
)

#: Models whose runtime dominates. Submitted first so a pool finishes sooner;
#: submission order cannot affect results, which are assembled by key.
_HEAVY_FIRST = ("lstm", "mlp", "xgboost", "arima", "local_linear_trend", "local_level", "theta")


# -- worker processes ---------------------------------------------------------

_WORKER: dict[str, object] = {}


def _init_worker(frame: pd.DataFrame, config_payload: dict) -> None:
    config = WalkForwardConfig.from_dict(config_payload)
    _WORKER["frame"] = frame
    _WORKER["config"] = config
    _WORKER["schedule"] = schedule_for(frame, config)
    _WORKER["cache"] = _InformationCache(frame, config)
    warm_imports(list(config.models))


def _run_spec(spec: JobSpec) -> JobResult:
    return run_job(
        spec,
        _WORKER["schedule"],  # type: ignore[arg-type]
        _WORKER["config"],  # type: ignore[arg-type]
        _WORKER["cache"],  # type: ignore[arg-type]
    )


def _run_attack(task: tuple[str, int, str, int]) -> AdversaryReport:
    model_id, horizon, window, fold = task
    return run_adversaries(
        model_id,
        _WORKER["frame"],  # type: ignore[arg-type]
        _WORKER["schedule"],  # type: ignore[arg-type]
        _WORKER["config"],  # type: ignore[arg-type]
        horizon=horizon,
        window=WindowSpec.parse(window),
        fold_index=fold,
    )


class _SingleThreadedPool:
    """A spawned process pool whose workers start with single-threaded BLAS."""

    def __init__(self, frame: pd.DataFrame, config: WalkForwardConfig, workers: int) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self._frame, self._config, self._workers = frame, config, workers
        self._saved: dict[str, str | None] = {}
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> ProcessPoolExecutor:
        self._saved = {k: os.environ.get(k) for k in SINGLE_THREAD_ENV}
        os.environ.update(SINGLE_THREAD_ENV)
        self._pool = ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self._frame, self._config.as_dict()),
        )
        return self._pool

    def __exit__(self, *exc: object) -> None:
        assert self._pool is not None
        self._pool.shutdown(wait=True)
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# -- the benchmark ------------------------------------------------------------


def job_specs(config: WalkForwardConfig) -> list[JobSpec]:
    return [JobSpec(m, h, w) for m in config.models for h in config.horizons for w in config.windows]


def attack_plan(config: WalkForwardConfig) -> list[tuple[str, int, str, int]]:
    """Where each model is attacked: the shortest and longest horizon, on the
    smallest rolling window (so there is an outside to poison), at the last fold."""
    rolling = [w for w in config.windows if w.rows is not None]
    window = min(rolling, key=lambda w: w.rows or 0) if rolling else config.windows[0]
    horizons = sorted({min(config.horizons), max(config.horizons)})
    return [(m, h, window.label, config.n_refits - 1) for m in config.models for h in horizons]


@dataclass
class BenchmarkOutcome:
    config: WalkForwardConfig
    schedule: OriginSchedule
    jobs: list[JobResult]
    records: pd.DataFrame
    scores: list[ConfigScores]
    comparisons: list[Comparison]
    family: dict
    leakage_reports: list[AdversaryReport]
    leakage: dict[str, str]
    gates: list[GateResult]
    decision: dict
    timings: dict = field(default_factory=dict)


def _order(model_id: str) -> int:
    return _HEAVY_FIRST.index(model_id) if model_id in _HEAVY_FIRST else len(_HEAVY_FIRST)


def run_benchmark(frame: pd.DataFrame, config: WalkForwardConfig, *, workers: int = 1) -> BenchmarkOutcome:
    """Every job, every adversary, the analysis and the decision."""
    started = time.perf_counter()
    schedule = schedule_for(frame, config)
    specs = job_specs(config)
    plan = attack_plan(config)

    with _SingleThreadedPool(frame, config, workers) as pool:
        submitted = sorted(specs, key=lambda s: (_order(s.model_id), s.key))
        job_futures = {spec.key: pool.submit(_run_spec, spec) for spec in submitted}
        attack_futures = {
            task: pool.submit(_run_attack, task)
            for task in sorted(plan, key=lambda t: (_order(t[0]), t))
        }
        jobs = [job_futures[spec.key].result() for spec in specs]
        reports = [attack_futures[task].result() for task in plan]
    jobs_seconds = time.perf_counter() - started

    records = pd.concat([j.records for j in jobs], ignore_index=True)
    records = records.sort_values(["model_id", "horizon", "window", "origin"], kind="mergesort")
    records = records.reset_index(drop=True)

    leakage: dict[str, str] = {}
    for model_id in config.models:
        mine = [r for r in reports if r.model_id == model_id]
        leakage[model_id] = (
            LEAKAGE_NOT_CHECKED if not mine else LEAKAGE_PASSED if all(r.passed for r in mine) else LEAKAGE_FAILED
        )

    scores = score_configs(records, schedule, baseline=config.baseline)
    comparisons, family = correct(compare_all(records, config), alpha=config.gates.alpha)
    gates = evaluate_gates(
        scores,
        comparisons,
        gates=config.gates,
        late_block=schedule.block_names[-1],
        leakage=leakage,
        clean={j.spec.key: j.clean for j in jobs},
        baseline=config.baseline,
    )
    verdict = decide(gates)

    return BenchmarkOutcome(
        config=config,
        schedule=schedule,
        jobs=jobs,
        records=records,
        scores=scores,
        comparisons=comparisons,
        family=family,
        leakage_reports=reports,
        leakage=leakage,
        gates=gates,
        decision=verdict,
        timings={
            "wall_clock_seconds": round(time.perf_counter() - started, 2),
            "jobs_and_adversaries_seconds": round(jobs_seconds, 2),
            "workers": workers,
            "over_budget_folds": [
                {
                    "job": "|".join(str(k) for k in j.spec.key),
                    "fold": f.fold,
                    "seconds": round(f.total_seconds, 1),
                    "budget_seconds": f.budget_seconds,
                }
                for j in jobs
                for f in j.folds
                if f.over_budget
            ],
            "training_time_capped_folds": [
                {"job": "|".join(str(k) for k in j.spec.key), "fold": f.fold}
                for j in jobs
                for f in j.folds
                if f.hit_training_time_cap
            ],
            "jobs": {
                "|".join(str(k) for k in j.spec.key): [f.timing() for f in j.folds] for j in jobs
            },
        },
    )


# -- writing ------------------------------------------------------------------


def _environment() -> dict:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "xgboost"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "absent"
    return {"platform": platform.platform(), "versions": versions}


def canonical_contents(outcome: BenchmarkOutcome, input_manifest: InputManifest) -> dict[str, bytes]:
    """The bytes of every canonical file, keyed by name."""
    config = outcome.config
    fold_statuses = [
        {
            "model_id": j.spec.model_id,
            "horizon": j.spec.horizon,
            "window": j.spec.window.label,
            "strategy": j.strategy,
            "folds": [f.canonical() for f in j.folds],
        }
        for j in outcome.jobs
    ]
    contents = {
        "config.json": config.canonical_json(),
        "input_manifest.json": input_manifest.canonical_json(),
        "schedule.json": canonical_json(outcome.schedule.as_dict()),
        "metrics.json": canonical_json([s.as_dict() for s in outcome.scores]),
        "comparisons.json": canonical_json(
            {"family": outcome.family, "comparisons": [c.as_dict() for c in outcome.comparisons]}
        ),
        "gates.json": canonical_json({"thresholds": config.gates.as_dict(), "results": [g.as_dict() for g in outcome.gates]}),
        "decision.json": canonical_json(outcome.decision),
        "leakage.json": canonical_json(
            {"status": outcome.leakage, "reports": [r.as_dict() for r in outcome.leakage_reports]}
        ),
        "folds.json": canonical_json(fold_statuses),
    }
    parts = {name: text.encode("utf-8") for name, text in contents.items()}
    parts["predictions.csv"] = canonical_csv(outcome.records)
    return parts


def build_manifest(outcome: BenchmarkOutcome, input_manifest: InputManifest, parts: dict[str, bytes]) -> dict:
    return {
        "kind": RUN_KIND,
        "scientific_status": "EXPLORATORY",
        "decision": outcome.decision["decision"],
        "config_digest": outcome.config.digest(),
        "input_frame_sha256": input_manifest.frame_sha256,
        "files": {name: sha256_hex(parts[name]) for name in sorted(parts)},
        "result_digest": result_digest(parts),
        "digest_covers": sorted(parts),
        "digest_excludes": ["run_info.json", "README.md", "manifest.json"],
        "predictions_storage": "predictions.csv.gz (deterministic gzip); the digest covers the uncompressed bytes",
        "promoted_models": [],
        "live_trading_enabled": False,
    }


def write_run(
    outcome: BenchmarkOutcome,
    directory: Path | str,
    *,
    input_manifest: InputManifest,
    readme: Callable[[dict], str] | None = None,
) -> dict:
    """Write every file, the manifest last. Returns the manifest.

    ``readme`` is given the finished manifest, so the README can quote the
    result digest it sits beside.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    parts = canonical_contents(outcome, input_manifest)
    for name, data in parts.items():
        if name == "predictions.csv":
            (out / "predictions.csv.gz").write_bytes(deterministic_gzip(data))
        else:
            (out / name).write_bytes(data)

    run_info = {
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "timings": outcome.timings,
        "note": "Timings describe the machine and are outside the result digest.",
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8")

    manifest = build_manifest(outcome, input_manifest, parts)
    if readme is not None:
        (out / "README.md").write_text(readme(manifest), encoding="utf-8")
    # Last. Its presence is the claim that the run finished.
    (out / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    return manifest


def stored_name(name: str) -> str:
    """The file a canonical part is stored in: predictions are gzipped on disk."""
    return "predictions.csv.gz" if name == "predictions.csv" else name


def read_canonical_parts(directory: Path | str, *, missing_ok: bool = False) -> dict[str, bytes]:
    """Every canonical part, uncompressed. With ``missing_ok``, absent ones are left out."""
    out = Path(directory)
    parts: dict[str, bytes] = {}
    for name in CANONICAL_FILES:
        path = out / stored_name(name)
        if missing_ok and not path.exists():
            continue
        data = path.read_bytes()
        parts[name] = gzip.decompress(data) if name == "predictions.csv" else data
    return parts


def verify_run(directory: Path | str) -> dict:
    """Recompute every canonical hash and the result digest from disk.

    A canonical file that is not on disk is named, not skipped silently: the files
    that are present are still checked against the manifest, but the result
    digest cannot be recomputed without all of them, so the run is not verified.
    A clone of the committed BTC run is in exactly that state -- its predictions
    are regenerated from the pinned snapshot, not redistributed.
    """
    out = Path(directory)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    parts = read_canonical_parts(out, missing_ok=True)
    absent = [name for name in CANONICAL_FILES if name not in parts]
    mismatched = [
        name for name in sorted(parts) if sha256_hex(parts[name]) != manifest["files"].get(name)
    ]
    digest = None if absent else result_digest(parts)
    return {
        "absent_files": absent,
        "result_digest_recorded": manifest["result_digest"],
        "result_digest_recomputed": digest,
        "mismatched_files": mismatched,
        "verified": not absent and not mismatched and digest == manifest["result_digest"],
    }


def assert_nothing_promoted(outcome: BenchmarkOutcome) -> None:
    if outcome.decision.get("promoted_models"):
        raise AssertionError("A7 produced a promoted model; it must not")
    if outcome.decision.get("live_trading_enabled"):
        raise AssertionError("A7 enabled live trading; it must not")


def clean_statuses(outcome: BenchmarkOutcome) -> dict[str, int]:
    """How many folds ended in each status, across every job."""
    counts: dict[str, int] = {status.value: 0 for status in ModelStatus}
    for job in outcome.jobs:
        for fold in job.folds:
            counts[fold.status] = counts.get(fold.status, 0) + 1
    return {k: v for k, v in counts.items() if v}


__all__ = [
    "CANONICAL_FILES",
    "RUN_KIND",
    "SINGLE_THREAD_ENV",
    "BenchmarkOutcome",
    "assert_nothing_promoted",
    "attack_plan",
    "build_manifest",
    "canonical_contents",
    "clean_statuses",
    "job_specs",
    "read_canonical_parts",
    "run_benchmark",
    "verify_run",
    "write_run",
]
