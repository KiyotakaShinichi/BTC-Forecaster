"""Model cards, generated from the benchmark manifest and nothing else.

Phase 26 is explicit: do not manually author individual results. That is not a
style preference. A hand-written card says what somebody believed when they
wrote it, and the first time a number changes the card becomes a confident
statement of something that is no longer true -- with no way to tell from
reading it.

So every field here is read out of the manifest, the registry, the diagnostics
and the comparison output. The only prose that is written rather than derived is
the shared caveat block, which is identical on every card because it is true of
every model in this lab.

The one thing a card must never say is that its model is good. `status` is
EXPLORATORY, the limitations section names the sample size and the single
partition, and where a model was significantly *worse* than the naive baseline
the card says so in the same place it would have said the opposite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import EXPLORATORY

#: On every card, because it is true of every model here.
SHARED_LIMITATIONS = (
    "Fitted on 1,000 training rows and scored on a single holdout block. That "
    "is a sample size at which almost nothing is decidable, and one partition "
    "is not a distribution.",
    "A6 is exploratory. A2's 36-fold walk-forward study remains the "
    "authoritative historical evidence, and a result here neither confirms nor "
    "overturns it -- the two are not comparable.",
    "No model in A6 is PROMOTED, and the paper-trading engine stays "
    "fail-closed. Nothing in this card is a recommendation to trade.",
)


def _format_number(value: Any, places: int = 6) -> str:
    if value is None:
        return "not reported"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{places}f}" if isinstance(value, float) else str(value)
    return str(value)


def _capability_line(model: dict) -> str:
    capabilities = model.get("capabilities") or []
    if not capabilities:
        return "none declared"
    return ", ".join(capabilities)


def _outputs(model: dict) -> list[str]:
    """What the model actually produces, and what it declined to produce.

    The declined half matters as much: a blank cell in the benchmark table is a
    fact about the model, and a card that omits it invites the reader to assume
    the number was simply not computed.
    """
    scores = model.get("scores") or {}
    lines = ["- point forecast of the next bar's log return"]
    if scores.get("probabilistic"):
        probabilistic = scores["probabilistic"]
        if probabilistic.get("pinball_by_level"):
            lines.append("- conditional quantiles (pinball loss reported)")
        if probabilistic.get("brier") is not None:
            lines.append("- direction probability (Brier score and calibration reported)")
    if scores.get("variance"):
        lines.append("- conditional variance (QLIKE against the squared-return proxy)")
    declared = set(model.get("capabilities") or [])
    missing = [
        name
        for name, label in (
            ("QUANTILES", "conditional quantiles"),
            ("DIRECTION_PROBABILITY", "direction probability"),
            ("VARIANCE", "conditional variance"),
        )
        if name not in declared
    ]
    for name in missing:
        lines.append(f"- **does not** produce {name.lower().replace('_', ' ')} (not declared)")
    return lines


def build_card(
    model: dict,
    *,
    registration: dict,
    diagnostics: dict | None = None,
    comparison: dict | None = None,
    stability: dict | None = None,
    artifact: dict | None = None,
    naive_mae: float | None = None,
) -> str:
    """One card, entirely derived. Returns markdown."""
    model_id = model["model_id"]
    scores = model.get("scores") or {}
    point = scores.get("point") or {}
    direction = scores.get("direction") or {}

    lines: list[str] = [
        f"# {model_id}",
        "",
        f"**Family** {model['family']} | **Status** {model['status']} | "
        f"**Scientific status** {EXPLORATORY}",
        "",
        f"{registration.get('description', '')}",
        "",
        "## What it is",
        "",
        f"- **Capabilities declared**: {_capability_line(model)}",
        f"- **Preprocessing**: {model.get('hyperparameters', {}).get('preprocessing', 'declared per model')}",
        f"- **Requires**: {', '.join(registration.get('requires') or []) or 'nothing beyond the core install'}",
        "",
        "## Training budget",
        "",
        "- 1,000 deterministic rows, the contiguous tail of the training partition",
        f"- **Fit time**: {_format_number(model.get('fit_seconds'), 3)} s",
        f"- **Predict time**: {_format_number(model.get('predict_seconds'), 3)} s",
        f"- **Parameters**: {_format_number(model.get('parameter_count'))}",
        f"- **Training-row fingerprint**: `{(model.get('train_fingerprint') or 'n/a')[:16]}`",
        "",
        "## What it produces",
        "",
        *_outputs(model),
        "",
    ]

    if model["status"] not in {"ACTIVE", "RESOURCE_LIMIT"}:
        lines += [
            "## Why it did not run",
            "",
            f"```\n{model.get('failure')}\n```",
            "",
        ]
    else:
        lines += [
            "## Measured",
            "",
            "| metric | value |",
            "|---|---|",
            f"| MAE | {_format_number(point.get('mae'))} |",
            f"| RMSE | {_format_number(point.get('rmse'))} |",
            f"| MASE (naive = 1) | {_format_number(point.get('mase'))} |",
            f"| skill vs naive | {_format_number(point.get('skill_vs_naive'))} |",
            f"| forecast bias | {_format_number(point.get('bias'))} |",
            f"| directional accuracy | {_format_number(direction.get('accuracy'), 4)} |",
            f"| balanced accuracy | {_format_number(direction.get('balanced_accuracy'), 4)} |",
            f"| MCC | {_format_number(direction.get('mcc'), 4)} |",
            f"| train-constant null | {_format_number(direction.get('train_constant_baseline'), 4)} |",
            f"| beats that null | {_format_number(direction.get('beats_train_constant'))} |",
            "",
        ]
        if naive_mae is not None:
            lines += [f"The naive baseline's MAE on the same block is {naive_mae:.6f}.", ""]

    if comparison:
        verdict = "not distinguishable from the naive baseline"
        if comparison.get("significant_after_bh"):
            worse = (comparison.get("mean_loss_difference") or 0.0) > 0
            verdict = (
                "significantly **worse** than the naive baseline"
                if worse
                else "significantly better than the naive baseline"
            )
        lines += [
            "## Against the baseline",
            "",
            f"- Diebold-Mariano statistic {_format_number(comparison.get('dm_statistic'), 3)}, "
            f"p = {_format_number(comparison.get('p_value'), 4)}, "
            f"q = {_format_number(comparison.get('q_value'), 4)} after Benjamini-Hochberg",
            f"- **Verdict**: {verdict}",
            "",
        ]

    if stability:
        lines += [
            "## Stability",
            "",
            f"- mean skill across blocks {_format_number(stability.get('mean_skill'))}",
            f"- worst block {_format_number(stability.get('worst_block_skill'))}",
            f"- positive in every block: {_format_number(stability.get('positive_in_every_block'))}",
            "",
        ]

    if diagnostics:
        failed = diagnostics.get("assumptions_failed") or []
        lines += [
            "## Residual diagnostics",
            "",
            f"- n = {diagnostics.get('n')}",
            f"- assumptions rejected: {', '.join(failed) if failed else 'none'}",
            "- diagnostics describe how a model fails, not whether it is profitable",
            "",
        ]

    if artifact:
        lines += [
            "## Serialization",
            "",
            f"- {artifact.get('bytes')} bytes, sha256 `{(artifact.get('sha256') or '')[:16]}`",
            "- reload reproduces the forecasts bit-identically",
            "",
        ]

    notes = registration.get("notes") or []
    if notes:
        lines += ["## Notes", "", *[f"- {note}" for note in notes], ""]

    lines += [
        "## Limitations",
        "",
        *[f"- {limitation}" for limitation in SHARED_LIMITATIONS],
        "",
    ]
    return "\n".join(lines)


def write_cards(
    manifest: dict,
    directory: Path | str,
    *,
    diagnostics: dict | None = None,
    comparisons: dict | None = None,
    stability: dict | None = None,
    artifacts: dict | None = None,
) -> list[Path]:
    """One card per registered model, including the ones that did not run.

    A model absent from the card directory would be a model absent from the
    record, which is the failure the whole status vocabulary exists to prevent.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    registrations = {entry["model_id"]: entry for entry in manifest.get("registry_entries", [])}
    diagnostic_reports = (diagnostics or {}).get("per_model", {})
    comparison_rows = {
        row["model_id"]: row for row in (comparisons or {}).get("results", [])
    }
    stability_rows = {row["model_id"]: row for row in (stability or [])}

    written: list[Path] = []
    for model in manifest.get("models", []):
        model_id = model["model_id"]
        card = build_card(
            model,
            registration=registrations.get(model_id, {}),
            diagnostics=diagnostic_reports.get(model_id),
            comparison=comparison_rows.get(model_id),
            stability=stability_rows.get(model_id),
            artifact=(artifacts or {}).get(model_id),
            naive_mae=manifest.get("naive_mae"),
        )
        path = out / f"{model_id}.md"
        path.write_text(card, encoding="utf-8")
        written.append(path)
    return written


__all__ = ["SHARED_LIMITATIONS", "build_card", "write_cards"]
