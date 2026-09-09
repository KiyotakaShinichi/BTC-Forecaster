"""Paired comparison, stability, and whether the models actually differ.

Three questions the results table cannot answer on its own.

**Is any model distinguishable from the baseline?** Diebold-Mariano on paired
forecast origins, reusing `evaluation.inference.diebold_mariano` -- which
already carries the HAC variance and the nested-model warning that A2 needed.
Every candidate is compared against the same baseline over the same window,
because comparing two models on different evaluation blocks is not a comparison.

**How many of those p-values are arithmetic?** Forty comparisons at alpha=0.05
produce two significant results from nothing. Raw p-values and
Benjamini-Hochberg q-values are reported together with the size of the family,
so a reader cannot mistake the first for a discovery.

**Do different architectures make different mistakes?** Pairwise error
correlation, sign disagreement, and shared large-error events. This is
preparation for a possible future ensemble study and explicitly not one: if
forty models make the same errors, an ensemble of them is one model with extra
steps, and that is worth knowing before anyone builds it.

Stability is reported by temporal block -- early, middle, late thirds of the
holdout -- with the worst-block figure alongside the mean. A model whose skill
comes from one regime and reverses in another has a mean that describes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..diagnostics.multiple_testing import benjamini_hochberg, expected_false_positives
from ..evaluation.inference import diebold_mariano, is_nested, loss_series

#: Below this many paired origins, a DM test is not worth reporting: its HAC
#: variance estimate is unstable and its p-value is a number rather than
#: evidence.
MINIMUM_PAIRED_ORIGINS = 100

INSUFFICIENT_FORWARD_EVIDENCE = "INSUFFICIENT_FORWARD_EVIDENCE"

#: The baseline every candidate is compared against. The naive zero-return
#: forecast, which is A2's hypothesis and the one that has never been beaten.
DEFAULT_BASELINE = "naive_last_value"


@dataclass(frozen=True)
class Comparison:
    """One candidate against one baseline, over one shared window."""

    model_id: str
    baseline_id: str
    n_origins: int
    status: str
    statistic: float | None = None
    p_value: float | None = None
    mean_loss_difference: float | None = None
    nested: bool = False

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "baseline_id": self.baseline_id,
            "n_paired_origins": self.n_origins,
            "status": self.status,
            "dm_statistic": self.statistic,
            "p_value": self.p_value,
            "mean_loss_difference": self.mean_loss_difference,
            "nested_in_baseline": self.nested,
        }


def compare_against_baseline(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    *,
    baseline_id: str = DEFAULT_BASELINE,
    loss: str = "absolute",
) -> list[Comparison]:
    """Diebold-Mariano for every candidate against one baseline.

    Paired: the same origins, the same window, the same loss. A comparison
    across different evaluation blocks is not a comparison, and it is the
    easiest way to make a model look better than it is.
    """
    if baseline_id not in forecasts:
        raise KeyError(f"baseline {baseline_id!r} did not produce a forecast")
    baseline = forecasts[baseline_id]
    n = len(actual)

    comparisons: list[Comparison] = []
    for model_id, prediction in sorted(forecasts.items()):
        if model_id == baseline_id:
            continue
        if n < MINIMUM_PAIRED_ORIGINS:
            comparisons.append(
                Comparison(
                    model_id=model_id,
                    baseline_id=baseline_id,
                    n_origins=n,
                    status=INSUFFICIENT_FORWARD_EVIDENCE,
                )
            )
            continue
        try:
            # The existing implementation takes loss *series*, not forecasts,
            # which is what lets the same test serve absolute and squared loss
            # without a second code path.
            result = diebold_mariano(
                loss_series(actual, prediction, loss=loss),
                loss_series(actual, baseline, loss=loss),
                model_a=model_id,
                model_b=baseline_id,
                loss=loss,
                nested=is_nested(model_id, baseline_id),
            )
            comparisons.append(
                Comparison(
                    model_id=model_id,
                    baseline_id=baseline_id,
                    n_origins=n,
                    status="OK",
                    statistic=float(result.statistic),
                    p_value=float(result.p_value),
                    mean_loss_difference=float(result.mean_loss_difference),
                    nested=bool(result.nested),
                )
            )
        except Exception as exc:  # noqa: BLE001 -- recorded, not raised
            comparisons.append(
                Comparison(
                    model_id=model_id,
                    baseline_id=baseline_id,
                    n_origins=n,
                    status=f"ERROR: {type(exc).__name__}",
                )
            )
    return comparisons


def correct_for_multiplicity(comparisons: list[Comparison], *, alpha: float = 0.05) -> dict:
    """Raw p-values, BH q-values, and the size of the family they came from.

    Reported together on purpose. Forty comparisons at alpha=0.05 yield two
    significant results from nothing at all, and a table of raw p-values invites
    the reader to treat those two as discoveries.
    """
    usable = [c for c in comparisons if c.p_value is not None and np.isfinite(c.p_value)]
    if not usable:
        return {
            "family_size": 0,
            "status": INSUFFICIENT_FORWARD_EVIDENCE,
            "note": "no comparison produced a usable p-value",
        }

    p_values = np.array([c.p_value for c in usable], dtype=float)
    # (adjusted, rejected) -- the order the implementation returns them in.
    q_values, rejected = benjamini_hochberg(p_values, alpha=alpha)
    return {
        "family_size": len(usable),
        "alpha": alpha,
        "raw_significant": int((p_values < alpha).sum()),
        "expected_false_positives_at_alpha": float(
            expected_false_positives(len(usable), alpha)
        ),
        "significant_after_bh": int(rejected.sum()),
        "results": [
            {
                **comparison.as_dict(),
                "q_value": float(q),
                "significant_after_bh": bool(flag),
            }
            for comparison, q, flag in zip(usable, q_values, rejected, strict=True)
        ],
        "note": (
            "a raw p-value below alpha in a family this size is expected; the "
            "q-value is what a claim should rest on"
        ),
    }


def stability_by_block(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    *,
    blocks: int = 3,
) -> pd.DataFrame:
    """MAE per temporal block, with the worst block reported beside the mean.

    A model whose skill comes from one regime and reverses in another has a mean
    that describes nothing, and the mean is what a results table shows.
    """
    n = len(actual)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    naive = np.abs(actual)

    rows = []
    for model_id, prediction in sorted(forecasts.items()):
        errors = np.abs(actual - prediction)
        per_block = []
        row: dict = {"model_id": model_id}
        for i in range(blocks):
            start, end = edges[i], edges[i + 1]
            block_mae = float(errors[start:end].mean())
            block_naive = float(naive[start:end].mean())
            skill = 1.0 - block_mae / block_naive if block_naive > 0 else float("nan")
            row[f"block_{i + 1}_mae"] = block_mae
            row[f"block_{i + 1}_skill"] = skill
            per_block.append(skill)
        row["mean_skill"] = float(np.mean(per_block))
        row["worst_block_skill"] = float(np.min(per_block))
        row["skill_std_across_blocks"] = float(np.std(per_block))
        # The number that matters: a model positive on average and negative
        # somewhere is not a model that worked.
        row["positive_in_every_block"] = bool(np.all(np.array(per_block) > 0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_skill", ascending=False).reset_index(drop=True)


def _correlation(rows: np.ndarray) -> np.ndarray:
    """Pairwise correlation with constant rows reported as NaN rather than a warning.

    Three baselines in this zoo forecast a constant, and `naive_last_value`
    forecasts exactly zero. A constant has no correlation with anything, and
    numpy expresses that by dividing by a zero standard deviation -- correct
    answer, RuntimeWarning per call. NaN is the honest value and saying so
    explicitly keeps the warning out of every benchmark run.
    """
    deviations = rows.std(axis=1)
    usable = deviations > 0
    out = np.full((len(rows), len(rows)), np.nan, dtype=float)
    if usable.sum() >= 2:
        sub = np.corrcoef(rows[usable])
        indices = np.flatnonzero(usable)
        out[np.ix_(indices, indices)] = sub
    for i in range(len(rows)):
        if not usable[i]:
            continue
        out[i, i] = 1.0
    return out


def error_diversity(
    actual: np.ndarray, forecasts: dict[str, np.ndarray], *, large_error_quantile: float = 0.95
) -> dict:
    """Do different families make different mistakes?

    Preparation for a possible future ensemble study, and explicitly not one. If
    forty models produce errors correlating at 0.99, an ensemble of them is one
    model with extra steps -- which is worth knowing before anyone builds it,
    and is not something a results table shows.
    """
    model_ids = sorted(forecasts)
    if len(model_ids) < 2:
        return {"status": "at least two models are needed to compare mistakes"}

    errors = np.vstack([actual - forecasts[m] for m in model_ids])
    predictions = np.vstack([forecasts[m] for m in model_ids])

    error_correlation = pd.DataFrame(
        _correlation(errors), index=model_ids, columns=model_ids
    )
    forecast_correlation = pd.DataFrame(
        _correlation(predictions), index=model_ids, columns=model_ids
    )

    signs = np.sign(predictions)
    n_models = len(model_ids)
    disagreement = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            disagreement[i, j] = float(np.mean(signs[i] != signs[j]))

    threshold = float(np.quantile(np.abs(errors), large_error_quantile))
    large = np.abs(errors) >= threshold
    shared = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            union = np.sum(large[i] | large[j])
            shared[i, j] = float(np.sum(large[i] & large[j]) / union) if union else 0.0

    off_diagonal = ~np.eye(n_models, dtype=bool)
    return {
        "models": model_ids,
        "mean_pairwise_error_correlation": float(
            np.nanmean(error_correlation.to_numpy()[off_diagonal])
        ),
        "min_pairwise_error_correlation": float(
            np.nanmin(error_correlation.to_numpy()[off_diagonal])
        ),
        "mean_pairwise_sign_disagreement": float(disagreement[off_diagonal].mean()),
        "mean_shared_large_error_jaccard": float(shared[off_diagonal].mean()),
        "large_error_threshold": threshold,
        "large_error_quantile": large_error_quantile,
        "error_correlation": error_correlation.round(4).to_dict(),
        "forecast_correlation": forecast_correlation.round(4).to_dict(),
        "interpretation": (
            "high error correlation with high sign agreement means the zoo is "
            "one model wearing forty names, and an ensemble would add nothing. "
            "This is measured here so a future ensemble study starts from "
            "evidence rather than from optimism. A6 builds no ensemble."
        ),
    }


__all__ = [
    "DEFAULT_BASELINE",
    "INSUFFICIENT_FORWARD_EVIDENCE",
    "MINIMUM_PAIRED_ORIGINS",
    "Comparison",
    "compare_against_baseline",
    "correct_for_multiplicity",
    "error_diversity",
    "stability_by_block",
]
