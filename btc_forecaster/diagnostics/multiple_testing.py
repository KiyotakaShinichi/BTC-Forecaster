"""Multiple-testing corrections.

Any procedure that examines many hypotheses and reports the ones that came out
significant needs one of these. The pre-Track-A pipeline had at least three such
procedures running uncorrected:

* 60 PACF lags scanned against a 5% band, and the survivors treated as
  discovered structure. About three are expected on pure noise.
* 48 candidate cutoff dates scored, and the maximum reported as the result.
* A diagnostic battery whose individual p-values were read one at a time.

Implemented here rather than taken from ``statsmodels.stats.multitest`` so the
correction travels with the diagnostics report as a small, readable dependency,
and so NaN p-values (from tests that failed to converge) are handled explicitly
rather than silently poisoning the ranking.
"""

from __future__ import annotations

import numpy as np


def bonferroni(p_values, *, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Family-wise error rate control. Conservative, and simple to defend.

    Multiplies each p-value by the number of tests. Controls the probability of
    *any* false positive, at the cost of power -- with many tests it will miss
    real effects. Prefer it when a single false claim would be expensive.
    """
    raw = np.asarray(p_values, dtype=float)
    n = int(np.isfinite(raw).sum())
    if n == 0:
        return raw.copy(), np.zeros_like(raw, dtype=bool)

    adjusted = np.minimum(raw * n, 1.0)
    return adjusted, np.where(np.isfinite(adjusted), adjusted < alpha, False)


def benjamini_hochberg(p_values, *, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """False discovery rate control. The right default for exploratory work.

    Controls the expected *proportion* of rejections that are false, rather than
    the probability of any false rejection. Substantially more powerful than
    Bonferroni when many hypotheses are genuinely non-null, which is the usual
    situation for a diagnostic battery.

    NaN p-values are excluded from the ranking and reported as non-significant,
    so a test that failed to converge cannot shift the threshold for the others.
    """
    raw = np.asarray(p_values, dtype=float)
    adjusted = np.full(raw.shape, np.nan, dtype=float)
    rejected = np.zeros(raw.shape, dtype=bool)

    finite = np.where(np.isfinite(raw))[0]
    n = len(finite)
    if n == 0:
        return adjusted, rejected

    order = finite[np.argsort(raw[finite])]
    ranks = np.arange(1, n + 1, dtype=float)

    scaled = raw[order] * n / ranks
    # Enforce monotonicity from the largest p-value downward.
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.minimum(monotone, 1.0)

    adjusted[order] = monotone
    rejected[order] = monotone < alpha
    return adjusted, rejected


def expected_false_positives(n_tests: int, alpha: float = 0.05) -> float:
    """How many rejections pure noise would produce. The number to compare against."""
    return float(n_tests * alpha)


def family_wise_error_rate(n_tests: int, alpha: float = 0.05) -> float:
    """Probability of at least one false positive across ``n_tests`` independent tests."""
    return float(1.0 - (1.0 - alpha) ** n_tests)


__all__ = [
    "benjamini_hochberg",
    "bonferroni",
    "expected_false_positives",
    "family_wise_error_rate",
]
