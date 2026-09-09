"""Residual diagnostics, and the discipline of saying INSUFFICIENT_N.

The tests themselves already exist. `btc_forecaster.diagnostics.suite` has ADF,
KPSS, Ljung-Box, ARCH-LM and Jarque-Bera, all tested; re-implementing them here
would create a second thing that can disagree.

What this module adds is the part that is easy to get wrong in a forty-model
zoo, and it is not statistical machinery -- it is restraint:

**A p-value from too few observations is not a small p-value, it is noise.**
Ljung-Box at 20 lags on 30 residuals is arithmetic without evidence. So every
test declares a minimum sample and returns ``INSUFFICIENT_N`` below it rather
than a number that will be read as a result.

**A diagnostic is not a verdict.** Residuals that pass Ljung-Box do not make a
model good, and residuals that fail ARCH-LM do not make it useless -- almost
every model in this zoo will fail ARCH-LM, because daily crypto returns are
volatility-clustered and a conditional-mean model does not claim otherwise. The
report says which assumptions failed and stops there.

**Multiple testing applies here too.** Forty models times seven tests is 280
p-values, and at alpha=0.05 fourteen of them are expected to be significant by
chance alone. The count of expected false positives is reported beside the count
of rejections, so the reader is not invited to be surprised by fourteen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..diagnostics.multiple_testing import benjamini_hochberg, expected_false_positives
from ..diagnostics.suite import (
    acf_values,
    adf_test,
    arch_lm_test,
    jarque_bera_test,
    kpss_test,
    ljung_box_test,
    pacf_values,
    significance_band,
)

#: Below this, a residual series does not support a hypothesis test worth
#: reporting. Not a convention borrowed from anywhere -- it is roughly the point
#: at which Ljung-Box with a sensible lag count has more observations than lags
#: by a comfortable margin.
MINIMUM_OBSERVATIONS = 50

#: Ljung-Box lag count. Kept well under n/5 so the statistic is not dominated by
#: its own degrees of freedom.
DEFAULT_LAGS = 10

INSUFFICIENT_N = "INSUFFICIENT_N"


@dataclass(frozen=True)
class DiagnosticOutcome:
    """One test's result, or an honest refusal to run it."""

    name: str
    status: str
    statistic: float | None = None
    p_value: float | None = None
    interpretation: str = ""
    n: int = 0

    @property
    def ran(self) -> bool:
        return self.status != INSUFFICIENT_N

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "interpretation": self.interpretation,
            "n": self.n,
        }


def _insufficient(name: str, n: int) -> DiagnosticOutcome:
    return DiagnosticOutcome(
        name=name,
        status=INSUFFICIENT_N,
        interpretation=(
            f"{n} observations is below the {MINIMUM_OBSERVATIONS} this test needs "
            "to say anything; a p-value from too few points is noise, not evidence"
        ),
        n=n,
    )


def _run(name: str, residuals: np.ndarray, test: Any) -> DiagnosticOutcome:
    n = int(len(residuals))
    if n < MINIMUM_OBSERVATIONS:
        return _insufficient(name, n)
    try:
        result = test(residuals)
    except Exception as exc:  # noqa: BLE001 -- a failed diagnostic is recorded
        return DiagnosticOutcome(
            name=name, status="ERROR", interpretation=f"{type(exc).__name__}: {exc}", n=n
        )
    # `rejects_null`, not `rejected`. A getattr default of False here meant no
    # test ever rejected anything, which is the quietest possible way for a
    # diagnostic battery to be useless -- every model would have reported clean
    # residuals.
    return DiagnosticOutcome(
        name=name,
        status="REJECT" if result.rejects_null else "NOT_REJECTED",
        statistic=float(result.statistic),
        p_value=float(result.p_value),
        interpretation=result.conclusion,
        n=n,
    )


def diagnose(residuals: np.ndarray | pd.Series, *, alpha: float = 0.05) -> dict:
    """Run the battery on one model's residuals.

    ACF and PACF are reported as values with their white-noise band rather than
    as a pass/fail, because they are a shape to look at rather than a hypothesis
    to reject.
    """
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))

    outcomes = [
        _run("ljung_box", values, lambda r: ljung_box_test(r, lags=DEFAULT_LAGS, alpha=alpha)),
        _run("arch_lm", values, lambda r: arch_lm_test(r, lags=DEFAULT_LAGS, alpha=alpha)),
        _run("jarque_bera", values, lambda r: jarque_bera_test(r, alpha=alpha)),
        _run("adf", values, lambda r: adf_test(r, alpha=alpha)),
        _run("kpss", values, lambda r: kpss_test(r, alpha=alpha)),
    ]

    correlogram: dict = {"status": INSUFFICIENT_N}
    if n >= MINIMUM_OBSERVATIONS:
        if float(values.std()) <= 0.0:
            # A constant residual series has no autocorrelation to report. The
            # arithmetic divides by a zero variance and produces NaN with a
            # warning; DEGENERATE is the same information without the noise.
            correlogram = {
                "status": "DEGENERATE",
                "reason": "the residual series is constant; autocorrelation is undefined",
            }
        else:
            lags = min(20, max(1, n // 5))
            correlogram = {
                "status": "OK",
                "band": float(significance_band(n)),
                "acf": [float(v) for v in acf_values(values, nlags=lags).to_numpy()],
                "pacf": [float(v) for v in pacf_values(values, nlags=lags).to_numpy()],
                "lags": lags,
            }

    return {
        "n": n,
        "mean": float(values.mean()) if n else float("nan"),
        "std": float(values.std(ddof=1)) if n > 1 else float("nan"),
        "tests": [outcome.as_dict() for outcome in outcomes],
        "correlogram": correlogram,
        "assumptions_failed": [o.name for o in outcomes if o.status == "REJECT"],
        "note": (
            "diagnostics, not a verdict. Failing ARCH-LM is expected for every "
            "conditional-mean model on a volatility-clustered series and says "
            "nothing about forecast quality; passing Ljung-Box does not make a "
            "model useful."
        ),
    }


def diagnose_all(
    residuals_by_model: dict[str, np.ndarray], *, alpha: float = 0.05
) -> dict:
    """The battery across the zoo, with the multiple-testing arithmetic attached.

    Forty models times five tests is 200 p-values. At alpha=0.05, ten
    rejections are expected from nothing at all, so the expected count is
    reported beside the observed one and the p-values are corrected -- otherwise
    the residual table becomes a list of discoveries.
    """
    per_model = {
        model_id: diagnose(residuals, alpha=alpha)
        for model_id, residuals in sorted(residuals_by_model.items())
    }

    flat: list[tuple[str, str, float]] = []
    for model_id, report in per_model.items():
        for test in report["tests"]:
            if test["p_value"] is not None and np.isfinite(test["p_value"]):
                flat.append((model_id, test["name"], float(test["p_value"])))

    corrected: dict = {"comparisons": 0}
    if flat:
        p_values = np.array([p for _, _, p in flat])
        q_values, rejected = benjamini_hochberg(p_values, alpha=alpha)
        corrected = {
            "comparisons": len(flat),
            "alpha": alpha,
            "raw_rejections": int((p_values < alpha).sum()),
            "expected_false_positives_at_alpha": float(
                expected_false_positives(len(flat), alpha)
            ),
            "rejections_after_bh": int(rejected.sum()),
            "results": [
                {
                    "model_id": model_id,
                    "test": name,
                    "p_value": float(p),
                    "q_value": float(q),
                    "significant_after_bh": bool(flag),
                }
                for (model_id, name, p), q, flag in zip(flat, q_values, rejected, strict=True)
            ],
        }

    return {
        "minimum_observations": MINIMUM_OBSERVATIONS,
        "per_model": per_model,
        "multiple_testing": corrected,
        "note": (
            "a residual diagnostic describes how a model failed, not whether it "
            "is profitable; and with this many tests, some rejections are "
            "arithmetic rather than evidence"
        ),
    }


__all__ = [
    "DEFAULT_LAGS",
    "INSUFFICIENT_N",
    "MINIMUM_OBSERVATIONS",
    "DiagnosticOutcome",
    "diagnose",
    "diagnose_all",
]
