"""Reusable time-series diagnostics.

Each test returns a :class:`DiagnosticResult` that states its own null
hypothesis and what rejecting it means, because the direction of a test's null
is the single easiest thing to get backwards. ADF and KPSS are the canonical
trap: ADF's null is *non-stationarity*, KPSS's null is *stationarity*, so
"p < 0.05" means opposite things for the two. Running both is standard practice
precisely because either alone is easy to misread.

On mechanical use
-----------------
These exist to describe data, not to select models. A diagnostic suite run over
one series produces a dozen p-values, and picking a model by "whichever test
came out significant" is a multiple-comparisons procedure with no correction --
the same error the original pipeline made when it scanned 60 PACF lags at a 5%
band and treated the survivors as discovered structure.

:class:`DiagnosticReport` therefore carries the number of tests run and can
apply a family-wise or false-discovery-rate correction, and every report says
plainly how many hypotheses were examined. Use the results as evidence to weigh,
never as a rule to dispatch on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .multiple_testing import benjamini_hochberg, bonferroni


@dataclass(frozen=True)
class DiagnosticResult:
    """One statistical test, with its null hypothesis attached."""

    name: str
    statistic: float
    p_value: float
    null_hypothesis: str
    reject_means: str
    alpha: float = 0.05
    detail: dict = field(default_factory=dict)

    @property
    def rejects_null(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < self.alpha)

    @property
    def conclusion(self) -> str:
        if not np.isfinite(self.p_value):
            return f"{self.name}: inconclusive (test did not produce a p-value)"
        if self.rejects_null:
            return f"{self.name}: p={self.p_value:.4g} < {self.alpha} -- {self.reject_means}"
        return (
            f"{self.name}: p={self.p_value:.4g} >= {self.alpha} -- "
            f"cannot reject: {self.null_hypothesis}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "null_hypothesis": self.null_hypothesis,
            "reject_means": self.reject_means,
            "alpha": self.alpha,
            "rejects_null": self.rejects_null,
            **self.detail,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """A family of diagnostics run together, and how many hypotheses that was."""

    subject: str
    results: tuple[DiagnosticResult, ...]
    n_observations: int

    @property
    def n_tests(self) -> int:
        return len(self.results)

    def get(self, name: str) -> DiagnosticResult | None:
        return next((r for r in self.results if r.name == name), None)

    def p_values(self) -> dict[str, float]:
        return {r.name: r.p_value for r in self.results}

    def adjusted(self, method: str = "benjamini-hochberg", alpha: float = 0.05) -> dict:
        """Apply a multiple-testing correction across this family.

        Running eleven tests at alpha=0.05 gives a ~43% chance of at least one
        spurious rejection. Reporting the raw p-values without saying how many
        were computed is how that becomes a finding.
        """
        names = [r.name for r in self.results]
        raw = [r.p_value for r in self.results]

        if method == "bonferroni":
            adjusted, rejected = bonferroni(raw, alpha=alpha)
        elif method == "benjamini-hochberg":
            adjusted, rejected = benjamini_hochberg(raw, alpha=alpha)
        else:
            raise ValueError(f"unknown correction method {method!r}")

        return {
            "method": method,
            "alpha": alpha,
            "n_tests": self.n_tests,
            "results": [
                {
                    "name": name,
                    "p_value": p,
                    "p_adjusted": p_adj,
                    "significant_after_correction": bool(flag),
                }
                for name, p, p_adj, flag in zip(names, raw, adjusted, rejected, strict=True)
            ],
        }

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "n_observations": self.n_observations,
            "n_tests": self.n_tests,
            "results": [r.to_dict() for r in self.results],
            "multiple_testing_note": (
                f"{self.n_tests} hypotheses were tested on this series. Raw p-values are "
                "uncorrected; call adjusted() before treating any single rejection as a "
                "finding, and do not select models by dispatching on these outcomes."
            ),
        }

    def summary(self) -> str:
        lines = [f"{self.subject} ({self.n_observations} observations, {self.n_tests} tests)"]
        lines.extend(f"  {r.conclusion}" for r in self.results)
        return "\n".join(lines)


def _clean(series: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.Series(np.asarray(series, dtype=float)).replace([np.inf, -np.inf], np.nan)
    return values.dropna().to_numpy()


# ------------------------------------------------------------- stationarity


def adf_test(series, *, alpha: float = 0.05, regression: str = "c") -> DiagnosticResult:
    """Augmented Dickey-Fuller. Null: the series HAS a unit root (non-stationary)."""
    from statsmodels.tsa.stattools import adfuller

    values = _clean(series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value, used_lag, n_obs, critical, _ = adfuller(values, regression=regression)

    return DiagnosticResult(
        name="adf",
        statistic=float(statistic),
        p_value=float(p_value),
        null_hypothesis="the series has a unit root (is non-stationary)",
        reject_means="evidence of stationarity",
        alpha=alpha,
        detail={
            "used_lag": int(used_lag),
            "n_obs": int(n_obs),
            "critical_values": {k: float(v) for k, v in critical.items()},
        },
    )


def kpss_test(series, *, alpha: float = 0.05, regression: str = "c") -> DiagnosticResult:
    """KPSS. Null: the series IS stationary -- the opposite of ADF's null.

    Reported alongside ADF deliberately. Agreement between the two is
    informative; disagreement usually means the series is neither cleanly
    stationary nor cleanly a unit root, which is worth knowing.
    """
    from statsmodels.tsa.stattools import kpss

    values = _clean(series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value, lags, critical = kpss(values, regression=regression, nlags="auto")

    return DiagnosticResult(
        name="kpss",
        statistic=float(statistic),
        p_value=float(p_value),
        null_hypothesis="the series is stationary",
        reject_means="evidence of non-stationarity",
        alpha=alpha,
        detail={
            "lags": int(lags),
            "critical_values": {k: float(v) for k, v in critical.items()},
            "note": "p-value is interpolated and clipped to [0.01, 0.10] by statsmodels",
        },
    )


# ---------------------------------------------------------- autocorrelation


def ljung_box_test(series, *, lags: int = 20, alpha: float = 0.05) -> DiagnosticResult:
    """Ljung-Box. Null: no autocorrelation up to ``lags``.

    On model residuals, rejecting means the model left structure behind.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    values = _clean(series)
    lags = min(lags, max(1, len(values) // 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = acorr_ljungbox(values, lags=[lags], return_df=True)

    return DiagnosticResult(
        name="ljung_box",
        statistic=float(table["lb_stat"].iloc[-1]),
        p_value=float(table["lb_pvalue"].iloc[-1]),
        null_hypothesis=f"no autocorrelation up to lag {lags}",
        reject_means="serial correlation is present",
        alpha=alpha,
        detail={"lags": int(lags)},
    )


def durbin_watson_statistic(series) -> DiagnosticResult:
    """Durbin-Watson: ~2 means no first-order autocorrelation, <2 positive, >2 negative.

    Has no p-value, so it is reported as a statistic with the interpretation
    stated rather than dressed up as a hypothesis test.
    """
    from statsmodels.stats.stattools import durbin_watson

    statistic = float(durbin_watson(_clean(series)))
    return DiagnosticResult(
        name="durbin_watson",
        statistic=statistic,
        p_value=float("nan"),
        null_hypothesis="no first-order autocorrelation (statistic near 2.0)",
        reject_means="n/a -- this statistic has no p-value",
        detail={
            "interpretation": (
                "positive autocorrelation" if statistic < 1.5
                else "negative autocorrelation" if statistic > 2.5
                else "no strong first-order autocorrelation"
            )
        },
    )


def acf_values(series, *, nlags: int = 40) -> pd.Series:
    from statsmodels.tsa.stattools import acf

    values = _clean(series)
    nlags = min(nlags, max(1, len(values) // 2 - 1))
    return pd.Series(acf(values, nlags=nlags), name="acf")


def pacf_values(series, *, nlags: int = 40, method: str = "ywm") -> pd.Series:
    from statsmodels.tsa.stattools import pacf

    values = _clean(series)
    nlags = min(nlags, max(1, len(values) // 2 - 1))
    return pd.Series(pacf(values, nlags=nlags, method=method), name="pacf")


def significance_band(n_observations: int, *, alpha: float = 0.05) -> float:
    """The +/- band outside which an ACF/PACF value is nominally significant."""
    from scipy.stats import norm

    return float(norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n_observations))


# ------------------------------------------------------- heteroskedasticity


def arch_lm_test(series, *, lags: int = 12, alpha: float = 0.05) -> DiagnosticResult:
    """Engle's ARCH-LM. Null: no conditional heteroskedasticity.

    Rejecting is the evidence that justifies fitting a GARCH model at all. Worth
    checking rather than assuming: the original pipeline fitted GARCH(1,1)
    unconditionally.
    """
    from statsmodels.stats.diagnostic import het_arch

    values = _clean(series)
    lags = min(lags, max(1, len(values) // 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm_stat, lm_p, f_stat, f_p = het_arch(values, nlags=lags)

    return DiagnosticResult(
        name="arch_lm",
        statistic=float(lm_stat),
        p_value=float(lm_p),
        null_hypothesis=f"no ARCH effects up to lag {lags} (homoskedastic)",
        reject_means="volatility clusters -- a conditional variance model is justified",
        alpha=alpha,
        detail={"lags": int(lags), "f_statistic": float(f_stat), "f_p_value": float(f_p)},
    )


# ---------------------------------------------------------------- normality


def jarque_bera_test(series, *, alpha: float = 0.05) -> DiagnosticResult:
    """Jarque-Bera. Null: the series is normally distributed.

    Financial returns essentially always reject this, so a rejection is not news.
    It is reported because the *degree* of excess kurtosis says how badly a
    Gaussian interval will understate tail risk -- which is exactly what a
    normal-shock Monte Carlo does.
    """
    from statsmodels.stats.stattools import jarque_bera

    values = _clean(series)
    statistic, p_value, skew, kurtosis = jarque_bera(values)

    return DiagnosticResult(
        name="jarque_bera",
        statistic=float(statistic),
        p_value=float(p_value),
        null_hypothesis="the series is normally distributed",
        reject_means="non-normal: skewed and/or fat-tailed",
        alpha=alpha,
        detail={
            "skew": float(skew),
            "kurtosis": float(kurtosis),
            "excess_kurtosis": float(kurtosis - 3.0),
        },
    )


# ------------------------------------------------------------------ bundles


def diagnose_returns(returns: pd.Series, *, alpha: float = 0.05, max_lag: int = 20) -> DiagnosticReport:
    """The standard battery for a return series."""
    values = returns.dropna()
    results = [
        adf_test(values, alpha=alpha),
        kpss_test(values, alpha=alpha),
        ljung_box_test(values, lags=max_lag, alpha=alpha),
        arch_lm_test(values, alpha=alpha),
        jarque_bera_test(values, alpha=alpha),
        durbin_watson_statistic(values),
    ]
    return DiagnosticReport(subject="returns", results=tuple(results), n_observations=len(values))


def diagnose_prices(close: pd.Series, *, alpha: float = 0.05) -> DiagnosticReport:
    """Stationarity of log price. Expected to look like a unit root."""
    values = np.log(close.dropna())
    results = [adf_test(values, alpha=alpha), kpss_test(values, alpha=alpha)]
    return DiagnosticReport(subject="log_price", results=tuple(results), n_observations=len(values))


def diagnose_residuals(
    residuals: pd.Series, *, alpha: float = 0.05, max_lag: int = 20
) -> DiagnosticReport:
    """The battery for model residuals.

    Well-behaved residuals show no remaining autocorrelation (Ljung-Box does not
    reject). ARCH effects usually remain even in a good mean model, which is what
    a separate volatility model is for.
    """
    values = residuals.dropna()
    results = [
        ljung_box_test(values, lags=max_lag, alpha=alpha),
        arch_lm_test(values, alpha=alpha),
        jarque_bera_test(values, alpha=alpha),
        durbin_watson_statistic(values),
        adf_test(values, alpha=alpha),
    ]
    return DiagnosticReport(subject="residuals", results=tuple(results), n_observations=len(values))


__all__ = [
    "DiagnosticReport",
    "DiagnosticResult",
    "acf_values",
    "adf_test",
    "arch_lm_test",
    "diagnose_prices",
    "diagnose_residuals",
    "diagnose_returns",
    "durbin_watson_statistic",
    "jarque_bera_test",
    "kpss_test",
    "ljung_box_test",
    "pacf_values",
    "significance_band",
]
