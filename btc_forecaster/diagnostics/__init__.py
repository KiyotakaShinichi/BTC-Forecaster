"""Time-series diagnostics, with multiple-testing control.

These describe data; they do not select models. See the module docstring of
btc_forecaster.diagnostics.suite for why that distinction is enforced here.
"""

from .multiple_testing import (
    benjamini_hochberg,
    bonferroni,
    expected_false_positives,
    family_wise_error_rate,
)
from .suite import (
    DiagnosticReport,
    DiagnosticResult,
    acf_values,
    adf_test,
    arch_lm_test,
    diagnose_prices,
    diagnose_residuals,
    diagnose_returns,
    durbin_watson_statistic,
    jarque_bera_test,
    kpss_test,
    ljung_box_test,
    pacf_values,
    significance_band,
)

__all__ = [
    "DiagnosticReport",
    "DiagnosticResult",
    "acf_values",
    "adf_test",
    "arch_lm_test",
    "benjamini_hochberg",
    "bonferroni",
    "diagnose_prices",
    "diagnose_residuals",
    "diagnose_returns",
    "durbin_watson_statistic",
    "expected_false_positives",
    "family_wise_error_rate",
    "jarque_bera_test",
    "kpss_test",
    "ljung_box_test",
    "pacf_values",
    "significance_band",
]
