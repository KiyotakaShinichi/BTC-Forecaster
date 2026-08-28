"""Run plots. matplotlib is optional and imported lazily.

Every function returns ``None`` when matplotlib is unavailable rather than
raising, so a headless or minimal install still produces the numerical
artifacts, which are the ones that matter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _pyplot(show: bool = False):
    try:
        import matplotlib
    except ImportError:
        return None
    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_forecast(
    history: pd.Series,
    forecast: pd.DataFrame,
    path: Path,
    *,
    title: str = "Forecast",
    history_bars: int = 365,
    show: bool = False,
) -> Path | None:
    plt = _pyplot(show)
    if plt is None:
        return None

    figure, axis = plt.subplots(figsize=(13, 6.5))
    tail = history.iloc[-history_bars:]
    axis.plot(tail.index, tail.to_numpy(), label=f"Historical (last {len(tail)} bars)", linewidth=1.8)
    axis.plot(forecast.index, forecast["point"].to_numpy(), label="Forecast", linewidth=1.8, color="crimson")

    if {"lower", "upper"} <= set(forecast.columns):
        axis.fill_between(
            forecast.index,
            forecast["lower"].to_numpy(),
            forecast["upper"].to_numpy(),
            alpha=0.18,
            color="crimson",
            label="Prediction interval",
        )

    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.set_xlabel("Date (UTC)")
    axis.set_ylabel("Price (USD)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.show() if show else plt.close(figure)
    return path


def plot_pacf_diagnostic(
    returns: pd.Series,
    path: Path,
    *,
    max_lag: int = 60,
    title: str = "PACF of returns",
    show: bool = False,
) -> Path | None:
    plt = _pyplot(show)
    if plt is None:
        return None

    from ..diagnostics.suite import pacf_values, significance_band

    values = pacf_values(returns, nlags=max_lag)
    band = significance_band(len(returns.dropna()))

    figure, axis = plt.subplots(figsize=(9, 4.5))
    lags = np.arange(1, len(values))
    axis.bar(lags, values.to_numpy()[1:], width=0.7)
    axis.axhline(band, linestyle="--", color="red", alpha=0.7, label=f"95% band (+/-{band:.3f})")
    axis.axhline(-band, linestyle="--", color="red", alpha=0.7)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.set_xlabel("Lag")
    axis.set_ylabel("Partial autocorrelation")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.show() if show else plt.close(figure)
    return path


def plot_model_comparison(
    summary: pd.DataFrame,
    path: Path,
    *,
    metric: str = "mae",
    baseline: str | None = None,
    show: bool = False,
) -> Path | None:
    """Walk-forward error per model, with the baseline marked.

    Replaces the old learning curve as the headline diagnostic. A learning curve
    says how one model responds to training size; this says whether the model is
    worth running at all.
    """
    plt = _pyplot(show)
    if plt is None or summary.empty or metric not in summary.columns:
        return None

    ordered = summary.sort_values(metric)
    figure, axis = plt.subplots(figsize=(9, 0.55 * len(ordered) + 2.2))

    colours = ["#888888" if name == baseline else "#2c6fbb" for name in ordered.index]
    axis.barh(list(ordered.index), ordered[metric].to_numpy(), color=colours)

    if f"{metric}_std" in ordered.columns:
        axis.errorbar(
            ordered[metric].to_numpy(),
            range(len(ordered)),
            xerr=ordered[f"{metric}_std"].to_numpy(),
            fmt="none",
            ecolor="black",
            alpha=0.5,
            capsize=3,
        )

    if baseline in ordered.index:
        axis.axvline(
            float(ordered.loc[baseline, metric]),
            linestyle="--",
            color="black",
            alpha=0.6,
            label=f"{baseline} baseline",
        )
        axis.legend(fontsize=9)

    axis.set_xlabel(f"{metric.upper()} (mean across folds, lower is better)")
    axis.set_title("Walk-forward model comparison", fontsize=12, fontweight="bold")
    axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.show() if show else plt.close(figure)
    return path


def plot_interval_calibration(
    per_fold: pd.DataFrame,
    path: Path,
    *,
    show: bool = False,
) -> Path | None:
    """Observed interval coverage per model against the nominal level.

    Nothing in the original pipeline measured this, which is how a 95% band that
    did not widen with horizon went unnoticed.
    """
    plt = _pyplot(show)
    if plt is None or per_fold.empty or "interval_coverage" not in per_fold.columns:
        return None

    scored = per_fold[per_fold["error"].isna()]
    if scored.empty:
        return None

    grouped = scored.groupby("model")["interval_coverage"].agg(["mean", "std"]).sort_values("mean")
    nominal = float(scored["interval_level"].dropna().iloc[0]) if "interval_level" in scored else 0.95

    figure, axis = plt.subplots(figsize=(9, 0.55 * len(grouped) + 2.2))
    axis.barh(list(grouped.index), grouped["mean"].to_numpy(), color="#2c6fbb")
    axis.errorbar(
        grouped["mean"].to_numpy(),
        range(len(grouped)),
        xerr=grouped["std"].fillna(0.0).to_numpy(),
        fmt="none",
        ecolor="black",
        alpha=0.5,
        capsize=3,
    )
    axis.axvline(nominal, linestyle="--", color="red", label=f"nominal {nominal:.0%}")
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Observed coverage (closer to nominal is better)")
    axis.set_title("Prediction interval calibration", fontsize=12, fontweight="bold")
    axis.grid(alpha=0.3, axis="x")
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.show() if show else plt.close(figure)
    return path


__all__ = [
    "plot_forecast",
    "plot_interval_calibration",
    "plot_model_comparison",
    "plot_pacf_diagnostic",
]
