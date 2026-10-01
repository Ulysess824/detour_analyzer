"""Seaborn plot of the test-period actuals against the base and tuned forecasts of each tree model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.panel_utils import Panel
from src.utils.feature_utils import make_frame
from src.utils.tree_utils import TREE_BASE, fit_predict_trees
from src.utils.tuning_utils import rolling_test

MONTH_NAMES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MODEL_NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest"}
COLORS = {"Real": "#222222", "Base": "#898781", "Optuna": "#2a78d6"}


def month_name(label) -> str:
    r"""'2025-07' -> 'jul 2025'."""
    period = pd.Period(str(label)[:7])
    return f"{MONTH_NAMES[period.month - 1]} {period.year}"


def collect_forecasts(S: Panel, results: dict, horizon: int, n_test: int, seed: int, series: int | None = None):
    r"""
    Refit every model for each test month with the base and the tuned parameters.

    With `series` set, the values are those of that single series; otherwise the monthly total.
    Returns (long table of values, accuracy summary).
    """
    n_months = S.monthly.shape[1]
    months = list(range(n_months - n_test, n_months))
    frame = make_frame(S, horizon)

    def actual(t):
        return float(S.total_monthly[t]) if series is None else float(S.monthly[series, t])

    def forecast(res, t):
        block = res[res["t"] == t] if series is None else res[(res["t"] == t) & (res["series"] == series)]
        return block["p"].sum()

    real = np.array([actual(t) for t in months])
    rows = [{"mes": month_name(S.labels[t]), "modelo": "Real", "variante": "Real", "consumo": actual(t)} for t in months]
    summary = []
    for kind in MODEL_NAMES:
        if kind not in results:
            continue
        for variant, params in [("Base", TREE_BASE[kind]), ("Optuna", results[kind]["best_params"])]:
            predict = lambda train, rows_, seed_, k=kind, p=params: fit_predict_trees(k, p, train, rows_, seed_)  # noqa: E731
            res = rolling_test(predict, S, frame, horizon, months, [seed])[0]
            pred = np.array([forecast(res, t) for t in months])
            rows += [
                {"mes": month_name(S.labels[t]), "modelo": MODEL_NAMES[kind], "variante": variant, "consumo": float(v)}
                for t, v in zip(months, pred)
            ]
            summary.append({"modelo": MODEL_NAMES[kind], "variante": variant, "accuracy": 1 - np.abs(real - pred).sum() / real.sum()})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def plot_forecasts(long: pd.DataFrame, summary: pd.DataFrame, horizon: int, out: Path, scope: str = "total mensual") -> None:
    r"""One panel per model: actual (black), base (grey dashed) and Optuna (blue); accuracy in the legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    models = [m for m in MODEL_NAMES.values() if m in set(long["modelo"])]
    real = long[long["variante"] == "Real"]
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.1 * len(models)), sharex=True, constrained_layout=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        accuracy = summary[summary["modelo"] == model].set_index("variante")["accuracy"]
        panel = pd.concat([real, long[long["modelo"] == model]])
        label = {"Real": "Real", **{v: f"{v} ({accuracy[v]:.1%})" for v in ("Base", "Optuna")}}
        panel = panel.assign(variante=panel["variante"].map(label))
        palette = {label[v]: c for v, c in COLORS.items()}
        dashes = {label["Real"]: "", label["Base"]: (4, 2), label["Optuna"]: ""}
        sns.lineplot(
            data=panel, x="mes", y="consumo", hue="variante", style="variante", markers=True, dashes=dashes,
            palette=palette, linewidth=2, markersize=7, ax=ax,
        )  # fmt: skip
        ax.set_title(f"{model} - {scope}, h={horizon} (test)", loc="left", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("consumo")
        ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9, title=None)
    plt.setp(np.atleast_1d(axes)[-1].get_xticklabels(), rotation=45)
    fig.savefig(out, dpi=110)
