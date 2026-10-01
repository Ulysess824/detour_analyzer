"""Test-period actual vs forecast of each tree model, base vs Optuna-tuned (seaborn small multiples).

Refits every model over the test months (rolling origin, horizon h) with the baseline parameters and
with the tuned ones saved by tune_trees.py, and plots the monthly total against the actual.

    python scripts/plot_pred_vs_real.py data/consumos_long.csv results/tuning_trees.json -o pred_vs_real.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_monthly as fm  # noqa: E402
import tune_trees as tt  # noqa: E402

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest"}
PALETTE = {"Real": "#222222", "Base": "#898781", "Optuna": "#2a78d6"}


def month_name(label) -> str:
    ts = pd.Period(str(label)[:7])
    return f"{MESES[ts.month - 1]} {ts.year}"


def collect(S, results: dict, h: int, n_test: int, seed: int, series: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = S.Y.shape[1]
    months = list(range(n - n_test, n))
    frame = fm.make_frame(S, h)
    actual = (lambda t: float(S.T[t])) if series is None else (lambda t: float(S.Y[series, t]))
    pick = (lambda res, t: res.loc[res["t"] == t, "p"].sum()) if series is None else (
        lambda res, t: res.loc[(res["t"] == t) & (res["series"] == series), "p"].sum())
    rows = [{"mes": month_name(S.labels[t]), "modelo": "Real", "variante": "Real", "consumo": actual(t)} for t in months]
    real = np.array([r["consumo"] for r in rows])
    summary = []
    for kind in NAMES:
        if kind not in results:
            continue
        for variante, params in [("Base", tt.BASE[kind]), ("Optuna", results[kind]["best_params"])]:
            res = tt.test_run(kind, params, frame, S, h, months, [seed])[0]
            pred = np.array([pick(res, t) for t in months])
            rows += [{"mes": month_name(S.labels[t]), "modelo": NAMES[kind], "variante": variante, "consumo": float(p)} for t, p in zip(months, pred)]
            summary.append({"modelo": NAMES[kind], "variante": variante, "accuracy": 1 - np.abs(real - pred).sum() / real.sum()})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def plot(long: pd.DataFrame, summary: pd.DataFrame, h: int, out: Path, scope: str = "total mensual") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    models = [m for m in NAMES.values() if m in set(long["modelo"])]
    real = long[long["variante"] == "Real"]
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.1 * len(models)), sharex=True, constrained_layout=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        acc = summary[summary["modelo"] == m].set_index("variante")["accuracy"]
        sub = pd.concat([real, long[long["modelo"] == m]])
        sub = sub.assign(variante=sub["variante"].map(lambda v: v if v == "Real" else f"{v} ({acc[v]:.1%})"))
        pal = {f"{k} ({acc[k]:.1%})" if k != "Real" else k: c for k, c in PALETTE.items()}
        sns.lineplot(data=sub, x="mes", y="consumo", hue="variante", style="variante", markers=True, dashes={"Real": "", **{k: ("" if k.startswith("Optuna") else (4, 2)) for k in pal}},
                     palette=pal, linewidth=2, markersize=7, ax=ax)
        ax.set_title(f"{m} - {scope}, h={h} (test)", loc="left", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("consumo")
        ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9, title=None)
    plt.setp(np.atleast_1d(axes)[-1].get_xticklabels(), rotation=45)
    fig.savefig(out, dpi=110)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", type=Path)
    p.add_argument("results", type=Path, help="JSON written by tune_trees.py")
    p.add_argument("-o", "--out", type=Path, default=Path("pred_vs_real.png"))
    p.add_argument("--h", type=int, default=1, choices=[1, 3])
    p.add_argument("--n-test", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--planta", default=None, help="plot one series instead of the total (needs --sku)")
    p.add_argument("--sku", default=None)
    p.add_argument("--csv", type=Path, default=None, help="also write the plotted values")
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    import json

    S = fm.build_panel(pd.read_csv(a.input, parse_dates=["fecha"]))
    df = pd.read_csv(a.input, parse_dates=["fecha"])
    series, scope = None, "total mensual"
    if a.planta or a.sku:
        keys = df.groupby(fm.GROUP_COLS).size().index  # same order as the panel rows
        series = keys.get_loc((a.planta, a.sku))
        scope = f"{a.planta} / {a.sku}"
    long, summary = collect(S, json.loads(a.results.read_text()), a.h, a.n_test, a.seed, series)
    print(long.pivot_table(index="mes", columns=["modelo", "variante"], values="consumo").round(0).to_string())
    print(summary.round(4).to_string(index=False))
    if a.csv:
        long.to_csv(a.csv, index=False)
    plot(long, summary, a.h, a.out, scope)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
