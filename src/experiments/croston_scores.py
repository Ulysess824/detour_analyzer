"""Scores of the weekly Croston backtest: accuracy tables, interval coverage, monthly comparison and the pre-set decision criteria."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics_utils import rank_by_wape, score
from src.experiments.croston_backtest import MODELS_WITH_INTERVAL

# the Croston models with the estimated alpha: the fixed-alpha variants are a sensitivity check, not candidates
CROSTON_MODELS = ("croston", "sb0", "sb1", "modified", "modified_log", "croston_model", "log_croston_model")
REFERENCES = ("naive", "mean4", "mean8", "ses")
COVERAGE_RANGE = (0.85, 1.0)  # fixed before looking at the results


def _scores(rows: dict[str, tuple[np.ndarray, np.ndarray]], rmse: bool = False) -> pd.DataFrame:
    out = []
    for model, (y, p) in rows.items():
        s = score(y, p)
        extra = {"rmse": float(np.sqrt(np.mean((y - p) ** 2)))} if rmse else {}
        out.append({"model": model, "n": len(y), "mae": float(np.mean(np.abs(y - p))), **extra, **s})
    return rank_by_wape(out)


def weekly_scores(weekly: pd.DataFrame, h: int, rmse: bool = False) -> pd.DataFrame:
    r"""Accuracy table at horizon `h` over the target periods that every model forecasts (same periods for all, so they are comparable)."""
    table = weekly[weekly["h"] == h]
    key = list(zip(table["series"], table["target"])) if "series" in table else list(table["target"])  # several series: (series, target)
    table = table.assign(_key=key)
    common = set.intersection(*(set(g["_key"]) for _, g in table.groupby("model")))
    table = table[table["_key"].isin(common)]
    return _scores({m: (g["real"].to_numpy(), g["forecast"].to_numpy()) for m, g in table.groupby("model")}, rmse)


def wide_table(weekly: pd.DataFrame, h: int) -> pd.DataFrame:
    r"""One row per target week: real, the forecast of every model and the limits of the models with an interval (`<model>_lower/_upper`)."""
    table = weekly[weekly["h"] == h]
    wide = table.pivot(index="target", columns="model", values="forecast")
    wide.insert(0, "real", table.drop_duplicates("target").set_index("target")["real"])
    for model in MODELS_WITH_INTERVAL:
        g = table[table["model"] == model].set_index("target")
        if not g.empty:
            wide[f"{model}_lower"], wide[f"{model}_upper"] = g["lower"], g["upper"]
    return wide.round(3)


def interval_scores(weekly: pd.DataFrame, h: int) -> pd.DataFrame:
    r"""Empirical coverage of the 95% interval and mean width, per model with an interval."""
    rows = []
    for model, g in weekly[(weekly["h"] == h) & weekly["model"].isin(MODELS_WITH_INTERVAL)].groupby("model"):
        inside = (g["real"] >= g["lower"]) & (g["real"] <= g["upper"])
        rows.append({"model": model, "n": len(g), "coverage": float(inside.mean()), "mean_width": float((g["upper"] - g["lower"]).mean()),
                     "weeks_with_zero_lower": int((g["lower"] <= 0).sum())})
    return pd.DataFrame(rows).set_index("model").round(4) if rows else pd.DataFrame()


def monthly_scores(monthly: pd.DataFrame) -> pd.DataFrame:
    r"""Accuracy over the months of `monthly_comparison`: every column except month, origin and real is a forecast."""
    models = [c for c in monthly.columns if c not in ("series", "month", "week", "origin", "real")]
    monthly = monthly.dropna(subset=models)  # rows where a model could not be fitted are left out for every model
    return _scores({m: (monthly["real"].to_numpy(), monthly[m].to_numpy()) for m in models})


def decision_criteria(scores_h1: pd.DataFrame, intervals_h1: pd.DataFrame, references: tuple[str, ...] = ("naive", "mean4")) -> dict:
    r"""
    The three criteria of docs/plan_croston.md, evaluated as they were fixed: (1) some Croston model beats naive and mean4 in weekly
    accuracy, (2) its 95% coverage is within COVERAGE_RANGE, (3) otherwise Croston stays as a reference.
    """
    acc = scores_h1["accuracy"]
    bar = max(acc[r] for r in references)
    winners = [m for m in CROSTON_MODELS if m in acc.index and acc[m] > bar]
    covered = [m for m in winners if m in intervals_h1.index and COVERAGE_RANGE[0] <= intervals_h1.loc[m, "coverage"] <= COVERAGE_RANGE[1]]
    return {"bar_accuracy": float(bar), "beat_naive_and_mean4": winners, "with_coverage_in_range": covered,
            "criterion_1": bool(winners), "criterion_2": bool(covered), "meets_preset_criteria": bool(covered)}


def daily_decision(week_scores: pd.DataFrame, month_scores: pd.DataFrame, intervals_h1: pd.DataFrame,
                   references: tuple[str, ...] = ("naive", "mean6", "mean24", "ses")) -> dict:
    r"""
    The criteria of the daily run (docs/plan_croston.md, fixed before the results): the SAME Croston model must (1) beat every
    reference in weekly accuracy, (2) match or beat ml_mean in monthly accuracy and (3) have 95% daily coverage within COVERAGE_RANGE.
    """
    week_acc, month_acc = week_scores["accuracy"], month_scores["accuracy"]
    bar = max(week_acc[r] for r in references)
    c1 = [m for m in CROSTON_MODELS if m in week_acc.index and week_acc[m] > bar]
    c2 = [m for m in CROSTON_MODELS if m in month_acc.index and month_acc[m] >= month_acc["ml_mean"]]
    c3 = [m for m in CROSTON_MODELS if m in intervals_h1.index and COVERAGE_RANGE[0] <= intervals_h1.loc[m, "coverage"] <= COVERAGE_RANGE[1]]
    both = [m for m in c1 if m in c2 and m in c3]
    return {"week_bar_accuracy": float(bar), "month_ml_mean_accuracy": float(month_acc["ml_mean"]), "criterion_1": c1, "criterion_2": c2,
            "criterion_3": c3, "meets_all": both, "adopted": bool(both)}
