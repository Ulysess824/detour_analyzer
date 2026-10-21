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


def _scores(rows: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    out = []
    for model, (y, p) in rows.items():
        s = score(y, p)
        out.append({"model": model, "n": len(y), "mae": float(np.mean(np.abs(y - p))), **s})
    return rank_by_wape(out)


def weekly_scores(weekly: pd.DataFrame, h: int) -> pd.DataFrame:
    r"""Accuracy table at horizon `h` over the target weeks that every model forecasts (same weeks for all, so they are comparable)."""
    table = weekly[weekly["h"] == h]
    common = set.intersection(*(set(g["target"]) for _, g in table.groupby("model")))
    table = table[table["target"].isin(common)]
    return _scores({m: (g["real"].to_numpy(), g["forecast"].to_numpy()) for m, g in table.groupby("model")})


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
    models = [c for c in monthly.columns if c not in ("month", "origin", "real")]
    return _scores({m: (monthly["real"].to_numpy(), monthly[m].to_numpy()) for m in models})


def decision_criteria(scores_h1: pd.DataFrame, intervals_h1: pd.DataFrame) -> dict:
    r"""
    The three criteria of docs/plan_croston.md, evaluated as they were fixed: (1) some Croston model beats naive and mean4 in weekly
    accuracy, (2) its 95% coverage is within COVERAGE_RANGE, (3) otherwise Croston stays as a reference.
    """
    acc = scores_h1["accuracy"]
    bar = max(acc["naive"], acc["mean4"])
    winners = [m for m in CROSTON_MODELS if m in acc.index and acc[m] > bar]
    covered = [m for m in winners if m in intervals_h1.index and COVERAGE_RANGE[0] <= intervals_h1.loc[m, "coverage"] <= COVERAGE_RANGE[1]]
    return {"bar_accuracy": float(bar), "beat_naive_and_mean4": winners, "with_coverage_in_range": covered,
            "criterion_1": bool(winners), "criterion_2": bool(covered), "meets_preset_criteria": bool(covered)}
