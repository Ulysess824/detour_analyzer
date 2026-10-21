"""
Croston experiment over several series: run the daily protocol on each, pool the results and compare models across series.

Pooled accuracy = 1 - sum|y - p| / sum y over every series and period (large series dominate); the per-series accuracy, the number of
series where a model beats another and a sign test count every series once. The bootstrap resamples whole series, not months.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from src.experiments import croston_backtest as cb
from src.experiments import croston_scores as cs
from src.experiments import daily_utils as du
from src.experiments.croston_intervals import DRAWS
from src.experiments.weekly_utils import weekly_series

WINDOWS = (6, 24)  # moving averages in working days: one week and four weeks
BATCH_REFERENCES = ("naive", "mean6", "mean24", "ses")
MIN_SERIES_SHARE = 0.5  # criterion 2: the model must beat ml_mean in at least this share of the series


def run_series(df: pd.DataFrame, planta: str, sku: str, ml_mean: pd.Series, test_start: str, draws: int = DRAWS, seed: int = 0) -> dict:
    r"""
    The daily protocol for one series: rolling daily forecasts (h=1), weeks and months aggregated from them. Returns the tables
    `daily`, `weeks` and `months`, each with a `series` column ("planta|sku").
    """
    key = f"{planta}|{sku}"
    series = du.daily_series(df, planta, sku)
    real_monthly = df[(df["planta"] == planta) & (df["sku"] == sku)].groupby(df["fecha"].dt.to_period("M"))["consumo"].sum()
    out = {
        "daily": cb.rolling_origin(series, test_start, (1,), draws, seed, windows=WINDOWS),
        "weeks": du.week_comparison(series, weekly_series(df, planta, sku), test_start, draws, seed, WINDOWS),
        "months": du.month_comparison(series, real_monthly.reindex(ml_mean.index, fill_value=0.0), ml_mean, draws, seed, WINDOWS),
    }
    for table in out.values():
        table.insert(0, "series", key)
    return out


def per_series_accuracy(table: pd.DataFrame) -> pd.DataFrame:
    r"""Accuracy (1 - WAPE) of every model in every series: rows = series, columns = models. Rows with a missing forecast are left out."""
    models = [c for c in table.columns if c not in ("series", "month", "week", "origin", "real")]
    table = table.dropna(subset=models)
    grouped = table.groupby("series")
    real = grouped["real"].sum()
    return pd.DataFrame({m: 1.0 - grouped.apply(lambda g, m=m: (g["real"] - g[m]).abs().sum()) / real for m in models}).round(4)


def win_count(per_series: pd.DataFrame, model: str, versus: str) -> int:
    r"""Series where `model` has a strictly higher accuracy than `versus`."""
    return int((per_series[model] > per_series[versus]).sum())


def sign_test_p(wins: int, n: int) -> float:
    r"""Two-sided exact sign test p-value of `wins` victories in `n` series (ties are dropped by the caller)."""
    k = max(wins, n - wins)
    return float(min(1.0, 2.0 * sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n))


def bootstrap_accuracy_diff(table: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    r"""
    Pooled accuracy of model `a` minus model `b`, with a 90% interval from resampling whole series with replacement. Positive means `a`
    is better.
    """
    table = table.dropna(subset=[a, b])
    g = table.groupby("series")
    err_a = g.apply(lambda x: (x["real"] - x[a]).abs().sum()).to_numpy()
    err_b = g.apply(lambda x: (x["real"] - x[b]).abs().sum()).to_numpy()
    real = g["real"].sum().to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(real), size=(n_boot, len(real)))
    diff = (err_b[picks].sum(axis=1) - err_a[picks].sum(axis=1)) / real[picks].sum(axis=1)
    point = (err_b.sum() - err_a.sum()) / real.sum()
    lo, hi = np.quantile(diff, [0.05, 0.95])
    return {"diff": float(point), "low90": float(lo), "high90": float(hi), "series": int(len(real))}


def batch_decision(week_scores: pd.DataFrame, month_scores: pd.DataFrame, month_per_series: pd.DataFrame, intervals: pd.DataFrame,
                   references: tuple[str, ...] = BATCH_REFERENCES) -> dict:
    r"""
    The criteria fixed before the run: the SAME Croston model (estimated alpha) must (1) beat every reference in pooled weekly
    accuracy, (2) match or beat ml_mean in pooled monthly accuracy AND beat it in at least MIN_SERIES_SHARE of the series, and
    (3) have pooled 95% daily coverage within COVERAGE_RANGE.
    """
    week_acc, month_acc = week_scores["accuracy"], month_scores["accuracy"]
    bar = max(week_acc[r] for r in references)
    n = len(month_per_series)
    c1 = [m for m in cs.CROSTON_MODELS if m in week_acc.index and week_acc[m] > bar]
    c2 = [m for m in cs.CROSTON_MODELS if m in month_acc.index and month_acc[m] >= month_acc["ml_mean"]
          and win_count(month_per_series, m, "ml_mean") >= MIN_SERIES_SHARE * n]
    c3 = [m for m in cs.CROSTON_MODELS if m in intervals.index and cs.COVERAGE_RANGE[0] <= intervals.loc[m, "coverage"] <= cs.COVERAGE_RANGE[1]]
    both = [m for m in c1 if m in c2 and m in c3]
    return {"series": n, "week_bar_accuracy": float(bar), "month_ml_mean_accuracy": float(month_acc["ml_mean"]), "criterion_1": c1,
            "criterion_2": c2, "criterion_3": c3, "meets_all": both, "adopted": bool(both)}
