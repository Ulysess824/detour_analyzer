"""Scoring helpers for the rolling-origin forecasts: test months, reconciliation, and accuracy at SKU, planta and total level."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.feature_utils import direct_forecasts
from src.utils.metrics_utils import rank_by_wape, score
from src.utils.panel_utils import Panel


def test_months_of(S: Panel, n_test: int) -> list[int]:
    r"""Indices of the last `n_test` months."""
    n_months = S.monthly.shape[1]
    return list(range(n_months - n_test, n_months))


def reconcile(res: pd.DataFrame, S: Panel, horizon: int, test_months: list[int], col: str) -> pd.Series:
    r"""
    Scale each planta's SKU forecasts so they add up to the direct planta forecast.

    The direct planta model (mean6 times last year's seasonal ratio) is the planta control
    total; it is allocated to the SKUs in proportion to their own forecasts.
    """
    out = res[col].copy()
    for t in test_months:
        control = direct_forecasts(S.planta_monthly, t - horizon, horizon, S.business_days)["mean6_x_seasonal"]
        in_month = (res["t"] == t).to_numpy()
        planta_sums = res.loc[in_month].groupby("planta_code")[col].sum()
        factor = pd.Series(control[planta_sums.index.to_numpy()], index=planta_sums.index)
        factor = factor / planta_sums.replace(0.0, np.nan)
        row_factor = res.loc[in_month, "planta_code"].map(factor).fillna(1.0).to_numpy()
        out.loc[in_month] = res.loc[in_month, col].to_numpy() * row_factor
    return out


def level_actual_and_bottom_up(S: Panel, res: pd.DataFrame, t: int, level: str, models: list[str]):
    r"""Actual values of month `t` at planta or total level, and the bottom-up sums of each model."""
    block = res[res["t"] == t]
    if level == "planta":
        actual = S.planta_monthly[:, t]
        sums = {n: block.groupby("planta_code")[f"p_{n}"].sum() for n in models}
        return actual, {n: sums[n].reindex(range(len(S.plantas))).fillna(0.0).to_numpy() for n in models}
    return np.array([S.total_monthly[t]]), {n: np.array([block[f"p_{n}"].sum()]) for n in models}


def score_models_by_level(S: Panel, res: pd.DataFrame, models: list[str], test_months: list[int]) -> dict[str, pd.DataFrame]:
    r"""Ranked accuracy tables of `models` at SKU level and, summing the SKU forecasts, at planta and total level."""
    tables = {"sku": rank_by_wape([{"model": n, **score(res["y"].to_numpy(), res[f"p_{n}"].to_numpy())} for n in models])}
    for level in ("planta", "total"):
        actual, bottom = [], {n: [] for n in models}
        for t in test_months:
            month_actual, month_bottom = level_actual_and_bottom_up(S, res, t, level, models)
            actual.append(month_actual)
            for n in models:
                bottom[n].append(month_bottom[n])
        y = np.concatenate(actual)
        tables[level] = rank_by_wape([{"model": n, **score(y, np.concatenate(bottom[n]))} for n in models])
    return tables
