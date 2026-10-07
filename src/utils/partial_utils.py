"""Features of the month in progress at the forecast origin (consumption up to a cutoff day, e.g. mid-month)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.panel_utils import Panel

PARTIAL_FEATURES = ["mtd", "mtd_rate", "mtd_occ", "mtd_proj"]


def month_to_date(S: Panel, day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Consumption and days with consumption of the first `day` calendar days of every month.

    Returns (mtd, active, elapsed): mtd and active are (n_series, n_months); elapsed (n_months,) is the
    number of non-Sunday days among those first days. Only days up to the cutoff are read.
    """
    n_series, n_months = S.monthly.shape
    mtd, active, elapsed = np.zeros((n_series, n_months)), np.zeros((n_series, n_months)), np.zeros(n_months)
    for m in range(n_months):
        lo = int(S.month_first_date[m])
        hi = lo + min(day, int(S.calendar_days[m]))
        block = S.daily[:, lo:hi]
        mtd[:, m] = block.sum(axis=1, dtype=float)
        active[:, m] = (block > 0).sum(axis=1)
        elapsed[m] = int((S.dates[lo:hi].dayofweek != 6).sum())
    return mtd, active, elapsed


def add_partial_features(frame: pd.DataFrame, S: Panel, day: int) -> pd.DataFrame:
    r"""
    Add the month-to-date features of the month after each row's origin (the month in progress at mid-month).

        mtd       consumption of the first `day` days
        mtd_rate  mtd / non-Sunday days elapsed
        mtd_occ   days with consumption / non-Sunday days elapsed
        mtd_proj  mtd_rate * non-Sunday days of the whole month (a projection of the month total)
    """
    mtd, active, elapsed = month_to_date(S, day)
    month = frame["o"].to_numpy() + 1
    series = frame["series"].to_numpy()
    days = elapsed[month]
    out = frame.copy()
    out["mtd"] = mtd[series, month]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mtd_rate"] = np.where(days > 0, out["mtd"] / days, np.nan)
        out["mtd_occ"] = np.where(days > 0, active[series, month] / days, np.nan)
    out["mtd_proj"] = out["mtd_rate"] * S.business_days[month]
    return out
