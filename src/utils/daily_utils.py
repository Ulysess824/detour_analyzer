"""Turn monthly rows into daily rows (one per calendar day of the target month)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.panel_utils import Panel

DAY_FEATURES = ["dow", "dom", "day_idx", "days_left"]


def expand_to_days(S: Panel, rows: pd.DataFrame, features: list[str], frac: float | None, seed: int):
    r"""
    Repeat each monthly row once per calendar day of its target month.

    Returns (X, y, row_position): the features plus the day features, the zero-filled daily
    consumption, and for every daily row the position of its monthly row in `rows`.
    With `frac` set, only that random share of the daily rows is kept (for faster training).
    """
    target = rows["t"].to_numpy()
    days_in_month = S.calendar_days[target]
    row_pos = np.repeat(np.arange(len(rows)), days_in_month)
    month_start = np.cumsum(days_in_month) - days_in_month
    day_idx = np.arange(row_pos.size) - np.repeat(month_start, days_in_month)
    if frac is not None:
        keep = np.random.default_rng(seed).random(row_pos.size) < frac
        row_pos, day_idx = row_pos[keep], day_idx[keep]
    date_pos = S.month_first_date[target[row_pos]] + day_idx

    base = rows[features].to_numpy(dtype=np.float32)[row_pos]
    day = np.column_stack(
        [
            S.dates.dayofweek.to_numpy()[date_pos],
            S.dates.day.to_numpy()[date_pos],
            day_idx,
            days_in_month[row_pos] - 1 - day_idx,
        ]
    ).astype(np.float32)
    X = pd.DataFrame(np.hstack([base, day]), columns=features + DAY_FEATURES)
    X["planta_code"] = X["planta_code"].astype(int)
    y = S.daily[rows["series"].to_numpy()[row_pos], date_pos]
    return X, y, row_pos
