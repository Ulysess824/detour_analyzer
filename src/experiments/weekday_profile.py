"""
Weekday profile of a daily series and its use on flat forecasts.

The factor of weekday k is the mean of the series on that weekday over the mean of all days, shrunk toward 1 by n_k / (n_k + shrink)
(n_k = days of that weekday in the history) and rescaled to mean 1 over the working week. Applied to a flat forecast it moves
consumption between weekdays and leaves the weekly total unchanged. Only the history up to the origin is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHRINK = 20.0  # fixed before looking at the results


def weekday_profile(values: np.ndarray, dows: np.ndarray, weekdays: int = 5, shrink: float = SHRINK) -> np.ndarray:
    r"""
    Factor per weekday 0..weekdays-1 (mean 1). `values` can be consumption or a 0/1 consumption indicator (a probability profile);
    a history with no consumption gives all ones.
    """
    values, dows = np.asarray(values, dtype=float), np.asarray(dows)
    mean = float(values.mean()) if len(values) else 0.0
    if mean <= 0:
        return np.ones(weekdays)
    w = np.ones(weekdays)
    for k in range(weekdays):
        mask = dows == k
        n = int(mask.sum())
        if n:
            w[k] = 1.0 + (float(values[mask].mean()) / mean - 1.0) * n / (n + shrink)
    return w * weekdays / w.sum()


def add_profile_rows(daily: pd.DataFrame, series: pd.Series, weekdays: int = 5, shrink: float = SHRINK) -> pd.DataFrame:
    r"""
    Daily rolling table (`rolling_origin` output) plus, for every model, a copy named `<model>_dow` whose forecast and limits are
    multiplied by the factor of the target's weekday, computed from the series up to each row's origin.
    """
    values, dows, index = series.to_numpy(dtype=float), series.index.dayofweek, series.index
    profiles = {o: weekday_profile(values[: index.get_loc(o) + 1], dows[: index.get_loc(o) + 1], weekdays, shrink) for o in daily["origin"].unique()}
    w = np.array([profiles[o][t.dayofweek] for o, t in zip(daily["origin"], daily["target"])])
    scaled = daily.copy()
    scaled["model"] = scaled["model"] + "_dow"
    for col in ("forecast", "lower", "upper"):
        scaled[col] = scaled[col] * w
    return pd.concat([daily, scaled], ignore_index=True)
