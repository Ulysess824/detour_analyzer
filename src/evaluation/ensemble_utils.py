"""
Forecast combinations (ensembles) of several models.

Methods, as in the usual forecast-combination literature:
    mean      simple average, equal weight to every member
    median    median of the members (robust to one bad member)
    trimmed   trimmed mean: drops the highest and lowest 20% of the members, then averages
    weighted  weighted average with weights proportional to 1 / past MAE of each member
              (members that did better before get more weight)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

TRIM = 0.2
METHODS = ["mean", "median", "trimmed", "weighted"]


def combine(P: np.ndarray, method: str, weights: np.ndarray | None = None) -> np.ndarray:
    r"""Combine the columns of P (rows x members) into one forecast per row."""
    if method == "mean":
        return P.mean(axis=1)
    if method == "median":
        return np.median(P, axis=1)
    if method == "trimmed":
        return trim_mean(P, TRIM, axis=1)
    if method == "weighted":
        return P @ weights
    raise ValueError(method)


def inverse_error_weights(past_abs_error: np.ndarray) -> np.ndarray:
    r"""Weights proportional to 1 / MAE of each member, from past absolute errors (rows x members)."""
    inverse = 1.0 / np.maximum(past_abs_error.mean(axis=0), 1e-9)
    return inverse / inverse.sum()


def add_ensembles(res: pd.DataFrame, groups: dict[str, list[str]], horizon: int) -> pd.DataFrame:
    r"""
    Add one column p_<group>_<method> per ensemble.

    `res` has one row per (series, target month t) with the target `y` and one column
    p_<member> per model. The weights of month t come only from months <= t - horizon, which
    are already observed at the origin of month t; before any history the weights are equal.
    """
    res = res.copy()
    months = sorted(res["t"].unique())
    for group, members in groups.items():
        cols = [f"p_{m}" for m in members]
        for method in METHODS:
            res[f"p_{group}_{method}"] = np.nan
        for t in months:
            now = (res["t"] == t).to_numpy()
            past = res[res["t"] <= t - horizon]
            if len(past):
                weights = inverse_error_weights(np.abs(past[cols].to_numpy() - past[["y"]].to_numpy()))
            else:
                weights = np.full(len(cols), 1.0 / len(cols))
            P = res.loc[now, cols].to_numpy()
            for method in METHODS:
                res.loc[now, f"p_{group}_{method}"] = combine(P, method, weights)
    return res
