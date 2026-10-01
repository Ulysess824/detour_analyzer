"""Error metrics and ranking tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    r"""
    WAPE = sum|error| / sum(actual); accuracy = 1 - WAPE.

    bias_pct = sum(actual - forecast) / sum(actual) * 100, so a positive bias means the
    forecast is too low.
    """
    error = y - p
    wape = float(np.abs(error).sum() / y.sum())
    return {
        "n": len(y),
        "mae": float(np.abs(error).mean()),
        "wape": wape,
        "accuracy": 1.0 - wape,
        "bias_pct": float(error.sum() / y.sum() * 100.0),
    }


def rank_by_wape(rows: list[dict]) -> pd.DataFrame:
    r"""Table with a 1-based `rank` column (lowest WAPE first), sorted by it."""
    table = pd.DataFrame(rows).set_index("model")
    table.insert(0, "rank", table["wape"].rank(method="min").astype(int))
    return table.sort_values("rank").round(4)


def row_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    r"""Metrics of the row-level (daily) comparison: n, mae, rmse, wape and mean error."""
    error = y - p
    return {
        "n": len(y),
        "mae": np.mean(np.abs(error)),
        "rmse": np.sqrt(np.mean(error**2)),
        "wape": np.sum(np.abs(error)) / np.sum(np.abs(y)),
        "bias": np.mean(error),
    }


def rank_by_mae(table: pd.DataFrame) -> pd.DataFrame:
    r"""Add a 1-based `rank` column (lowest MAE first) and sort the table by it."""
    out = table.copy()
    out.insert(0, "rank", out["mae"].rank(method="min").astype(int))
    return out.sort_values("rank").round(4)
