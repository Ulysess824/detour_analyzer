"""Compare the forecast of the planners with the forecast of the models, for one month."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.metrics_utils import score
from src.utils.panel_utils import GROUP_COLS, Panel


def planner_key(sku: str) -> str:
    r"""
    Key shared by the planner SKU and the dataset SKU.

    Planner: "TSL/01/1/80gsm/2450mm". Dataset: "TSL/01/80gsm/2450mm/1200-1450".
    The planner name has an extra "/1" and no core width, so both are reduced to type/01/gsm/width.
    """
    parts = sku.lower().split("/")
    return "/".join(p for i, p in enumerate(parts) if not (i == 2 and p == "1"))


def dataset_key(sku: str) -> str:
    r"""Drop the core width (last part) of the dataset SKU."""
    return sku.lower().rsplit("/", 1)[0]


def series_table(S: Panel, df: pd.DataFrame) -> pd.DataFrame:
    r"""One row per series index with its planta, sku and the key used to match the planner SKU."""
    pairs = df.groupby(GROUP_COLS).size().index.to_frame(index=False)
    pairs["series"] = np.arange(len(pairs))
    pairs["key"] = pairs["sku"].map(dataset_key)
    return pairs


def compare_month(planner: pd.DataFrame, res: pd.DataFrame, pairs: pd.DataFrame, month: int, models: list[str]):
    r"""
    One row per planner SKU: real consumption, planner forecast and the forecast of each model.

    Series of the same planner SKU (same type, gsm and width, different core) are added up.
    Returns (table, list of planner SKUs with no match in the data).
    """
    planner = planner.assign(key=planner["sku_planner"].map(planner_key))
    block = res[res["t"] == month]
    rows, missing = [], []
    for _, row in planner.iterrows():
        series = pairs[(pairs["planta"] == row["planta"]) & (pairs["key"] == row["key"])]["series"].tolist()
        if not series:
            missing.append(row["sku_planner"])
            continue
        mine = block[block["series"].isin(series)]
        record = {"sku": row["sku_planner"], "series": len(series), "real": mine["y"].sum(), "planner": row["forecast_to"]}
        record.update({m: mine[f"p_{m}"].sum() for m in models})
        rows.append(record)
    return pd.DataFrame(rows), missing


def accuracy_table(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    r"""Accuracy, WAPE and bias of the planner and of each model over the planner SKUs, ranked by accuracy."""
    rows = [{"model": c, **score(table["real"].to_numpy(), table[c].to_numpy())} for c in columns]
    out = pd.DataFrame(rows).sort_values("wape", ignore_index=True)
    out.insert(0, "rank", out["wape"].rank(method="min").astype(int))
    return out[["rank", "model", "n", "mae", "wape", "accuracy", "bias_pct"]]
