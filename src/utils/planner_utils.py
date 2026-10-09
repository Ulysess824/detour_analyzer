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
    r"""Keep the first four parts of the dataset SKU (type, sub-grade, gsm, width); the core width and suffixes are dropped."""
    return "/".join(sku.lower().split("/")[:4])


def series_table(S: Panel, df: pd.DataFrame) -> pd.DataFrame:
    r"""One row per series index with its planta, sku, the key used to match the planner SKU and its first month."""
    pairs = df.groupby(GROUP_COLS).size().index.to_frame(index=False)
    pairs["series"] = np.arange(len(pairs))
    pairs["key"] = pairs["sku"].map(dataset_key)
    pairs["first_month"] = S.first_month[pairs["series"].to_numpy()]
    return pairs


def alt_key(key: str) -> str:
    r"""The key without its sub-grade (second part): type/gsm/width. Used only to explain unmatched planner SKUs."""
    parts = key.split("/")
    return "/".join(parts[:1] + parts[2:])


def first_consumption_dates(df: pd.DataFrame) -> pd.Series:
    r"""First day with consumption of every (planta, key), over all the series that share the key (index: planta, key)."""
    return df.assign(key=df["sku"].map(dataset_key)).groupby(["planta", "key"])["fecha"].min()


# Reasons a planner SKU-month is left out of a comparison (shown in the exclusion files).
NO_SERIES = "sin serie en los datos"
SUBGRADE = "subgrado distinto"
LOW_HISTORY = "poca historia"
NO_MODEL = "sin predicción del modelo"


def compare_month_audit(
    planner: pd.DataFrame,
    res: pd.DataFrame,
    pairs: pd.DataFrame,
    month: int,
    models: list[str],
    min_history: int = 0,
    strategies: list[str] | None = None,
    horizon: int = 1,
):
    r"""
    One row per planner SKU with real consumption, planner forecast and the forecast of each model, plus the SKUs left out.

    Series of the same planner SKU (same planta, type, gsm and width, different core) are added up. `strategies` keeps only
    those planner strategies (None keeps all). A planner SKU is left out, with its reason, when
      - NO_SERIES: no series of the planta has that key (the SKU never consumed);
      - SUBGRADE: no series has that key but one has the same type, gsm and width with another sub-grade (not matched);
      - LOW_HISTORY: its series have less than `min_history` months before the origin (month - horizon), the origin included;
      - NO_MODEL: otherwise the models have no forecast for it (it was not alive at the origin).
    Returns (table, excluded); `excluded` has planta, mes, sku, strategy, planner, reason and detail.
    """
    planner = planner.assign(key=planner["sku_planner"].map(planner_key))
    if strategies is not None:
        planner = planner[planner.get("strategy", pd.Series("", index=planner.index)).isin(strategies)]
    pairs = pairs.assign(alt=pairs["key"].map(alt_key))
    block = res[res["t"] == month]
    origin = month - horizon
    rows, excluded = [], []
    for _, row in planner.iterrows():
        mine_pairs = pairs[pairs["planta"] == row["planta"]]
        found = mine_pairs[mine_pairs["key"] == row["key"]]
        info = {"planta": row["planta"], "mes": row.get("mes", ""), "sku": row["sku_planner"], "strategy": row.get("strategy", ""), "planner": row["forecast_to"]}
        if found.empty:
            near = mine_pairs[mine_pairs["alt"] == alt_key(row["key"])]
            if len(near):
                excluded.append({**info, "reason": SUBGRADE, "detail": "en los datos: " + ", ".join(sorted({"/".join(sku.split("/")[:4]) for sku in near["sku"]}))})
            else:
                excluded.append({**info, "reason": NO_SERIES, "detail": ""})
            continue
        history = origin - int(found["first_month"].min()) + 1
        if history < min_history:
            excluded.append({**info, "reason": LOW_HISTORY, "detail": f"{max(history, 0)} meses antes del origen"})
            continue
        mine = block[block["series"].isin(found["series"])]
        if mine.empty:
            excluded.append({**info, "reason": NO_MODEL, "detail": ""})
            continue
        record = {"planta": row["planta"], "sku": row["sku_planner"], "strategy": row.get("strategy", ""), "series": len(found), "history": history, "real": mine["y"].sum(), "planner": row["forecast_to"]}
        record.update({m: mine[f"p_{m}"].sum() for m in models})
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(excluded, columns=["planta", "mes", "sku", "strategy", "planner", "reason", "detail"])


def compare_month(planner: pd.DataFrame, res: pd.DataFrame, pairs: pd.DataFrame, month: int, models: list[str]):
    r"""
    `compare_month_audit` without filters. Returns (table, planner SKUs with no match in the data); the SKUs the models
    cannot forecast (not alive at the origin) are left out of the table instead of counting as a real of 0.
    """
    table, excluded = compare_month_audit(planner, res, pairs, month, models)
    return table, excluded.loc[excluded["reason"].isin([NO_SERIES, SUBGRADE]), "sku"].tolist()


def attach_source(table: pd.DataFrame, source: pd.DataFrame, column: str) -> pd.DataFrame:
    r"""Add the forecast of `source` (planta, mes, sku_planner, forecast_to) as `column`, matched by planta, month and SKU; NaN when absent."""
    forecast = source[["planta", "mes", "sku_planner", "forecast_to"]].rename(columns={"sku_planner": "sku", "forecast_to": column})
    return table.merge(forecast, on=["planta", "mes", "sku"], how="left")


MISSING_INTERNAL = "faltante en el modelo interno"


def add_internal(table: pd.DataFrame, internal: pd.DataFrame, month: str):
    r"""
    Add the internal model's forecast of `month` to `table` (columns `internal` and `internal_status`, "ok" or "faltante").

    A SKU is "faltante" when its forecast is blank in the internal file (0 in every month there) or the SKU is not in the file.
    Returns (table, one row per "faltante" SKU with planta, mes, sku, strategy, planner, reason and detail).
    """
    table = attach_source(table, internal[internal["mes"] == month], "internal")
    table["internal_status"] = np.where(table["internal"].notna(), "ok", "faltante")
    known = set(zip(internal["planta"], internal["sku_planner"]))
    gone = table[table["internal"].isna()]
    detail = ["0 en todos los meses" if (p, s) in known else "no está en el archivo del modelo interno" for p, s in zip(gone["planta"], gone["sku"])]
    missing = pd.DataFrame({"planta": gone["planta"], "mes": month, "sku": gone["sku"], "strategy": gone["strategy"], "planner": gone["planner"], "reason": MISSING_INTERNAL, "detail": detail})
    return table, missing


def attach_planner(table: pd.DataFrame, planner: pd.DataFrame) -> pd.DataFrame:
    r"""Add the planner forecast of the same (planta, month, SKU) to `table` (columns planta, mes, sku); NaN when absent."""
    return attach_source(table, planner, "planner")


def accuracy_table(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    r"""Accuracy, WAPE and bias of the planner and of each model over the planner SKUs, ranked by accuracy."""
    rows = [{"model": c, **score(table["real"].to_numpy(), table[c].to_numpy())} for c in columns]
    out = pd.DataFrame(rows).sort_values("wape", ignore_index=True)
    out.insert(0, "rank", out["wape"].rank(method="min").astype(int))
    return out[["rank", "model", "n", "mae", "wape", "accuracy", "bias_pct"]]
