"""Read the workbook with the proposed forecast of the company's internal model (every planta, one column per month)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.planner_io_utils import build_sku

KEY_COLUMNS = ["Customer", "Grade", "Sub-grade", "Grammage", "Width", "Product"]
OUTPUT_COLUMNS = ["planta", "mes", "sku_planner", "product_sap", "forecast_to"]


def parse_month_column(label) -> str:
    r"""Month label ("2026-06") from a column header such as "06.2026"."""
    found = re.fullmatch(r"\s*(\d{1,2})\.(\d{4})\s*", str(label))
    if not found or not 1 <= int(found.group(1)) <= 12:
        raise ValueError(f"cannot read the month from the column {label!r}")
    return f"{found.group(2)}-{int(found.group(1)):02d}"


def read_internal_workbook(path: str | Path) -> pd.DataFrame:
    r"""
    One row per (planta, SKU, month) of the first sheet: planta, mes, sku_planner, product_sap, forecast_to.

    The first row holds a label ("Proposed Forecast") and the second the column names; the closing "Result" row and the empty
    row after it are dropped, after checking that the Result row equals the sum of the monthly columns.
    """
    raw = pd.read_excel(path, sheet_name=0, header=1)
    if list(raw.columns[: len(KEY_COLUMNS)]) != KEY_COLUMNS:
        raise ValueError(f"{path}: expected the columns {KEY_COLUMNS}, found {list(raw.columns)}")
    months = {col: parse_month_column(col) for col in raw.columns[len(KEY_COLUMNS) :]}
    if not months:
        raise ValueError(f"{path}: no month columns after {KEY_COLUMNS}")
    rows = raw[raw["Grade"].notna()].copy()
    totals = raw[raw["Customer"].astype(str).str.strip().str.lower() == "result"]
    if len(totals):
        for col, month in months.items():
            if not np.isclose(float(totals[col].iloc[0]), float(rows[col].sum()), atol=0.5):
                raise ValueError(f"{path}: the Result row of {month} ({totals[col].iloc[0]}) is not the sum of the rows ({rows[col].sum()})")
    forecast = rows[list(months)].apply(pd.to_numeric, errors="coerce")
    if forecast.isna().any().any() or (forecast < 0).any().any():
        raise ValueError(f"{path}: the monthly forecasts have missing or negative values")
    rows["sku_planner"] = build_sku(rows["Grade"], rows["Sub-grade"], rows["Grammage"], rows["Width"])
    rows["planta"] = rows["Customer"].astype(str).str.strip()
    repeated = rows[rows.duplicated(["planta", "sku_planner"], keep=False)]
    if len(repeated):
        raise ValueError(f"{path}: {len(repeated)} rows repeat a (planta, SKU): {repeated['sku_planner'].unique()[:5].tolist()}")
    rows["product_sap"] = rows["Product"].astype(int)
    long = rows.melt(id_vars=["planta", "sku_planner", "product_sap"], value_vars=list(months), var_name="mes", value_name="forecast_to")
    long["mes"] = long["mes"].map(months)
    long["forecast_to"] = long["forecast_to"].astype(float)
    return long[OUTPUT_COLUMNS].sort_values(["planta", "sku_planner", "mes"], ignore_index=True)


def planta_is_all_zero(long: pd.DataFrame, planta: str) -> bool:
    r"""True when every forecast of `planta` is 0 (the workbook has the planta but no values for it)."""
    mine = long[long["planta"] == planta]
    return len(mine) > 0 and bool((mine["forecast_to"] == 0).all())


def replace_planta(long: pd.DataFrame, other: pd.DataFrame, planta: str) -> pd.DataFrame:
    r"""`long` with every row of `planta` taken from `other` (same columns) instead."""
    return pd.concat([long[long["planta"] != planta], other[other["planta"] == planta][OUTPUT_COLUMNS]], ignore_index=True).sort_values(["planta", "sku_planner", "mes"], ignore_index=True)


def mark_missing(long: pd.DataFrame) -> pd.DataFrame:
    r"""
    Blank the forecast (NaN, "faltante") of every SKU whose forecast is 0 in every month: an exact zero in all months means the
    model gave no forecast, not a forecast of zero. SKUs with a zero in only some months keep their zeros.
    """
    all_zero = long.groupby(["planta", "sku_planner"])["forecast_to"].transform(lambda s: bool((s == 0).all()))
    out = long.copy()
    out.loc[all_zero, "forecast_to"] = np.nan
    return out
