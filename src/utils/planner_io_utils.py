"""Read the monthly planner assignment workbooks (one sheet per month, every planta) into the planner csv layout."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Month names of the workbook headers ("JUN´26 allocation", "SEPT´26 allocation").
MONTH_NUMBERS = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AGO": 8, "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DIC": 12}
INPUT_COLUMNS = ["Customer", "Grade", "Sub-grade", "Grammage", "Width", "Product", "Strategy", "Plant Forecast"]
OUTPUT_COLUMNS = ["planta", "mes", "sku_planner", "product_sap", "strategy", "forecast_to"]
N_COLUMNS = len(INPUT_COLUMNS) + 1  # the ninth column is "<MONTH>´<YY> allocation", the mill that produces (not used)


def parse_month(header: str) -> str:
    r"""Month label ("2026-09") from the header of the ninth column, for example "SEPT´26 allocation"."""
    found = re.match(r"\s*([A-Za-z]+)\W*(\d{2})\b", str(header))
    if not found or found.group(1).upper() not in MONTH_NUMBERS:
        raise ValueError(f"cannot read the month from the header {header!r}")
    return f"{2000 + int(found.group(2))}-{MONTH_NUMBERS[found.group(1).upper()]:02d}"


def read_assignment(path: str | Path) -> pd.DataFrame:
    r"""
    One row per (planta, planner SKU) of the first sheet, with the columns of `data/planner_forecast_<month>.csv`.

    `Customer` is the planta. The SKU is `grade/sub-grade (two digits)/gsm/width`, the planner SKU format of the
    existing files; the allocation column (the producing mill) is dropped. The month comes from the header.
    """
    raw = pd.read_excel(path, sheet_name=0, usecols=range(N_COLUMNS))
    if list(raw.columns[: len(INPUT_COLUMNS)]) != INPUT_COLUMNS:
        raise ValueError(f"{path}: expected the columns {INPUT_COLUMNS}, found {list(raw.columns)}")
    month = parse_month(raw.columns[N_COLUMNS - 1])
    raw = raw.dropna(subset=["Customer"])
    forecast = pd.to_numeric(raw["Plant Forecast"], errors="coerce")
    if forecast.isna().any() or (forecast < 0).any():
        raise ValueError(f"{path}: Plant Forecast has missing or negative values in {int(forecast.isna().sum() + (forecast < 0).sum())} rows")
    sku = (
        raw["Grade"].astype(str).str.strip()
        + "/"
        + raw["Sub-grade"].astype(int).astype(str).str.zfill(2)
        + "/"
        + raw["Grammage"].astype(int).astype(str)
        + "gsm/"
        + raw["Width"].astype(int).astype(str)
        + "mm"
    )
    out = pd.DataFrame(
        {
            "planta": raw["Customer"].astype(str).str.strip(),
            "mes": month,
            "sku_planner": sku,
            "product_sap": raw["Product"].astype(int),
            "strategy": raw["Strategy"].fillna("").astype(str).str.strip(),
            "forecast_to": forecast.astype(float),
        }
    )[OUTPUT_COLUMNS].reset_index(drop=True)
    repeated = out[out.duplicated(["planta", "sku_planner"], keep=False)]
    if len(repeated):
        raise ValueError(f"{path}: {len(repeated)} rows repeat a (planta, SKU): {repeated['sku_planner'].unique()[:5].tolist()}")
    return out
