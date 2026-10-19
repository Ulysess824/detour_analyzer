"""Forecast of the internal model of the company (SCAN, months 2026-06 to 2026-09)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MONTH_COLUMNS = ["2026-06", "2026-07", "2026-08", "2026-09"]


def product_to_sku(planner_files: list[Path]) -> dict:
    r"""SAP product code -> `sku_planner` (grade/sub-grade/gsm/width), read from the planner files."""
    planner = pd.concat([pd.read_csv(f)[["product_sap", "sku_planner"]] for f in planner_files], ignore_index=True)
    return planner.drop_duplicates("product_sap").set_index("product_sap")["sku_planner"].to_dict()


def load_internal_forecast(path, planner_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Long table (planta, mes, sku_planner, product_sap, strategy, forecast_to), the layout of the planner files, and the
    rows whose SAP code is not in the planner files.

    The file has the grade, gsm and width of each product but no sub-grade, and some labels differ from the data (for example
    three THP products appear as TSL), so the SKU is taken from the SAP code, which is what the planner files and the data share.
    """
    wide = pd.read_csv(path)
    sku = product_to_sku(planner_files)
    wide["sku_planner"] = wide["product_sap"].map(sku)
    unmatched = wide[wide["sku_planner"].isna()]
    known = wide.dropna(subset=["sku_planner"])
    long = known.melt(id_vars=["planta", "product_sap", "sku_planner"], value_vars=MONTH_COLUMNS, var_name="mes", value_name="forecast_to")
    long["strategy"] = ""
    return long[["planta", "mes", "sku_planner", "product_sap", "strategy", "forecast_to"]], unmatched
