"""Forecast proposed by the internal model of the company (all plantas, same SKU grain as the planner file)."""

from __future__ import annotations

import pandas as pd

MONTH_COLUMNS = ["2026-06", "2026-07", "2026-08", "2026-09"]


def load_internal_forecast(path) -> pd.DataFrame:
    r"""
    Long table (planta, mes, sku_planner, product_sap, strategy, forecast_to) from `data/internal_model_forecast.csv`.

    The columns have the layout of the planner files, so `planner_utils.compare_month` reads both. The SKU is
    `grade/sub-grade/gsm/width` as in the planner files (no core width).
    """
    wide = pd.read_csv(path)
    months = [c for c in wide.columns if c in MONTH_COLUMNS or c[:2] == "20"]
    long = wide.melt(id_vars=["planta", "grade", "sub_grade", "grammage", "width", "product_sap"], value_vars=months, var_name="mes", value_name="forecast_to")
    long["sku_planner"] = long.apply(lambda r: f"{r['grade']}/{int(r['sub_grade']):02d}/{int(r['grammage'])}gsm/{int(r['width'])}mm", axis=1)
    long["strategy"] = ""
    return long[["planta", "mes", "sku_planner", "product_sap", "strategy", "forecast_to"]]
