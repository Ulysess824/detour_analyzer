"""The Historial screen: every saved forecast with when it was made, with which data and parameters, and how it turned out."""

from __future__ import annotations

import json

import pandas as pd

from src.app.paths import AppPaths
from src.app.tracking_service import tracking_table, to_measure
from src.app.consumption_service import ConsumptionStore
from src.evaluation.tracking_run import BACKTEST_RUN


def forecast_history(paths: AppPaths, store: ConsumptionStore | None) -> list[dict]:
    r"""Saved forecasts, newest month first: total, data used, parameters used and, once measured, the accuracy of `ml_mean`."""
    tracking = tracking_table(paths)
    production = tracking[(tracking["run_id"] != BACKTEST_RUN) & (tracking["modelo"] == "ml_mean") & (tracking["planta"] == "")] if len(tracking) else tracking
    pending = set(to_measure(paths, store)) if store else set()
    out = []
    for month, strategy, csv in paths.forecast_files():
        table = pd.read_csv(csv, usecols=["ml_mean"])
        row = {"mes": month, "estrategia": strategy, "total_TO": round(float(table["ml_mean"].sum()), 1), "series": int(len(table)), "creado_en": None, "datos_hasta": None, "parametros": None,
               "ventana_ajuste": None, "medido": None, "por_medir": month in pending and strategy == "h1"}
        manifest = csv.with_suffix(".manifest.json")
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            row["creado_en"] = data["run"]["created_at"]
            row["datos_hasta"] = data["data"]["last_date"]
            row["parametros"] = data["params"]["tuning_file"].replace("\\", "/").split("/")[-1]
            row["ventana_ajuste"] = next(iter(data["params"].get("tune_window", {}).values()), None)
        scores = production[production["mes"] == month].set_index("nivel")["acierto"] if len(production) else None
        if scores is not None and len(scores):
            row["medido"] = {level: round(float(scores[level]) * 100, 1) for level in ("sku", "planta", "total") if level in scores.index}
        out.append(row)
    return out
