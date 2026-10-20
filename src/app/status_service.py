"""What the Inicio screen shows: data, last forecast, tracking of the last closed month and the hyperparameter retune schedule."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from src.app.paths import AppPaths
from src.engine.db_utils import read_tracking_csv

RETUNE_MONTHS = 6
WARN_DAYS = 30


def add_months(day: date, months: int) -> date:
    r"""`day` plus whole months, clipped to the end of a shorter month."""
    stamp = pd.Timestamp(day) + pd.DateOffset(months=months)
    return stamp.date()


def retune_status(paths: AppPaths, today: date) -> dict:
    r"""Last retune date, next due date and the state: al_dia, por_vencer (30 days or less), vencido, or sin_registro."""
    path = paths.adjustments_json
    if not path.exists():
        return {"estado": "sin_registro", "ultimo": None, "proximo": None, "dias_restantes": None, "intervalo_meses": RETUNE_MONTHS}
    record = json.loads(path.read_text(encoding="utf-8"))
    months = int(record.get("intervalo_meses", RETUNE_MONTHS))
    last = date.fromisoformat(record["ultimo_ajuste"])
    due = add_months(last, months)
    left = (due - today).days
    state = "vencido" if left < 0 else "por_vencer" if left <= WARN_DAYS else "al_dia"
    return {"estado": state, "ultimo": str(last), "proximo": str(due), "dias_restantes": left, "intervalo_meses": months}


def last_forecast(paths: AppPaths) -> dict | None:
    r"""The newest `results/forecast_<month>.csv` with the figures of its manifest (when it has one)."""
    files = sorted(paths.results.glob("forecast_????-??.csv"))
    if not files:
        return None
    csv = files[-1]
    month = csv.stem.removeprefix("forecast_")
    out = {"mes": month, "archivo": csv.name, "creado_en": None, "total_ml_mean_TO": None, "series": None}
    manifest = csv.with_suffix(".manifest.json")
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        out["creado_en"] = data["run"]["created_at"]
        out["total_ml_mean_TO"] = data["output"]["total_ml_mean"]
        out["series"] = data["output"]["rows"]
    else:
        table = pd.read_csv(csv)
        out["total_ml_mean_TO"] = round(float(table["ml_mean"].sum()), 3)
        out["series"] = len(table)
    return out


def _percent(value: float) -> float | None:
    r"""An accuracy as a percentage; None when it is not a number (JSON cannot carry NaN or infinity)."""
    return round(float(value) * 100, 1) if pd.notna(value) and abs(value) != float("inf") else None


def last_tracking(paths: AppPaths) -> dict | None:
    r"""Scores of `ml_mean` in the newest tracked month of the production runs (the backtest run counts only when nothing else exists)."""
    if not paths.tracking_csv.exists():
        return None
    table = read_tracking_csv(paths.tracking_csv)
    table = table[table["modelo"] == "ml_mean"]
    if table.empty:
        return None
    real = table[table["run_id"] != "backtest-h1"]
    source = real if len(real) else table
    month = source["mes"].max()
    rows = source[source["mes"] == month]
    pick = lambda level: rows[(rows["nivel"] == level) & (rows["planta"] == "")]
    score = lambda level: _percent(pick(level)["acierto"].iloc[0]) if len(pick(level)) else None
    alerts = sorted({code for cell in rows["alerta"].fillna("") for code in str(cell).split(";") if code})
    return {
        "mes": month,
        "origen": "produccion" if len(real) else "backtest",
        "acierto_sku": score("sku"),
        "acierto_planta": score("planta"),
        "acierto_total": score("total"),
        "alertas": alerts,
    }


def pending_tracking(paths: AppPaths, last_complete_month: str | None, forecast: dict | None) -> str | None:
    r"""The month of the saved forecast when it has already closed in the data but is not tracked yet."""
    if forecast is None or last_complete_month is None or forecast["mes"] > last_complete_month:
        return None
    if paths.tracking_csv.exists():
        done = read_tracking_csv(paths.tracking_csv)
        if ((done["mes"] == forecast["mes"]) & (done["run_id"] != "backtest-h1")).any():
            return None
    return forecast["mes"]


def app_status(paths: AppPaths, info: dict | None, today: date) -> dict:
    r"""Everything the Inicio screen needs in one dictionary; `info` is `ConsumptionStore.info()` (None when there is no data yet)."""
    forecast = last_forecast(paths)
    complete = None
    if info is not None:
        complete = info["meses"][-2] if info["mes_parcial"] else info["meses"][-1]
    return {
        "datos": None if info is None else {**{k: v for k, v in info.items() if k != "meses"}, "ultimo_mes_completo": complete},
        "proximo_pronostico": str(pd.Period(complete, freq="M") + 1) if complete else None,
        "pronostico": forecast,
        "seguimiento": last_tracking(paths),
        "seguimiento_pendiente": pending_tracking(paths, complete, forecast),
        "ajuste": {**retune_status(paths, today), "candidato": paths.tuning_candidate.exists()},
    }
