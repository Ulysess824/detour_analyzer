"""Tracking for the Seguimiento screen: the history of accuracy, the detail of one month, the biggest misses and the measuring of a closed month."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.app.consumption_service import ConsumptionStore
from src.app.forecast_service import list_forecasts, load_forecast
from src.app.paths import AppPaths
from src.engine.db_utils import read_tracking_csv
from src.evaluation.tracking_run import BACKTEST_RUN, save_tracking, track_saved_forecast
from src.evaluation.tracking_utils import MODELS, THRESHOLDS

LEVELS = ["sku", "planta", "total"]
TOP_ERRORS = 15


def _pct(value: float) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value) * 100, 1)


def _num(value: float) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 1)


def tracking_table(paths: AppPaths) -> pd.DataFrame:
    return read_tracking_csv(paths.tracking_csv) if paths.tracking_csv.exists() else pd.DataFrame()


def tracked_months(paths: AppPaths) -> set[str]:
    r"""Months of the saved forecasts that already have a production tracking (the backtest does not count)."""
    table = tracking_table(paths)
    return set(table.loc[table["run_id"] != BACKTEST_RUN, "mes"]) if len(table) else set()


def to_measure(paths: AppPaths, store: ConsumptionStore) -> list[str]:
    r"""Months with a saved forecast whose real month has already closed in the data and that were not measured yet."""
    done = tracked_months(paths)
    return [f["mes"] for f in list_forecasts(paths) if store.is_closed(f["mes"]) and f["mes"] not in done]


def history(paths: AppPaths, store: ConsumptionStore | None) -> dict:
    r"""Accuracy of `ml_mean` by month and level, split into the historical test (backtest) and the production months, plus the alert thresholds."""
    table = tracking_table(paths)
    out = {"umbrales": {k: round(v * 100, 1) for k, v in THRESHOLDS.items()}, "backtest": [], "produccion": [], "por_medir": to_measure(paths, store) if store else []}
    if table.empty:
        return out
    ml = table[(table["modelo"] == "ml_mean") & (table["planta"] == "") & table["nivel"].isin(LEVELS)]
    for (run, month), block in ml.groupby(["run_id", "mes"]):
        scores = block.set_index("nivel")["acierto"]
        alerts = sorted({c for cell in table[(table["run_id"] == run) & (table["mes"] == month)]["alerta"] for c in str(cell).split(";") if c})
        row = {"mes": month, "sku": _pct(scores.get("sku")), "planta": _pct(scores.get("planta")), "total": _pct(scores.get("total")), "alertas": alerts}
        out["backtest" if run == BACKTEST_RUN else "produccion"].append(row)
    for key in ("backtest", "produccion"):
        out[key].sort(key=lambda r: r["mes"])
    return out


def top_errors(paths: AppPaths, store: ConsumptionStore, month: str, n: int = TOP_ERRORS) -> list[dict]:
    r"""The series where the saved forecast of a closed month missed most in tonnes (empty when there is no forecast file or the month is open)."""
    if not paths.forecast_csv(month).exists() or not store.is_closed(month):
        return []
    forecast = load_forecast(paths, month)
    real = store.month_values(month)
    forecast["real"] = [real.get(key, 0.0) for key in zip(forecast["planta"], forecast["sku"])]
    forecast["error"] = forecast["ml_mean"] - forecast["real"]
    worst = forecast.reindex(forecast["error"].abs().sort_values(ascending=False).index).head(n)
    return [{"planta": r["planta"], "sku": r["sku"], "real_TO": _num(r["real"]), "pronostico_TO": _num(r["ml_mean"]), "error_TO": _num(r["error"])} for _, r in worst.iterrows()]


def detail(paths: AppPaths, store: ConsumptionStore | None, month: str | None, origin: str | None = None) -> dict:
    r"""One tracked month: every model by level, the plantas of `ml_mean` from the worst, the alerts and the biggest misses. `origin` is produccion or backtest."""
    table = tracking_table(paths)
    if table.empty:
        raise LookupError("Todavía no hay seguimiento. Cuando cierre el mes pronosticado, mídelo aquí.")
    if month is None:
        production = table[table["run_id"] != BACKTEST_RUN]
        month = (production if len(production) else table)["mes"].max()
    rows = table[table["mes"] == month]
    if rows.empty:
        raise LookupError(f"No hay seguimiento de {month}.")
    has_production = (rows["run_id"] != BACKTEST_RUN).any()
    origin = origin or ("produccion" if has_production else "backtest")
    rows = rows[(rows["run_id"] != BACKTEST_RUN) if origin == "produccion" else (rows["run_id"] == BACKTEST_RUN)]
    if rows.empty:
        raise LookupError(f"No hay seguimiento de {month} con origen {origin}.")
    rows = rows[rows["run_id"] == rows["run_id"].iloc[-1]]  # one run: the last one stored for the month
    pooled = rows[rows["planta"] == ""]
    models = []
    for model in MODELS:
        mine = pooled[pooled["modelo"] == model].set_index("nivel")
        if mine.empty:
            continue
        models.append({"modelo": model, **{lv: _pct(mine["acierto"].get(lv)) for lv in LEVELS}, "sesgo_total_pct": _num(mine["bias_pct"].get("total"))})
    total = pooled[(pooled["modelo"] == "ml_mean") & (pooled["nivel"] == "total")]
    plantas = rows[(rows["modelo"] == "ml_mean") & (rows["nivel"] == "planta") & (rows["planta"] != "")]
    plantas = [{"planta": r["planta"], "acierto": _pct(r["acierto"]), "real_TO": _num(r["real_TO"]), "pronostico_TO": _num(r["pronostico_TO"]), "sesgo_pct": _num(r["bias_pct"])}
               for _, r in plantas.sort_values("acierto").iterrows()]
    flagged = rows[(rows["modelo"] == "ml_mean") & (rows["alerta"] != "")]
    alerts = [{"codigo": c, "nivel": r["nivel"], "planta": r["planta"]} for _, r in flagged.iterrows() for c in r["alerta"].split(";")]
    informative = [{"codigo": "planta_revisar", "nivel": "planta", "planta": p["planta"]} for p in plantas if p["acierto"] is not None and p["acierto"] < THRESHOLDS["planta_revisar"] * 100]
    missing = rows[rows["nivel"] == "sin_pronostico"]
    return {
        "mes": month,
        "origen": origin,
        "disponibles": sorted(set(table["mes"]), reverse=True),
        "modelos": models,
        "total": None if total.empty else {"real_TO": _num(total["real_TO"].iloc[0]), "pronostico_TO": _num(total["pronostico_TO"].iloc[0]), "sesgo_pct": _num(total["bias_pct"].iloc[0])},
        "plantas": plantas,
        "alertas": alerts or informative,
        "sin_pronostico": None if missing.empty or not missing["n"].iloc[0] else {"series": int(missing["n"].iloc[0]), "real_TO": _num(missing["real_TO"].iloc[0])},
        "errores": top_errors(paths, store, month) if origin == "produccion" and store is not None else [],
    }


def measure(paths: AppPaths, store: ConsumptionStore, month: str) -> dict:
    r"""Score the saved forecast of a closed month against the real and store it (history csv, and the database when there is one); returns the new detail."""
    if not paths.forecast_csv(month).exists():
        raise ValueError(f"No hay un pronóstico guardado para {month}.")
    if not store.is_closed(month):
        raise ValueError(f"El mes {month} todavía no está completo en los datos. Carga el mes completo en la sección Datos.")
    try:
        run = track_saved_forecast(paths.forecast_csv(month), store.table, store.panel)
    except SystemExit as stop:  # the engine stops with a message
        raise ValueError(str(stop)) from None
    save_tracking([run], paths.tracking_csv, paths.database if paths.database.exists() else None)
    return {"corrida": run.run_id, "alertas": int(run.fired), "detalle": detail(paths, store, month, "produccion")}
