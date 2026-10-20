"""Forecasts for the Predicción screen: the saved `forecast_<month>.csv` files, their detail against the real history, and the run itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.app.consumption_service import ConsumptionStore
from src.app.paths import AppPaths
from src.engine.prediction_run import run_prediction

MODELS = ["lgbm", "xgb", "rf"]
TABLE_LIMIT = 50


class ForecastMissing(LookupError):
    pass


def list_forecasts(paths: AppPaths) -> list[dict]:
    r"""Saved forecasts, newest month first, with when each was made and its total."""
    out = []
    for csv in sorted(paths.results.glob("forecast_????-??.csv"), reverse=True):
        month = csv.stem.removeprefix("forecast_")
        manifest = csv.with_suffix(".manifest.json")
        created = json.loads(manifest.read_text(encoding="utf-8"))["run"]["created_at"] if manifest.exists() else None
        out.append({"mes": month, "creado_en": created, "total_TO": round(float(pd.read_csv(csv, usecols=["ml_mean"])["ml_mean"].sum()), 1)})
    return out


def load_forecast(paths: AppPaths, month: str) -> pd.DataFrame:
    csv = paths.forecast_csv(month)
    if not csv.exists():
        raise ForecastMissing(f"No hay un pronóstico guardado para {month}.")
    return pd.read_csv(csv)


def _number(value: float) -> float | None:
    return None if value is None or np.isnan(value) else round(float(value), 1)


def _change(forecast: float, real: float | None) -> float | None:
    return None if not real else round((forecast / real - 1) * 100, 1)


def forecast_detail(paths: AppPaths, store: ConsumptionStore, month: str | None, planta: str | None, sku: str | None, limit: int = TABLE_LIMIT) -> dict:
    r"""
    The forecast of `month` (the newest by default) for a planta and SKU selection: totals by model, the spread of the three trees, the real history to
    draw next to it and a table (plantas when nothing is chosen, SKUs otherwise) with the real of the last complete month.
    """
    saved = list_forecasts(paths)
    if not saved:
        raise ForecastMissing("Todavía no hay pronósticos. Genera el de este mes.")
    month = month or saved[0]["mes"]
    forecast = load_forecast(paths, month)
    planta, sku, _ = store.select(planta, sku)
    mask = np.ones(len(forecast), dtype=bool)
    if planta:
        mask &= forecast["planta"].to_numpy() == planta
    if sku:
        mask &= forecast["sku"].to_numpy() == sku
    chosen = forecast[mask]
    totals = {model: float(chosen[model].sum()) for model in [*MODELS, "ml_mean", "naive"]}
    history = store.monthly(planta, sku, breakdown=False)
    last = history["resumen"]["ultimo_mes_completo"]
    last_real = history["resumen"]["ultimo_mes_TO"]
    spread = [totals[m] for m in MODELS]
    S = store.panel
    t = S.labels.index(last) if last in S.labels else None
    position = {key: i for i, key in enumerate(S.keys)}

    def real_of(keys: list[tuple[str, str]]) -> float | None:
        if t is None:
            return None
        return float(np.nansum([S.monthly[position[k], t] for k in keys if k in position]))

    if planta is None and sku is None:
        rows = []
        for name, block in chosen.groupby("planta"):
            real = float(S.planta_monthly[S.plantas.index(name), t]) if t is not None and name in S.plantas else None
            rows.append({"planta": name, "pronostico_TO": _number(block["ml_mean"].sum()), "real_ultimo_TO": _number(real), "cambio_pct": _change(block["ml_mean"].sum(), real)})
        rows.sort(key=lambda r: -(r["pronostico_TO"] or 0))
        table = {"tipo": "planta", "filas": rows, "total_filas": len(rows)}
    else:
        top = chosen.sort_values("ml_mean", ascending=False)
        rows = []
        for _, r in top.head(limit).iterrows():
            real = real_of([(r["planta"], r["sku"])])
            rows.append({"planta": r["planta"], "sku": r["sku"], "pronostico_TO": _number(r["ml_mean"]), "real_ultimo_TO": _number(real),
                         "cambio_pct": _change(r["ml_mean"], real)})
        table = {"tipo": "sku", "filas": rows, "total_filas": len(top)}

    return {
        "mes": month,
        "disponibles": [f["mes"] for f in saved],
        "creado_en": next(f["creado_en"] for f in saved if f["mes"] == month),
        "filtro": {"planta": planta, "sku": sku},
        "pronostico": {
            "ml_mean_TO": _number(totals["ml_mean"]),
            "modelos_TO": {m: _number(totals[m]) for m in [*MODELS, "naive"]},
            "minimo_TO": _number(min(spread)),
            "maximo_TO": _number(max(spread)),
            "series": int(len(chosen)),
        },
        "ultimo_real": {"mes": last, "TO": last_real, "cambio_pct": _change(totals["ml_mean"], last_real)},
        "historia": {"etiquetas": history["etiquetas"], "valores": history["valores"], "mes_parcial": history["resumen"]["mes_parcial"]},
        "tabla": table,
    }


def next_month(store: ConsumptionStore) -> str:
    return str(pd.Period(store.last_complete(), freq="M") + 1)


def generate(paths: AppPaths, store: ConsumptionStore, progress: Callable[[str, float], None]) -> dict:
    r"""Forecast the month after the last complete one (the same run as scripts/predict_next_month.py) and return what the screen needs."""
    if store.is_partial():
        raise ValueError(f"El último mes de los datos ({store.panel.labels[-1]}) está incompleto. Carga el mes completo en la sección Datos.")
    run = run_prediction(paths.consumption_csv, paths.tuning_json, paths.forecast_csv(next_month(store)), repo_dir=paths.root, progress=progress)
    return {"mes": run.month, "series": int(len(run.result)), "total_TO": round(float(run.result["ml_mean"].sum()), 1),
            "avisos_de_datos": len(run.validation.warnings)}
