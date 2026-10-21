"""Forecasts for the Predicción screen: the saved `forecast_<month>[_<strategy>].csv` files, their detail against the real history, and the run itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.app import settings_service
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
    for month, strategy, csv in paths.forecast_files():
        manifest = csv.with_suffix(".manifest.json")
        record = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        out.append({"mes": month, "estrategia": strategy, "corte": record.get("strategy", {}).get("cutoff_day"),
                    "creado_en": record.get("run", {}).get("created_at"), "total_TO": round(float(pd.read_csv(csv, usecols=["ml_mean"])["ml_mean"].sum()), 1)})
    return out


def load_forecast(paths: AppPaths, month: str, strategy: str = "h1") -> pd.DataFrame:
    csv = paths.forecast_csv(month, strategy)
    if not csv.exists():
        raise ForecastMissing(f"No hay un pronóstico guardado para {month}.")
    return pd.read_csv(csv)


def _number(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 1)


def _change(forecast: float, real: float | None) -> float | None:
    return None if not real else round((forecast / real - 1) * 100, 1)


def forecast_detail(paths: AppPaths, store: ConsumptionStore, month: str | None, planta: str | None, sku: str | None, limit: int = TABLE_LIMIT, strategy: str | None = None) -> dict:
    r"""
    The forecast of `month` (the newest by default; the one-month forecast when the month has several) for a planta and SKU selection: totals by model, the spread of the three trees, the real history to
    draw next to it and a table (plantas when nothing is chosen, SKUs otherwise) with the real of the last complete month.
    """
    saved = list_forecasts(paths)
    if not saved:
        raise ForecastMissing("Todavía no hay pronósticos. Genera el de este mes.")
    month = month or saved[0]["mes"]
    of_month = [f for f in saved if f["mes"] == month]
    strategy = strategy or (of_month[0]["estrategia"] if of_month else "h1")
    forecast = load_forecast(paths, month, strategy)
    planta, sku, _ = store.select(planta, sku)
    mask = np.ones(len(forecast), dtype=bool)
    if planta:
        mask &= forecast["planta"].to_numpy() == planta
    if sku:
        mask &= forecast["sku"].to_numpy() == sku
    chosen = forecast[mask]
    totals = {model: float(chosen[model].sum()) for model in [*MODELS, "ml_mean", "naive"]}
    history = store.monthly(planta, sku, breakdown=False)
    last = reference_month(store, month)  # the month before the forecast, even when later months are loaded already
    last_real = history["valores"][history["etiquetas"].index(last)] if last in history["etiquetas"] else None
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
        "estrategia": strategy,
        "disponibles": list(dict.fromkeys(f["mes"] for f in saved)),
        "disponibles_detalle": [{"mes": f["mes"], "estrategia": f["estrategia"]} for f in saved],
        "creado_en": next(f["creado_en"] for f in saved if f["mes"] == month and f["estrategia"] == strategy),
        "corte": next(f["corte"] for f in saved if f["mes"] == month and f["estrategia"] == strategy),
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


def reference_month(store: ConsumptionStore, month: str) -> str:
    r"""The real month a forecast is compared with: the one before it (the last complete month when that one is not in the data)."""
    before = str(pd.Period(month, freq="M") - 1)
    return before if store.is_closed(before) else store.last_complete()


def next_month(store: ConsumptionStore) -> str:
    return str(pd.Period(store.last_complete(), freq="M") + 1)


def plan_now(paths: AppPaths, store: ConsumptionStore):
    r"""The strategy, cutoff and target month the saved settings give for the data loaded (ValueError, in Spanish, when they do not fit)."""
    return settings_service.current_plan(store, settings_service.load(paths))


def generate(paths: AppPaths, store: ConsumptionStore, progress: Callable[[str, float], None]) -> dict:
    r"""Forecast with the saved settings (the same run as scripts/predict_next_month.py) and return what the screen needs."""
    settings = settings_service.load(paths)
    plan = settings_service.current_plan(store, settings)
    target = paths.forecast_csv(plan.target, plan.strategy)
    run = run_prediction(paths.consumption_csv, paths.tuning_for_forecast(), target, repo_dir=paths.root, progress=progress,
                         strategy=settings["estrategia"], cutoff_day=settings["corte_manual"])
    return {"mes": run.month, "estrategia": run.plan.strategy, "corte": run.plan.cutoff_day, "avisos": run.plan.notes, "series": int(len(run.result)),
            "total_TO": round(float(run.result["ml_mean"].sum()), 1), "avisos_de_datos": len(run.validation.warnings)}
