"""Hyperparameter retune for the Ajuste screen: run it in the background, show the candidate, adopt or discard it, and keep the schedule."""

from __future__ import annotations

import json
import shutil
from datetime import date
from typing import Callable

import numpy as np

from src.app.consumption_service import ConsumptionStore
from src.app.paths import AppPaths
from src.evaluation.tuning_run import NAMES, run_tuning
from src.evaluation.tuning_utils import to_jsonable

TEST_MONTHS = 12
PRESETS = {
    "completo": {"nombre": "Completo", "trials": {"lgbm": 100, "xgb": 100, "rf": 40}, "minutos": 20,
                 "texto": "La búsqueda de siempre: 100 pruebas para LightGBM y XGBoost y 40 para Random Forest. Es la que se recomienda."},
    "rapido": {"nombre": "Rápido", "trials": {"lgbm": 15, "xgb": 15, "rf": 8}, "minutos": 5,
               "texto": "Una búsqueda corta para probar que todo funciona. Sus parámetros suelen ser peores: no se recomienda adoptarla."},
}


def presets() -> list[dict]:
    return [{"id": key, "nombre": p["nombre"], "minutos": p["minutos"], "texto": p["texto"], "trials": p["trials"]} for key, p in PRESETS.items()]


def retune(paths: AppPaths, store: ConsumptionStore, preset: str, progress: Callable[[str, float], None]) -> dict:
    r"""Tune the three models on the data up to the last complete month and save the result as the candidate (it changes nothing until adopted)."""
    if preset not in PRESETS:
        raise ValueError(f"No existe el tipo de ajuste '{preset}'.")
    if store.is_partial():
        raise ValueError(f"El último mes de los datos ({store.panel.labels[-1]}) está incompleto. Carga el mes completo en la sección Datos.")
    trials = PRESETS[preset]["trials"]
    results = to_jsonable(run_tuning(store.panel, list(trials), trials, n_test=TEST_MONTHS, progress=progress, log=lambda message: None))
    stamp = date.today().isoformat()
    for model in results.values():
        model["meta"] = {**model["meta"], "preset": preset, "generado_en": stamp}
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.tuning_candidate.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return candidate_summary(paths)


def _mean(values) -> float:
    return float(np.mean(values))


def candidate_summary(paths: AppPaths) -> dict | None:
    r"""What the screen shows to decide: for each model the SKU, planta and total accuracy in the test months with the baseline and with the tuned parameters."""
    if not paths.tuning_candidate.exists():
        return None
    data = json.loads(paths.tuning_candidate.read_text(encoding="utf-8"))
    models, gains = [], []
    for kind, result in data.items():
        base, tuned = result["h1"]["base"], result["h1"]["tuned"]
        row = {"modelo": kind, "nombre": NAMES.get(kind, kind), "pruebas": result["n_trials"], "minutos": result["minutes"]}
        for level in ("sku", "planta", "total"):
            row[f"{level}_base"] = round(_mean(base[f"{level}_acc"]) * 100, 1)
            row[f"{level}_ajustado"] = round(_mean(tuned[f"{level}_acc"]) * 100, 1)
        row["mejora_sku_pts"] = round(row["sku_ajustado"] - row["sku_base"], 1)
        gains.append(row["mejora_sku_pts"])
        models.append(row)
    meta = next(iter(data.values()))["meta"]
    return {"modelos": models, "mejora_sku_pts": round(_mean(gains), 1), "ventana_ajuste": meta["tune_window"], "ventana_prueba": meta["test_window"],
            "tipo": meta.get("preset"), "generado_en": meta.get("generado_en")}


def history(paths: AppPaths) -> list[dict]:
    if not paths.adjustments_json.exists():
        return []
    return sorted(json.loads(paths.adjustments_json.read_text(encoding="utf-8")).get("historial", []), key=lambda r: r["fecha"], reverse=True)


def adopt(paths: AppPaths, today: date) -> dict:
    r"""Make the candidate the production parameters: the previous ones are kept in `results/ajustes_anteriores/` and the retune date and history are updated."""
    summary = candidate_summary(paths)
    if summary is None:
        raise LookupError("No hay parámetros nuevos esperando una decisión.")
    if paths.production_tuning.exists():
        paths.tuning_archive.mkdir(parents=True, exist_ok=True)
        target = paths.tuning_archive / f"tuning_produccion_{today.isoformat()}.json"
        count = 1
        while target.exists():
            count += 1
            target = paths.tuning_archive / f"tuning_produccion_{today.isoformat()}_{count}.json"
        shutil.move(str(paths.production_tuning), str(target))
    shutil.move(str(paths.tuning_candidate), str(paths.production_tuning))
    record = json.loads(paths.adjustments_json.read_text(encoding="utf-8")) if paths.adjustments_json.exists() else {"intervalo_meses": 6, "historial": []}
    record["ultimo_ajuste"] = today.isoformat()
    record.setdefault("historial", []).append({
        "fecha": today.isoformat(), "origen": "app", "tipo": summary["tipo"], "ventana": summary["ventana_ajuste"], "prueba": summary["ventana_prueba"],
        "mejora_sku_pts": summary["mejora_sku_pts"],
    })
    paths.adjustments_json.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def discard(paths: AppPaths) -> None:
    if not paths.tuning_candidate.exists():
        raise LookupError("No hay parámetros nuevos esperando una decisión.")
    paths.tuning_candidate.unlink()
