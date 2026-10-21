"""Ajustes screen: which forecast strategy the app uses and the cutoff day, saved in `config/ajustes_app.json`, with the plan they give for the data loaded."""

from __future__ import annotations

import json

from src.app.consumption_service import ConsumptionStore
from src.app.paths import AppPaths
from src.engine.month_state import MIN_CUTOFF_DAY, STRATEGIES, MonthState, StrategyPlan, detect_month_state, resolve_strategy

DEFAULTS = {"estrategia": "automatica", "corte_manual": None}

# Test 2025-10..2026-09 (12 months, 24,407 SKU-months), SKU / planta / total accuracy in percent: results/curva_corte_parcial.csv and the README. The
# one-month and nowcast figures are from compare_horizon_strategies.py and nowcast_input_ml_mean.py (the nowcast one only at SKU level).
CURVE = [
    {"dia": 5, "sku": 66.8, "planta": 89.8, "total": 93.8}, {"dia": 8, "sku": 67.2, "planta": 89.9, "total": 94.2},
    {"dia": 10, "sku": 67.8, "planta": 90.5, "total": 94.8}, {"dia": 12, "sku": 68.2, "planta": 90.8, "total": 95.1},
    {"dia": 15, "sku": 68.7, "planta": 90.9, "total": 96.0}, {"dia": 18, "sku": 69.2, "planta": 91.0, "total": 95.9},
    {"dia": 20, "sku": 69.4, "planta": 91.3, "total": 96.6}, {"dia": 22, "sku": 69.6, "planta": 91.2, "total": 96.4},
    {"dia": 25, "sku": 70.2, "planta": 91.8, "total": 97.5}, {"dia": 28, "sku": 70.7, "planta": 92.1, "total": 97.6},
]
OPTIONS = [
    {"id": "automatica", "nombre": "Automática (recomendada)", "mes_en_curso": False, "recomendada": True, "acierto": None,
     "texto": f"Elige según los datos: con el último mes completo pronostica el mes siguiente; con el mes en curso usa sus primeros días desde el día {MIN_CUTOFF_DAY} y la directa antes."},
    {"id": "h1", "nombre": "Un mes de anticipación", "mes_en_curso": False, "recomendada": False, "acierto": {"sku": 71.0, "planta": 91.8, "total": 97.5},
     "texto": "Pronostica el mes siguiente al último mes completo. Es la más precisa, pero necesita el mes anterior terminado."},
    {"id": "directa", "nombre": "Dos meses, directa", "mes_en_curso": False, "recomendada": False, "acierto": {"sku": 66.1, "planta": 90.0, "total": 94.1},
     "texto": "Pronostica dos meses después del último mes completo, sin mirar el mes en curso."},
    {"id": "parcial", "nombre": "Dos meses, directa y días del mes en curso", "mes_en_curso": True, "recomendada": False, "acierto": {"sku": 68.7, "planta": 90.9, "total": 96.0},
     "texto": f"Añade el consumo de los primeros días del mes en curso. Mejora a la directa desde el día {MIN_CUTOFF_DAY}; el acierto que se muestra es con corte el día 15."},
    {"id": "nowcast_media", "nombre": "Dos meses, mes en curso completado con la media", "mes_en_curso": True, "recomendada": False, "acierto": {"sku": 68.1, "planta": None, "total": None},
     "texto": "Completa el mes en curso con la media diaria de los días vistos y lo usa como si fuera un mes observado. El mes completado es una estimación."},
    {"id": "iterada", "nombre": "Dos meses, iterada (no recomendada)", "mes_en_curso": False, "recomendada": False, "acierto": {"sku": 65.4, "planta": 89.8, "total": 94.4},
     "texto": "Pronostica un mes adelante y usa ese pronóstico como dato para el siguiente. No mejora a la directa."},
]
assert [o["id"] for o in OPTIONS] == STRATEGIES


def validate(estrategia: str, corte_manual: int | None) -> dict:
    r"""The settings as they are saved; raises ValueError (with a message for the screen) when they make no sense."""
    if estrategia not in STRATEGIES:
        raise ValueError(f"La estrategia '{estrategia}' no existe.")
    if corte_manual is not None and (isinstance(corte_manual, bool) or not isinstance(corte_manual, int) or not 1 <= corte_manual <= 31):
        raise ValueError("El día de corte debe ser un número entero entre 1 y 31.")
    return {"estrategia": estrategia, "corte_manual": corte_manual}


def load(paths: AppPaths) -> dict:
    r"""The saved settings; the defaults when the file is missing or damaged (the screen never fails because of it)."""
    try:
        record = json.loads(paths.settings_json.read_text(encoding="utf-8"))
        return validate(record.get("estrategia", DEFAULTS["estrategia"]), record.get("corte_manual"))
    except (OSError, ValueError, AttributeError):
        return dict(DEFAULTS)


def save(paths: AppPaths, estrategia: str, corte_manual: int | None) -> dict:
    settings = validate(estrategia, corte_manual)
    paths.settings_json.parent.mkdir(parents=True, exist_ok=True)
    paths.settings_json.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    return settings


def month_state(store: ConsumptionStore) -> MonthState:
    return detect_month_state(store.table)


def current_plan(store: ConsumptionStore, settings: dict) -> StrategyPlan:
    r"""The strategy that would run now with these settings on the data loaded (ValueError when the cutoff does not fit the data)."""
    return resolve_strategy(month_state(store), settings["estrategia"], settings["corte_manual"])


def _plan_record(plan: StrategyPlan) -> dict:
    return {"estrategia": plan.strategy, "corte": plan.cutoff_day, "mes_origen": plan.origin, "mes_pronosticado": plan.target, "avisos": plan.notes}


def describe(paths: AppPaths, store: ConsumptionStore) -> dict:
    r"""What the Ajustes screen shows: the saved settings, the options with their accuracy, the state of the data and the plan they give."""
    settings = load(paths)
    out = {"ajustes": settings, "opciones": OPTIONS, "curva_parcial": CURVE, "corte_minimo": MIN_CUTOFF_DAY, "estado": None, "plan": None, "error": None}
    if not paths.consumption_csv.exists():
        return out
    state = month_state(store)
    out["estado"] = {"estado": state.status, "mes": state.month, "ultimo_registro": str(state.last_date.date()), "ultimo_mes_completo": state.last_complete,
                     "dia_de_corte": state.cutoff_day, "dias_vistos": state.days_seen, "dias_restantes": state.days_left}
    try:
        out["plan"] = _plan_record(current_plan(store, settings))
    except ValueError as error:
        out["error"] = str(error)
    return out
