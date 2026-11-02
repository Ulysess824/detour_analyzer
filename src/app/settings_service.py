"""Ajustes screen: which forecast strategy the app uses and the cutoff day, saved in `config/ajustes_app.json`, with the plan they give for the data loaded."""

from __future__ import annotations

import json

from src.app.consumption_service import ConsumptionStore
from src.app.paths import AppPaths
from src.engine.month_state import MIN_CUTOFF_DAY, STRATEGIES, MonthState, StrategyPlan, detect_month_state, resolve_strategy

DEFAULTS = {"estrategia": "automatica", "corte_manual": None}

# Test 2025-10..2026-09 (12 months, 24,407 SKU-months), SKU / planta / total accuracy in percent: README. The nowcast figure is from nowcast_input_ml_mean.py
# (SKU level only).
OPTIONS = [
    {"id": "automatica", "nombre": "Automática", "mes_en_curso": False, "recomendada": True, "acierto": None,
     "texto": "Elige según los datos: con el último mes completo pronostica el mes siguiente; con el mes en curso lo completa con la media diaria de los días vistos y pronostica dos meses adelante."},
    {"id": "h1", "nombre": "Un mes de anticipación", "mes_en_curso": False, "recomendada": False, "acierto": {"sku": 71.0, "planta": 91.8, "total": 97.5},
     "texto": "Pronostica el mes siguiente al último mes completo. Es la más precisa, pero necesita el mes anterior terminado."},
    {"id": "nowcast_media", "nombre": "Dos meses, mes en curso completado con la media", "mes_en_curso": True, "recomendada": False, "acierto": {"sku": 68.1, "planta": None, "total": None},
     "texto": "Completa el mes en curso con la media diaria de los días vistos y lo usa como si fuera un mes observado. El mes completado es una estimación."},
    {"id": "iterada", "nombre": "Dos meses, iterada (no recomendada)", "mes_en_curso": False, "recomendada": False, "acierto": {"sku": 65.4, "planta": 89.8, "total": 94.4},
     "texto": "Pronostica un mes adelante y usa ese pronóstico como dato para el siguiente."},
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
    r"""Validate and write the settings."""
    settings = validate(estrategia, corte_manual)
    paths.settings_json.parent.mkdir(parents=True, exist_ok=True)
    paths.settings_json.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    return settings


def month_state(store: ConsumptionStore) -> MonthState:
    return detect_month_state(store.table)


def current_plan(store: ConsumptionStore, settings: dict) -> StrategyPlan:
    r"""The strategy that would run now with these settings on the data loaded (ValueError when the cutoff does not fit the data)."""
    state = month_state(store)
    try:
        return resolve_strategy(state, settings["estrategia"], settings["corte_manual"])
    except ValueError as error:  # the only way to get here with validated settings is a cutoff after the last record (the data changed)
        raise ValueError(f"El día de corte ({settings['corte_manual']}) no puede ser posterior al último registro de los datos (día {state.cutoff_day}, {state.last_date.date()}). "
                         "Cámbialo en Ajustes o vuelve al corte automático.") from error


def _plan_record(plan: StrategyPlan) -> dict:
    return {"estrategia": plan.strategy, "corte": plan.cutoff_day, "mes_origen": plan.origin, "mes_pronosticado": plan.target, "avisos": plan.notes}


def plan_summary(paths: AppPaths, store: ConsumptionStore) -> dict | None:
    r"""The plan for the status of the app: None without data, with `error` set (and no strategy) when the saved cutoff does not fit the data."""
    if not paths.consumption_csv.exists():
        return None
    try:
        return {**_plan_record(current_plan(store, load(paths))), "error": None}
    except ValueError as error:
        return {"estrategia": None, "corte": None, "mes_origen": None, "mes_pronosticado": None, "avisos": [], "error": str(error)}


def describe(paths: AppPaths, store: ConsumptionStore) -> dict:
    r"""What the Ajustes screen shows: the saved settings, the options with their accuracy, the state of the data and the plan they give."""
    settings = load(paths)
    out = {"ajustes": settings, "opciones": OPTIONS, "corte_minimo": MIN_CUTOFF_DAY, "estado": None, "plan": None, "error": None}
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
