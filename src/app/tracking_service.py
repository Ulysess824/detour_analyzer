"""Tracking for the Seguimiento screen: the history of accuracy, the detail of one month, the biggest misses and the measuring of a closed month."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.app.consumption_service import ConsumptionStore
from src.app.forecast_service import list_forecasts, load_forecast
from src.app.paths import AppPaths
from src.engine.db_utils import read_tracking_csv
from src.evaluation.tracking_run import is_backtest, save_tracking, track_saved_forecast
from src.evaluation.tracking_utils import MODELS, THRESHOLDS, THRESHOLDS_TWO_MONTHS, coverage_ok, forecast_coverage

LEVELS = ["sku", "planta", "total"]
STRATEGY_ORDER = ["h1", "directa", "parcial", "nowcast_media", "iterada"]
TOP_ERRORS = 15


def _pct(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value) * 100, 1)


def _num(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 1)


def tracking_table(paths: AppPaths) -> pd.DataFrame:
    return read_tracking_csv(paths.tracking_csv) if paths.tracking_csv.exists() else pd.DataFrame()


def _production(table: pd.DataFrame) -> pd.DataFrame:
    r"""The rows of forecasts that were really made (not the historical tests)."""
    return table[~table["run_id"].map(is_backtest)] if len(table) else table


def tracked_pairs(paths: AppPaths) -> set[tuple[str, str]]:
    r"""(month, strategy) of the saved forecasts that already have a production tracking (the backtests do not count)."""
    production = _production(tracking_table(paths))
    return set(zip(production["mes"], production["estrategia"])) if len(production) else set()


def belongs_to_data(paths: AppPaths, store: ConsumptionStore, month: str, strategy: str = "h1") -> bool:
    r"""Whether the series of the saved forecast of `month` and `strategy` exist in the loaded data (a forecast made with other data cannot be measured)."""
    return coverage_ok(*forecast_coverage(store.panel, load_forecast(paths, month, strategy)))


def to_measure(paths: AppPaths, store: ConsumptionStore) -> list[dict]:
    r"""The saved forecasts (month and strategy) of these data whose real month has already closed in the data and that were not measured yet."""
    done = tracked_pairs(paths)
    return [{"mes": f["mes"], "estrategia": f["estrategia"]} for f in list_forecasts(paths)
            if store.is_closed(f["mes"]) and (f["mes"], f["estrategia"]) not in done and belongs_to_data(paths, store, f["mes"], f["estrategia"])]


def _pooled(table: pd.DataFrame, run_id: str, month: str) -> dict:
    scores = table[(table["run_id"] == run_id) & (table["mes"] == month) & (table["modelo"] == "ml_mean") & (table["planta"] == "")].set_index("nivel")["acierto"]
    return {level: _pct(scores.get(level)) for level in LEVELS}


def _average(rows: list[dict]) -> dict | None:
    r"""Mean of the monthly accuracies of a strategy (a month counts once), None when there are no months."""
    if not rows:
        return None
    mean = lambda level: _pct(np.nanmean([np.nan if r[level] is None else r[level] / 100 for r in rows]) if any(r[level] is not None for r in rows) else None)  # noqa: E731
    return {"meses": len(rows), **{level: mean(level) for level in LEVELS}}


def history(paths: AppPaths, store: ConsumptionStore | None) -> dict:
    r"""
    Accuracy of `ml_mean` by month, level and strategy, split into the historical tests (backtest) and the production months, the alert thresholds and a
    comparison of the strategies. A (month, strategy) that was measured twice keeps its newest run.
    """
    table = tracking_table(paths)
    out = {"umbrales": {k: round(v * 100, 1) for k, v in THRESHOLDS.items()}, "umbrales_dos_meses": {k: round(v * 100, 1) for k, v in THRESHOLDS_TWO_MONTHS.items()},
           "backtest": [], "produccion": [], "por_medir": to_measure(paths, store) if store else [], "comparacion": []}
    if table.empty:
        return out
    ml = table[(table["modelo"] == "ml_mean") & (table["planta"] == "") & table["nivel"].isin(LEVELS)]
    seen: dict[tuple, str] = {}
    for (run, month), block in ml.groupby(["run_id", "mes"]):
        strategy = block["estrategia"].iloc[0]
        if not is_backtest(run):  # the newest run of the (month, strategy)
            key = (month, strategy)
            if key in seen and ml[(ml["run_id"] == seen[key]) & (ml["mes"] == month)]["creado_en"].iloc[0] >= block["creado_en"].iloc[0]:
                continue
            seen[key] = run
        alerts = sorted({c for cell in table[(table["run_id"] == run) & (table["mes"] == month)]["alerta"] for c in str(cell).split(";") if c})
        corte = block["corte"].iloc[0]
        row = {"mes": month, "estrategia": strategy, "corte": None if pd.isna(corte) else int(corte), **_pooled(table, run, month), "alertas": alerts, "_run": run}
        out["backtest" if is_backtest(run) else "produccion"].append(row)
    production = {(r["mes"], r["estrategia"]): r["_run"] for r in out["produccion"]}
    out["produccion"] = [r for r in out["produccion"] if production[(r["mes"], r["estrategia"])] == r["_run"]]
    for key in ("backtest", "produccion"):
        out[key].sort(key=lambda r: (r["mes"], STRATEGY_ORDER.index(r["estrategia"]) if r["estrategia"] in STRATEGY_ORDER else 99))
        for r in out[key]:
            r.pop("_run", None)
    strategies = sorted({r["estrategia"] for key in ("backtest", "produccion") for r in out[key]}, key=lambda x: STRATEGY_ORDER.index(x) if x in STRATEGY_ORDER else 99)
    out["comparacion"] = [{"estrategia": name, "produccion": _average([r for r in out["produccion"] if r["estrategia"] == name]),
                           "prueba_historica": _average([r for r in out["backtest"] if r["estrategia"] == name])} for name in strategies]
    return out


def top_errors(paths: AppPaths, store: ConsumptionStore, month: str, strategy: str = "h1", n: int = TOP_ERRORS) -> list[dict]:
    r"""The series where the saved forecast of a closed month and strategy missed most in tonnes (empty when there is no forecast file or the month is open)."""
    if not paths.forecast_csv(month, strategy).exists() or not store.is_closed(month):
        return []
    forecast = load_forecast(paths, month, strategy)
    real = store.month_values(month)
    forecast["real"] = [real.get(key, 0.0) for key in zip(forecast["planta"], forecast["sku"])]
    forecast["error"] = forecast["ml_mean"] - forecast["real"]
    worst = forecast.reindex(forecast["error"].abs().sort_values(ascending=False).index).head(n)
    return [{"planta": r["planta"], "sku": r["sku"], "real_TO": _num(r["real"]), "pronostico_TO": _num(r["ml_mean"]), "error_TO": _num(r["error"])} for _, r in worst.iterrows()]


def detail(paths: AppPaths, store: ConsumptionStore | None, month: str | None, origin: str | None = None, strategy: str | None = None) -> dict:
    r"""
    One tracked month and strategy: every model by level, the plantas of `ml_mean` from the worst, the alerts, the biggest misses and how the other
    strategies did in the same month. `origin` is produccion or backtest; `strategy` defaults to the one-month forecast when it was tracked.
    """
    table = tracking_table(paths)
    if table.empty:
        raise LookupError("Todavía no hay seguimiento. Cuando cierre el mes pronosticado, mídelo aquí.")
    if month is None:
        production = _production(table)
        month = (production if len(production) else table)["mes"].max()
    rows = table[table["mes"] == month]
    if rows.empty:
        raise LookupError(f"No hay seguimiento de {month}.")
    has_production = rows["run_id"].map(lambda r: not is_backtest(r)).any()
    origin = origin or ("produccion" if has_production else "backtest")
    rows = rows[rows["run_id"].map(lambda r: not is_backtest(r))] if origin == "produccion" else rows[rows["run_id"].map(is_backtest)]
    if rows.empty:
        raise LookupError(f"No hay seguimiento de {month} con origen {origin}.")
    available = sorted(set(rows["estrategia"]), key=lambda x: STRATEGY_ORDER.index(x) if x in STRATEGY_ORDER else 99)
    strategy = strategy or ("h1" if "h1" in available else available[0])
    mine = rows[rows["estrategia"] == strategy]
    if mine.empty:
        raise LookupError(f"No hay seguimiento de {month} con la estrategia {strategy}.")
    newest = mine.loc[mine["creado_en"].idxmax(), "run_id"]  # one run: the newest stored for the month and strategy
    chosen = mine[mine["run_id"] == newest]
    pooled = chosen[chosen["planta"] == ""]
    models = []
    for model in MODELS:
        by_level = pooled[pooled["modelo"] == model].set_index("nivel")
        if by_level.empty:
            continue
        models.append({"modelo": model, **{lv: _pct(by_level["acierto"].get(lv)) for lv in LEVELS}, "sesgo_total_pct": _num(by_level["bias_pct"].get("total"))})
    total = pooled[(pooled["modelo"] == "ml_mean") & (pooled["nivel"] == "total")]
    plantas = chosen[(chosen["modelo"] == "ml_mean") & (chosen["nivel"] == "planta") & (chosen["planta"] != "")]
    plantas = [{"planta": r["planta"], "acierto": _pct(r["acierto"]), "real_TO": _num(r["real_TO"]), "pronostico_TO": _num(r["pronostico_TO"]), "sesgo_pct": _num(r["bias_pct"])}
               for _, r in plantas.sort_values("acierto").iterrows()]
    flagged = chosen[(chosen["modelo"] == "ml_mean") & (chosen["alerta"] != "")]
    alerts = [{"codigo": c, "nivel": r["nivel"], "planta": r["planta"]} for _, r in flagged.iterrows() for c in r["alerta"].split(";")]
    review_below = (THRESHOLDS if strategy == "h1" else THRESHOLDS_TWO_MONTHS)["planta_revisar"] * 100
    informative = [{"codigo": "planta_revisar", "nivel": "planta", "planta": p["planta"]} for p in plantas if p["acierto"] is not None and p["acierto"] < review_below]
    missing = chosen[chosen["nivel"] == "sin_pronostico"]
    cutoff = chosen["corte"].iloc[0]
    others = [{"estrategia": name, **_pooled(table, rows[rows["estrategia"] == name].loc[rows[rows["estrategia"] == name]["creado_en"].idxmax(), "run_id"], month)}
              for name in available if name != strategy]
    return {
        "mes": month,
        "origen": origin,
        "estrategia": strategy,
        "corte": None if pd.isna(cutoff) else int(cutoff),
        "estrategias": available,
        "otras_estrategias": others,
        "disponibles": sorted(set(table["mes"]), reverse=True),
        "modelos": models,
        "total": None if total.empty else {"real_TO": _num(total["real_TO"].iloc[0]), "pronostico_TO": _num(total["pronostico_TO"].iloc[0]), "sesgo_pct": _num(total["bias_pct"].iloc[0])},
        "plantas": plantas,
        "alertas": alerts or informative,
        "sin_pronostico": None if missing.empty or not missing["n"].iloc[0] else {"series": int(missing["n"].iloc[0]), "real_TO": _num(missing["real_TO"].iloc[0])},
        "errores": top_errors(paths, store, month, strategy) if origin == "produccion" and store is not None else [],
    }


def measure(paths: AppPaths, store: ConsumptionStore, month: str, strategy: str = "h1") -> dict:
    r"""Score the saved forecast of a closed month and strategy against the real and store it (history csv, and the database when there is one); returns the new detail."""
    if not paths.forecast_csv(month, strategy).exists():
        raise ValueError(f"No hay un pronóstico guardado para {month} con la estrategia {strategy}.")
    if not store.is_closed(month):
        raise ValueError(f"El mes {month} todavía no está completo en los datos. Carga el mes completo en la sección Datos.")
    found, total = forecast_coverage(store.panel, load_forecast(paths, month, strategy))
    if not coverage_ok(found, total):
        raise ValueError(f"El pronóstico de {month} no corresponde a los datos cargados: solo {found} de sus {total} series existen en ellos. "
                         "Probablemente se generó con otros datos, así que no se puede medir.")
    try:
        run = track_saved_forecast(paths.forecast_csv(month, strategy), store.table, store.panel)
    except SystemExit as stop:  # the engine stops with a message
        raise ValueError(str(stop)) from None
    save_tracking([run], paths.tracking_csv, paths.database if paths.database.exists() else None)
    return {"corrida": run.run_id, "estrategia": run.strategy, "alertas": int(run.fired), "detalle": detail(paths, store, month, "produccion", run.strategy)}
