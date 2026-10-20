"""
Tracking of a forecast against the real once its month has closed.

The forecast saved for a month (`results/forecast_<month>.csv`) is compared with the consumption that really happened: accuracy (1 - WAPE) and
bias at SKU level (the forecast series pooled), planta level (bottom-up planta totals, pooled and one by one) and total level, for every model.
`bias_pct` follows the project rule (actual - forecast over actual, so positive means the forecast was too low). Alerts compare `ml_mean` with
thresholds taken from the 12 backtest months (results/member_forecasts.csv), where none of them fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.data_checks import incomplete_last_month_message
from src.engine.db_utils import TRACKING_COLUMNS, now_utc, read_tracking_csv
from src.engine.panel_utils import Panel
from src.evaluation.metrics_utils import score

MODELS = ["ml_mean", "lgbm", "xgb", "rf", "naive"]
# strict "below" thresholds; observed over the 12 backtest months (ml_mean): SKU 66.2-74.7%, planta 87.3-95.3%, total 91.2-99.6%, a planta-month below 60% in 3 of 180
THRESHOLDS = {"sku": 0.64, "planta": 0.85, "total": 0.90, "planta_revisar": 0.60}
ALERT_TEXT = {
    "sku_bajo": "SKU accuracy of ml_mean is below the threshold",
    "planta_bajo": "planta accuracy of ml_mean is below the threshold",
    "total_bajo": "total accuracy of ml_mean is below the threshold",
    "peor_que_naive": "ml_mean is worse than the naive at SKU level",
    "planta_revisar": "a planta with accuracy of ml_mean below the review threshold",
}


def check_month_closed(df: pd.DataFrame, month: str) -> None:
    r"""Stop unless the consumption table covers `month` completely (it is in the data and its last working days are there)."""
    period = pd.Period(month, freq="M")
    inside = df[df["fecha"] <= period.end_time]
    if inside.empty or inside["fecha"].max() < period.start_time:
        raise SystemExit(f"The month {month} is not in the data yet: the table ends on {df['fecha'].max().date()}. Add the data of the month first.")
    message = incomplete_last_month_message(inside)
    if message:
        raise SystemExit(message.replace("The last month", "The month"))


def _finite(value: float) -> float:
    r"""NaN for a ratio that is not a number: with no real consumption WAPE and bias are undefined, not infinite."""
    return float(value) if np.isfinite(value) else np.nan


def _row(month: str, nivel: str, planta: str, modelo: str, y: np.ndarray, p: np.ndarray) -> dict:
    s = score(y, p)
    return {
        "mes": month, "nivel": nivel, "planta": planta, "modelo": modelo, "acierto": _finite(s["accuracy"]), "wape": _finite(s["wape"]),
        "bias_pct": _finite(s["bias_pct"]), "n": s["n"], "real_TO": float(y.sum()), "pronostico_TO": float(p.sum()),
    }


MIN_COVERAGE = 0.5  # a forecast made from these data has all its series in them; with fewer than half it comes from other data


def forecast_coverage(S: Panel, forecast: pd.DataFrame) -> tuple[int, int]:
    r"""(series of the forecast that exist in the panel, series in the forecast)."""
    known = set(S.keys)
    return sum((p, s) in known for p, s in zip(forecast["planta"], forecast["sku"])), len(forecast)


def coverage_ok(found: int, total: int) -> bool:
    return total > 0 and found / total >= MIN_COVERAGE


def check_forecast_matches(S: Panel, forecast: pd.DataFrame, month: str) -> None:
    r"""Stop when the forecast does not belong to the data: scoring it would compare series that do not exist with a real of zero."""
    found, total = forecast_coverage(S, forecast)
    if not coverage_ok(found, total):
        raise SystemExit(f"The forecast of {month} does not belong to these data: only {found} of its {total} series exist in them. It was probably made with other data.")


def month_scores(S: Panel, forecast: pd.DataFrame, month: str, models: list[str] = MODELS) -> pd.DataFrame:
    r"""
    Scores of a closed month, one row per level, planta and model.

    `forecast` has planta, sku and one column per model. Levels: `sku` (the forecast series pooled), `planta` (all plantas pooled, `planta` = ''),
    `planta` with the planta code (one planta alone) and `total`. The volume of series that consumed in the month but have no forecast is reported
    in one `sin_pronostico` row. `S` must contain the month.
    """
    t = S.labels.index(month)
    position = {key: i for i, key in enumerate(S.keys)}
    index = np.array([position.get((p, s), -1) for p, s in zip(forecast["planta"], forecast["sku"])])
    real = np.where(index >= 0, np.nan_to_num(S.monthly[np.maximum(index, 0), t]), 0.0)
    planta_code = {name: i for i, name in enumerate(S.plantas)}
    codes = np.array([planta_code.get(p, -1) for p in forecast["planta"]])
    rows = []
    for model in [m for m in models if m in forecast.columns]:
        p = forecast[model].to_numpy(dtype=float)
        bottom_up = np.array([p[codes == c].sum() for c in range(len(S.plantas))])
        rows.append(_row(month, "sku", "", model, real, p))
        rows.append(_row(month, "planta", "", model, S.planta_monthly[:, t], bottom_up))
        rows.append(_row(month, "total", "", model, np.array([S.total_monthly[t]]), np.array([p.sum()])))
        for c, name in enumerate(S.plantas):
            if S.planta_monthly[c, t] > 0:
                rows.append(_row(month, "planta", name, model, S.planta_monthly[c:c + 1, t], bottom_up[c:c + 1]))
    covered = set(index[index >= 0])
    alive = np.where(np.nan_to_num(S.monthly[:, t]) > 0)[0]
    missing = [i for i in alive if i not in covered]
    rows.append({"mes": month, "nivel": "sin_pronostico", "planta": "", "modelo": "-", "acierto": np.nan, "wape": np.nan, "bias_pct": np.nan,
                 "n": len(missing), "real_TO": float(np.nan_to_num(S.monthly[missing, t]).sum()) if missing else 0.0, "pronostico_TO": 0.0})
    return pd.DataFrame(rows)


def find_alerts(scores: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    r"""Alerts of `ml_mean` in one month: `tipo` is 'alerta' or 'info' (the plantas to review). Strictly below the threshold fires; equal does not."""
    th = {**THRESHOLDS, **(thresholds or {})}
    ml, found = scores[scores["modelo"] == "ml_mean"], []

    def add(codigo, tipo, nivel, planta, valor, umbral):
        found.append({"codigo": codigo, "tipo": tipo, "nivel": nivel, "planta": planta, "valor": valor, "umbral": umbral, "texto": ALERT_TEXT[codigo]})

    for nivel, code, key in (("sku", "sku_bajo", "sku"), ("planta", "planta_bajo", "planta"), ("total", "total_bajo", "total")):
        row = ml[(ml["nivel"] == nivel) & (ml["planta"] == "")]
        if len(row) and row.iloc[0]["acierto"] < th[key]:
            add(code, "alerta", nivel, "", row.iloc[0]["acierto"], th[key])
    naive = scores[(scores["modelo"] == "naive") & (scores["nivel"] == "sku")]
    sku = ml[ml["nivel"] == "sku"]
    if len(naive) and len(sku) and sku.iloc[0]["acierto"] < naive.iloc[0]["acierto"]:
        add("peor_que_naive", "alerta", "sku", "", sku.iloc[0]["acierto"], naive.iloc[0]["acierto"])
    for _, row in ml[(ml["nivel"] == "planta") & (ml["planta"] != "")].iterrows():
        if row["acierto"] < th["planta_revisar"]:
            add("planta_revisar", "info", "planta", row["planta"], row["acierto"], th["planta_revisar"])
    return pd.DataFrame(found, columns=["codigo", "tipo", "nivel", "planta", "valor", "umbral", "texto"])


def tracking_rows(run_id: str, scores: pd.DataFrame, alerts: pd.DataFrame, created_at: str | None = None) -> pd.DataFrame:
    r"""The scores as rows for storage: the run, the alert code on the `ml_mean` row it refers to, and the time."""
    if np.isinf(scores.select_dtypes("number").to_numpy(dtype=float)).any():
        raise ValueError("The scores have infinite values; they would break the history file and its JSON.")
    rows = scores.copy()
    rows.insert(0, "run_id", run_id)
    rows["alerta"] = ""
    for _, a in alerts.iterrows():
        hit = (rows["mes"] == scores["mes"].iloc[0]) & (rows["modelo"] == "ml_mean") & (rows["nivel"] == a["nivel"]) & (rows["planta"] == a["planta"])
        rows.loc[hit, "alerta"] = rows.loc[hit, "alerta"].map(lambda current: f"{current};{a['codigo']}" if current else a["codigo"])
    rows["creado_en"] = created_at or now_utc()
    return rows[TRACKING_COLUMNS]


def write_tracking_csv(path, rows: pd.DataFrame) -> None:
    r"""Add `rows` to the history file, replacing the rows of the same (run_id, mes); sorted so the file changes little between runs."""
    from pathlib import Path

    path = Path(path)
    if path.exists():
        old = read_tracking_csv(path)
        keys = set(rows[["run_id", "mes"]].itertuples(index=False, name=None))
        old = old[[(r, m) not in keys for r, m in zip(old["run_id"], old["mes"])]]
        rows = pd.concat([old, rows], ignore_index=True)
    rows = rows.sort_values(["mes", "run_id", "nivel", "planta", "modelo"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False, float_format="%.6f")


def backtest_forecasts(S: Panel, cache: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r"""The one-month-ahead backtest cache (results/member_forecasts.csv) as one forecast table per target month."""
    out = {}
    for t, block in cache.groupby("t"):
        keys = np.asarray(S.keys, dtype=object)[block["series"].to_numpy()]
        table = pd.DataFrame({"planta": keys[:, 0], "sku": keys[:, 1], "mes": S.labels[t]})
        for kind in ("lgbm", "xgb", "rf"):
            table[kind] = block[f"p_{kind}"].to_numpy()
        table["ml_mean"] = block[["p_lgbm", "p_xgb", "p_rf"]].mean(axis=1).to_numpy()
        table["naive"] = block["p_naive"].to_numpy()
        out[S.labels[t]] = table
    return out
