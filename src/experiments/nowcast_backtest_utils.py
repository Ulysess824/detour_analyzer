"""
Backtest of the mean nowcast over several months, by planta and SKU.

For each month the rest is forecast from the first days (src/experiments/now_casting_current_month.py) and the full month is
the real first days plus the forecast of the rest. The full month is also compared with two references available before the
month starts: the one-month-ahead `ml_mean` (cache of run_ensembles.py) and the previous complete month. Scores are the
accuracy 1 - WAPE at SKU, planta and total level and the bias of the total (forecast / real - 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.panel_utils import Panel
from src.experiments.now_casting_current_month import MODELS, accuracy, by_planta, forecast_media, forecast_mediana, nowcast_month, seen_days

NOWCAST_MODELS = MODELS  # media, mediana, montecarlo_historia_mes, montecarlo_solo_mes
REFERENCES = ["ml_mean_h1", "mes_anterior"]
LIGHT_MODELS = ["media", "mediana"]
TOLERANCE = 0.005  # a tie: within half a point of accuracy


def h1_reference(cache: pd.DataFrame) -> pd.DataFrame:
    r"""`ml_mean` of the one-month-ahead cache (series, t, ml_mean_h1): the mean of LightGBM, XGBoost and Random Forest."""
    out = cache[["series", "t"]].copy()
    out["ml_mean_h1"] = cache[["p_lgbm", "p_xgb", "p_rf"]].mean(axis=1).to_numpy()
    return out


def _assemble(S: Panel, month: int, series: np.ndarray, first: np.ndarray, real_rest: np.ndarray, forecasts: dict, h1: pd.DataFrame | None) -> pd.DataFrame:
    keys = np.asarray(S.keys, dtype=object)[series]
    out = pd.DataFrame(
        {
            "mes": S.labels[month],
            "planta": keys[:, 0],
            "sku": keys[:, 1],
            "series": series,
            "planta_code": S.planta_code[series],
            "real_primeros": first,
            "real_resto": real_rest,
            "real_mes": first + real_rest,
        }
    )
    for model, f in forecasts.items():
        out[f"resto_{model}"] = f
        out[f"mes_{model}"] = first + f
    out["mes_mes_anterior"] = S.monthly[series, month - 1]
    if h1 is not None:
        mine = h1[h1["t"] == month].drop_duplicates("series").set_index("series")["ml_mean_h1"]
        out["mes_ml_mean_h1"] = mine.reindex(series).to_numpy()
    return out


def month_table(S: Panel, month: int, cutoff_day: int = 15, draws: int = 200, seed: int = 0, h1: pd.DataFrame | None = None) -> pd.DataFrame:
    r"""One row per series alive in the previous month: real first days, real rest and month, and every forecast (rest and month)."""
    nc = nowcast_month(S, month, cutoff_day, draws, seed)
    return _assemble(S, month, nc.series, nc.first, nc.real_rest, nc.forecasts, h1)


def light_month_table(S: Panel, month: int, cutoff_day: int, h1: pd.DataFrame | None = None) -> pd.DataFrame:
    r"""Same table with only the models that do not simulate (media and mediana): fast, for other cutoff days."""
    series = np.where(~np.isnan(S.monthly[:, month - 1]))[0]
    start = int(S.month_first_date[month])
    cutoff, end = start + cutoff_day, start + int(S.calendar_days[month])
    days, left = seen_days(S, month, cutoff_day, series)
    forecasts = {"media": forecast_media(days, left), "mediana": forecast_mediana(days, left)}
    first = S.daily[series, start:cutoff].sum(axis=1, dtype=float)
    real_rest = S.daily[series, cutoff:end].sum(axis=1, dtype=float)
    return _assemble(S, month, series, first, real_rest, forecasts, h1)


def _pooled_accuracy(real: np.ndarray, forecast: np.ndarray) -> float:
    return float(1.0 - np.abs(real - forecast).sum() / real.sum()) if real.sum() > 0 else np.nan


def _columns(long: pd.DataFrame, target: str) -> tuple[str, dict[str, str]]:
    r"""The real column and {model: forecast column} that exist in `long` for target 'mes' or 'resto'."""
    real = f"real_{target}"
    prefix = f"{target}_"
    models = {c[len(prefix):]: c for c in long.columns if c.startswith(prefix) and c != real}
    return real, models


def global_scores(long: pd.DataFrame, n_plantas: int, target: str = "mes") -> pd.DataFrame:
    r"""
    Accuracy of every model for each month and pooled over the months ('todos'): SKU level (all series pooled), planta level
    (planta totals pooled) and total level, plus the bias of the total and the forecast and real totals (TO). The pooled
    total accuracy is 1 - sum|F - R| / sum R over the month totals.
    """
    real_col, models = _columns(long, target)
    rows = []
    for month, part in list(long.groupby("mes")) + [("todos", long)]:
        for model, col in models.items():
            ok = part[col].notna()
            sub = part[ok]
            if sub.empty:
                continue
            real, f = sub[real_col].to_numpy(), sub[col].to_numpy()
            cells = sub.assign(_r=real, _f=f).groupby(["mes", "planta_code"])[["_r", "_f"]].sum()
            totals = sub.assign(_r=real, _f=f).groupby("mes")[["_r", "_f"]].sum()
            rows.append(
                {
                    "mes": month,
                    "objetivo": target,
                    "modelo": model,
                    "acierto_SKU": _pooled_accuracy(real, f),
                    "acierto_planta": _pooled_accuracy(cells["_r"].to_numpy(), cells["_f"].to_numpy()),
                    "acierto_total": _pooled_accuracy(totals["_r"].to_numpy(), totals["_f"].to_numpy()),
                    "sesgo_total": f.sum() / real.sum() - 1.0 if real.sum() > 0 else np.nan,
                    "pronostico_TO": f.sum(),
                    "real_TO": real.sum(),
                    "series": len(sub),
                }
            )
    return pd.DataFrame(rows)


def planta_scores(long: pd.DataFrame, target: str = "mes") -> pd.DataFrame:
    r"""Per planta, month and model: planta-total accuracy, accuracy of its SKUs pooled, bias, forecast and real (TO); 'todos' pools the months."""
    real_col, models = _columns(long, target)
    rows = []
    for planta, part_p in long.groupby("planta"):
        for month, part in list(part_p.groupby("mes")) + [("todos", part_p)]:
            for model, col in models.items():
                sub = part[part[col].notna()]
                real, f = sub[real_col].to_numpy(), sub[col].to_numpy()
                if len(sub) == 0 or real.sum() <= 0:
                    continue
                cells = sub.assign(_r=real, _f=f).groupby("mes")[["_r", "_f"]].sum()
                rows.append(
                    {
                        "planta": planta,
                        "mes": month,
                        "objetivo": target,
                        "modelo": model,
                        "acierto_planta": _pooled_accuracy(cells["_r"].to_numpy(), cells["_f"].to_numpy()),
                        "acierto_SKU": _pooled_accuracy(real, f),
                        "sesgo": f.sum() / real.sum() - 1.0,
                        "pronostico_TO": f.sum(),
                        "real_TO": real.sum(),
                        "series": len(sub),
                    }
                )
    return pd.DataFrame(rows)


def series_summary(long: pd.DataFrame, target: str = "mes") -> pd.DataFrame:
    r"""Per (planta, sku) pooled over the months: real, and for every model the forecast, absolute error, accuracy and bias."""
    real_col, models = _columns(long, target)
    out = long.groupby(["planta", "sku"]).agg(meses=("mes", "nunique"), **{f"{real_col}_TO": (real_col, "sum")})
    for model, col in models.items():
        ok = long[col].notna()
        g = long[ok].assign(_f=long.loc[ok, col], _e=(long.loc[ok, col] - long.loc[ok, real_col]).abs()).groupby(["planta", "sku"])
        total_real = long[ok].groupby(["planta", "sku"])[real_col].sum()
        out[f"pron_{model}_TO"] = g["_f"].sum()
        out[f"acierto_{model}"] = (1.0 - g["_e"].sum() / total_real).where(total_real > 0)
        out[f"sesgo_{model}"] = (g["_f"].sum() / total_real - 1.0).where(total_real > 0)
    return out.reset_index()


def closer_counts(long: pd.DataFrame, a: str, b: str, target: str = "mes") -> dict:
    r"""SKU-months where model `a` is closer to the real than model `b`, `b` closer, and ties (counted for nobody)."""
    real_col, models = _columns(long, target)
    sub = long[long[models[a]].notna() & long[models[b]].notna()]
    ea, eb = (sub[models[a]] - sub[real_col]).abs().to_numpy(), (sub[models[b]] - sub[real_col]).abs().to_numpy()
    return {a: int((ea < eb).sum()), b: int((eb < ea).sum()), "empates": int((ea == eb).sum()), "total": len(sub)}


def decision_criteria(month_rest: pd.DataFrame, month_full: pd.DataFrame, planta_full: pd.DataFrame) -> pd.DataFrame:
    r"""
    The four criteria fixed before the run.

    1. media wins or ties (within half a point) with mediana and both Monte Carlo at SKU, planta and total, pooled, on the rest.
    2. media of the full month equals or beats ml_mean one month ahead at planta level and at total level in at least 6 of the months.
    3. the bias of the total of the full month is within +-5% in more than half of the months.
    4. plantas where the full month of media is worse than ml_mean one month ahead (pooled planta accuracy).
    """
    rows = []
    pooled = month_rest[month_rest["mes"] == "todos"].set_index("modelo")
    wins = {lv: bool(all(pooled.loc["media", lv] >= pooled.loc[m, lv] - TOLERANCE for m in NOWCAST_MODELS if m != "media")) for lv in ("acierto_SKU", "acierto_planta", "acierto_total")}
    rows.append({"criterio": "1. media gana o empata con mediana y Monte Carlo (resto, acumulado)", "resultado": all(wins.values()), "detalle": str(wins)})

    monthly = month_full[month_full["mes"] != "todos"]
    m = monthly[monthly["modelo"] == "media"].set_index("mes")
    r = monthly[monthly["modelo"] == "ml_mean_h1"].set_index("mes")
    n = len(m)
    planta_ok, total_ok = int((m["acierto_planta"] >= r["acierto_planta"]).sum()), int((m["acierto_total"] >= r["acierto_total"]).sum())
    need = int(np.ceil(2 * n / 3))
    rows.append({"criterio": f"2. nowcast >= ml_mean a un mes en planta y total en al menos {need} de {n} meses", "resultado": planta_ok >= need and total_ok >= need,
                 "detalle": f"planta {planta_ok}/{n}, total {total_ok}/{n}"})

    within = int((m["sesgo_total"].abs() <= 0.05).sum())
    rows.append({"criterio": "3. sesgo del total del mes dentro de +-5% en mas de la mitad de los meses", "resultado": within > n / 2, "detalle": f"{within}/{n} meses"})

    p = planta_full[(planta_full["mes"] == "todos")].pivot(index="planta", columns="modelo", values="acierto_planta")
    worse = sorted(p.index[p["media"] < p["ml_mean_h1"]])
    rows.append({"criterio": "4. plantas donde el nowcast es peor que ml_mean a un mes (mes completo, acumulado)", "resultado": None, "detalle": ", ".join(worse) or "ninguna"})
    return pd.DataFrame(rows)
