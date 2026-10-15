"""
Nowcasting month benchmark: forecast the rest of the month in progress from its first days, with four models.

At a cutoff (e.g. the 15th) the days after it are unknown. Every model sees the same information rule: nothing after the
cutoff. Models (all give the consumption of the days that are left, per series, in TO):

    media                      every day left equals the mean of the non-Sunday days already seen (zeros included)
    mediana                    every day left equals the median of those days
    montecarlo_historia_mes    mean of simulated scenarios; the simulator learns from the 365 days before the cutoff and the
                               days seen of the month (src/utils/montecarlo_utils.py)
    montecarlo_solo_mes        same simulator, learning from the days seen of the month in progress alone

Planta and total forecasts are sums of the series forecasts (for the Monte Carlo models, the mean of the summed scenarios,
which is the same number). Series are those alive in the previous month.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.montecarlo_utils import simulate_rest_of_month
from src.utils.panel_utils import Panel

MODELS = ["media", "mediana", "montecarlo_historia_mes", "montecarlo_solo_mes"]
MONTE_CARLO = ["montecarlo_historia_mes", "montecarlo_solo_mes"]


@dataclass
class MonthNowcast:
    month: int
    cutoff_day: int
    series: np.ndarray  # positions of the series alive in the previous month
    planta: np.ndarray  # planta code of each of them
    first: np.ndarray  # real consumption of the first days
    real_rest: np.ndarray  # real consumption of the days left (known only afterwards, for scoring)
    forecasts: dict[str, np.ndarray]  # model -> forecast of the rest, per series
    scenarios: dict[str, np.ndarray]  # Monte Carlo model -> (draws, n_series)


def seen_days(S: Panel, month: int, cutoff_day: int, series: np.ndarray) -> tuple[np.ndarray, int]:
    r"""Daily consumption of the non-Sunday days already seen of `month` (series x days, zeros included) and the non-Sunday days left."""
    start = int(S.month_first_date[month])
    cutoff, end = start + cutoff_day, start + int(S.calendar_days[month])
    weekday = np.asarray(S.dates[start:cutoff].dayofweek)
    days = S.daily[series, start:cutoff][:, weekday != 6].astype(float)
    left = int((np.asarray(S.dates[cutoff:end].dayofweek) != 6).sum())
    return days, left


def forecast_media(days: np.ndarray, left: int) -> np.ndarray:
    r"""Mean of the days seen times the days left."""
    return days.mean(axis=1) * left


def forecast_mediana(days: np.ndarray, left: int) -> np.ndarray:
    r"""Median of the days seen times the days left (the median of an intermittent series is often 0)."""
    return np.median(days, axis=1) * left


def nowcast_month(S: Panel, month: int, cutoff_day: int = 15, draws: int = 200, seed: int = 0) -> MonthNowcast:
    r"""Run the four models for `month` with only the first `cutoff_day` days known; the real rest is kept for scoring."""
    series = np.where(~np.isnan(S.monthly[:, month - 1]))[0]
    start = int(S.month_first_date[month])
    cutoff, end = start + cutoff_day, start + int(S.calendar_days[month])
    days, left = seen_days(S, month, cutoff_day, series)

    scenarios = {
        "montecarlo_historia_mes": simulate_rest_of_month(S, month, cutoff_day, draws, seed)[:, series],
        "montecarlo_solo_mes": simulate_rest_of_month(S, month, cutoff_day, draws, seed, lookback=cutoff_day, shrink=0.0)[:, series],
    }
    forecasts = {"media": forecast_media(days, left), "mediana": forecast_mediana(days, left)}
    forecasts.update({name: sim.mean(axis=0) for name, sim in scenarios.items()})
    return MonthNowcast(
        month=month,
        cutoff_day=cutoff_day,
        series=series,
        planta=S.planta_code[series],
        first=S.daily[series, start:cutoff].sum(axis=1, dtype=float),
        real_rest=S.daily[series, cutoff:end].sum(axis=1, dtype=float),
        forecasts=forecasts,
        scenarios=scenarios,
    )


def by_planta(values: np.ndarray, planta: np.ndarray, n_plantas: int) -> np.ndarray:
    r"""Sum an array (n_series,) into (n_plantas,)."""
    out = np.zeros(n_plantas)
    np.add.at(out, planta, values)
    return out


def accuracy(real: np.ndarray, forecast: np.ndarray) -> float:
    r"""1 - WAPE = 1 - sum|real - forecast| / sum(real)."""
    return float(1.0 - np.abs(real - forecast).sum() / real.sum())


def level_scores(nc: MonthNowcast, n_plantas: int) -> pd.DataFrame:
    r"""
    One row per model with the accuracy at SKU level (all series pooled), at planta level (all plantas pooled) and at total
    level, the bias of the total (forecast / real - 1) and, for the Monte Carlo models, how often the real total of the rest
    falls inside the 10-90% band of the summed scenarios.
    """
    real_p = by_planta(nc.real_rest, nc.planta, n_plantas)
    rows = {}
    for model in MODELS:
        f = nc.forecasts[model]
        f_p = by_planta(f, nc.planta, n_plantas)
        row = {
            "acierto_SKU": accuracy(nc.real_rest, f),
            "acierto_planta": accuracy(real_p, f_p),
            "acierto_total": 1.0 - abs(f.sum() - nc.real_rest.sum()) / nc.real_rest.sum(),
            "sesgo_total": f.sum() / nc.real_rest.sum() - 1.0,
            "pronostico_resto_TO": f.sum(),
        }
        if model in nc.scenarios:
            totals = nc.scenarios[model].sum(axis=1)
            lo, hi = np.percentile(totals, [10, 90])
            row["real_en_banda_total"] = float(lo <= nc.real_rest.sum() <= hi)
        rows[model] = row
    return pd.DataFrame(rows).T


def planta_table(nc: MonthNowcast, plantas: list[str]) -> pd.DataFrame:
    r"""Per planta: real rest and, for every model, the accuracy of the planta total and the accuracy of its SKUs pooled."""
    rows = []
    for code, name in enumerate(plantas):
        mine = nc.planta == code
        if not mine.any() or nc.real_rest[mine].sum() <= 0:
            continue
        real = nc.real_rest[mine]
        row = {"planta": name, "series": int(mine.sum()), "real_resto_TO": real.sum()}
        for model in MODELS:
            f = nc.forecasts[model][mine]
            row[f"planta:{model}"] = 1.0 - abs(f.sum() - real.sum()) / real.sum()
            row[f"SKU:{model}"] = accuracy(real, f)
        rows.append(row)
    return pd.DataFrame(rows).set_index("planta")


def ranking(scores: pd.Series) -> pd.Series:
    r"""Position of every model (1 = highest accuracy); ties share the best position."""
    return scores.rank(ascending=False, method="min").astype(int)
