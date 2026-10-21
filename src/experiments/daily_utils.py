"""
Daily series of one planta and SKU over working days (Monday to Saturday), built from the daily master (data/consumos_long.csv).

A day without a row of consumption is 0 from the first consumption of the SKU onwards. Sunday consumption is added to the next Monday
(the totals are kept and nothing is known before it is observed); a Sunday that is the last day of the data has no next Monday and is dropped.
Holidays are not modelled: they are days at 0 like any other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.engine.panel_utils import build_panel
from src.evaluation.cache_utils import check_alignment
from src.experiments import croston_intervals as ci
from src.experiments.croston_backtest import month_business_days, origin_forecasts

WEEK_DAYS = 6  # working days of a week
MIN_HISTORY_DAYS = 30  # origins with fewer working days of history are skipped


def daily_series(df: pd.DataFrame, planta: str, sku: str) -> pd.Series:
    r"""
    Consumption (TO) per working day of one series, indexed by date, from its first consumption day to the last working day of the
    data. `df` is the whole master (its last date ends the series).
    """
    rows = df[(df["planta"] == planta) & (df["sku"] == sku)]
    if rows.empty:
        raise ValueError(f"no rows for planta {planta!r}, SKU {sku!r}")
    daily = rows.groupby("fecha")["consumo"].sum()
    last = df["fecha"].max().normalize()
    sunday = daily.index.dayofweek == 6
    folded = pd.concat([daily[~sunday], pd.Series(daily[sunday].to_numpy(), index=daily.index[sunday] + pd.Timedelta(days=1))])
    folded = folded.groupby(level=0).sum()
    folded = folded[folded.index <= last]  # a Sunday on the last day of the data has no Monday to join
    days = pd.date_range(folded.index.min(), last, freq="D")
    days = days[days.dayofweek != 6]
    return folded.reindex(days, fill_value=0.0).rename(f"{planta}|{sku}")


def _flat_totals(series: pd.Series, pos: int, days: int, draws: int, seed: int, windows: tuple[int, ...]) -> dict[str, float]:
    r"""Forecast of every model at the origin `pos` (horizon 1, flat) times the number of days it covers."""
    rows = origin_forecasts(series.to_numpy(dtype=float)[: pos + 1], 1, draws, seed * 1_000_003 + pos * 10 + 1, windows=windows)
    return {r["model"]: r["forecast"] * days for r in rows}


def week_comparison(series: pd.Series, real_weekly: pd.Series, test_start: str, draws: int = ci.DRAWS, seed: int = 0,
                    windows: tuple[int, ...] = (6, 24)) -> pd.DataFrame:
    r"""
    One row per week with Monday >= test_start: real total (`real_weekly`, indexed by Monday, Monday to Sunday) and every model's
    flat daily forecast at the last working day before the Monday times the 6 days of the week.
    """
    rows = []
    for monday in real_weekly.index[real_weekly.index >= pd.Timestamp(test_start)]:
        pos = int(np.searchsorted(series.index, monday)) - 1  # last working day before the Monday
        if pos + 1 < MIN_HISTORY_DAYS:
            continue
        rows.append({"week": monday, "origin": series.index[pos], "real": float(real_weekly[monday]),
                     **_flat_totals(series, pos, WEEK_DAYS, draws, seed, windows)})
    return pd.DataFrame(rows)


def month_comparison(series: pd.Series, real_monthly: pd.Series, ml_mean: pd.Series, draws: int = ci.DRAWS, seed: int = 0,
                     windows: tuple[int, ...] = (6, 24)) -> pd.DataFrame:
    r"""
    One row per month of `ml_mean` (indexed by Period): real total, ml_mean and every model's flat daily forecast at the last working
    day before the month times the working days of the month. ml_mean has the same information: data up to the end of the previous month.
    """
    rows = []
    for month in ml_mean.index:
        pos = int(np.searchsorted(series.index, month.start_time)) - 1
        if pos + 1 < MIN_HISTORY_DAYS:
            continue
        rows.append({"month": str(month), "origin": series.index[pos], "real": float(real_monthly[month]), "ml_mean": float(ml_mean[month]),
                     **_flat_totals(series, pos, month_business_days(month), draws, seed, windows)})
    return pd.DataFrame(rows)


def ml_mean_by_month(predictions: Path, planta: str, sku: str, df: pd.DataFrame, first_month: str) -> pd.Series:
    r"""ml_mean (mean of lgbm, xgb, rf) of the one-month-ahead cache for the target months from `first_month`, indexed by Period."""
    S = build_panel(df)
    res = pd.read_csv(predictions)
    check_alignment(res, S)
    series = S.keys.index((planta, sku))
    rows = res[res["series"] == series]
    months = pd.PeriodIndex([S.labels[t] for t in rows["t"]], freq="M")
    ml = pd.Series(rows[["p_lgbm", "p_xgb", "p_rf"]].mean(axis=1).to_numpy(), index=months)
    return ml[ml.index >= pd.Period(first_month, "M")]
