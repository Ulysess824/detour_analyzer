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
from src.experiments.weekday_profile import weekday_profile

WEEK_DAYS = 6  # working days of a week (Monday to Saturday); `weekdays=5` runs use Monday to Friday
MIN_HISTORY_DAYS = 30  # origins with fewer working days of history are skipped


def daily_series(df: pd.DataFrame, planta: str, sku: str, weekdays: int = 6) -> pd.Series:
    r"""
    Consumption (TO) per working day of one series, indexed by date, from its first consumption day to the last working day of the
    data. `df` is the whole master (its last date ends the series). Working days are Monday to Saturday (`weekdays=6`) or Monday to
    Friday (`weekdays=5`); consumption on a day off goes to the next Monday.
    """
    rows = df[(df["planta"] == planta) & (df["sku"] == sku)]
    if rows.empty:
        raise ValueError(f"no rows for planta {planta!r}, SKU {sku!r}")
    daily = rows.groupby("fecha")["consumo"].sum()
    last = df["fecha"].max().normalize()
    off = daily.index.dayofweek >= weekdays
    to_monday = pd.to_timedelta(7 - daily.index[off].dayofweek, unit="D")
    folded = pd.concat([daily[~off], pd.Series(daily[off].to_numpy(), index=daily.index[off] + to_monday)])
    folded = folded.groupby(level=0).sum()
    folded = folded[folded.index <= last]  # a day off at the end of the data has no Monday to join
    days = pd.date_range(folded.index.min(), last, freq="D")
    days = days[days.dayofweek < weekdays]
    return folded.reindex(days, fill_value=0.0).rename(f"{planta}|{sku}")


def _flat_totals(series: pd.Series, pos: int, days: int, draws: int, seed: int, windows: tuple[int, ...],
                 profile_scale: float | None = None) -> dict[str, float]:
    r"""
    Forecast of every model at the origin `pos` (horizon 1, flat) times the number of days it covers. With `profile_scale` (the sum
    of the weekday factors of those days) every model also gets a `<model>_dow` total.
    """
    rows = origin_forecasts(series.to_numpy(dtype=float)[: pos + 1], 1, draws, seed * 1_000_003 + pos * 10 + 1, windows=windows)
    out = {r["model"]: r["forecast"] * days for r in rows}
    if profile_scale is not None:
        out.update({r["model"] + "_dow": r["forecast"] * profile_scale for r in rows})
    return out


def week_comparison(series: pd.Series, real_weekly: pd.Series, test_start: str, draws: int = ci.DRAWS, seed: int = 0,
                    windows: tuple[int, ...] = (6, 24), weekdays: int = WEEK_DAYS) -> pd.DataFrame:
    r"""
    One row per week with Monday >= test_start: real total (`real_weekly`, indexed by Monday, Monday to Sunday) and every model's
    flat daily forecast at the last working day before the Monday times the working days of the week (`weekdays`). A weekday profile
    does not change a week's total, so there are no `_dow` columns here.
    """
    rows = []
    for monday in real_weekly.index[real_weekly.index >= pd.Timestamp(test_start)]:
        pos = int(np.searchsorted(series.index, monday)) - 1  # last working day before the Monday
        if pos + 1 < MIN_HISTORY_DAYS:
            continue
        rows.append({"week": monday, "origin": series.index[pos], "real": float(real_weekly[monday]),
                     **_flat_totals(series, pos, weekdays, draws, seed, windows)})
    return pd.DataFrame(rows)


def month_comparison(series: pd.Series, real_monthly: pd.Series, ml_mean: pd.Series, draws: int = ci.DRAWS, seed: int = 0,
                     windows: tuple[int, ...] = (6, 24), weekdays: int = 6, shrink: float | None = None) -> pd.DataFrame:
    r"""
    One row per month of `ml_mean` (indexed by Period): real total, ml_mean and every model's flat daily forecast at the last working
    day before the month times the working days of the month. ml_mean has the same information: data up to the end of the previous month.
    With `shrink`, every model also gets a `<model>_dow` total: its daily forecast times the sum of the weekday factors of the month's days.
    """
    rows = []
    for month in ml_mean.index:
        pos = int(np.searchsorted(series.index, month.start_time)) - 1
        if pos + 1 < MIN_HISTORY_DAYS:
            continue
        scale = None
        if shrink is not None:
            w = weekday_profile(series.to_numpy(dtype=float)[: pos + 1], series.index.dayofweek[: pos + 1], weekdays, shrink)
            days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
            scale = float(w[days[days.dayofweek < weekdays].dayofweek].sum())
        rows.append({"month": str(month), "origin": series.index[pos], "real": float(real_monthly[month]), "ml_mean": float(ml_mean[month]),
                     **_flat_totals(series, pos, month_business_days(month, weekdays), draws, seed, windows, scale)})
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
