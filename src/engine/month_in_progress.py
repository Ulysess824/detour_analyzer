"""
The month in progress inside the panel: hide it, fill it with an estimate, complete it with the mean of its first days.

At mid-month the last complete month is `o` and the month `o + 1` has only its first days. The panel built from the data holds the
partial total of that month in `monthly`, so the forecasts hide it (`mask_panel`) and read its first days from `daily`. The nowcast
completes it as (real first days) + (mean daily consumption of the non-Sunday days seen x non-Sunday days left).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.engine.panel_utils import Panel


def mask_panel(S: Panel, last_known: int) -> Panel:
    r"""Copy of the panel in which every month after `last_known` is unknown (NaN)."""
    cut = last_known + 1
    monthly, active = S.monthly.copy(), S.active_days.copy()
    planta, total = S.planta_monthly.copy(), S.total_monthly.copy()
    monthly[:, cut:], active[:, cut:], planta[:, cut:], total[cut:] = np.nan, np.nan, np.nan, np.nan
    return dataclasses.replace(S, monthly=monthly, active_days=active, planta_monthly=planta, total_monthly=total)


def with_pseudo_month(S: Panel, month: int, series: np.ndarray, values: np.ndarray, active_days: np.ndarray) -> Panel:
    r"""Copy of the panel in which `month` holds forecasts (as if observed) for `series`; the other series stay NaN there."""
    monthly, active = S.monthly.copy(), S.active_days.copy()
    planta, total = S.planta_monthly.copy(), S.total_monthly.copy()
    monthly[series, month] = values
    active[series, month] = active_days
    planta[:, month] = np.bincount(S.planta_code[series], weights=values, minlength=len(S.plantas))
    total[month] = values.sum()
    return dataclasses.replace(S, monthly=monthly, active_days=active, planta_monthly=planta, total_monthly=total)


def complete_calendar(S: Panel) -> Panel:
    r"""
    Panel whose calendar reaches the end of the last month, with zero consumption on the days not exported yet.

    `build_panel` stops the calendar at the last record, so a month in progress looks shorter than it is (its `calendar_days`, `business_days`
    and `dates`). The nowcast needs the days left and the month-to-date projection the business days of the whole month. The added days are
    never read as consumption: every reader stops at the cutoff day.
    """
    end = pd.Period(S.labels[-1], freq="M").end_time.normalize()
    if S.dates[-1] >= end:
        return S
    extra = pd.date_range(S.dates[-1] + pd.Timedelta(days=1), end)
    calendar, business = S.calendar_days.copy(), S.business_days.copy()
    calendar[-1] += len(extra)
    business[-1] += int((extra.dayofweek != 6).sum())
    daily = np.concatenate([S.daily, np.zeros((S.daily.shape[0], len(extra)), dtype=S.daily.dtype)], axis=1)
    return dataclasses.replace(S, dates=S.dates.append(extra), daily=daily, calendar_days=calendar, business_days=business)


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


def completed_month(S: Panel, month: int, cutoff_day: int, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Monthly total and days with consumption of `month` completed from its first `cutoff_day` days, for `series`.

    total = consumption of the first days + mean of the non-Sunday days seen x non-Sunday days left;
    active days = days with consumption seen, scaled from the days seen to all the non-Sunday days of the month.
    Only the days before the cutoff are read.
    """
    start = int(S.month_first_date[month])
    cutoff = start + cutoff_day
    days, left = seen_days(S, month, cutoff_day, series)
    first = S.daily[series, start:cutoff].sum(axis=1, dtype=float)
    seen_active = (S.daily[series, start:cutoff] > 0).sum(axis=1)
    business = float(S.business_days[month])
    active = np.minimum(seen_active / max(days.shape[1], 1) * business, business)
    return first + forecast_media(days, left), active
