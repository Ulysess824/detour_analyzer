"""Monthly panel of consumption: one row per (planta, sku) series, one column per month."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUP_COLS = ["planta", "sku"]


def load_consumption(path) -> pd.DataFrame:
    r"""Read the long csv (planta, sku, fecha, consumo) written by transform_consumos.py."""
    return pd.read_csv(path, parse_dates=["fecha"])


@dataclass
class Panel:
    r"""
    Everything the forecasting code needs, as numpy arrays indexed by series and month.

    monthly          (n_series, n_months) monthly totals; NaN before a series first appears
    first_month      (n_series,) index of the first month of each series
    planta_code      (n_series,) integer code of the planta of each series
    plantas          planta names, in code order
    planta_monthly   (n_plantas, n_months) monthly totals per planta
    total_monthly    (n_months,) monthly total over everything
    month_of_year    (n_months,) 1..12
    labels           (n_months,) "2025-07" style labels
    business_days    (n_months,) days per month excluding Sundays
    calendar_days    (n_months,) days per month
    active_days      (n_series, n_months) days with positive consumption; NaN before the first month
    daily            (n_series, n_days) zero-filled daily consumption over the whole calendar
    dates            every calendar day covered
    month_first_date (n_months,) position in `dates` of the first day of each month
    keys             (planta, sku) of every series, in row order (the series index used everywhere)
    """

    monthly: np.ndarray
    first_month: np.ndarray
    planta_code: np.ndarray
    plantas: list[str]
    planta_monthly: np.ndarray
    total_monthly: np.ndarray
    month_of_year: np.ndarray
    labels: list[str]
    business_days: np.ndarray
    calendar_days: np.ndarray
    active_days: np.ndarray
    daily: np.ndarray
    dates: pd.DatetimeIndex
    month_first_date: np.ndarray
    keys: list[tuple[str, str]]


def build_panel(df: pd.DataFrame) -> Panel:
    r"""Build the panel; a series is NaN before it first appears and 0 in its empty months."""
    start = df["fecha"].min().to_period("M")
    month_idx = ((df["fecha"].dt.year - start.year) * 12 + (df["fecha"].dt.month - start.month)).to_numpy()
    n_months = int(month_idx.max()) + 1
    month_cols = np.arange(n_months)[None, :]

    # Monthly totals per series.
    table = (
        df.assign(m=month_idx).groupby(GROUP_COLS + ["m"])["consumo"].sum().unstack("m").reindex(columns=range(n_months))
    )
    raw = table.to_numpy(dtype=float)
    first_month = np.argmax(~np.isnan(raw), axis=1)
    monthly = np.where(month_cols >= first_month[:, None], np.nan_to_num(raw, nan=0.0), np.nan)
    monthly_zero = np.nan_to_num(monthly, nan=0.0)

    # Totals per planta and overall.
    plantas = sorted(table.index.get_level_values("planta").unique())
    planta_code = np.asarray(pd.Categorical(table.index.get_level_values("planta"), categories=plantas).codes)
    planta_monthly = np.zeros((len(plantas), n_months))
    np.add.at(planta_monthly, planta_code, monthly_zero)

    # Calendar: days per month, with and without Sundays.
    dates = pd.date_range(start.to_timestamp(), df["fecha"].max())
    date_month = ((dates.year - start.year) * 12 + (dates.month - start.month)).to_numpy()
    calendar_days = np.bincount(date_month, minlength=n_months)
    business_days = np.bincount(date_month[dates.dayofweek != 6], minlength=n_months)
    month_first_date = np.searchsorted(date_month, np.arange(n_months))

    # Zero-filled daily matrix and the number of active days per month.
    series_idx = table.index.get_indexer(pd.MultiIndex.from_frame(df[GROUP_COLS]))
    date_idx = dates.get_indexer(df["fecha"])
    value = df["consumo"].to_numpy()
    daily = np.zeros((len(table), len(dates)), dtype=np.float32)
    np.add.at(daily, (series_idx, date_idx), value.astype(np.float32))
    active = np.zeros((len(table), n_months))
    np.add.at(active, (series_idx, month_idx), (value > 0).astype(float))
    active_days = np.where(month_cols >= first_month[:, None], active, np.nan)

    if not np.allclose(np.add.reduceat(daily, month_first_date, axis=1), monthly_zero, rtol=1e-4, atol=0.05):
        raise ValueError("daily matrix does not add up to the monthly totals")

    return Panel(
        monthly=monthly,
        first_month=first_month,
        planta_code=planta_code,
        plantas=plantas,
        planta_monthly=planta_monthly,
        total_monthly=monthly_zero.sum(axis=0),
        month_of_year=((start.month - 1 + np.arange(n_months)) % 12) + 1,
        labels=[str(p) for p in pd.period_range(start, periods=n_months, freq="M")],
        business_days=business_days,
        calendar_days=calendar_days,
        active_days=active_days,
        daily=daily,
        dates=dates,
        month_first_date=month_first_date,
        keys=list(table.index),
    )


# Window helpers: the window is the months lo..hi (inclusive) of the last axis; lo is clipped
# at 0 and an empty window gives NaN.


def _window(a: np.ndarray, lo: int, hi: int) -> np.ndarray | None:
    lo = max(lo, 0)
    return a[..., lo : hi + 1] if hi >= lo else None


def window_mean(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    w = _window(a, lo, hi)
    if w is None:
        return np.full(a.shape[:-1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(w, axis=-1)


def window_std(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    w = _window(a, lo, hi)
    if w is None:
        return np.full(a.shape[:-1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(w, axis=-1)


def window_rate(a: np.ndarray, business_days: np.ndarray, lo: int, hi: int) -> np.ndarray:
    r"""Sum over the window divided by the non-Sunday days of the months each series was alive."""
    w = _window(a, lo, hi)
    if w is None:
        return np.full(a.shape[:-1], np.nan)
    total = np.nansum(w, axis=-1)
    days = ((~np.isnan(w)) * business_days[max(lo, 0) : hi + 1]).sum(axis=-1)
    out = np.full(total.shape, np.nan)
    np.divide(total, days, out=out, where=days > 0)
    return out
