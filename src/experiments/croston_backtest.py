"""
Rolling-origin weekly backtest of the Croston models of one series, against simple references and the monthly ml_mean.

Every forecast of week `s` at horizon `h` uses only the weeks up to the origin `s - h`; models are refitted at each origin (expanding
window). The monthly comparison with ml_mean puts the weekly forecast of the last complete week before the month on the same footing
as the monthly model: flat forecast x non-Sunday days of the month / 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import croston_intervals as ci
from src.experiments import croston_utils as cu

FIXED_ALPHAS = (0.1, 0.5)  # sensitivity of F_cr; 0.5 is the value used in the paper for F_sb0 and F_sb1
MODELS_WITH_INTERVAL = ("modified", "modified_log", "croston_model", "log_croston_model")


def _simple_models(y: np.ndarray) -> dict[str, float]:
    out = {"naive": float(y[-1]), "mean4": float(np.mean(y[-4:])), "mean8": float(np.mean(y[-8:]))}
    out["ses"] = cu.ses(y, cu.fit_alpha(y))[0] if len(y) >= 2 else float(y[-1])
    return out


def origin_forecasts(y: np.ndarray, h: int, draws: int, seed: int, level: float = 0.95) -> list[dict]:
    r"""
    Forecast of the week `h` weeks after the end of `y` by every model: a list of {model, forecast, lower, upper} (limits NaN for
    models without interval). The Croston models are left out when `y` has fewer than cu.MIN_DEMANDS positive weeks.
    """
    rows = [{"model": m, "forecast": v, "lower": np.nan, "upper": np.nan} for m, v in _simple_models(y).items()]
    fit, fit_log = cu.fit(y), cu.fit(y, log=True)
    if fit is None:
        return rows
    add = lambda model, forecast, lower=np.nan, upper=np.nan: rows.append(  # noqa: E731
        {"model": model, "forecast": forecast, "lower": lower, "upper": upper})
    add("croston", cu.f_croston(fit))
    add("sb0", cu.f_sb0(fit))
    add("sb1", cu.f_sb1(fit))
    for a in FIXED_ALPHAS:
        add(f"croston_a{a}", cu.f_croston(cu.fit(y, alpha=a)))
    add("sb0_a0.5", cu.f_sb0(cu.fit(y, alpha=0.5)))
    add("sb1_a0.5", cu.f_sb1(cu.fit(y, alpha=0.5)))
    add("modified", cu.m_modified(fit, h), *ci.modified_interval(fit, h, level))
    add("modified_log", cu.m_modified_log(fit_log, h), *ci.modified_interval(fit_log, h, level))
    for name, f in (("croston_model", fit), ("log_croston_model", fit_log)):
        lower, upper, mean = ci.simulated_interval(ci.simulate_paths(f, h, draws, seed), h, level)
        add(name, mean, lower, upper)
    return rows


def rolling_weekly(series: pd.Series, test_start: str, horizons=(1, 4), draws: int = ci.DRAWS, seed: int = 0) -> pd.DataFrame:
    r"""
    Long table of the rolling-origin forecasts: one row per (target week, h, model) with `origin`, `target`, `real`, `forecast`,
    `lower`, `upper`. Targets are the weeks of `series` with Monday >= test_start; the origin is `h` weeks before the target.
    """
    values = series.to_numpy(dtype=float)
    rows = []
    for h in horizons:
        for pos in np.flatnonzero(series.index >= pd.Timestamp(test_start)):
            origin = pos - h
            if origin < 1:
                continue
            for r in origin_forecasts(values[: origin + 1], h, draws, seed * 1_000_003 + pos * 10 + h):
                rows.append({"origin": series.index[origin], "target": series.index[pos], "h": h, **r, "real": values[pos]})
    return pd.DataFrame(rows)


def month_business_days(month: pd.Period) -> int:
    r"""Days of the month that are not Sunday."""
    days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
    return int((days.dayofweek != 6).sum())


def monthly_comparison(series: pd.Series, real_monthly: pd.Series, ml_mean: pd.Series, draws: int = ci.DRAWS, seed: int = 0,
                       days_per_week: int = 6) -> pd.DataFrame:
    r"""
    Month by month: real total, ml_mean and the weekly models' flat forecast scaled to the month. The weekly forecast (horizon 1) is
    made at the last complete week that ends before the month starts, with the weeks up to it. `real_monthly` and `ml_mean` are
    indexed by pandas Period (month); months with fewer than two weeks before them are left out.
    """
    rows = []
    for month in ml_mean.index:
        before = np.flatnonzero(series.index + pd.Timedelta(days=6) < month.start_time)  # weeks that end before the month
        if len(before) < 2:
            continue
        pos = int(before[-1])
        scale = month_business_days(month) / days_per_week
        row = {"month": str(month), "origin": series.index[pos], "real": float(real_monthly[month]), "ml_mean": float(ml_mean[month])}
        for r in origin_forecasts(series.to_numpy(dtype=float)[: pos + 1], 1, draws, seed * 1_000_003 + pos * 10 + 1):
            row[r["model"]] = r["forecast"] * scale
        rows.append(row)
    return pd.DataFrame(rows)
