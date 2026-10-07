"""Features and simple baseline forecasts for the monthly forecasting frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.panel_utils import Panel, window_mean, window_rate, window_std

MIN_ORIGIN = 3

SERIES_FEATURES = [
    "lag1", "lag2", "lag3",
    "mean3", "mean6", "mean12", "std6",
    "zero_share12", "months_active", "ly",
    "planta_code", "month_t",
    "ratio_planta", "ratio_global", "lyrel_planta", "lyrel_global",
    "n_days_t",
    "rate3", "rate6", "rate12",
    "occ3", "occ6", "occ12",
    "size6", "ly_rate",
]  # fmt: skip

BASELINES = ["naive", "mean3", "mean6", "mean12", "perday6", "seasonal_naive", "mean6_x_global", "mean6_x_planta"]
DIRECT = ["naive", "mean3", "mean6", "mean12", "perday6", "seasonal_naive", "mean6_x_seasonal"]


def direct_forecasts(a: np.ndarray, origin: int, horizon: int, business_days: np.ndarray) -> dict[str, np.ndarray]:
    r"""Apply the simple models to aggregated series `a` of shape (k, n_months)."""
    target = origin + horizon
    mean6 = window_mean(a, origin - 5, origin)
    last_year = a[:, target - 12] if target - 12 >= 0 else np.full(a.shape[0], np.nan)
    if origin - 17 >= 0:
        base = window_mean(a, origin - 17, origin - 12)
    else:
        base = np.full(a.shape[0], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = last_year / base
    return {
        "naive": a[:, origin],
        "mean3": window_mean(a, origin - 2, origin),
        "mean6": mean6,
        "mean12": window_mean(a, origin - 11, origin),
        "perday6": window_rate(a, business_days, origin - 5, origin) * business_days[target],
        "seasonal_naive": np.where(np.isnan(last_year), mean6, last_year),
        "mean6_x_seasonal": np.where(np.isfinite(ratio), mean6 * ratio, mean6),
    }


def _seasonal_ratios(S: Panel, origin: int, target: int) -> dict:
    r"""Last year's seasonal ratios of each planta and of the grand total (NaN when unavailable)."""
    n = len(S.plantas)
    total = S.total_monthly[None, :]
    nan_p, nan_g = np.full(n, np.nan), np.nan
    out = {"ratio_p": nan_p, "ratio_g": nan_g, "lyrel_p": nan_p, "lyrel_g": nan_g}
    with np.errstate(divide="ignore", invalid="ignore"):
        if target - 12 >= 0 and origin - 17 >= 0:
            out["ratio_p"] = S.planta_monthly[:, target - 12] / window_mean(S.planta_monthly, origin - 17, origin - 12)
            out["ratio_g"] = (S.total_monthly[target - 12] / window_mean(total, origin - 17, origin - 12))[0]
        if target - 15 >= 0:
            out["lyrel_p"] = S.planta_monthly[:, target - 12] / window_mean(S.planta_monthly, target - 15, target - 9)
            out["lyrel_g"] = S.total_monthly[target - 12] / window_mean(total, target - 15, target - 9)[0]
    return out


def _origin_frame(S: Panel, origin: int, horizon: int) -> pd.DataFrame:
    r"""Rows of every series alive at `origin`, with features known at the origin."""
    target = origin + horizon
    alive = np.flatnonzero(S.first_month <= origin)
    ya = S.monthly[alive]
    active = S.active_days[alive]
    planta = S.planta_code[alive]
    bd = S.business_days
    missing = np.full(len(alive), np.nan)

    def lag(k: int) -> np.ndarray:
        return ya[:, origin - k] if origin - k >= 0 else missing

    mean3, mean6, mean12 = (window_mean(ya, origin - k, origin) for k in (2, 5, 11))
    rate3, rate6, rate12 = (window_rate(ya, bd, origin - k, origin) for k in (2, 5, 11))
    occ3, occ6, occ12 = (window_rate(active, bd, origin - k, origin) for k in (2, 5, 11))
    zeros = np.where(np.isnan(ya), np.nan, (ya == 0).astype(float))
    last_year = ya[:, target - 12] if target - 12 >= 0 else missing
    last_year_rate = last_year / bd[target - 12] if target - 12 >= 0 else missing
    with np.errstate(divide="ignore", invalid="ignore"):
        size6 = np.where(occ6 > 0, rate6 / occ6, np.nan)

    ratios = _seasonal_ratios(S, origin, target)
    ratio_planta = ratios["ratio_p"][planta]

    frame = pd.DataFrame(
        {
            "series": alive,
            "o": origin,
            "t": target,
            "planta_code": planta,
            "lag1": lag(0),
            "lag2": lag(1),
            "lag3": lag(2),
            "mean3": mean3,
            "mean6": mean6,
            "mean12": mean12,
            "std6": window_std(ya, origin - 5, origin),
            "zero_share12": window_mean(zeros, origin - 11, origin),
            "months_active": origin - S.first_month[alive] + 1,
            "ly": last_year,
            "month_t": S.month_of_year[target],
            "ratio_planta": ratio_planta,
            "ratio_global": ratios["ratio_g"],
            "lyrel_planta": ratios["lyrel_p"][planta],
            "lyrel_global": ratios["lyrel_g"],
            "n_days_t": bd[target],
            "rate3": rate3,
            "rate6": rate6,
            "rate12": rate12,
            "occ3": occ3,
            "occ6": occ6,
            "occ12": occ12,
            "size6": size6,
            "ly_rate": last_year_rate,
            "y": ya[:, target],
        }
    )
    # Baseline forecasts, stored next to the features.
    frame["p_naive"] = frame["lag1"]
    frame["p_mean3"], frame["p_mean6"], frame["p_mean12"] = mean3, mean6, mean12
    frame["p_perday6"] = rate6 * bd[target]
    frame["p_seasonal_naive"] = np.where(np.isnan(last_year), mean6, last_year)
    frame["p_mean6_x_global"] = np.where(np.isfinite(ratios["ratio_g"]), mean6 * ratios["ratio_g"], mean6)
    frame["p_mean6_x_planta"] = np.where(np.isfinite(ratio_planta), mean6 * ratio_planta, mean6)
    return frame


def make_frame(S: Panel, horizon: int) -> pd.DataFrame:
    r"""
    One row per (origin, series alive at that origin).

    Every window ends at or before the origin, so nothing from the target month
    (origin + horizon) is used as a feature; `y` is that target month's total.
    """
    n_months = S.monthly.shape[1]
    frames = [_origin_frame(S, origin, horizon) for origin in range(MIN_ORIGIN, n_months - horizon)]
    return pd.concat(frames, ignore_index=True)


def frame_at_origin(S: Panel, origin: int, horizon: int) -> pd.DataFrame:
    r"""Rows of every series alive at one `origin` (the building block of `make_frame`)."""
    return _origin_frame(S, origin, horizon)
