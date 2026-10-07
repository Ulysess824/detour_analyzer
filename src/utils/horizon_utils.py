"""
Forecast the month after next, as a planner's mid-month deadline requires.

At mid-month `m` the last complete month is `o = m - 1`, and the month to forecast is `t = o + 2`.
Strategies compared (the model of the report, `ml_mean`, is the mean of LightGBM, XGBoost and Random Forest):

    direct      one model trained with the target two months ahead
    iterated    a one-month-ahead model applied twice: its forecast of month `o + 1` is used as if it were observed
    partial     the direct model plus the consumption of the first days of month `o + 1` (mid-month cutoff)
    local       per-series models (SES, damped Holt, ARIMA) and simple baselines, to compare with the global trees

Every forecast of month `t` uses only what is known at the end of month `o` (and, for `partial`, the first days of `o + 1`).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.utils.econometric_utils import ECONOMETRIC_MODELS, econometric_forecasts
from src.utils.feature_utils import SERIES_FEATURES, frame_at_origin
from src.utils.member_utils import ML_MODELS, ml_forecast
from src.utils.panel_utils import Panel
from src.utils.partial_utils import PARTIAL_FEATURES

HORIZON = 2
BASELINES = ["naive", "mean6", "seasonal_naive"]


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


def ml_forecasts(train: pd.DataFrame, rows: pd.DataFrame, tuned: dict, features: list[str] | None = None) -> dict:
    r"""Forecast of each machine-learning model plus their mean (`ml_mean`) for `rows`."""
    out = {kind: ml_forecast(kind, train, rows, tuned, features) for kind in ML_MODELS}
    out["ml_mean"] = np.mean([out[kind] for kind in ML_MODELS], axis=0)
    return out


def iterated_forecast(S: Panel, frame1: pd.DataFrame, origin: int, tuned: dict) -> pd.Series:
    r"""
    Two applications of the one-month-ahead model, trained on the targets known at `origin`.

    Month `origin + 1` is filled with the first forecasts (active days from the recent occupancy) and the
    features of origin `origin + 1` are rebuilt from the panel with the later months hidden.
    Returns the forecast of month `origin + 2`, indexed by series.
    """
    train = frame1[frame1["t"] <= origin]
    rows = frame1[frame1["o"] == origin].reset_index(drop=True)
    first = ml_forecasts(train, rows, tuned)["ml_mean"]
    days = S.business_days[origin + 1]
    active = np.minimum(rows["occ6"].fillna(0.0).to_numpy() * days, days)
    pseudo = with_pseudo_month(mask_panel(S, origin), origin + 1, rows["series"].to_numpy(), first, active)
    rows2 = frame_at_origin(pseudo, origin + 1, 1)
    rows2 = rows2[rows2["series"].isin(rows["series"])].reset_index(drop=True)
    second = ml_forecasts(train, rows2, tuned)["ml_mean"]
    return pd.Series(second, index=rows2["series"].to_numpy())


def strategy_forecasts(
    S: Panel,
    frame1: pd.DataFrame,
    frame2: pd.DataFrame,
    frame2_partial: pd.DataFrame,
    tuned: dict,
    test_months: list[int],
    n_jobs: int = 4,
    local: bool = False,
    log=print,
) -> pd.DataFrame:
    r"""
    One row per (series, test month t) with the actual `y` and a column p_<strategy>.

    frame1 / frame2 are `make_frame` at horizon 1 / 2; frame2_partial is frame2 with the month-to-date features.
    The series are those alive at the last complete month `t - 2`, the same for every strategy.
    """
    parts = []
    for t in test_months:
        origin = t - HORIZON
        rows = frame2[frame2["o"] == origin].reset_index(drop=True)
        block = rows[["series", "o", "t", "planta_code", "y"]].copy()

        direct = ml_forecasts(frame2[frame2["t"] <= origin], rows, tuned)
        for kind in ML_MODELS:
            block[f"p_direct_{kind}"] = direct[kind]
        block["p_direct"] = direct["ml_mean"]

        rows_p = frame2_partial[frame2_partial["o"] == origin].reset_index(drop=True)
        partial = ml_forecasts(frame2_partial[frame2_partial["t"] <= origin], rows_p, tuned, SERIES_FEATURES + PARTIAL_FEATURES)
        block["p_partial"] = partial["ml_mean"]

        block["p_iterated"] = iterated_forecast(S, frame1, origin, tuned).reindex(block["series"]).to_numpy()

        for name in BASELINES:
            block[f"p_{name}"] = rows[f"p_{name}"].to_numpy()
        if local:
            econ = econometric_forecasts(S, rows, origin, HORIZON, n_jobs)
            for name in ECONOMETRIC_MODELS:
                block[f"p_local_{name}"] = econ[name].to_numpy()
            block["p_local_mean"] = econ[ECONOMETRIC_MODELS].mean(axis=1).to_numpy()
        parts.append(block)
        log(f"    target {S.labels[t]} done")
    return pd.concat(parts, ignore_index=True)
