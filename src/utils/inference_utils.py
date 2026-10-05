"""Forecast of the next month from the full history (what a monthly run in production does)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.utils.feature_utils import make_frame
from src.utils.member_utils import ML_MODELS, ml_forecast
from src.utils.panel_utils import Panel


def check_last_month_complete(df: pd.DataFrame) -> None:
    r"""
    Stop when the last month of the export is not finished.

    The month is incomplete when a day after the last record, up to the end of that month, is
    not a Sunday (Sundays have almost no consumption, so a month can end with empty days).
    """
    last = df["fecha"].max()
    rest = pd.date_range(last + pd.Timedelta(days=1), last + pd.offsets.MonthEnd(0))
    missing = rest[rest.dayofweek != 6]
    if len(missing):
        raise SystemExit(
            f"The last month ({last.strftime('%Y-%m')}) is incomplete: the data ends on {last.date()} and "
            f"{len(missing)} working days are missing. Export the complete month, or pass --as-of with the last complete month."
        )


def extend_panel(S: Panel, n_ahead: int) -> Panel:
    r"""
    Add `n_ahead` empty months (NaN consumption) after the last one, so the frame can build the rows of a future target.

    Only the arrays the monthly features read are extended; `daily`, `dates` and `month_first_date` are left as they are.
    """
    n = S.monthly.shape[1]
    start = pd.Period(S.labels[0], freq="M")
    periods = pd.period_range(start, periods=n + n_ahead, freq="M")
    days = [pd.date_range(p.start_time, p.end_time.normalize()) for p in periods[n:]]
    pad = lambda a: np.concatenate([a, np.full(a.shape[:-1] + (n_ahead,), np.nan)], axis=-1)  # noqa: E731
    return dataclasses.replace(
        S,
        monthly=pad(S.monthly),
        active_days=pad(S.active_days),
        planta_monthly=pad(S.planta_monthly),
        total_monthly=pad(S.total_monthly),
        month_of_year=np.concatenate([S.month_of_year, [p.month for p in periods[n:]]]),
        labels=[str(p) for p in periods],
        business_days=np.concatenate([S.business_days, [int((d.dayofweek != 6).sum()) for d in days]]),
        calendar_days=np.concatenate([S.calendar_days, [len(d) for d in days]]),
    )


def forecast_next_month(S: Panel, tuned: dict, horizon: int = 1) -> pd.DataFrame:
    r"""
    Train on every target month already observed and forecast the month `horizon` after the last one.

    One row per series alive at the last month, with a column per machine-learning model and
    `ml_mean` (their average, the model used in the report). The tuned parameters come from `tuned`
    (the baseline parameters when a model has none).
    """
    n = S.monthly.shape[1]
    extended = extend_panel(S, horizon)
    frame = make_frame(extended, horizon)
    train = frame[frame["t"] <= n - 1]  # targets already observed
    rows = frame[frame["o"] == n - 1].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "planta": [S.keys[s][0] for s in rows["series"]],
            "sku": [S.keys[s][1] for s in rows["series"]],
            "mes": extended.labels[n - 1 + horizon],
        }
    )
    for kind in ML_MODELS:
        out[kind] = ml_forecast(kind, train, rows, tuned)
    out["ml_mean"] = out[ML_MODELS].mean(axis=1)
    out["naive"] = rows["p_naive"].to_numpy()
    return out
