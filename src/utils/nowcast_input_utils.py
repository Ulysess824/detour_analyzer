"""
Month in progress completed with the mean of its first days, used as the input of ml_mean for the month after it.

At mid-month `m` the last complete month is `o = m - 1` and the month in progress is `o + 1`. The month in progress is
completed as (real first days) + (mean daily consumption of those days x non-Sunday days left), series by series, and
enters the one-month-ahead model as if it were an observed month; the model forecasts `o + 2`. It is trained only on real
complete months (targets up to `o`), so the completed month is the only estimated input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.feature_utils import frame_at_origin
from src.utils.horizon_utils import mask_panel, ml_forecasts, with_pseudo_month
from src.utils.now_casting_current_month import forecast_media, seen_days
from src.utils.panel_utils import Panel


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


def nowcast_input_forecast(S: Panel, frame1: pd.DataFrame, origin: int, tuned: dict, cutoff_day: int = 15) -> pd.DataFrame:
    r"""
    Forecast of month `origin + 2` from the last complete month `origin` and the completed month `origin + 1`.

    `frame1` is `make_frame(S, 1)`. Returns one row per series alive at `origin` with the forecast of each tree model and
    their mean (`ml_mean`). The model is trained on targets known at `origin` (real complete months only).
    """
    train = frame1[frame1["t"] <= origin]
    rows = frame1[frame1["o"] == origin]
    series = rows["series"].to_numpy()
    values, active = completed_month(S, origin + 1, cutoff_day, series)
    pseudo = with_pseudo_month(mask_panel(S, origin), origin + 1, series, values, active)
    rows2 = frame_at_origin(pseudo, origin + 1, 1)
    rows2 = rows2[rows2["series"].isin(series)].reset_index(drop=True)
    out = ml_forecasts(train, rows2, tuned)
    return pd.DataFrame({"series": rows2["series"].to_numpy(), **out})
