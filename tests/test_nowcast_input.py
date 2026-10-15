"""Month completed with the mean as the input of ml_mean: only real data up to the cutoff, and the completion is what the docstring says."""

import dataclasses

import numpy as np

from src.utils.feature_utils import make_frame
from src.utils.nowcast_input_utils import completed_month, nowcast_input_forecast

DAY = 15


def test_completed_month_is_first_days_plus_the_mean_of_the_days_seen(toy_panel):
    S = toy_panel
    month = S.monthly.shape[1] - 2
    series = np.arange(S.daily.shape[0])
    total, active = completed_month(S, month, DAY, series)
    start = int(S.month_first_date[month])
    block = S.daily[:, start : start + DAY].astype(float)
    nonsun = np.asarray(S.dates[start : start + DAY].dayofweek != 6)
    left = int(S.business_days[month]) - int(nonsun.sum())
    assert np.allclose(total, block.sum(axis=1) + block[:, nonsun].mean(axis=1) * left)
    assert (active >= 0).all() and (active <= S.business_days[month]).all()


def test_the_real_rest_of_the_month_in_progress_and_later_cannot_change_the_forecast(toy_panel):
    S = toy_panel
    n = S.monthly.shape[1]
    origin = n - 3  # last complete month; the month in progress is n - 2 and the target n - 1
    cutoff = int(S.month_first_date[origin + 1]) + DAY
    monthly, active = S.monthly.copy(), S.active_days.copy()
    planta, total, daily = S.planta_monthly.copy(), S.total_monthly.copy(), S.daily.copy()
    monthly[:, origin + 1 :] *= 3
    active[:, origin + 1 :] *= 3
    planta[:, origin + 1 :] *= 3
    total[origin + 1 :] *= 3
    daily[:, cutoff:] *= 3
    S2 = dataclasses.replace(S, monthly=monthly, active_days=active, planta_monthly=planta, total_monthly=total, daily=daily)

    a = nowcast_input_forecast(S, make_frame(S, 1), origin, {}, DAY)
    b = nowcast_input_forecast(S2, make_frame(S2, 1), origin, {}, DAY)
    assert np.allclose(a["ml_mean"], b["ml_mean"])
    assert (a["ml_mean"] >= 0).all() and len(a) > 0
