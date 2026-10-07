"""Two-month-ahead strategies: nothing after the last complete month is used, except the first days of the next one."""

import dataclasses

import numpy as np
import pandas as pd

from src.utils.feature_utils import make_frame
from src.utils.horizon_utils import mask_panel, strategy_forecasts, with_pseudo_month
from src.utils.partial_utils import PARTIAL_FEATURES, add_partial_features, month_to_date

DAY = 15


def _perturb(S, first_month, daily_from):
    r"""Copy of the panel with everything from `first_month` (monthly) and from daily position `daily_from` multiplied by 3."""
    monthly, active = S.monthly.copy(), S.active_days.copy()
    planta, total, daily = S.planta_monthly.copy(), S.total_monthly.copy(), S.daily.copy()
    monthly[:, first_month:] *= 3
    active[:, first_month:] *= 3
    planta[:, first_month:] *= 3
    total[first_month:] *= 3
    daily[:, daily_from:] *= 3
    return dataclasses.replace(S, monthly=monthly, active_days=active, planta_monthly=planta, total_monthly=total, daily=daily)


def test_mask_panel_hides_the_future_and_keeps_the_past(toy_panel):
    S = toy_panel
    M = mask_panel(S, 20)
    assert np.array_equal(M.monthly[:, :21], S.monthly[:, :21], equal_nan=True)
    assert np.isnan(M.monthly[:, 21:]).all() and np.isnan(M.total_monthly[21:]).all() and np.isnan(M.active_days[:, 21:]).all()


def test_pseudo_month_fills_only_the_given_series_and_updates_totals(toy_panel):
    M = mask_panel(toy_panel, 20)
    series = np.array([0, 1, 2])
    P = with_pseudo_month(M, 21, series, np.array([5.0, 6.0, 7.0]), np.array([3.0, 3.0, 3.0]))
    assert np.allclose(P.monthly[series, 21], [5.0, 6.0, 7.0])
    assert np.isnan(np.delete(P.monthly[:, 21], series)).all()
    assert np.isclose(P.total_monthly[21], 18.0) and np.isclose(P.planta_monthly[:, 21].sum(), 18.0)


def test_month_to_date_matches_a_manual_sum_and_reads_only_the_first_days(toy_panel):
    S = toy_panel
    mtd, active, elapsed = month_to_date(S, DAY)
    m = 12
    lo = S.month_first_date[m]
    assert np.allclose(mtd[:, m], S.daily[:, lo : lo + DAY].sum(axis=1), atol=1e-4)
    assert elapsed[m] == int((S.dates[lo : lo + DAY].dayofweek != 6).sum())
    S2 = _perturb(S, m, lo + DAY)  # later days change, the first 15 days do not
    assert np.allclose(month_to_date(S2, DAY)[0][:, m], mtd[:, m])
    S3 = _perturb(S, m, lo)  # now the first days change too
    assert not np.allclose(month_to_date(S3, DAY)[0][:, m], mtd[:, m])


def test_partial_features_describe_the_month_after_the_origin(toy_panel):
    S = toy_panel
    f2 = add_partial_features(make_frame(S, 2), S, DAY)
    row = f2.iloc[len(f2) // 2]
    mtd, active, elapsed = month_to_date(S, DAY)
    m, s = int(row["o"]) + 1, int(row["series"])
    assert np.isclose(row["mtd"], mtd[s, m])
    assert np.isclose(row["mtd_proj"], mtd[s, m] / elapsed[m] * S.business_days[m])
    assert set(PARTIAL_FEATURES) <= set(f2.columns)


def test_strategies_ignore_the_future_of_the_last_complete_month(toy_panel):
    r"""Tripling everything after month `o` (and the days after the cutoff of month `o+1`) cannot move any forecast."""
    S = toy_panel
    target = S.monthly.shape[1] - 1
    origin = target - 2
    first_day = S.month_first_date[origin + 1]
    S2 = _perturb(S, origin + 1, first_day + DAY)

    def run(P):
        f1, f2 = make_frame(P, 1), make_frame(P, 2)
        return strategy_forecasts(P, f1, f2, add_partial_features(f2, P, DAY), {}, [target], local=False, log=lambda *_: None)

    a, b = run(S), run(S2)
    assert not np.allclose(a["y"], b["y"])  # the actuals of the target did change
    for col in ("p_direct", "p_iterated", "p_partial", "p_naive", "p_mean6", "p_seasonal_naive"):
        assert np.allclose(a[col], b[col]), col
    assert (a[["p_direct", "p_iterated", "p_partial"]] >= 0).all().all()
