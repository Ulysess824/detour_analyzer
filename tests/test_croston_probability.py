"""Probability and size forecasts of Croston: the identity with the flat forecast, no look-ahead, scores by hand, weekday profile."""

import numpy as np
import pandas as pd
import pytest

from src.experiments import croston_probability as cp
from src.experiments import croston_utils as cu
from src.experiments import weekday_profile as wp

DAYS = pd.bdate_range("2025-01-06", periods=200)  # Monday to Friday


def _series(seed=0, p=0.3):
    rng = np.random.default_rng(seed)
    return pd.Series(np.where(rng.random(len(DAYS)) < p, rng.gamma(3.0, 1.0, len(DAYS)), 0.0), index=DAYS)


def test_probability_times_size_is_the_croston_forecast():
    s = _series()
    table = cp.rolling_probabilities(s, str(DAYS[150].date()))
    pos = list(DAYS).index(table["target"].iloc[0])
    fit = cu.fit(s.to_numpy()[:pos])
    assert table["p_croston"].iloc[0] * table["s_z"].iloc[0] == pytest.approx(cu.f_croston(fit))


def test_rolling_probabilities_use_only_the_past():
    s = _series()
    start = str(DAYS[150].date())
    a = cp.rolling_probabilities(s, start)
    changed = s.copy()
    changed.iloc[150:] = 99.0  # the target day and everything after it
    b = cp.rolling_probabilities(changed, start)
    first = a.iloc[0]
    assert first["target"] == DAYS[150] and first["origin"] == DAYS[149]
    for col in ("p_croston", "p_freq", "p_croston_dow", "p_freq_dow", "s_z", "s_mean", "s_median", "s_last3"):
        assert b.iloc[0][col] == pytest.approx(first[col])  # only the realised value differs
    assert b.iloc[0]["y"] == 99.0


def test_probabilities_stay_in_zero_one_and_skip_series_without_history():
    t = cp.rolling_probabilities(_series(p=0.9), str(DAYS[150].date()))
    assert t[[c for c in t.columns if c.startswith("p_")]].apply(lambda c: c.between(0, 1).all()).all()
    empty = pd.Series(np.zeros(len(DAYS)), index=DAYS)
    assert cp.rolling_probabilities(empty, str(DAYS[150].date())).empty


def test_probability_scores_by_hand():
    table = pd.DataFrame({"has": [1, 0, 0, 0], "p_freq": [0.25] * 4, "p_croston": [0.5, 0.5, 0.0, 0.0]})
    out = cp.probability_scores(table, models=("croston", "freq"))
    assert out.loc["freq", "brier"] == pytest.approx((0.5625 + 3 * 0.0625) / 4)
    assert out.loc["croston", "brier"] == pytest.approx((0.25 + 0.25) / 4)
    assert out.loc["freq", "brier_skill"] == pytest.approx(0.0)
    assert out.loc["croston", "brier_skill"] == pytest.approx(1 - 0.125 / 0.1875)
    assert out.loc["croston", "observed"] == 0.25 and out.loc["croston", "mean_forecast"] == 0.25


def test_calibration_bins_count_days_and_compare_forecast_with_frequency():
    table = pd.DataFrame({"has": [1, 0, 1, 1, 0, 0], "p_x": [0.05, 0.05, 0.9, 0.9, 0.9, 0.9]})
    out = cp.calibration(table, "x")
    assert list(out["n"]) == [2, 4]
    assert out.loc[0, "observed"] == 0.5 and out.loc[1, "observed"] == 0.5
    assert out.loc[1, "mean_forecast"] == pytest.approx(0.9)


def test_size_scores_only_use_days_with_consumption():
    table = pd.DataFrame({"y": [0.0, 2.0, 4.0], "has": [0, 1, 1], "s_z": [9.0, 3.0, 3.0], "s_mean": [2.0, 2.0, 2.0]})
    out = cp.size_scores(table, models=("z", "mean"))
    assert out.loc["z", "n"] == 2 and out.loc["z", "wape"] == pytest.approx(2 / 6, abs=1e-4)
    assert out.loc["mean", "wape"] == pytest.approx(2 / 6, abs=1e-4) and out.loc["z", "mae"] == 1.0


def test_weekday_profile_moves_consumption_between_days_and_keeps_the_mean():
    dows = np.tile([0, 1, 2, 3, 4], 40)
    values = np.where(dows == 4, 0.0, 5.0)  # nothing on Fridays
    w = wp.weekday_profile(values, dows, weekdays=5, shrink=0.0)
    assert w[4] == 0.0 and w.mean() == pytest.approx(1.0) and w[0] == pytest.approx(1.25)
    shrunk = wp.weekday_profile(values, dows, weekdays=5, shrink=1e9)  # no evidence: all ones
    assert np.allclose(shrunk, 1.0)
    assert np.allclose(wp.weekday_profile(np.zeros(10), np.arange(10) % 5), 1.0)


def test_add_profile_rows_scale_forecast_and_limits_by_the_origin_profile():
    s = pd.Series(np.where(np.arange(len(DAYS)) % 5 == 4, 0.0, 5.0), index=DAYS)  # Fridays at 0
    target = DAYS[150]  # a Monday
    daily = pd.DataFrame({"origin": [DAYS[149]] * 2, "target": [target, DAYS[154]], "h": 1, "model": "croston",
                          "forecast": [2.0, 2.0], "lower": [0.0, 0.0], "upper": [4.0, 4.0], "real": 1.0})
    out = wp.add_profile_rows(daily, s, weekdays=5, shrink=0.0)
    dow = out[out["model"] == "croston_dow"].reset_index(drop=True)
    assert len(out) == 4 and (out["model"].iloc[:2] == "croston").all()
    assert dow.loc[0, "forecast"] == pytest.approx(2.5) and dow.loc[0, "upper"] == pytest.approx(5.0)  # Monday: 1.25 x
    assert dow.loc[1, "forecast"] == 0.0  # the Friday of that week
