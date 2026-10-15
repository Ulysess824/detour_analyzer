"""Monte Carlo completion of the rest of a month: no peeking past the cutoff, reproducible, only on days the planta works."""

import numpy as np

from src.utils.montecarlo_utils import simulate_rest_of_month, working_day_profile

DAY = 15


def _target(S):
    return S.monthly.shape[1] - 1


def test_days_after_the_cutoff_cannot_change_the_simulation(toy_panel):
    import dataclasses

    S, m = toy_panel, _target(toy_panel)
    cutoff = int(S.month_first_date[m]) + DAY
    daily = S.daily.copy()
    daily[:, cutoff:] *= 3  # the real rest of the month and everything after it
    S2 = dataclasses.replace(S, daily=daily)
    a = simulate_rest_of_month(S, m, DAY, draws=30, seed=1)
    b = simulate_rest_of_month(S2, m, DAY, draws=30, seed=1)
    assert np.array_equal(a, b)


def test_same_seed_same_draws_and_other_seed_other_draws(toy_panel):
    S, m = toy_panel, _target(toy_panel)
    a = simulate_rest_of_month(S, m, DAY, draws=20, seed=3)
    assert np.array_equal(a, simulate_rest_of_month(S, m, DAY, draws=20, seed=3))
    assert not np.array_equal(a, simulate_rest_of_month(S, m, DAY, draws=20, seed=4))
    assert a.shape == (20, S.daily.shape[0]) and (a >= 0).all()


def test_a_weekday_without_consumption_in_the_data_gets_zero_probability(toy_panel):
    S, m = toy_panel, _target(toy_panel)
    cutoff = int(S.month_first_date[m]) + DAY
    profile = working_day_profile(S, cutoff)
    assert (profile[:, 6] == 0).all()  # the toy table has no Sundays
    assert (profile[:, :6] > 0.5).all()


def test_a_series_that_never_consumed_stays_at_zero(toy_panel):
    import dataclasses

    S, m = toy_panel, _target(toy_panel)
    daily = S.daily.copy()
    daily[0, :] = 0.0
    S2 = dataclasses.replace(S, daily=daily)
    assert (simulate_rest_of_month(S2, m, DAY, draws=20, seed=0)[:, 0] == 0).all()
