"""Croston intervals: closed form against the normal interval it reduces to, zero mass, simulation against Theorem 3, seeds, guards."""

import math
from statistics import NormalDist

import numpy as np
import pytest

from src.experiments import croston_intervals as ci
from src.experiments import croston_utils as cu


def _fit(z=10.0, p=1.0, sigma=1.0, alpha=0.3, log=False, sigma_q=1.0, idle=0):
    return cu.CrostonFit(alpha, z, p, sigma, sigma_q, p, 10, log, idle, p_log=math.log(p) if log else None, sigma_q_log=sigma_q if log else None)


def test_modified_interval_is_the_normal_interval_when_every_week_has_demand():
    lower, upper = ci.modified_interval(_fit(z=10.0, p=1.0, sigma=1.0), h=1, level=0.95)
    assert lower == pytest.approx(10 - 1.959964, abs=1e-5)
    assert upper == pytest.approx(10 + 1.959964, abs=1e-5)


def test_modified_log_interval_is_lognormal_when_every_week_has_demand():
    lower, upper = ci.modified_interval(_fit(z=2.0, p=1.0, sigma=0.4, log=True), h=1)
    assert lower == pytest.approx(math.exp(2 - 1.959964 * 0.4), rel=1e-5)
    assert upper == pytest.approx(math.exp(2 + 1.959964 * 0.4), rel=1e-5)


def test_modified_interval_widens_with_the_horizon():
    f = _fit(p=1.0, sigma=1.0, alpha=0.5)
    (l1, u1), (l4, u4) = ci.modified_interval(f, h=1), ci.modified_interval(f, h=4)
    assert u4 - l4 > u1 - l1


def test_modified_interval_with_many_zero_weeks_reaches_zero_and_covers_the_point_forecast():
    f = _fit(z=10.0, p=2.0, sigma=1.0)
    lower, upper = ci.modified_interval(f, h=1)
    assert lower <= 0.0
    assert upper > cu.m_modified(f)
    flog = _fit(z=2.0, p=2.0, sigma=0.4, log=True)
    lower_log, upper_log = ci.modified_interval(flog, h=1)
    assert lower_log == 0.0 and upper_log > cu.m_modified_log(flog)


def test_modified_interval_with_a_demand_chance_below_the_tail_is_zero():
    assert ci.modified_interval(_fit(p=50.0), h=1, level=0.95) == (0.0, 0.0)


def test_simulation_is_reproducible_and_depends_on_the_seed():
    f = _fit(z=5.0, p=2.0, sigma=0.5)
    a, b = ci.simulate_paths(f, 4, draws=500, seed=1), ci.simulate_paths(f, 4, draws=500, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, ci.simulate_paths(f, 4, draws=500, seed=2))
    assert a.shape == (500, 4)


def test_simulated_mean_matches_theorem_3_for_one_step():
    f = _fit(z=10.0, p=2.0, sigma=0.01, sigma_q=1.0, alpha=0.3)  # rho = Phi((1 - 2) / 1) = 0.1587
    paths = ci.simulate_paths(f, 1, draws=40_000, seed=3)
    assert paths[:, 0].mean() == pytest.approx(cu.m1_croston(f), abs=0.1)
    assert cu.m1_croston(f) == pytest.approx(NormalDist().cdf(-1.0) * 10.0)


def test_simulated_mean_matches_theorem_3_for_the_log_model():
    f = _fit(z=2.0, p=math.e, sigma=0.05, sigma_q=0.5, alpha=0.3, log=True)  # log interval level = 1
    paths = ci.simulate_paths(f, 1, draws=40_000, seed=4)
    assert paths[:, 0].mean() == pytest.approx(cu.m1_log_croston(f), rel=0.03)


def test_when_the_chance_of_demand_is_one_half_half_the_weeks_are_zero():
    paths = ci.simulate_paths(_fit(z=10.0, p=1.0, sigma=0.01, sigma_q=0.01), 1, draws=20_000, seed=5)
    assert (paths[:, 0] == 0).mean() == pytest.approx(0.5, abs=0.02)  # P = 1 rounds up to 2 half of the times: the model of the paper


def test_simulated_interval_truncates_negative_limits_only_when_asked():
    paths = np.array([[-2.0], [-1.0], [0.0], [5.0]] * 100)
    lower, upper, mean = ci.simulated_interval(paths, 1, truncate=True)
    assert lower == 0.0 and upper > 0 and mean == pytest.approx(0.5)
    assert ci.simulated_interval(paths, 1, truncate=False)[0] < 0


def test_idle_weeks_move_the_next_arrival_earlier_in_the_horizon():
    base = _fit(z=10.0, p=2.5, sigma=0.01, sigma_q=0.01)
    fresh = ci.simulate_paths(base, 1, draws=2_000, seed=6)[:, 0]
    idle = ci.simulate_paths(_fit(z=10.0, p=2.5, sigma=0.01, sigma_q=0.01, idle=2), 1, draws=2_000, seed=6)[:, 0]
    assert (fresh > 0).mean() == pytest.approx(0.0, abs=0.01)  # interval 2.5 rounds up to 3: next demand at week 3
    assert (idle > 0).mean() > 0.9  # last demand 2 weeks ago: next one at week 1
