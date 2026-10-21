"""Croston point forecasts: hand-computed states, no leakage, the corrections of the paper, the guards."""

import math

import numpy as np
import pytest

from src.experiments import croston_utils as cu

Y = np.array([0, 4, 0, 0, 6, 2, 0, 8], dtype=float)  # sizes 4, 6, 2, 8 at weeks 2, 5, 6, 8 -> intervals 3, 1, 2


def test_demands_sizes_and_intervals():
    sizes, q = cu.demands(Y)
    assert list(sizes) == [4, 6, 2, 8]
    assert list(q) == [3, 1, 2]


def test_croston_states_by_hand():
    f = cu.fit(Y, alpha=0.5)
    # Z: 4 -> 5 -> 3.5 -> 5.75 ; P starts at the first interval: 3 -> 2 -> 2
    assert f.z == pytest.approx(5.75)
    assert f.p == pytest.approx(2.0)
    assert cu.f_croston(f) == pytest.approx(5.75 / 2.0)
    assert f.p_bar == pytest.approx(2.0)
    assert f.n_demands == 4


def test_sigma_is_the_rms_of_the_one_step_errors_of_the_sizes():
    f = cu.fit(Y, alpha=0.5)
    errors = [6 - 4, 2 - 5, 8 - 3.5]
    assert f.sigma == pytest.approx(math.sqrt(np.mean(np.square(errors))))


def test_sb0_is_the_croston_forecast_times_one_minus_half_alpha():
    f = cu.fit(Y, alpha=0.2)
    assert cu.f_sb0(f) == pytest.approx((1 - 0.1) * cu.f_croston(f))


def test_sb1_is_equal_to_the_ratio_when_every_interval_is_one():
    f = cu.fit(np.array([3.0, 5.0, 4.0, 6.0]), alpha=0.3)
    assert f.p == pytest.approx(1.0)
    assert cu.f_sb1(f) == pytest.approx(cu.f_croston(f))


def test_constant_series_gives_the_constant():
    f = cu.fit(np.full(10, 7.0), alpha=0.4)
    assert cu.f_croston(f) == pytest.approx(7.0)
    assert cu.m_modified(f) == pytest.approx(7.0)
    assert f.sigma == pytest.approx(cu.SIGMA_FLOOR)


def test_modified_models_by_hand_with_no_zero_weeks():
    y = np.array([2.0, 4.0, 3.0, 5.0])
    f = cu.fit(y, alpha=0.5)
    assert cu.m_modified(f, h=1) == pytest.approx(f.z)  # p = 1
    flog = cu.fit(y, alpha=0.5, log=True)
    assert cu.m_modified_log(flog, h=1) == pytest.approx(math.exp(flog.z + flog.sigma ** 2 / 2))
    assert cu.m_modified_log(flog, h=3) > cu.m_modified_log(flog, h=1)  # the variance term grows with h


def test_one_step_croston_models_are_at_most_half_of_the_level_when_intervals_are_one():
    f = cu.fit(np.array([3.0, 5.0, 4.0, 6.0]), alpha=0.3)
    assert cu.m1_croston(f) == pytest.approx(0.5 * f.z)  # Phi(0) = 0.5 because P = 1: a property of the model in the paper
    flog = cu.fit(np.array([3.0, 5.0, 4.0, 6.0]), alpha=0.3, log=True)
    assert cu.m1_log_croston(flog) < math.exp(flog.z + flog.sigma ** 2 / 2)


def test_alpha_is_estimated_from_the_given_series_only_and_is_in_the_grid():
    y = np.array([0, 5, 1, 7, 0, 2, 9, 4, 0, 6], dtype=float)
    f = cu.fit(y)
    assert f.alpha in cu.ALPHA_GRID
    assert cu.fit(y[:-1], alpha=f.alpha).z != f.z  # the fit reads exactly the values it is given: callers cut the series at the origin
    assert cu.fit(y).alpha == f.alpha  # deterministic


def test_alpha_is_one_for_a_random_walk_and_small_for_noise():
    walk = np.cumsum(np.ones(30)) * 2.0
    assert cu.fit_alpha(walk) == 1.0  # the best level of a ramp is the last value
    noise = np.tile([4.0, 6.0], 15)
    assert cu.fit_alpha(noise) <= 0.5


def test_too_few_demands_gives_none():
    assert cu.fit(np.array([0.0, 3.0, 0.0, 5.0])) is None
    assert cu.fit(np.zeros(8)) is None
