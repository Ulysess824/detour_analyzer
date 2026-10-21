"""
Prediction intervals of the Croston models (Shenstone and Hyndman 2005): closed form for the two modified models (Theorem 4,
equations 22 and 23) and simulated sample paths for the Croston and log-Croston models (section "Prediction intervals").

The paper writes alpha for both the smoothing parameter and the significance level; here the smoothing parameter is `alpha` and the
level of the interval is `level` (gamma = 1 - level). Negative limits are truncated at 0 by the callers that need it (`truncate`).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from src.experiments.croston_utils import CrostonFit

DRAWS = 10_000  # sample paths used by the paper
_NORMAL = NormalDist()


def modified_interval(f: CrostonFit, h: int = 1, level: float = 0.95) -> tuple[float, float]:
    r"""
    (lower, upper) of the `level` interval for the demand h weeks after the origin: modified Croston model when `f.log` is False
    (equation 22), modified log-Croston when True (equation 23). The lower limit of the first one can be negative.
    """
    gamma, p = 1.0 - level, f.p_bar
    if gamma * p / 2.0 >= 1.0:  # the chance of demand is below gamma / 2: both quantiles are 0
        return 0.0, 0.0
    delta = f.sigma * math.sqrt(1.0 + f.alpha ** 2 * (h - 1) / p)
    k1 = _NORMAL.inv_cdf(gamma * p / 2.0)
    upper = f.z - k1 * delta
    zero_mass_small = p < 2.0 / (2.0 - gamma)  # I(x) = x only then; k2 is defined only then
    k2_term = f.z + _NORMAL.inv_cdf(1.0 - p + gamma * p / 2.0) * delta if zero_mass_small else None
    if f.log:
        lower = max(0.0, math.exp(k2_term)) if zero_mass_small else 0.0  # min[0, exp(.)] = 0
        return lower, math.exp(upper)
    lower = min(0.0, f.z + k1 * delta)
    if k2_term is not None:
        lower = max(lower, k2_term)
    return lower, upper


def simulate_paths(f: CrostonFit, h: int, draws: int = DRAWS, seed: int = 0) -> np.ndarray:
    r"""
    Sample paths (draws x h) of the Croston model (5) when `f.log` is False and of the log-Croston model (6) when True, weeks
    origin+1 .. origin+h. Sizes and intervals follow the states of the fit; each interval is rounded up to a positive integer.
    The last demand is `f.idle` weeks before the origin and arrivals before the origin are discarded (the paper's Theorem 3 puts the last
    demand at the origin; the difference is small when `idle` is 0 or 1).
    """
    rng = np.random.default_rng(seed)
    z = np.full(draws, f.z)
    pq = np.full(draws, f.p_log if f.log else f.p)
    sigma_q = f.sigma_q_log if f.log else f.sigma_q
    t = np.full(draws, -float(f.idle))  # week of the last demand relative to the origin
    paths = np.zeros((draws, h))
    alive = np.ones(draws, dtype=bool)
    while alive.any():
        n = int(alive.sum())
        e_y = rng.normal(0.0, f.sigma, n)
        e_q = rng.normal(0.0, sigma_q, n)
        size = z[alive] + e_y
        size = np.exp(size) if f.log else size
        interval = pq[alive] + e_q
        interval = np.exp(interval) if f.log else interval
        z[alive] += f.alpha * e_y
        pq[alive] += f.alpha * e_q
        t[alive] += np.maximum(1.0, np.ceil(interval))
        rows = np.flatnonzero(alive)
        landed = (t[rows] >= 1) & (t[rows] <= h)
        paths[rows[landed], t[rows[landed]].astype(int) - 1] = size[landed]
        alive[rows[t[rows] >= h]] = False
    return paths


def simulated_interval(paths: np.ndarray, h: int, level: float = 0.95, truncate: bool = True) -> tuple[float, float, float]:
    r"""(lower, upper, mean) of week h of the simulated paths; limits truncated at 0 by default (the paper does so for negative limits)."""
    week = paths[:, h - 1]
    gamma = 1.0 - level
    lower, upper = (float(q) for q in np.quantile(week, [gamma / 2.0, 1.0 - gamma / 2.0]))
    if truncate:
        lower, upper = max(lower, 0.0), max(upper, 0.0)
    return lower, upper, float(week.mean())
