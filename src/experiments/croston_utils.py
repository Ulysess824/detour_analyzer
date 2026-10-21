"""
Croston's method and the stochastic models behind it (Shenstone and Hyndman 2005, Journal of Forecasting 24, 389-402).

All functions read a series of non-negative weekly values up to the origin and nothing after it. Equation numbers refer to the paper.
The paper gives no estimation rule: alpha is the value in ALPHA_GRID that minimises the squared one-step errors of the demand sizes
(the intervals are almost always 1 here, so they carry no information about alpha), and the same alpha is used for the intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

ALPHA_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
MIN_DEMANDS = 3  # sizes need 2 errors and intervals need 2 values
SIGMA_FLOOR = 1e-6  # keeps Phi((1 - P) / sigma) defined when every interval is the same
C_SB1 = 100.0  # constant of F_sb1 suggested by Syntetos and Boylan (2001)

_NORMAL = NormalDist()


def demands(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Sizes Y*_j (positive values) and intervals Q_j (weeks between positive values); Q has one value less than the sizes."""
    y = np.asarray(y, dtype=float)
    weeks = np.flatnonzero(y > 0)
    return y[weeks], np.diff(weeks).astype(float)


def ses(x: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    r"""Simple exponential smoothing started at the first value: the final level and the one-step errors x_j - level_{j-1}."""
    level = float(x[0])
    errors = np.empty(len(x) - 1)
    for j in range(1, len(x)):
        errors[j - 1] = x[j] - level
        level += alpha * errors[j - 1]
    return level, errors


def fit_alpha(sizes: np.ndarray, grid: np.ndarray = ALPHA_GRID) -> float:
    r"""alpha of the grid with the smallest sum of squared one-step errors of `sizes` (the first of the grid in a tie)."""
    sse = [float(np.sum(ses(sizes, a)[1] ** 2)) for a in grid]
    return float(grid[int(np.argmin(sse))])


@dataclass(frozen=True)
class CrostonFit:
    alpha: float
    z: float  # Z_l: level of the (log) sizes
    p: float  # P_l: level of the intervals
    sigma: float  # standard deviation of the one-step errors of the (log) sizes
    sigma_q: float  # the same for the intervals
    p_bar: float  # mean interval (the geometric parameter p of the modified models)
    n_demands: int
    log: bool


def fit(y: np.ndarray, alpha: float | None = None, log: bool = False) -> CrostonFit | None:
    r"""
    Fit the series `y` (weekly values up to the origin). `alpha` None = estimate it; `log` smooths log sizes (equations 6 and 9).
    Returns None when there are fewer than MIN_DEMANDS positive values.
    """
    sizes, q = demands(y)
    if len(sizes) < MIN_DEMANDS:
        return None
    x = np.log(sizes) if log else sizes
    alpha = fit_alpha(x) if alpha is None else float(alpha)
    z, e = ses(x, alpha)
    p, eq = ses(q, alpha)  # P_2 = Q_2: the first demand has no interval before it
    sigma = max(math.sqrt(float(np.mean(e ** 2))), SIGMA_FLOOR)
    sigma_q = max(math.sqrt(float(np.mean(eq ** 2))) if len(eq) else 0.0, SIGMA_FLOOR)
    return CrostonFit(alpha, z, p, sigma, sigma_q, float(np.mean(q)), len(sizes), log)


def f_croston(f: CrostonFit) -> float:
    r"""F_cr = Z_l / P_l (equation 3)."""
    return f.z / f.p


def f_sb0(f: CrostonFit) -> float:
    r"""F_sb0 = (1 - alpha / 2) Z_l / P_l (Syntetos and Boylan 2000)."""
    return (1.0 - f.alpha / 2.0) * f.z / f.p


def f_sb1(f: CrostonFit, c: float = C_SB1) -> float:
    r"""F_sb1 = Z_l / (P_l c^(P_l - 1)), as printed in the paper (Syntetos and Boylan 2001)."""
    return f.z / (f.p * c ** (f.p - 1.0))


def m_modified(f: CrostonFit, h: int = 1) -> float:
    r"""Modified Croston model (7): m_h = Z_l / p (equation 11), p = mean interval. Needs a fit with log=False."""
    return f.z / f.p_bar


def m_modified_log(f: CrostonFit, h: int = 1) -> float:
    r"""Modified log-Croston model (9): m_h = exp{Z_l + sigma^2 [p + alpha^2 (h-1)] / (2p)} / p (equation 16). Needs log=True."""
    p = f.p_bar
    return math.exp(f.z + f.sigma ** 2 * (p + f.alpha ** 2 * (h - 1)) / (2.0 * p)) / p


def m1_croston(f: CrostonFit) -> float:
    r"""Croston model (5), one step ahead (Theorem 3): m_1 = Phi[(1 - P_l) / sigma_Q] Z_l. It is at most Z_l / 2 because P_l >= 1."""
    return _NORMAL.cdf((1.0 - f.p) / f.sigma_q) * f.z


def m1_log_croston(f: CrostonFit) -> float:
    r"""log-Croston model (6), one step ahead (Theorem 3): m_1 = Phi(-P_l / sigma_Q) exp{Z_l + sigma^2 / 2}."""
    return _NORMAL.cdf(-f.p / f.sigma_q) * math.exp(f.z + f.sigma ** 2 / 2.0)
