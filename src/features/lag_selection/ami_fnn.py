from __future__ import annotations

import numpy as np

from .base import LagSelectionStrategy, LagSpec


def _average_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    r"""Compute average mutual information between two 1D arrays."""
    c_xy, bin_x, bin_y = np.histogram2d(x, y, bins=bins)
    c_x, _ = np.histogram(x, bins=bin_x)
    c_y, _ = np.histogram(y, bins=bin_y)

    n = len(x)
    p_xy = c_xy / n
    p_x = c_x / n
    p_y = c_y / n

    ami = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0.0 and p_x[i] > 0.0 and p_y[j] > 0.0:
                ami += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(ami)


def estimate_delay_ami(series: np.ndarray, max_tau: int = 30) -> int:
    r"""Estimate the delay tau using average mutual information first local minimum."""
    s = series[~np.isnan(series)]
    if s.size <= 2:
        return 1
    max_tau = min(max_tau, s.size // 3)
    if max_tau < 1:
        return 1

    ami_curve = []
    for k in range(1, max_tau + 1):
        ami = _average_mutual_information(s[:-k], s[k:])
        ami_curve.append(ami)

    for i in range(1, len(ami_curve) - 1):
        if ami_curve[i] < ami_curve[i - 1] and ami_curve[i] < ami_curve[i + 1]:
            return i + 1

    if len(ami_curve) > 0:
        return int(np.argmin(ami_curve) + 1)
    return 1


def estimate_dimension_fnn(
    series: np.ndarray,
    tau: int,
    max_m: int = 10,
    r_tol: float = 15.0,
    a_tol: float = 2.0,
) -> int:
    r"""Estimate embedding dimension m using false nearest neighbors method."""
    from scipy.spatial import KDTree

    s = series[~np.isnan(series)]
    n = s.size
    std_s = np.std(s)
    if std_s == 0.0:
        return 2

    fnn_percentages = []
    for m in range(1, max_m):
        start_idx = (m - 1) * tau
        if start_idx >= n - 1:
            break

        points_m = []
        for i in range(start_idx, n):
            vec = [s[i - j * tau] for j in range(m)]
            points_m.append(vec)

        points_m = np.array(points_m)
        num_points = len(points_m)
        if num_points < 3:
            break

        tree = KDTree(points_m)
        dists, indices = tree.query(points_m, k=2)
        r_d = dists[:, 1]
        nn_idx = indices[:, 1]

        false_neighbors = 0
        valid_points = 0

        for idx in range(num_points):
            orig_t = start_idx + idx
            if orig_t - m * tau < 0:
                continue

            nn_t = start_idx + nn_idx[idx]
            if nn_t - m * tau < 0:
                continue

            valid_points += 1
            dist_m = r_d[idx]
            if dist_m == 0.0:
                diff = abs(s[orig_t - m * tau] - s[nn_t - m * tau])
                if diff > 0.0:
                    false_neighbors += 1
                continue

            diff = abs(s[orig_t - m * tau] - s[nn_t - m * tau])
            cond1 = diff / dist_m > r_tol
            dist_m1 = np.sqrt(dist_m**2 + diff**2)
            cond2 = dist_m1 / std_s > a_tol

            if cond1 or cond2:
                false_neighbors += 1

        if valid_points == 0:
            break

        fnn_pct = false_neighbors / valid_points
        fnn_percentages.append(fnn_pct)

        if fnn_pct < 0.01:
            return m

    for m_idx, pct in enumerate(fnn_percentages):
        if pct < 0.10:
            return m_idx + 1

    if len(fnn_percentages) > 0:
        return int(np.argmin(fnn_percentages) + 1)
    return 2


class AmiFnnSelector(LagSelectionStrategy):
    r"""
    Takens delay-embedding selector: tau via Average Mutual Information, m via FNN.

    *   select: estimate tau and m, then produce uniform lags [1, 1+tau, ..., 1+(m-1)tau].
    *   aggregate: median tau and 75th-percentile m across series, rebuilt as uniform lags.
    *   Uses estimate_delay_ami / estimate_dimension_fnn defined in this module.
    *
    """

    name = "ami_fnn"

    def __init__(
        self,
        max_tau: int = 30,
        max_m: int = 10,
        windows: tuple[int, ...] = (3, 5, 10),
    ) -> None:
        self.max_tau = max_tau
        self.max_m = max_m
        self.windows = list(windows)

    def _uniform_lags(self, tau: int, m: int) -> list[int]:
        return [1 + k * tau for k in range(m)]

    def select(self, series: np.ndarray) -> LagSpec:
        tau = estimate_delay_ami(series, max_tau=self.max_tau)
        m = estimate_dimension_fnn(series, tau, max_m=self.max_m)
        return LagSpec(
            lags=self._uniform_lags(tau, m),
            windows=list(self.windows),
            method=self.name,
            metadata={"tau": tau, "m": m},
        )

    def aggregate(self, specs: list[LagSpec]) -> LagSpec:
        taus = [s.metadata["tau"] for s in specs]
        dims = [s.metadata["m"] for s in specs]
        tau = max(int(np.median(taus)), 1)
        m = max(int(np.percentile(dims, 75)), 2)
        return LagSpec(
            lags=self._uniform_lags(tau, m),
            windows=list(self.windows),
            method=self.name,
            metadata={"tau": tau, "m": m},
        )
