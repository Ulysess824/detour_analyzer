from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import LagSelectionStrategy, LagSpec, min_max_normalize


class GreyRelationalSelector(LagSelectionStrategy):
    r"""
    Grey Relational Analysis selector (Deng): rank lags by grey relational grade (GRG).

    *   Reference sequence: the series itself; comparison sequences: its lagged copies.
    *   For each candidate lag j, GRG = mean over time of the grey relational coefficient
        xi(k) = (Dmin + rho*Dmax) / (D_j(k) + rho*Dmax), with Dmin/Dmax global over all lags.
    *   Keeps lags whose grade >= grade_threshold (top_k as a cap), always keeping lag 1.
    *   Suited to short series where AMI/FNN are unreliable (grey theory targets small samples).
    *
    """

    name = "gra"

    def __init__(
        self,
        max_lag: int = 20,
        rho: float = 0.5,
        grade_threshold: float = 0.6,
        top_k: int | None = 8,
        windows: tuple[int, ...] = (3, 5, 10),
    ) -> None:
        self.max_lag = max_lag
        self.rho = rho
        self.grade_threshold = grade_threshold
        self.top_k = top_k
        self.windows = list(windows)

    def _grades(self, series: np.ndarray) -> dict[int, float]:
        s = series[~np.isnan(series)]
        max_lag = min(self.max_lag, s.size // 3)
        if max_lag < 1:
            return {1: 1.0}

        # Reference aligned so every candidate lag is available
        ref = min_max_normalize(s[max_lag:])
        diffs: dict[int, np.ndarray] = {}
        for j in range(1, max_lag + 1):
            cand = min_max_normalize(s[max_lag - j: s.size - j])
            diffs[j] = np.abs(ref - cand)

        all_d = np.concatenate(list(diffs.values()))
        d_min = float(all_d.min())
        d_max = float(all_d.max())
        denom_const = self.rho * d_max

        grades: dict[int, float] = {}
        for j, d in diffs.items():
            coeff = (d_min + denom_const) / (d + denom_const)
            grades[j] = float(coeff.mean())
        return grades

    def _select_from_grades(self, grades: dict[int, float]) -> list[int]:
        selected = [j for j, g in grades.items() if g >= self.grade_threshold]
        if not selected:
            selected = [int(max(grades, key=grades.get))]
        if self.top_k is not None and len(selected) > self.top_k:
            selected = sorted(selected, key=lambda j: grades[j], reverse=True)[: self.top_k]
        if 1 not in selected:
            selected.append(1)
        return sorted(selected)

    def select(self, series: np.ndarray) -> LagSpec:
        grades = self._grades(series)
        return LagSpec(
            lags=self._select_from_grades(grades),
            windows=list(self.windows),
            method=self.name,
            metadata={"grades": grades},
        )

    def aggregate(self, specs: list[LagSpec]) -> LagSpec:
        acc: dict[int, list[float]] = defaultdict(list)
        for sp in specs:
            for j, g in sp.metadata["grades"].items():
                acc[j].append(g)
        mean_grades = {j: float(np.mean(v)) for j, v in acc.items()}
        return LagSpec(
            lags=self._select_from_grades(mean_grades),
            windows=list(self.windows),
            method=self.name,
            metadata={"grades": mean_grades},
        )
