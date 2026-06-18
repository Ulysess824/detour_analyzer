from __future__ import annotations

from collections import Counter

import numpy as np
from statsmodels.tsa.stattools import pacf

from .base import LagSelectionStrategy, LagSpec


class AcfPacfSelector(LagSelectionStrategy):
    r"""
    Box-Jenkins selector: keep lags with a statistically significant partial autocorrelation.

    *   select: compute the PACF with confidence intervals; a lag is significant when its
        confidence band excludes zero. This estimates the effective AR order directly.
    *   aggregate: keep lags significant in at least half of the series (majority support).
    *
    """

    name = "acf_pacf"

    def __init__(
        self,
        max_lag: int = 20,
        alpha: float = 0.05,
        windows: tuple[int, ...] = (3, 5, 10),
    ) -> None:
        self.max_lag = max_lag
        self.alpha = alpha
        self.windows = list(windows)

    def select(self, series: np.ndarray) -> LagSpec:
        s = series[~np.isnan(series)]
        # statsmodels requires nlags < n/2
        nlags = min(self.max_lag, s.size // 2 - 1)
        if nlags < 1:
            return LagSpec([1], list(self.windows), self.name, {"nlags": 0})

        values, confint = pacf(s, nlags=nlags, alpha=self.alpha)
        selected: list[int] = []
        for j in range(1, nlags + 1):
            lo, hi = confint[j]
            # Significant if the confidence interval around the estimate excludes zero
            if lo > 0 or hi < 0:
                selected.append(j)
        if not selected:
            selected = [1]
        return LagSpec(
            lags=sorted(selected),
            windows=list(self.windows),
            method=self.name,
            metadata={"pacf": values.tolist(), "nlags": nlags},
        )

    def aggregate(self, specs: list[LagSpec]) -> LagSpec:
        support: Counter = Counter()
        for sp in specs:
            support.update(sp.lags)
        n = len(specs)
        selected = [j for j, cnt in support.items() if cnt >= n / 2]
        if not selected:
            selected = [1]
        return LagSpec(
            lags=sorted(selected),
            windows=list(self.windows),
            method=self.name,
            metadata={"support": dict(support), "n_series": n},
        )
