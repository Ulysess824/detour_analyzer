from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

FrameLike = pl.DataFrame | pd.DataFrame


def to_polars(df: FrameLike) -> pl.DataFrame:
    r"""Accept a polars or pandas DataFrame and return a polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(
        f"Expected a polars or pandas DataFrame, got {type(df).__name__}."
    )


def is_pandas(df: FrameLike) -> bool:
    r"""True when the input is a pandas DataFrame (used to round-trip the return type)."""
    return isinstance(df, pd.DataFrame)


@dataclass(frozen=True)
class LagSpec:
    r"""
    Common output contract produced by every lag-selection strategy.

    *   lags: explicit integer shifts to materialise as lag_{n} columns (e.g. [1, 4, 7]).
    *   windows: rolling window sizes to materialise as rolling_{mean,std,max}_{w}.
    *   method: name of the strategy that produced the spec.
    *   metadata: free-form diagnostics (tau, m, grey grades, pacf support, ...).
    *
    """

    lags: list[int]
    windows: list[int]
    method: str
    metadata: dict = field(default_factory=dict)


class LagSelectionStrategy(ABC):
    r"""
    Strategy interface for deriving a lag/window spec from consumption series.

    *   select: estimate a spec from a single chronological series.
    *   aggregate: reduce per-series specs to one global, rectangular spec.
    *   fit_panel: convenience that runs select over the longest series of a panel
        and aggregates them into a single global LagSpec (train data only).
    *
    """

    name: str

    @abstractmethod
    def select(self, series: np.ndarray) -> LagSpec:
        r"""Estimate the lag/window spec from a single chronological series."""

    @abstractmethod
    def aggregate(self, specs: list[LagSpec]) -> LagSpec:
        r"""Reduce per-series specs to one global, rectangular spec."""

    def fit_panel(
        self,
        df: FrameLike,
        target_col: str = "consumo_real",
        group_cols: tuple[str, str] = ("planta", "sku"),
        date_col: str = "fecha",
        max_series: int = 40,
        min_len: int = 30,
    ) -> LagSpec:
        r"""
        Estimate a global LagSpec from a panel by selecting per series and aggregating.

        *   Accepts a polars or pandas DataFrame (converted internally).
        *   Iterates over the longest series (up to max_series) to keep estimation stable.
        *   Falls back to a single-series spec when only one series qualifies.
        *
        """
        df = to_polars(df)
        lengths = (
            df.group_by(list(group_cols)).len().sort("len", descending=True)
        )
        groups = lengths.head(max_series).select(list(group_cols)).rows()

        specs: list[LagSpec] = []
        for keys in groups:
            mask = pl.lit(True)
            for col, value in zip(group_cols, keys):
                mask = mask & (pl.col(col) == value)
            s = (
                df.filter(mask)
                .sort(date_col)[target_col]
                .to_numpy()
                .astype(float)
            )
            if s.size < min_len:
                continue
            specs.append(self.select(s))

        if not specs:
            # Not enough history anywhere: fall back to a minimal persistence spec
            return LagSpec(lags=[1], windows=[3, 5, 10], method=self.name, metadata={})
        if len(specs) == 1:
            return specs[0]
        return self.aggregate(specs)


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    r"""Grey relational generating: scale a sequence to [0, 1] (flat -> zeros)."""
    rng = float(x.max() - x.min())
    if rng == 0.0:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def build_lag_features(
    df: FrameLike,
    spec: LagSpec,
    consumo_col: str = "consumo_real",
    forecast_col: str = "forecast_mensual",
    date_col: str = "fecha",
    group_cols: tuple[str, str] = ("planta", "sku"),
) -> tuple[FrameLike, list[str]]:
    r"""
    Compact, evaluation-only feature builder driven by a LagSpec.

    *   Accepts a polars or pandas DataFrame and returns the same flavour it received.
    *   Materialises lag_{n} columns for each shift in spec.lags and
        rolling_{mean,std,max}_{w} for each window in spec.windows.
    *   Adds daily_forecast plus two calendar features so the comparison across
        strategies is fair (everything but the lag block is held constant).
    *   This is intentionally separate from build_features (additive design): it exists
        only so LagFeatureValidator can score lag strategies head to head.
    *
    """
    return_pandas = is_pandas(df)
    df = to_polars(df)
    groups = list(group_cols)
    df = df.sort(groups + [date_col])

    base_lag1 = pl.col(consumo_col).shift(1).over(groups)
    exprs: list[pl.Expr] = [base_lag1.alias("_consumo_lag1")]
    df = df.with_columns(exprs)

    feature_cols: list[str] = []

    lag_exprs = [
        pl.col(consumo_col).shift(n).over(groups).alias(f"lag_{n}")
        for n in spec.lags
    ]
    feature_cols.extend(f"lag_{n}" for n in spec.lags)

    rolling_exprs: list[pl.Expr] = []
    for w in spec.windows:
        rolling_exprs.extend([
            pl.col("_consumo_lag1").rolling_mean(window_size=w).over(groups).alias(f"rolling_mean_{w}"),
            pl.col("_consumo_lag1").rolling_std(window_size=w).over(groups).alias(f"rolling_std_{w}"),
            pl.col("_consumo_lag1").rolling_max(window_size=w).over(groups).alias(f"rolling_max_{w}"),
        ])
        feature_cols.extend([f"rolling_mean_{w}", f"rolling_std_{w}", f"rolling_max_{w}"])

    calendar_exprs = [
        (pl.col(forecast_col) / 20.0).alias("daily_forecast"),
        pl.col(date_col).dt.day().alias("day_of_month"),
        (pl.col(date_col).dt.month_end() - pl.col(date_col)).dt.total_days().alias("days_to_end_of_month"),
    ]
    feature_cols.extend(["daily_forecast", "day_of_month", "days_to_end_of_month"])

    # Guarantee a lag_1 column for the persistence baseline (not added as a feature
    # unless the spec already requested it, to avoid duplicating the column).
    if 1 not in spec.lags:
        lag_exprs.append(pl.col(consumo_col).shift(1).over(groups).alias("lag_1"))

    df = df.with_columns(lag_exprs + rolling_exprs + calendar_exprs)
    df = df.drop(["_consumo_lag1"])
    if return_pandas:
        return df.to_pandas(), feature_cols
    return df, feature_cols
