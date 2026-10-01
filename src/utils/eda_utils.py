"""Descriptive statistics of the consumption table, per series and per planta."""

from __future__ import annotations

import pandas as pd


def _pct_zero(values: pd.Series) -> float:
    return (values == 0).mean() * 100


def eda_by_planta_sku(df: pd.DataFrame) -> pd.DataFrame:
    r"""One row per (planta, sku): count, date span, coverage and consumption statistics."""
    group = df.groupby(["planta", "sku"])
    consumo = group["consumo"]
    first, last = group["fecha"].min(), group["fecha"].max()

    out = pd.DataFrame(
        {
            "n_obs": consumo.count(),
            "fecha_min": first,
            "fecha_max": last,
            "span_days": (last - first).dt.days + 1,
            "mean": consumo.mean().round(3),
            "std": consumo.std().round(3),
            "min": consumo.min(),
            "max": consumo.max(),
            "pct_zero": consumo.apply(_pct_zero).round(1),
        }
    )
    out["coverage_pct"] = (out["n_obs"] / out["span_days"] * 100).round(1)
    out["cv"] = (out["std"] / out["mean"]).round(3)
    return out.reset_index().sort_values(["planta", "sku"])


def eda_by_planta(df: pd.DataFrame) -> pd.DataFrame:
    r"""One row per planta, aggregated across all its sku."""
    group = df.groupby("planta")
    consumo = group["consumo"]

    out = pd.DataFrame(
        {
            "n_sku": group["sku"].nunique(),
            "n_obs": consumo.count(),
            "fecha_min": group["fecha"].min(),
            "fecha_max": group["fecha"].max(),
            "consumo_total": consumo.sum().round(1),
            "consumo_mean": consumo.mean().round(3),
            "consumo_std": consumo.std().round(3),
            "consumo_min": consumo.min(),
            "consumo_max": consumo.max(),
            "pct_zero": consumo.apply(_pct_zero).round(1),
        }
    )
    out["cv"] = (out["consumo_std"] / out["consumo_mean"]).round(3)
    out["obs_por_sku"] = (out["n_obs"] / out["n_sku"]).round(1)
    return out.reset_index().sort_values("consumo_total", ascending=False)
