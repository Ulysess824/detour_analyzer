"""
Exploratory analysis of the long-format consumption table (planta, sku, fecha,
consumo) produced by transform_consumos.py, at two levels:

    - Per (planta, sku): one row per series with descriptive stats.
    - Per planta: aggregated across all sku of that planta.

Usage:
    python scripts/eda_consumos.py data/consumos_long.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fecha"])
    return df


def eda_by_planta_sku(df: pd.DataFrame) -> pd.DataFrame:
    r"""One row per (planta, sku): count, date span, coverage, and consumption stats."""
    g = df.groupby(["planta", "sku"])["consumo"]
    fecha_min = df.groupby(["planta", "sku"])["fecha"].min()
    fecha_max = df.groupby(["planta", "sku"])["fecha"].max()
    span_days = (fecha_max - fecha_min).dt.days + 1

    out = pd.DataFrame({
        "n_obs": g.count(),
        "fecha_min": fecha_min,
        "fecha_max": fecha_max,
        "span_days": span_days,
        "mean": g.mean().round(3),
        "std": g.std().round(3),
        "min": g.min(),
        "max": g.max(),
        "pct_zero": (g.apply(lambda s: (s == 0).mean() * 100)).round(1),
    })
    out["coverage_pct"] = (out["n_obs"] / out["span_days"] * 100).round(1)
    out["cv"] = (out["std"] / out["mean"]).round(3)

    return out.reset_index().sort_values(["planta", "sku"])


def eda_by_planta(df: pd.DataFrame) -> pd.DataFrame:
    r"""One row per planta: aggregated across all sku (n_sku, total/mean consumption, etc.)."""
    g = df.groupby("planta")["consumo"]
    n_sku = df.groupby("planta")["sku"].nunique()
    fecha_min = df.groupby("planta")["fecha"].min()
    fecha_max = df.groupby("planta")["fecha"].max()

    out = pd.DataFrame({
        "n_sku": n_sku,
        "n_obs": g.count(),
        "fecha_min": fecha_min,
        "fecha_max": fecha_max,
        "consumo_total": g.sum().round(1),
        "consumo_mean": g.mean().round(3),
        "consumo_std": g.std().round(3),
        "consumo_min": g.min(),
        "consumo_max": g.max(),
        "pct_zero": (g.apply(lambda s: (s == 0).mean() * 100)).round(1),
    })
    out["cv"] = (out["consumo_std"] / out["consumo_mean"]).round(3)
    out["obs_por_sku"] = (out["n_obs"] / out["n_sku"]).round(1)

    return out.reset_index().sort_values("consumo_total", ascending=False)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/eda_consumos.py <consumos_long.csv>")
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    df = load(input_path)

    by_planta_sku = eda_by_planta_sku(df)
    by_planta = eda_by_planta(df)

    out_dir = input_path.parent
    by_planta_sku.to_csv(out_dir / "eda_by_planta_sku.csv", index=False)
    by_planta.to_csv(out_dir / "eda_by_planta.csv", index=False)

    print(f"planta+sku series: {len(by_planta_sku):_}")
    print(f"plantas: {len(by_planta):_}")
    print(f"rows loaded: {len(df):_}")


if __name__ == "__main__":
    main()
