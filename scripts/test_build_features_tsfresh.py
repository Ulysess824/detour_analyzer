"""Synthetic-data smoke test for the tsfresh-based build_features."""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_features import build_features


def build_panel(n_days: int = 70, seed: int = 0) -> pl.DataFrame:
    """Two series with a strong AR(1) consumption so t+1 is predictable from the past."""
    rng = np.random.default_rng(seed)
    fechas = [date(2026, 1, 1) + timedelta(days=i) for i in range(n_days)]

    frames = []
    for planta, sku, base in [("PlantaA", "G1_S1_80_100", 20.0), ("PlantaB", "G2_S2_90_120", 8.0)]:
        s = np.empty(n_days)
        s[0] = base
        for t in range(1, n_days):
            s[t] = max(0.0, 0.8 * s[t - 1] + rng.normal(0, base * 0.1) + base * 0.2)
        df = pl.DataFrame({
            "planta": planta,
            "sku": sku,
            "fecha": fechas,
            "consumo_real": s,
            "consumo_real_to": s,
            "forecast_mensual": base * 20.0,
        }).with_columns(
            daily_forecast=pl.col("forecast_mensual") / 20.0,
            # Target for t+1: next-day consumption within the series
            target_consumo_real=pl.col("consumo_real").shift(-1).over(["planta", "sku"]),
        )
        frames.append(df)

    return pl.concat(frames).sort(["planta", "sku", "fecha"])


def main() -> None:
    panel = build_panel()
    # Chronological split: first 70% train, rest test (walk-forward style, no shuffling)
    cutoff = panel["fecha"].unique().sort()[int(len(panel["fecha"].unique()) * 0.7)]
    df_train = panel.filter(pl.col("fecha") <= cutoff)
    df_test = panel.filter(pl.col("fecha") > cutoff)

    # TRAIN pass: estimate + select features
    out_train, sku_stats = build_features(
        df_train, max_timeshift=10, min_timeshift=3
    )
    selected = sku_stats["selected_columns"]
    print("Selected features:", len(selected))
    print("Sample:", selected[:5])

    assert isinstance(sku_stats, dict)
    assert len(selected) > 0, "FRESH selection (with fallback) must yield features"
    assert set(selected).issubset(out_train.columns), "selected cols missing in train output"
    assert "lag_1" in out_train.columns

    # TEST pass: reuse the stored spec, no re-selection (no leakage)
    out_test, _ = build_features(df_test, sku_stats=sku_stats, max_timeshift=10, min_timeshift=3)
    assert set(selected).issubset(out_test.columns), "selected cols missing in test output"

    # Train and test must expose the exact same feature set
    train_feats = [c for c in out_train.columns if c in selected]
    test_feats = [c for c in out_test.columns if c in selected]
    assert train_feats == test_feats, "train/test feature sets differ"

    # pandas round-trip
    import pandas as pd
    out_pd, _ = build_features(df_train.to_pandas(), sku_stats=sku_stats,
                              max_timeshift=10, min_timeshift=3)
    assert isinstance(out_pd, pd.DataFrame), "pandas input must return pandas"

    print("\nTrain shape:", out_train.shape, "| Test shape:", out_test.shape)
    print("All assertions passed.")


if __name__ == "__main__":
    main()
