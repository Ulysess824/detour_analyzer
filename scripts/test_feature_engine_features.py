"""Synthetic-data smoke test for FeatureEngineBuilder (feature-engine lag/rolling features)."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_engine_features import FeatureEngineBuilder


def build_panel() -> pl.DataFrame:
    """Two plants x one SKU each, 40 daily observations per series (chronological)."""
    rng = np.random.default_rng(7)
    dates = pl.date_range(
        pl.date(2026, 1, 1), pl.date(2026, 2, 9), interval="1d", eager=True
    )
    rows: list[dict] = []
    series = {
        ("PlantaA", "SKU1"): 50.0,
        ("PlantaB", "SKU2"): 12.0,
    }
    for (planta, sku), base in series.items():
        consumo = base + rng.normal(0.0, base * 0.1, size=len(dates))
        for fecha, c in zip(dates, consumo):
            rows.append(
                {
                    "planta": planta,
                    "sku": sku,
                    "fecha": fecha,
                    "consumo_real": round(float(max(c, 0.0)), 3),
                    "forecast_mensual": base * 20.0,
                }
            )
    return pl.DataFrame(rows)


def check_no_cross_series_leakage(df_out: pl.DataFrame) -> None:
    """The first row of every series must have NaN lag_1 (no value borrowed from another series)."""
    first_rows = (
        df_out.sort(["planta", "sku", "fecha"])
        .group_by(["planta", "sku"], maintain_order=True)
        .first()
    )
    leaked = first_rows.filter(pl.col("consumo_real_lag_1").is_not_null())
    assert leaked.height == 0, "lag_1 leaked across a series boundary"
    print("Leakage check: first row of each series has null lag_1 (OK)")


def main() -> None:
    panel = build_panel()

    # Chronological walk-forward split: first 30 days train, last 10 days test (per series).
    cutoff = pl.date(2026, 1, 30)
    df_train = panel.filter(pl.col("fecha") <= cutoff)
    df_test = panel.filter(pl.col("fecha") > cutoff)

    builder = FeatureEngineBuilder(
        value_cols="consumo_real",
        group_cols=("planta", "sku"),
        date_col="fecha",
        lags=(1, 2, 3, 5, 7),
        windows=(3, 5, 10),
        window_functions=("mean", "std", "max"),
        add_expanding=True,
        expanding_functions=("mean",),
    )

    df_train_feat, feature_cols = builder.fit_transform(df_train)
    df_test_feat, feature_cols_test = builder.transform(df_test)

    print("Train input shape:", df_train.shape)
    print("Train output shape:", df_train_feat.shape)
    print("Number of engineered columns:", len(feature_cols))
    print("Feature columns:", feature_cols)

    # Structure preserved: same rows, original columns untouched, only new columns added.
    assert df_train_feat.height == df_train.height, "row count changed"
    assert set(panel.columns).issubset(set(df_train_feat.columns)), "original columns dropped"
    assert df_train_feat.width == df_train.width + len(feature_cols), "unexpected column count"
    assert feature_cols == feature_cols_test, "train/test feature sets diverge"

    # Naming matches the requested style: consumo_real_lag_1, consumo_real_window_3_mean, ...
    assert "consumo_real_lag_1" in feature_cols
    assert "consumo_real_window_3_mean" in feature_cols
    assert "consumo_real_expanding_mean" in feature_cols

    check_no_cross_series_leakage(df_train_feat)

    # lag_1 equals the previous day's consumo_real within a series.
    sample = (
        df_train_feat.filter((pl.col("planta") == "PlantaA") & (pl.col("sku") == "SKU1"))
        .sort("fecha")
        .select(["fecha", "consumo_real", "consumo_real_lag_1", "consumo_real_window_3_mean"])
        .head(6)
    )
    print("\nSample (PlantaA / SKU1):")
    print(sample)

    expected_lag = sample["consumo_real"][0]
    got_lag = sample["consumo_real_lag_1"][1]
    assert abs(expected_lag - got_lag) < 1e-9, "lag_1 does not match previous consumo_real"

    # pandas round-trip returns a pandas DataFrame.
    df_pd_out, _ = builder.transform(df_test.to_pandas())
    import pandas as pd

    assert isinstance(df_pd_out, pd.DataFrame), "pandas input did not round-trip to pandas"

    # Automatic mode: a large battery is generated without specifying lags/windows/functions.
    auto_builder = FeatureEngineBuilder(value_cols="consumo_real", auto=True)
    df_auto_train, auto_cols = auto_builder.fit_transform(df_train)
    df_auto_test, auto_cols_test = auto_builder.transform(df_test)

    print("\nAuto mode: generated", len(auto_cols), "feature columns with no manual spec.")
    print("Auto sample columns:", auto_cols[:6], "...")

    assert df_auto_train.height == df_train.height, "auto mode changed row count"
    assert df_auto_train.width == df_train.width + len(auto_cols), "auto column count mismatch"
    assert auto_cols == auto_cols_test, "auto train/test feature sets diverge"
    assert len(auto_cols) > len(feature_cols), "auto should yield far more features than manual"
    # Broad grid materialised: deep lags, large windows and higher-moment statistics appear.
    assert "consumo_real_lag_14" in auto_cols, "auto lags not exploded"
    assert "consumo_real_window_30_mean" in auto_cols, "auto windows not exploded"
    assert "consumo_real_window_7_skew" in auto_cols, "auto window functions not exploded"
    check_no_cross_series_leakage(df_auto_train)

    print("\nAll FeatureEngineBuilder checks passed.")


if __name__ == "__main__":
    main()
