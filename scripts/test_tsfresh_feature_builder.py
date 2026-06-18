"""Synthetic-data smoke test for TSFreshFeatureBuilder (additive tsfresh panel features).

Run from the repo root:
    python scripts/test_tsfresh_feature_builder.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_engine_features import TSFreshFeatureBuilder


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


def main() -> None:
    panel = build_panel()

    # Chronological walk-forward split: first 30 days train, last 10 days test (per series).
    cutoff = pl.date(2026, 1, 30)
    df_train = panel.filter(pl.col("fecha") <= cutoff)
    df_test = panel.filter(pl.col("fecha") > cutoff)

    builder = TSFreshFeatureBuilder(
        value_cols="consumo_real",
        group_cols=("planta", "sku"),
        date_col="fecha",
        fc_parameters="minimal",
        max_timeshift=10,
        min_timeshift=3,
        n_jobs=0,
    )

    df_train_feat, feature_cols = builder.fit_transform(df_train)
    df_test_feat, feature_cols_test = builder.transform(df_test)

    print("Train input shape :", df_train.shape)
    print("Train output shape:", df_train_feat.shape)
    print("Number of tsfresh feature columns:", len(feature_cols))
    print("Sample feature columns:", feature_cols[:6])

    # Same shape rule: identical row count, original columns kept, only features appended.
    assert df_train_feat.height == df_train.height, "row count changed on train"
    assert df_test_feat.height == df_test.height, "row count changed on test"
    assert set(panel.columns).issubset(set(df_train_feat.columns)), "original columns dropped"
    assert df_train_feat.width == df_train.width + len(feature_cols), "unexpected column count"
    assert feature_cols == feature_cols_test, "train/test feature sets diverge"
    assert all(c.startswith("consumo_real__") for c in feature_cols), "unexpected feature naming"

    # 'efficient' must produce a strictly larger battery than 'minimal'.
    eff = TSFreshFeatureBuilder(fc_parameters="efficient", max_timeshift=10, min_timeshift=3)
    df_eff, eff_cols = eff.fit_transform(df_train)
    print("\nMinimal feature count :", len(feature_cols))
    print("Efficient feature count:", len(eff_cols))
    assert len(eff_cols) > len(feature_cols), "efficient should yield more features than minimal"
    assert df_eff.height == df_train.height, "efficient changed row count"

    # First min_timeshift rows of each series have no trailing window -> null features.
    first_rows = (
        df_train_feat.sort(["planta", "sku", "fecha"])
        .group_by(["planta", "sku"], maintain_order=True)
        .first()
    )
    assert first_rows[feature_cols[0]].null_count() == first_rows.height, (
        "first row of each series should have null tsfresh features (no past window)"
    )
    print("\nLeakage/edge check: first row of each series has null features (OK)")

    # pandas round-trip returns a pandas DataFrame.
    import pandas as pd

    df_pd_out, _ = builder.transform(df_test.to_pandas())
    assert isinstance(df_pd_out, pd.DataFrame), "pandas input did not round-trip to pandas"

    print("\nAll TSFreshFeatureBuilder checks passed.")


if __name__ == "__main__":
    main()
