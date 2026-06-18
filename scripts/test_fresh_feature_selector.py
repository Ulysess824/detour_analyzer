"""Synthetic-data smoke test for FreshFeatureSelector.

Chains TSFreshFeatureBuilder (the builder under test in test_tsfresh_feature_builder.py)
with FreshFeatureSelector, proving the selector consumes the builder's (df, feature_cols)
output directly and reproduces the FRESH selection methodology.

Run from the repo root:
    python scripts/test_fresh_feature_selector.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_engine_features import (
    FreshFeatureSelector,
    TSFreshFeatureBuilder,
)


def build_panel(n_days: int = 70, seed: int = 0) -> pl.DataFrame:
    """Two strongly autocorrelated AR(1) series, so past-window stats predict the next day."""
    rng = np.random.default_rng(seed)
    fechas = [date(2026, 1, 1) + timedelta(days=i) for i in range(n_days)]

    frames = []
    for planta, sku, base in [("PlantaA", "SKU1", 50.0), ("PlantaB", "SKU2", 12.0)]:
        s = np.empty(n_days)
        s[0] = base
        for t in range(1, n_days):
            s[t] = max(0.0, 0.9 * s[t - 1] + rng.normal(0, base * 0.05) + base * 0.1)
        df = pl.DataFrame(
            {
                "planta": planta,
                "sku": sku,
                "fecha": fechas,
                "consumo_real": s,
                "forecast_mensual": base * 20.0,
            }
        ).with_columns(
            # Next-day target within the series (chronological, no leakage).
            target_consumo_real=pl.col("consumo_real").shift(-1).over(["planta", "sku"])
        )
        frames.append(df)
    return pl.concat(frames).sort(["planta", "sku", "fecha"])


def main() -> None:
    panel = build_panel()
    cutoff = pl.date(2026, 2, 19)  # ~70 percent for train (walk-forward, no shuffle)
    df_train = panel.filter(pl.col("fecha") <= cutoff)
    df_test = panel.filter(pl.col("fecha") > cutoff)

    # 1) Build the tsfresh battery (this is exactly the test_tsfresh_feature_builder output).
    builder = TSFreshFeatureBuilder(fc_parameters="efficient", max_timeshift=10, min_timeshift=3)
    df_train_feat, feature_cols = builder.fit_transform(df_train)
    df_test_feat, _ = builder.transform(df_test)
    print("Built features:", len(feature_cols), "| train shape:", df_train_feat.shape)

    # 2) FRESH selection against the next-day target (fit on TRAIN only).
    selector = FreshFeatureSelector(fdr_level=0.05)
    df_train_sel, kept = selector.fit_transform(
        df_train_feat, feature_cols, y="target_consumo_real"
    )
    df_test_sel, kept_test = selector.transform(df_test_feat)

    table = selector.relevance_table_
    print("Relevance table rows:", len(table), "| selected:", len(kept))
    print("Target kind:", table["target_kind"].iloc[0])
    print("Tests used:", sorted(table["test"].unique()))
    print("Top relevant features:")
    print(table.loc[table["relevant"]].head(5)[["feature", "test", "p_value", "p_value_adjusted"]])

    # --- Methodology / contract checks ---
    assert len(table) == len(feature_cols), "relevance table must score every candidate feature"
    assert set(kept).issubset(set(feature_cols)), "selected features must be a subset of candidates"
    assert kept == kept_test, "train/test selected sets must match (no re-selection on test)"
    assert len(kept) > 0, "strong AR signal should yield at least one relevant feature"

    # Real-valued target -> continuous features tested with Kendall's tau.
    assert table["target_kind"].iloc[0] == "real", "next-day consumption is a real target"
    assert "kendall" in set(table["test"]), "continuous feature vs real target should use Kendall"

    # Selector output = panel minus the rejected feature columns; rows preserved.
    rejected = [c for c in feature_cols if c not in set(kept)]
    assert df_train_sel.height == df_train_feat.height, "selector changed the row count"
    assert all(c in df_train_sel.columns for c in kept), "kept features dropped by mistake"
    assert all(c not in df_train_sel.columns for c in rejected), "rejected features not dropped"
    assert "planta" in df_train_sel.columns and "consumo_real" in df_train_sel.columns, (
        "non-feature columns must be preserved"
    )

    # Benjamini-Yekutieli sanity: adjusted p-values dominate raw ones, all within [0, 1].
    assert (table["p_value_adjusted"] >= table["p_value"] - 1e-9).all(), "BY must not shrink p-values"
    assert ((table["p_value_adjusted"] >= 0) & (table["p_value_adjusted"] <= 1)).all()
    # Every rejected H0 must clear the FDR threshold on its adjusted p-value.
    assert (table.loc[table["relevant"], "p_value_adjusted"] <= 0.05 + 1e-9).all(), (
        "relevant features must satisfy the FDR level"
    )

    # pandas round-trip.
    import pandas as pd

    df_pd_sel, _ = selector.transform(df_test_feat.to_pandas())
    assert isinstance(df_pd_sel, pd.DataFrame), "pandas input did not round-trip to pandas"

    # Passing y as an array instead of a column name must give the same selection.
    sel_arr = FreshFeatureSelector(fdr_level=0.05)
    y_arr = df_train_feat["target_consumo_real"].to_numpy()
    _, kept_arr = sel_arr.fit_transform(df_train_feat, feature_cols, y=y_arr)
    assert kept_arr == kept, "array target and column target must select identically"

    print("\nAll FreshFeatureSelector checks passed.")


if __name__ == "__main__":
    main()
