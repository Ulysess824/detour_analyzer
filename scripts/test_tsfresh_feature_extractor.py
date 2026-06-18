"""Synthetic-data smoke test for TSFreshFeatureExtractor.

Run from the repo root:
    python scripts/test_tsfresh_feature_extractor.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_features import TSFreshFeatureExtractor


def make_synthetic(n_skus: int = 3, n_days: int = 60, seed: int = 42) -> pl.DataFrame:
    r"""Build a polars panel with columns id (str), fecha (Date), consumo_real_to (Float64)."""
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 1)
    rows: list[dict] = []
    for s in range(n_skus):
        sku_id = f"cliente{s}_sku{s}"
        base = 50.0 + 10.0 * s
        for d in range(n_days):
            seasonal = 8.0 * np.sin(2 * np.pi * d / 7.0)
            noise = rng.normal(0.0, 3.0)
            value = max(0.0, base + seasonal + noise)
            rows.append(
                {
                    "id": sku_id,
                    "fecha": start + timedelta(days=d),
                    "consumo_real_to": float(value),
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("fecha").cast(pl.Date),
        pl.col("consumo_real_to").cast(pl.Float64),
    )


def main() -> None:
    df = make_synthetic()
    print(f"Synthetic panel: {df.height} rows, {df['id'].n_unique()} SKUs")

    extractor = TSFreshFeatureExtractor()

    x_filtrado = extractor.fit_transform(df)
    y = extractor.get_target()

    print("\n--- fit_transform ---")
    print(f"X_filtrado shape : {x_filtrado.shape}")
    print(f"y shape          : {y.shape}")
    print(f"selected columns : {len(extractor._selected_columns)}")
    assert x_filtrado.shape[0] == y.shape[0], "X and y must be aligned row-wise"
    assert x_filtrado.shape[0] > 0, "fit_transform produced no rows"
    assert list(x_filtrado.index) == list(y.index), "X and y indices must match"

    x_hoy = extractor.transform_latest(df)
    print("\n--- transform_latest ---")
    print(f"X_hoy shape      : {x_hoy.shape}")
    print(f"X_hoy index      : {list(x_hoy.index)}")
    assert x_hoy.shape[0] == df["id"].n_unique(), "one latest window per SKU expected"
    assert list(x_hoy.columns) == extractor._selected_columns, "columns must match selection"

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "extractor.joblib")
        extractor.save(path)
        reloaded = TSFreshFeatureExtractor.load(path)

    print("\n--- save / load ---")
    assert reloaded._selected_columns == extractor._selected_columns
    x_hoy_reloaded = reloaded.transform_latest(df)
    assert x_hoy_reloaded.shape == x_hoy.shape
    assert list(x_hoy_reloaded.columns) == list(x_hoy.columns)
    print("Reloaded extractor reproduces the same feature set.")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
