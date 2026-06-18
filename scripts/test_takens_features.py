r"""
Synthetic-data test for the Takens delay-embedding features in build_features.

*   Builds a deterministic system (noisy sine) so AMI/FNN have a known structure.
*   Verifies estimate_delay_ami and estimate_dimension_fnn return sane values.
*   Verifies build_features produces embed_* columns, no train/test leakage in (tau, m),
    and that the embedding metadata travels inside sku_stats.
"""

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import (
    build_features,
    estimate_delay_ami,
    estimate_dimension_fnn,
)


def _make_synthetic(n_plants: int = 2, n_skus: int = 5, n_days: int = 200) -> pl.DataFrame:
    r"""Build a synthetic consumption frame with a clear periodic signal per series."""
    rng = np.random.default_rng(42)
    dates = pl.date_range(
        pl.date(2025, 1, 1), pl.date(2025, 1, 1) + pl.duration(days=n_days - 1),
        interval="1d", eager=True,
    )
    rows = []
    for p in range(n_plants):
        for k in range(n_skus):
            t = np.arange(n_days)
            # Periodic deterministic core (period 14) + mild noise -> low-dim dynamics
            signal = 6.0 + 4.0 * np.sin(2.0 * np.pi * t / 14.0)
            signal = signal + rng.normal(0.0, 0.4, size=n_days)
            signal = np.clip(signal, 0.5, None)
            for i, d in enumerate(dates):
                rows.append({
                    "planta": f"P{p}",
                    "sku": f"SKU_{k:02d}",
                    "fecha": d,
                    "consumo_real": float(signal[i]),
                    "forecast_mensual": 100.0,
                })
    return pl.DataFrame(rows)


def main() -> None:
    # 1. Parameter estimators on a clean periodic series (period 14)
    t = np.arange(400)
    s = 5.0 + 3.0 * np.sin(2.0 * np.pi * t / 14.0)
    tau = estimate_delay_ami(s)
    m = estimate_dimension_fnn(s, tau)
    print(f"Clean sine (period 14): tau={tau}, m={m}")
    assert 1 <= tau <= 14, "tau should be a fraction of the period"
    assert 2 <= m <= 6, "a smooth periodic orbit needs a small embedding dimension"

    # 2. End-to-end build_features with a chronological train/test split
    df = _make_synthetic()
    cutoff = pl.date(2025, 6, 1)
    df_train_raw = df.filter(pl.col("fecha") <= cutoff)
    df_test_raw = df.filter(pl.col("fecha") > cutoff)

    df_train, sku_stats = build_features(df_train_raw)
    df_test, _ = build_features(df_test_raw, sku_stats=sku_stats)

    # 3. Embedding columns are present and consistent across train/test
    embed_cols_train = sorted([c for c in df_train.columns if c.startswith("embed_")])
    embed_cols_test = sorted([c for c in df_test.columns if c.startswith("embed_")])
    print(f"Embedding columns: {embed_cols_train}")
    assert embed_cols_train, "no embed_* columns were generated"
    assert embed_cols_train == embed_cols_test, "train/test embedding dimensions differ"

    # 4. (tau, m) metadata travels inside sku_stats and is not leaked as a feature
    assert "_embed_tau" in sku_stats.columns and "_embed_m" in sku_stats.columns
    assert "_embed_tau" not in df_train.columns, "embedding metadata leaked into features"
    used_m = int(sku_stats["_embed_m"][0])
    used_tau = int(sku_stats["_embed_tau"][0])
    print(f"Persisted embedding: tau={used_tau}, m={used_m}")
    assert len(embed_cols_train) == used_m, "number of embed_* columns must equal m"

    # 5. embed_0 equals lag_1 (most recent legitimate observation), no NaN beyond warmup
    same = df_train.select((pl.col("embed_0") == pl.col("lag_1")).fill_null(True).all()).item()
    assert same, "embed_0 must equal lag_1"

    print("\nSample of generated embedding features:")
    print(df_train.select(["planta", "sku", "fecha"] + embed_cols_train).drop_nulls().head(3))
    print("\nAll Takens feature tests passed.")


if __name__ == "__main__":
    main()
