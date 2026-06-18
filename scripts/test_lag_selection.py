r"""
Synthetic-data test for the lag_selection module (Strategy + validator design).

*   Exercises the three selectors (AMI+FNN, GRA, ACF/PACF) on a periodic panel.
*   Verifies each produces a valid LagSpec and that fit_panel aggregates across series.
*   Runs LagFeatureValidator end to end and checks it ranks strategies and picks a winner.
"""

import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.lag_selection import (
    AcfPacfSelector,
    AmiFnnSelector,
    GreyRelationalSelector,
    LagFeatureValidator,
    LagSpec,
    build_lag_features,
)


def _make_panel(n_plants: int = 2, n_skus: int = 6, n_days: int = 220) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    dates = pl.date_range(
        pl.date(2025, 1, 1), pl.date(2025, 1, 1) + pl.duration(days=n_days - 1),
        interval="1d", eager=True,
    )
    rows = []
    for p in range(n_plants):
        for k in range(n_skus):
            t = np.arange(n_days)
            signal = 6.0 + 4.0 * np.sin(2.0 * np.pi * t / 7.0)  # weekly cycle
            signal = np.clip(signal + rng.normal(0.0, 0.5, n_days), 0.5, None)
            for i, d in enumerate(dates):
                rows.append({
                    "planta": f"P{p}", "sku": f"SKU_{k:02d}", "fecha": d,
                    "consumo_real": float(signal[i]), "forecast_mensual": 100.0,
                })
    df = pl.DataFrame(rows).sort(["planta", "sku", "fecha"])
    # Regression target for t+1 (mirrors transform_data)
    df = df.with_columns(
        target_consumo_real=pl.col("consumo_real").shift(-1).over(["planta", "sku"])
    ).drop_nulls(subset=["target_consumo_real"])
    return df


def main() -> None:
    df = _make_panel()

    # 1. Each strategy produces a valid spec via fit_panel
    selectors = [AmiFnnSelector(), GreyRelationalSelector(), AcfPacfSelector()]
    for sel in selectors:
        spec = sel.fit_panel(df)
        print(f"{sel.name:10s} -> lags={spec.lags} windows={spec.windows}")
        assert isinstance(spec, LagSpec)
        assert len(spec.lags) >= 1 and all(lag >= 1 for lag in spec.lags)
        assert spec.method == sel.name

    # 2. build_lag_features always exposes a lag_1 baseline column
    spec = AcfPacfSelector().fit_panel(df)
    df_feat, feature_cols = build_lag_features(df, spec)
    assert "lag_1" in df_feat.columns, "baseline lag_1 column missing"
    assert all(c in df_feat.columns for c in feature_cols)
    print(f"\nFeature columns built: {feature_cols}")

    # 3. Validator ranks strategies and selects a winner
    validator = LagFeatureValidator(
        strategies=[AmiFnnSelector(), GreyRelationalSelector(), AcfPacfSelector()],
        model_factory=lambda: LinearRegression(),
        n_splits=4,
        primary_metric="wape",
    )
    report = validator.run(df)
    print("\nWalk-forward comparison:")
    print(report)
    assert report.height == 3
    assert validator.best_spec_ is not None
    assert report[0, "method"] == validator.best_spec_.method
    print(f"\nBest method: {validator.best_spec_.method} | lags={validator.best_spec_.lags}")

    # 4. The public API accepts pandas and returns pandas (round-trips the input flavour)
    import pandas as pd

    df_pd = df.to_pandas()
    spec_pd = AmiFnnSelector().fit_panel(df_pd)
    assert isinstance(spec_pd, LagSpec), "fit_panel must accept a pandas DataFrame"

    feat_pd, cols_pd = build_lag_features(df_pd, spec_pd)
    assert isinstance(feat_pd, pd.DataFrame), "build_lag_features must return pandas for pandas input"

    report_pd = LagFeatureValidator(
        strategies=[AmiFnnSelector(), GreyRelationalSelector()],
        model_factory=lambda: LinearRegression(),
        n_splits=4,
        primary_metric="wape",
    ).run(df_pd)
    assert isinstance(report_pd, pd.DataFrame), "run must return pandas for pandas input"
    print("\nPandas input accepted and returned correctly.")
    print("\nAll lag_selection tests passed.")


if __name__ == "__main__":
    main()
