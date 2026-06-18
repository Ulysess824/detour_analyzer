"""Synthetic-data validation for HuberRegressor on the full year 2026 (feature-engine features).

Mirrors test_1.py but swaps the model for the outlier-robust HuberRegressor, which won the
PyCaret comparison. Trains on a chronological walk-forward split and validates against the
lag-1 persistence baseline (must beat it on MAE and WAPE, per CLAUDE.md).
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_engine_features import FeatureEngineBuilder
from src.models.huber_regressor import HuberRegressor
from src.utils.regression_metrics import RegressionMetrics

CUTOFF = pl.date(2026, 10, 31)  # train: Jan-Oct 2026, test: Nov-Dec 2026


def build_year_2026() -> pl.DataFrame:
    """Daily consumption for the whole of 2026: 3 plants x 20 SKUs with monthly means and peaks."""
    rng = np.random.default_rng(42)
    plants = ["Planta_A", "Planta_B", "Planta_C"]
    skus = [f"SKU_{i:02d}" for i in range(1, 21)]
    dates = pl.date_range(
        pl.date(2026, 1, 1), pl.date(2026, 12, 31), interval="1d", eager=True
    )
    rows: list[dict] = []
    for plant in plants:
        for sku in skus:
            monthly_avg = {m: int(rng.integers(2, 9)) for m in range(1, 13)}
            forecast_mensual = float(np.mean(list(monthly_avg.values())) * 20.0)
            for fecha in dates:
                avg = monthly_avg[fecha.month]
                if rng.random() < 0.10:
                    low = min(3.0 * avg + 1.0, 29.0)
                    consumo = float(rng.uniform(low, 30.0))
                else:
                    consumo = float(np.clip(avg + rng.normal(0.0, 1.0), 1.0, 3.0 * avg - 0.1))
                rows.append(
                    {
                        "planta": plant,
                        "sku": sku,
                        "fecha": fecha,
                        "consumo_real": round(consumo, 4),
                        "forecast_mensual": forecast_mensual,
                    }
                )
    return pl.DataFrame(rows).sort(["planta", "sku", "fecha"])


def add_target_and_calendar(df: pl.DataFrame) -> pl.DataFrame:
    """Next-day regression target plus simple forecast/calendar features (leakage-safe)."""
    groups = ["planta", "sku"]
    return df.with_columns(
        pl.col("consumo_real").shift(-1).over(groups).alias("target_consumo_real"),
        (pl.col("forecast_mensual") / 20.0).alias("daily_forecast"),
        pl.col("fecha").dt.day().alias("day_of_month"),
        (pl.col("fecha").dt.month_end() - pl.col("fecha"))
        .dt.total_days()
        .alias("days_to_end_of_month"),
    )


def main() -> None:
    panel = add_target_and_calendar(build_year_2026())
    df_train_raw = panel.filter(pl.col("fecha") <= CUTOFF)

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
    builder.fit(df_train_raw)
    df_feat, lag_feature_cols = builder.transform(panel)

    feature_cols = lag_feature_cols + [
        "daily_forecast",
        "day_of_month",
        "days_to_end_of_month",
    ]

    df_train = df_feat.filter(
        (pl.col("fecha") <= CUTOFF) & pl.col("target_consumo_real").is_not_null()
    )
    df_test = df_feat.filter(
        (pl.col("fecha") > CUTOFF) & pl.col("target_consumo_real").is_not_null()
    )

    model = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=200)
    model.fit(df_train, x_predictor=feature_cols, y="target_consumo_real")

    print(model.summary())

    y_pred = model.predict(df_test)
    y_true = df_test["target_consumo_real"].to_numpy()
    y_baseline = df_test["consumo_real"].to_numpy()

    test_metrics = RegressionMetrics(
        model_name="HuberRegressor (2026 holdout)",
        n_features=len(feature_cols),
    ).compute(y_true, y_pred, y_baseline=y_baseline)

    print("\n" + test_metrics.summary())

    beats_mae = test_metrics.mae < test_metrics.baseline_mae
    beats_wape = test_metrics.wape < test_metrics.baseline_wape
    print(f"\nBeats baseline on MAE:  {beats_mae}  "
          f"(model {test_metrics.mae:.4f} vs baseline {test_metrics.baseline_mae:.4f})")
    print(f"Beats baseline on WAPE: {beats_wape}  "
          f"(model {test_metrics.wape:.4f} vs baseline {test_metrics.baseline_wape:.4f})")

    assert beats_mae, "Model does not beat the lag-1 baseline on MAE."
    assert beats_wape, "Model does not beat the lag-1 baseline on WAPE."
    print("\nValidation passed: HuberRegressor beats the lag-1 persistence baseline.")


if __name__ == "__main__":
    main()
