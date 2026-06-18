"""Compare regression models with PyCaret on the full year 2026 (feature-engine features).

Runs in the isolated .venv_pycaret environment (PyCaret pins older numpy/pandas/scikit-learn,
so it must NOT share the main project interpreter).

Pipeline
--------
1. Reuse the 2026 synthetic panel and feature builder from test_1.
2. Chronological walk-forward split (train: Jan-Oct, test: Nov-Dec 2026).
3. PyCaret setup with a time-series CV fold strategy (never shuffles) and compare_models to
   rank the candidate regressors.
4. Re-evaluate the top models on the untouched Nov-Dec holdout and report MAE/WAPE plus the
   lift over the lag-1 persistence baseline (the bar every model must clear, per CLAUDE.md).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pycaret.regression import compare_models, predict_model, pull, setup

# Load FeatureEngineBuilder directly from its file so the src.features package __init__
# (which imports tsfresh, absent from this isolated venv) is not triggered.
_fe_path = ROOT / "src" / "features" / "feature_engine_features.py"
_spec = importlib.util.spec_from_file_location("feature_engine_features", _fe_path)
_fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fe)
FeatureEngineBuilder = _fe.FeatureEngineBuilder

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


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Return (train_pdf, test_pdf, baseline_test, feature_cols) with a chronological split."""
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
    ).sort(["fecha", "planta", "sku"])
    df_test = df_feat.filter(
        (pl.col("fecha") > CUTOFF) & pl.col("target_consumo_real").is_not_null()
    ).sort(["fecha", "planta", "sku"])

    cols = feature_cols + ["target_consumo_real"]
    train_pdf = df_train.select(cols).to_pandas()
    test_pdf = df_test.select(cols).to_pandas()
    # Lag-1 persistence baseline: predict tomorrow as today's consumption.
    baseline_test = df_test["consumo_real"].to_numpy()
    return train_pdf, test_pdf, baseline_test, feature_cols


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error."""
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan


def main() -> None:
    train_pdf, test_pdf, baseline_test, feature_cols = build_dataset()
    print(f"Train rows: {len(train_pdf):_}  |  Test rows: {len(test_pdf):_}  "
          f"|  Features: {len(feature_cols):_}")

    # Time-series CV, never shuffled: the holdout/folds respect chronological order.
    setup(
        data=train_pdf,
        target="target_consumo_real",
        session_id=42,
        data_split_shuffle=False,
        fold_strategy="timeseries",
        fold=5,
        fold_shuffle=False,
        n_jobs=-1,
        verbose=False,
        html=False,
    )

    # Rank candidate regressors by cross-validated MAE.
    top_models = compare_models(sort="MAE", n_select=5)
    leaderboard = pull()
    print("\n=== PyCaret cross-validated leaderboard (train, time-series CV) ===")
    print(leaderboard.to_string())

    if not isinstance(top_models, list):
        top_models = [top_models]

    # Re-evaluate the top models on the untouched Nov-Dec holdout against the baseline.
    y_true = test_pdf["target_consumo_real"].to_numpy()
    base_mae = float(np.mean(np.abs(y_true - baseline_test)))
    base_wape = wape(y_true, baseline_test)

    print("\n=== Holdout (Nov-Dec 2026) vs lag-1 persistence baseline ===")
    print(f"{'Model':<28} {'MAE':>10} {'WAPE':>10} {'MAE lift %':>12} "
          f"{'WAPE lift %':>12} {'Beats?':>8}")
    print("-" * 84)
    results = []
    for model in top_models:
        name = type(model).__name__
        preds = predict_model(model, data=test_pdf, verbose=False)["prediction_label"].to_numpy()
        mae = float(np.mean(np.abs(y_true - preds)))
        w = wape(y_true, preds)
        mae_lift = (base_mae - mae) / base_mae * 100.0
        wape_lift = (base_wape - w) / base_wape * 100.0
        beats = mae < base_mae and w < base_wape
        results.append((name, mae, w, mae_lift, wape_lift, beats))
        print(f"{name:<28} {mae:>10.4f} {w:>10.4f} {mae_lift:>12.2f} "
              f"{wape_lift:>12.2f} {str(beats):>8}")

    print("-" * 84)
    print(f"{'Lag-1 baseline':<28} {base_mae:>10.4f} {base_wape:>10.4f}")

    best = min(results, key=lambda r: r[1])
    print(f"\nBest on holdout MAE: {best[0]} (MAE {best[1]:.4f}, WAPE {best[2]:.4f})")
    assert best[5], "Best model does not beat the lag-1 persistence baseline."
    print("Validation passed: best PyCaret model beats the lag-1 persistence baseline.")


if __name__ == "__main__":
    main()
