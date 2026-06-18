"""Compare regression models with PyCaret on the REAL data (historico_consumo.parquet).

Runs in the isolated .venv_pycaret environment (Python 3.11). The raw ERP export is passed
through transform_data, restricted to the most active series and a recent window to stay
tractable (the full panel is 2.7M rows x 4540 mostly-zero series), then features are built with
FeatureEngineBuilder and the candidate regressors are ranked with PyCaret.

Both src modules are loaded directly by file path so the src.features package __init__ (which
imports tsfresh, absent from this venv) is never triggered.
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

# Tunables: keep the comparison tractable on the real panel.
DATA_PATH = ROOT / "data" / "historico_consumo.parquet"
MIN_DATE = pl.date(2024, 6, 1)      # ignore older history
TEST_CUTOFF = pl.date(2026, 3, 31)  # train <= cutoff, test > cutoff (last ~2 months)
N_SERIES = 300                      # keep the N most active (planta, sku) series


def _load_module(name: str, relpath: str):
    """Import a single source module by path, bypassing the package __init__ side effects."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transform_data = _load_module(
    "data_transformation", "src/utils/data_transformation.py"
).transform_data
FeatureEngineBuilder = _load_module(
    "feature_engine_features", "src/features/feature_engine_features.py"
).FeatureEngineBuilder


def load_real_panel() -> pl.DataFrame:
    """Transform the raw export, drop dead series and trim to the recent window."""
    df_raw = pl.read_parquet(DATA_PATH)
    df = transform_data(df_raw, umbral_pico=2.0)

    df = df.filter(pl.col("fecha") >= MIN_DATE)

    # Keep the N most active series (highest total consumption) so the comparison is not
    # dominated by the many all-zero series; chronology within each series is preserved.
    activity = (
        df.group_by(["planta", "sku"])
        .agg(pl.col("consumo_real").sum().alias("total"))
        .sort("total", descending=True)
        .head(N_SERIES)
        .select(["planta", "sku"])
    )
    df = df.join(activity, on=["planta", "sku"], how="inner")

    return df.with_columns(
        pl.col("fecha").dt.day().alias("day_of_month"),
        (pl.col("fecha").dt.month_end() - pl.col("fecha"))
        .dt.total_days()
        .alias("days_to_end_of_month"),
    ).sort(["planta", "sku", "fecha"])


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Return (train_pdf, test_pdf, baseline_test, feature_cols) with a chronological split."""
    panel = load_real_panel()
    df_train_raw = panel.filter(pl.col("fecha") <= TEST_CUTOFF)

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
        (pl.col("fecha") <= TEST_CUTOFF) & pl.col("target_consumo_real").is_not_null()
    ).sort(["fecha", "planta", "sku"])
    df_test = df_feat.filter(
        (pl.col("fecha") > TEST_CUTOFF) & pl.col("target_consumo_real").is_not_null()
    ).sort(["fecha", "planta", "sku"])

    cols = feature_cols + ["target_consumo_real"]
    train_pdf = df_train.select(cols).to_pandas()
    test_pdf = df_test.select(cols).to_pandas()
    baseline_test = df_test["consumo_real"].to_numpy()  # lag-1 persistence
    return train_pdf, test_pdf, baseline_test, feature_cols


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error."""
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan


def main() -> None:
    print(f"Loading real data from {DATA_PATH.name} "
          f"(window >= {MIN_DATE}, top {N_SERIES} series)...")
    train_pdf, test_pdf, baseline_test, feature_cols = build_dataset()
    print(f"Train rows: {len(train_pdf):_}  |  Test rows: {len(test_pdf):_}  "
          f"|  Features: {len(feature_cols):_}")

    setup(
        data=train_pdf,
        target="target_consumo_real",
        session_id=42,
        data_split_shuffle=False,
        fold_strategy="timeseries",
        fold=3,
        fold_shuffle=False,
        n_jobs=-1,
        verbose=False,
        html=False,
    )

    top_models = compare_models(sort="MAE", n_select=5)
    leaderboard = pull()
    print("\n=== PyCaret cross-validated leaderboard (train, time-series CV) ===")
    print(leaderboard.to_string())

    if not isinstance(top_models, list):
        top_models = [top_models]

    y_true = test_pdf["target_consumo_real"].to_numpy()
    base_mae = float(np.mean(np.abs(y_true - baseline_test)))
    base_wape = wape(y_true, baseline_test)

    print("\n=== Holdout (after 2026-03-31) vs lag-1 persistence baseline ===")
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
    if best[5]:
        print("Best model beats the lag-1 persistence baseline.")
    else:
        print("WARNING: best model does NOT beat the lag-1 persistence baseline.")


if __name__ == "__main__":
    main()
