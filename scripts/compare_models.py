"""
Compare forecasting models on the raw observation rows (planta, sku, fecha, consumo) with a
chronological train/test split at --cutoff (never a random split).

Models: naive lag-1, Croston, hurdle (logistic occurrence x log-linear size), global LightGBM
and XGBoost with a Tweedie objective, and per-series ARIMA.
Each observed row is one time period; days without a row are not reconstructed.

Usage:
    python scripts/compare_models.py data/consumos_long.csv --cutoff 2026-04-30
"""

import argparse

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.metrics_utils import rank_by_mae, row_metrics
from src.utils.row_model_utils import (
    arima_forecast, build_row_features, croston_forecast, hurdle_forecast, lgbm_tweedie_forecast,
    load_rows, naive_lag1_forecast, xgboost_forecast,
)  # fmt: skip

MODELS = {
    "naive_lag1": "pred_naive",
    "croston": "pred_croston",
    "hurdle": "pred_hurdle",
    "lgbm_tweedie": "pred_lgbm",
    "xgboost_tweedie": "pred_xgb",
    "arima": "pred_arima",
}


def metrics_table(df: pd.DataFrame, models: dict, columns: list[str]) -> pd.DataFrame:
    r"""Metrics of each model on the given rows of the test set."""
    rows = []
    for name, col in models.items():
        sub = df[["consumo", col]].dropna()
        rows.append({"model": name, **row_metrics(sub["consumo"].to_numpy(), sub[col].to_numpy())})
    return pd.DataFrame(rows).set_index("model")[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--cutoff", type=str, default=None, help="last train date; default: end of the month before the last --test-months")
    parser.add_argument("--test-months", type=int, default=2, help="months at the end used as test when --cutoff is not given")
    args = parser.parse_args()

    rows = load_rows(args.input)
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else (rows["fecha"].max().to_period("M") - args.test_months).end_time.normalize()
    df = build_row_features(rows, cutoff)
    train_mask, test_mask = df["fecha"] <= cutoff, df["fecha"] > cutoff

    df["pred_naive"] = naive_lag1_forecast(df)
    df["pred_croston"] = croston_forecast(df)
    df["pred_hurdle"] = hurdle_forecast(df, train_mask)
    df["pred_lgbm"], importance = lgbm_tweedie_forecast(df, train_mask)
    df["pred_xgb"] = xgboost_forecast(df, train_mask)
    df["pred_arima"], arima_orders = arima_forecast(df, train_mask)

    models = MODELS

    print(f"train rows: {train_mask.sum():_} | test rows: {test_mask.sum():_} | cutoff: {cutoff.date()}\n")
    test = df[test_mask]

    table = metrics_table(test, models, ["n", "mae", "rmse", "wape", "bias"])
    baseline_mae = table.loc["naive_lag1", "mae"]
    table["mae_lift_pct"] = ((baseline_mae - table["mae"]) / baseline_mae * 100).round(1)
    print(rank_by_mae(table))

    # ARIMA skips short series, so repeat the comparison on the rows every model covers.
    common = test[["consumo", *models.values()]].dropna()
    print(f"\nSame-rows comparison (n={len(common):_}, where every model has a prediction):")
    print(rank_by_mae(metrics_table(common, models, ["mae", "rmse", "wape", "bias"])))

    print("\nARIMA order chosen by AIC:")
    print(arima_orders.value_counts().to_string())
    print("\nLightGBM feature importance (gain):")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
