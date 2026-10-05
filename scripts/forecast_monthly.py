"""
Monthly forecasting of consumption per (planta, sku), validated with a rolling origin.

Two ways of getting a monthly number are compared under identical conditions:
    - monthly-direct: forecast the monthly total itself (simple rules and a global LightGBM).
    - daily-then-aggregate (--daily): forecast every calendar day of the target month with a
      zero-filled daily model, then sum the days.

Validation (never a random split): for every test month t the models are fit using only months
up to t - h and asked for month t. The last --n-test months are the test months; h is the
horizon in months (1 = next month, 3 = a quarter ahead).

Models: naive, mean3/6/12, perday6 (daily rate times business days), seasonal_naive,
mean6_x_global / mean6_x_planta (mean6 times last year's seasonal ratio), lgbm_tweedie,
daily_lgbm (--daily) and *_reconciled (SKU forecasts rescaled to the direct planta forecast).
Planta and total are scored both directly (simple models on the aggregate) and bottom-up.

Metrics: WAPE = sum|error| / sum(actual); accuracy = 1 - WAPE; bias_pct = sum(actual -
forecast) / sum(actual), so a positive bias means the model under-forecasts.

Usage:
    python scripts/forecast_monthly.py data/consumos_long.csv --daily
"""

import argparse

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.evaluation_utils import evaluate, tercile_table
from src.utils.panel_utils import build_panel, load_consumption


def print_horizon(out: dict, horizon: int, daily: bool, n_plantas: int, n_test: int) -> None:
    print(f"\n===== horizon h={horizon} month(s) ahead =====")
    print("\nSummary (accuracy = 1 - WAPE; planta and total are bottom-up sums of the SKU forecasts):")
    print(out["summary"].round(4))
    print(f"\nSKU level (n = SKU-months, {out['sku']['n'].iloc[0]:_} each):")
    print(out["sku"])
    print(f"\nPlanta level ({n_plantas * n_test} planta-months):")
    print(out["planta"])
    print(f"\nTotal level ({n_test} months):")
    print(out["total"])
    print("\nAccuracy by planta:")
    print(out["by_planta"])
    print("\nMonth by month:")
    print(out["monthly"])

    models = ["mean6", "lgbm_tweedie", "lgbm_tweedie_reconciled"] + (["daily_lgbm", "daily_lgbm_reconciled"] if daily else [])
    print("\nSKU level by size (mean12 tercile within each month):")
    print(tercile_table(out["res"], models))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--daily", action="store_true", help="also run the daily-then-aggregate model (slower)")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    S = build_panel(load_consumption(args.input))
    n_series, n_months = S.monthly.shape
    print(f"months: {S.labels[0]} .. {S.labels[-1]} ({n_months}) | series: {n_series:_} | plantas: {len(S.plantas)}")
    print(f"test months: {S.labels[n_months - args.n_test]} .. {S.labels[-1]} ({args.n_test})")

    for horizon in args.horizons:
        print_horizon(evaluate(S, horizon, args.n_test, daily=args.daily), horizon, args.daily, len(S.plantas), args.n_test)


if __name__ == "__main__":
    main()
