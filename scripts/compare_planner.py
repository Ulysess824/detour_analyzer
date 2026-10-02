"""
Compare the forecast of the planners with the models for one month, SKU by SKU.

The planner file has one row per planner SKU (planta, sku_planner, forecast_to). The planner SKU
has no core width, so the series of the dataset that share type, gsm and width are added up.
The model forecasts are the ones saved by run_ensembles.py (h=1, so made one month before).

The comparison covers only the SKUs the planner forecasted, so every model and the planner are
scored on exactly the same SKUs and the same month.

Usage:
    python scripts/compare_planner.py data/consumos_long.csv data/planner_forecast_2026-09.csv --month 2026-09
"""

import argparse

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.ensemble_utils import add_ensembles
from src.utils.member_utils import ENSEMBLE_GROUPS
from src.utils.panel_utils import build_panel, load_consumption
from src.utils.planner_utils import accuracy_table, compare_month, series_table

HIGHLIGHT = ["lgbm", "xgb", "rf", "ml_mean", "ml_econ_mean", "ml_econ_weighted", "naive", "mean6", "ses"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("planner", type=Path)
    parser.add_argument("--month", default="2026-09", help="month the planner forecasted (YYYY-MM)")
    parser.add_argument("--predictions", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/planner_comparison.csv"))
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)

    df = load_consumption(args.input)
    S = build_panel(df)
    month = S.labels.index(args.month)
    res = add_ensembles(pd.read_csv(args.predictions), ENSEMBLE_GROUPS, horizon=1)
    all_models = [c[2:] for c in res.columns if c.startswith("p_")]

    table, missing = compare_month(pd.read_csv(args.planner), res, series_table(S, df), month, all_models)
    if missing:
        print("planner SKUs with no match in the data:", missing)
    print(f"month {args.month}: {len(table)} planner SKUs, real total {table['real'].sum():.1f}, planner {table['planner'].sum():.1f}\n")

    print("Accuracy on the planner SKUs (rank 1 = lowest WAPE):")
    ranking = accuracy_table(table, ["planner", *all_models])
    print(ranking)

    shown = table[["sku", "real", "planner", *HIGHLIGHT]].round(1)
    shown["planner_err_pct"] = ((table["planner"] - table["real"]) / table["real"] * 100).round(0)
    print("\nBy SKU (forecast minus real in percent for the planner):")
    print(shown.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
