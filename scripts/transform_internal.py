"""
Transform the workbook of the internal model's proposed forecast into data/internal_forecast_plantas.csv (long format).

The workbook has one row per SKU of every planta and one column per month. Two rules are applied:
  - a planta that is all zeros in the workbook (SCAN, at the moment) is replaced by the values of an earlier file with the
    layout of data/internal_model_forecast.csv (the SKU comes from the SAP code, read from the planner files);
  - a SKU whose forecast is 0 in every month is written as missing (empty), not as a forecast of 0.

Usage:
    python scripts/transform_internal.py proposed_forecast_internal_model_all_plants.xlsx
"""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.internal_io_utils import mark_missing, planta_is_all_zero, read_internal_workbook, replace_planta
from src.utils.internal_utils import load_internal_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--fallback", type=Path, default=Path("data/internal_model_forecast.csv"), help="earlier file used for the plantas that are all zeros")
    parser.add_argument("--planner-glob", default="data/planner_forecast_*.csv", help="planner files, to turn the SAP codes of the fallback into SKUs")
    parser.add_argument("--out", type=Path, default=Path("data/internal_forecast_plantas.csv"))
    args = parser.parse_args()

    long = read_internal_workbook(args.workbook)
    print(f"{args.workbook.name}: {long.groupby('planta')['sku_planner'].nunique().sum()} SKUs, {long['planta'].nunique()} plantas, months {sorted(long['mes'].unique())}")

    for planta in sorted(long["planta"].unique()):
        if not planta_is_all_zero(long, planta):
            continue
        if not args.fallback.exists():
            raise SystemExit(f"{planta} is all zeros in the workbook and {args.fallback} does not exist.")
        earlier, unmatched = load_internal_forecast(args.fallback, sorted(Path().glob(args.planner_glob)))
        long = replace_planta(long, earlier[["planta", "mes", "sku_planner", "product_sap", "forecast_to"]], planta)
        print(f"{planta} is all zeros in the workbook: replaced by {earlier[earlier['planta'] == planta]['sku_planner'].nunique()} SKUs of {args.fallback.name}"
              + (f" ({len(unmatched)} products without a SKU left out)" if len(unmatched) else ""))

    long = mark_missing(long)
    missing = long[long["forecast_to"].isna()].groupby("planta")["sku_planner"].nunique()
    print(f"SKUs with 0 in every month, written as missing: {int(missing.sum())}", missing.to_dict())
    long.to_csv(args.out, index=False, float_format="%.10g")
    print(f"wrote {args.out}: {len(long)} rows")


if __name__ == "__main__":
    main()
