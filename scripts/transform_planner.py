"""
Transform the monthly planner assignment workbooks into data/planner_forecast_<month>.csv (every planta of the workbook).

Each workbook has one sheet; the month comes from the header of its ninth column ("SEPT´26 allocation"), not from the file name.
All rows are kept, whatever their strategy (VMI, NO VMI, VMI EST): filter later with compare_planner.py --strategies.

Usage:
    python scripts/transform_planner.py Asignacion_jun_26.xlsx Asignacion_jul_26.xlsx --out-dir data
"""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.planner_io_utils import read_assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="one or more .xlsx planner workbooks")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    for path in args.inputs:
        table = read_assignment(path)
        month = table["mes"].iloc[0]
        out = args.out_dir / f"planner_forecast_{month}.csv"
        replaced = "replaces" if out.exists() else "writes"
        table.to_csv(out, index=False, float_format="%.10g")
        print(f"{path.name}: {replaced} {out} | {len(table)} rows, {table['planta'].nunique()} plantas, total {table['forecast_to'].sum():,.1f} TO")
        print("  rows by strategy:", table["strategy"].replace("", "(none)").value_counts().to_dict())


if __name__ == "__main__":
    main()
