"""
Write the planner / internal model / ml_mean comparison as an Excel workbook with three tables.

Sheet "Comparación": one row per planta, month and SKU with the real and the forecasts of our model (ml_mean), the planner and the
internal model; the internal model reads "faltante" where it has no forecast. Sheet "Excluidos": every row left out of the metrics,
with the reason. Both come from compare_planner.py --internal (results/planner_plantas_<month>.csv and
results/filas_excluidas_planificador.csv).

Usage:
    python scripts/export_comparison_excel.py --month 2026-09
"""

import argparse

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.excel_utils import COMPARISON_SHEET, add_error_columns, comparison_sheet, excluded_sheet, summary_sheet, write_workbook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--month", default="2026-09")
    parser.add_argument("--comparison", type=Path, default=None, help="default: results/planner_plantas_<month>.csv")
    parser.add_argument("--excluded", type=Path, default=Path("results/filas_excluidas_planificador.csv"))
    parser.add_argument("--out", type=Path, default=None, help="default: results/comparacion_<month>.xlsx")
    args = parser.parse_args()

    table = pd.read_csv(args.comparison or Path(f"results/planner_plantas_{args.month}.csv"))
    if "internal" not in table.columns:
        raise SystemExit("The comparison has no internal model column; run compare_planner.py with --internal first.")
    excluded = pd.read_csv(args.excluded)
    excluded = excluded[excluded["mes"] == args.month]
    out = args.out or Path(f"results/comparacion_{args.month}.xlsx")
    comparison = comparison_sheet(table)
    in_metrics = comparison[comparison["Modelo interno (TO)"] != "faltante"]
    sheets = {
        COMPARISON_SHEET: add_error_columns(comparison),
        "Excluidos": excluded_sheet(excluded),
        "Resumen por planta": summary_sheet(sorted(in_metrics["Planta"].unique()), len(comparison)),
    }
    write_workbook(out, sheets)
    print(f"wrote {out}: {len(comparison)} rows of predictions ({len(comparison) - len(in_metrics)} with the internal model faltante), {len(sheets['Excluidos'])} excluded rows")
    print(sheets["Excluidos"]["Motivo"].value_counts().to_string())


if __name__ == "__main__":
    main()
