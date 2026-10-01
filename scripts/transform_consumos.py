"""
Transform SAP consumption exports into one long csv with the columns planta, sku, fecha, consumo.

Wide and long exports are detected automatically (see src/utils/transform_utils.py).

Usage:
    python scripts/transform_consumos.py data/consumos_2026.xlsx -o data/consumos_long.csv

    # combine exports that do not overlap in time; --drop-zeros aligns the wide export
    # (explicit zeros) with the long ones (no zero rows)
    python scripts/transform_consumos.py --drop-zeros \\
        data/consumos_2024_2025.xlsx data/consumos_2026.xlsx -o data/consumos_long.csv
"""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.transform_utils import combine_exports, read_export, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="one or more .xlsx exports")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output .csv")
    parser.add_argument("--drop-zeros", action="store_true", help="discard rows whose raw value is exactly 0")
    args = parser.parse_args()

    parts = []
    for path in args.inputs:
        part = read_export(path, drop_zeros=args.drop_zeros)
        print(f"  {path.name}: {len(part):_} rows")
        parts.append(part)

    records = combine_exports(parts)
    write_csv(records, args.output)
    print(f"Wrote {len(records):_} rows to {args.output}")


if __name__ == "__main__":
    main()
