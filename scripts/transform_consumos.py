"""
Transform a SAP consumption export into a long-format table with columns:
planta, sku, fecha, consumo.

Three export layouts are supported and auto-detected (one wide, two long):

WIDE (sheet "RESULT") -- a pivot table with one column per business day:
    - Row 7:  day headers per monthly block ("01.01.2026", ..., "#" as a blank
              spacer column between months).
    - Row 8:  identifier headers (Sales Region, Customer, Customer name,
              Material, Material description).
    - Row 9+: one row per (Sales Region, Customer, Material) with the daily
              consumption value in each date column.
    - Last 2 rows: "Result" / "Overall Result" subtotals, excluded from output.

LONG -- already one row per (planta, material, day). Two variants are seen:
    - With Sales Region: headers on row 2 (Sales Region, Customer, Customer name,
      Day, Material, Material description, TO), data from row 3.
    - Without it: headers on row 1 (Customer, Customer name, Material, Material
      description, Day, TO), data from row 2.
    Columns are located by header name, so either variant works. Days with no
    movement simply have no row, so these layouts carry no explicit zeros (the
    wide one does).

In both layouts "sku" is the Material description (e.g.
"K/01/200gsm/1800mm/1200-1400"), not the numeric Material code -- verified
unique per planta.

Rows with no consumption value for a given date are dropped (not filled with 0).
Negative values (stock return / partial-consumption corrections, since stock is
not modeled here) are clipped to 0 rather than dropped, to keep the daily series
continuous for downstream lag/rolling features.

Requires: openpyxl (pip install openpyxl)

Usage:
    python scripts/transform_consumos.py data/consumos_2026.xlsx -o data/consumos_long.csv

    # combine exports that do not overlap in time; --drop-zeros aligns the wide
    # export (explicit zeros) with the long ones (no zero rows)
    python scripts/transform_consumos.py --drop-zeros \
        data/consumos_2024_2025.xlsx data/consumos_2026.xlsx -o data/consumos_long.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import openpyxl

SHEET_NAME = "RESULT"
DATE_ROW = 7
FIRST_DATA_ROW = 9
FIRST_DATE_COL = 6
PLANTA_COL = 2
SKU_COL = 5  # Material description (e.g. "K/01/200gsm/1800mm/1200-1400"), unique per planta

LONG_HEADER_SEARCH_ROWS = 3


def _parse_date(value: object) -> datetime | None:
    r"""Parse a day-header cell, skipping blanks and the monthly '#' spacer."""
    if not isinstance(value, str) or value in ("", "#"):
        return None
    try:
        return datetime.strptime(value, "%d.%m.%Y")
    except ValueError:
        return None


def _find_long_header(ws) -> tuple[int, dict[str, int]] | None:
    r"""
    Locate the header row of an already-long export and map field names to 0-based columns.

    *   The header row is the one, within the first rows, that holds both 'Customer' and
        'Day'. The wide layout has them on different rows, so it never matches.
    *   Columns are found by name, not position, because the long exports differ in
        their leading columns (with or without 'Sales Region').
    *   The material description sits in the column right after 'Material'; its own
        header is blank in every export seen so far.
    *
    """
    rows = ws.iter_rows(min_row=1, max_row=LONG_HEADER_SEARCH_ROWS, values_only=True)
    for row_number, row in enumerate(rows, start=1):
        names = [v if isinstance(v, str) else None for v in row]
        if "Customer" in names and "Day" in names and "Material" in names and "TO" in names:
            material = names.index("Material")
            return row_number, {
                "planta": names.index("Customer"),
                "fecha": names.index("Day"),
                "sku": material + 1,
                "consumo": names.index("TO"),
            }
    return None


def _transform_long(
    ws, header_row: int, cols: dict[str, int], drop_zeros: bool = False
) -> list[tuple[str, str, str, float]]:
    r"""Read the already-long layout into (planta, sku, fecha, consumo) records."""
    records: list[tuple[str, str, str, float]] = []
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        planta = row[cols["planta"]]
        sku = row[cols["sku"]]
        consumo = row[cols["consumo"]]
        if planta in (None, "Result") or sku is None or consumo is None:
            continue
        if drop_zeros and float(consumo) == 0.0:
            continue
        fecha = _parse_date(row[cols["fecha"]])
        if fecha is None:
            continue
        records.append(
            (str(planta), str(sku), fecha.strftime("%Y-%m-%d"), max(float(consumo), 0.0))
        )
    return records


def transform(input_path: Path, drop_zeros: bool = False) -> list[tuple[str, str, str, float]]:
    r"""
    Read any supported export layout into (planta, sku, fecha, consumo) records.

    *   drop_zeros discards rows whose raw value is exactly 0 (before negatives are
        clipped). The long exports omit no-movement days entirely while the wide one
        records them as 0, so this makes the wide export comparable when files of
        both kinds are combined.
    *
    """
    wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
    ws = wb[SHEET_NAME] if SHEET_NAME in wb.sheetnames else wb[wb.sheetnames[0]]

    long_header = _find_long_header(ws)
    if long_header is not None:
        records = _transform_long(ws, *long_header, drop_zeros=drop_zeros)
        records.sort(key=lambda r: (r[0], r[1], r[2]))
        return records

    date_row = next(ws.iter_rows(min_row=DATE_ROW, max_row=DATE_ROW, values_only=True))
    date_by_col: dict[int, datetime] = {}
    for col_idx, value in enumerate(date_row, start=1):
        if col_idx < FIRST_DATE_COL:
            continue
        parsed = _parse_date(value)
        if parsed is not None:
            date_by_col[col_idx] = parsed

    records: list[tuple[str, str, str, float]] = []
    for row in ws.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
        planta = row[PLANTA_COL - 1]
        sku = row[SKU_COL - 1]
        if planta in (None, "Result") or sku is None:
            continue
        for col_idx, fecha in date_by_col.items():
            consumo = row[col_idx - 1]
            if consumo is None:
                continue
            if drop_zeros and float(consumo) == 0.0:
                continue
            consumo = max(float(consumo), 0.0)
            records.append((str(planta), str(sku), fecha.strftime("%Y-%m-%d"), consumo))

    records.sort(key=lambda r: (r[0], r[1], r[2]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="one or more .xlsx exports")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output .csv")
    parser.add_argument(
        "--drop-zeros",
        action="store_true",
        help="discard rows whose raw value is exactly 0 (aligns wide exports with long ones)",
    )
    args = parser.parse_args()

    records: list[tuple[str, str, str, float]] = []
    for path in args.inputs:
        part = transform(path, drop_zeros=args.drop_zeros)
        print(f"  {path.name}: {len(part):_} rows")
        records.extend(part)

    keys = [(r[0], r[1], r[2]) for r in records]
    duplicated = len(keys) - len(set(keys))
    if duplicated:
        raise SystemExit(
            f"{duplicated:_} (planta, sku, fecha) keys appear in more than one input; "
            "the files overlap in time. Pass non-overlapping exports."
        )
    records.sort(key=lambda r: (r[0], r[1], r[2]))

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["planta", "sku", "fecha", "consumo"])
        writer.writerows(records)

    print(f"Wrote {len(records):_} rows to {output_path}")


if __name__ == "__main__":
    main()
