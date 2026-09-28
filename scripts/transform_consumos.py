"""
Transform a SAP consumption export into a long-format table with columns:
planta, sku, fecha, consumo.

Two export layouts are supported and auto-detected:

WIDE (sheet "RESULT") -- a pivot table with one column per business day:
    - Row 7:  day headers per monthly block ("01.01.2026", ..., "#" as a blank
              spacer column between months).
    - Row 8:  identifier headers (Sales Region, Customer, Customer name,
              Material, Material description).
    - Row 9+: one row per (Sales Region, Customer, Material) with the daily
              consumption value in each date column.
    - Last 2 rows: "Result" / "Overall Result" subtotals, excluded from output.

LONG -- already one row per (planta, material, day):
    - Row 2:  headers (Sales Region, Customer, Customer name, Day, Material,
              Material description, TO).
    - Row 3+: data. Days with no movement simply have no row, so this layout
              carries no explicit zeros (the wide one does).

In both layouts "sku" is the Material description (e.g.
"K/01/200gsm/1800mm/1200-1400"), not the numeric Material code -- verified
unique per planta.

Rows with no consumption value for a given date are dropped (not filled with 0).
Negative values (stock return / partial-consumption corrections, since stock is
not modeled here) are clipped to 0 rather than dropped, to keep the daily series
continuous for downstream lag/rolling features.

Requires: openpyxl (pip install openpyxl)

Usage:
    python scripts/transform_consumos.py data/consumos_2026.xlsx data/consumos_long.csv
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import openpyxl

SHEET_NAME = "RESULT"
DATE_ROW = 7
FIRST_DATA_ROW = 9
FIRST_DATE_COL = 6
PLANTA_COL = 2
SKU_COL = 5  # Material description (e.g. "K/01/200gsm/1800mm/1200-1400"), unique per planta

LONG_HEADER_ROW = 2
LONG_FIRST_DATA_ROW = 3
LONG_PLANTA_COL = 2
LONG_DATE_COL = 4
LONG_SKU_COL = 6
LONG_VALUE_COL = 7


def _parse_date(value: object) -> datetime | None:
    r"""Parse a day-header cell, skipping blanks and the monthly '#' spacer."""
    if not isinstance(value, str) or value in ("", "#"):
        return None
    try:
        return datetime.strptime(value, "%d.%m.%Y")
    except ValueError:
        return None


def _is_long_layout(ws) -> bool:
    r"""True when the sheet already has one row per day (a 'Day' header in row 2)."""
    header = next(ws.iter_rows(min_row=LONG_HEADER_ROW, max_row=LONG_HEADER_ROW, values_only=True))
    return "Day" in [v for v in header if isinstance(v, str)]


def _transform_long(ws) -> list[tuple[str, str, str, float]]:
    r"""Read the already-long layout into (planta, sku, fecha, consumo) records."""
    records: list[tuple[str, str, str, float]] = []
    for row in ws.iter_rows(min_row=LONG_FIRST_DATA_ROW, values_only=True):
        planta = row[LONG_PLANTA_COL - 1]
        sku = row[LONG_SKU_COL - 1]
        consumo = row[LONG_VALUE_COL - 1]
        if planta in (None, "Result") or sku is None or consumo is None:
            continue
        fecha = _parse_date(row[LONG_DATE_COL - 1])
        if fecha is None:
            continue
        records.append(
            (str(planta), str(sku), fecha.strftime("%Y-%m-%d"), max(float(consumo), 0.0))
        )
    return records


def transform(input_path: Path) -> list[tuple[str, str, str, float]]:
    r"""Read either export layout into (planta, sku, fecha, consumo) records."""
    wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
    ws = wb[SHEET_NAME] if SHEET_NAME in wb.sheetnames else wb[wb.sheetnames[0]]

    if _is_long_layout(ws):
        records = _transform_long(ws)
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
            consumo = max(float(consumo), 0.0)
            records.append((str(planta), str(sku), fecha.strftime("%Y-%m-%d"), consumo))

    records.sort(key=lambda r: (r[0], r[1], r[2]))
    return records


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/transform_consumos.py <input.xlsx> <output.csv>")
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    records = transform(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["planta", "sku", "fecha", "consumo"])
        writer.writerows(records)

    print(f"Wrote {len(records):_} rows to {output_path}")


if __name__ == "__main__":
    main()
