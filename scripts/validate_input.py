"""
Check a consumption table (csv or .db) and write the validation tables.

Runs every input rule of src/engine/data_checks.py without stopping at the first error, prints the errors and warnings with the
rows that broke them, and writes four tables (long format) to --out-dir: Sunday rows (they count as consumption), rows above the
peak limit, and the month-to-month change by planta and SKU and by planta. The last month must be complete unless
--allow-partial-last-month is given. Exit code 1 when there are errors.

Usage:
    python scripts/validate_input.py data/consumos_long.csv
    python scripts/validate_input.py data/detour.db --out-dir results --peak-limit 100
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.data_checks import PEAK_LIMIT, check_consumption
from src.engine.panel_utils import load_consumption

FILES = {
    "domingos": "validacion_domingos.csv",
    "picos": "validacion_picos.csv",
    "cambio_planta_sku": "validacion_cambio_planta_sku.csv",
    "cambio_planta": "validacion_cambio_planta.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="consumption csv or .db")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--peak-limit", type=float, default=PEAK_LIMIT, help="TO in one day for one SKU and planta that raises a warning")
    parser.add_argument("--allow-partial-last-month", action="store_true")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    df = load_consumption(args.input, validate=False)
    report = check_consumption(df, peak_limit=args.peak_limit, require_complete_last_month=not args.allow_partial_last_month, with_tables=True)
    print(f"{args.input}: {len(df):,} rows, {df['fecha'].min().date()} .. {df['fecha'].max().date()}" if "fecha" in df and len(df) else str(args.input))
    print(report.summary() or "no errors, no warnings")
    print(f"\nresult: {len(report.errors)} errors, {len(report.warnings)} warnings")

    if report.tables:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in report.tables.items():
            table.to_csv(args.out_dir / FILES[name], index=False, float_format="%.4f")
            print(f"saved {args.out_dir / FILES[name]}  ({len(table):,} rows)")
        change = report.tables["cambio_planta"]
        if len(change):
            last = change["mes"].max()
            print(f"\nchange by planta, {last} (TO):")
            print(change[change["mes"] == last].drop(columns="mes").sort_values("cambio_TO").to_string(index=False))
    if report.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
