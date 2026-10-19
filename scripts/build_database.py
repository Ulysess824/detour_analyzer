"""
Build the SQLite database with the real consumption from the master csv.

Creates the four tables (plantas, skus, consumos, consumos_predichos) and fills plantas, skus and consumos. The csv stays the
master: the database can be rebuilt at any time and is not tracked by git (data/*.db). After loading, the panel read back from the
database is compared with the panel built from the csv (same fingerprint, same sum per planta and month).

Usage:
    python scripts/build_database.py data/consumos_long.csv --db data/detour.db
    python scripts/build_database.py data/consumos_long.csv --db data/detour.db --rebuild
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.db_utils import connect, create_schema, load_tracking_csv, write_consumption
from src.engine.panel_utils import build_panel, load_consumption
from src.evaluation.cache_utils import panel_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="master csv (planta, sku, fecha, consumo)")
    parser.add_argument("--db", type=Path, default=Path("data/detour.db"))
    parser.add_argument("--rebuild", action="store_true", help="delete the database file first (also drops the predictions)")
    parser.add_argument("--tracking", type=Path, default=Path("results/seguimiento.csv"), help="tracking history restored into the database when the file exists")
    args = parser.parse_args()

    if args.rebuild and args.db.exists():
        args.db.unlink()
    records = load_consumption(args.input)
    con = connect(args.db)
    create_schema(con)
    counts = write_consumption(con, records)
    restored = load_tracking_csv(con, args.tracking) if args.tracking.exists() else 0
    con.close()
    print(f"{args.db}: {counts['plantas']} plantas, {counts['skus']:_} skus, {counts['consumos']:_} consumption rows" + (f", {restored} tracking rows restored" if restored else ""))

    S_csv, S_db = build_panel(records), build_panel(load_consumption(args.db))
    same = panel_fingerprint(S_csv) == panel_fingerprint(S_db) and np.allclose(S_csv.planta_monthly, S_db.planta_monthly)
    if not same:
        raise SystemExit("The panel read from the database differs from the panel of the csv.")
    print("check: the panel read from the database equals the panel of the csv")


if __name__ == "__main__":
    main()
