"""
Exploratory analysis of the long consumption table (planta, sku, fecha, consumo):
one row per series and one row per planta. Writes eda_by_planta_sku.csv and eda_by_planta.csv
next to the input file.

Usage:
    python scripts/eda_consumos.py data/consumos_long.csv
"""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.eda_utils import eda_by_planta, eda_by_planta_sku
from src.utils.panel_utils import load_consumption


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    args = parser.parse_args()

    df = load_consumption(args.input)
    by_planta_sku, by_planta = eda_by_planta_sku(df), eda_by_planta(df)
    by_planta_sku.to_csv(args.input.parent / "eda_by_planta_sku.csv", index=False)
    by_planta.to_csv(args.input.parent / "eda_by_planta.csv", index=False)

    print(f"planta+sku series: {len(by_planta_sku):_}")
    print(f"plantas: {len(by_planta):_}")
    print(f"rows loaded: {len(df):_}")


if __name__ == "__main__":
    main()
