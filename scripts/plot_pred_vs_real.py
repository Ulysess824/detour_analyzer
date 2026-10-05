"""
Test-period actual against the base and Optuna-tuned forecasts of each tree model (seaborn).

Refits every model over the test months (rolling origin, horizon h) with the baseline
parameters and with the tuned ones saved by tune_trees.py, and plots the monthly total, or
one series when --planta and --sku are given.

Usage:
    python scripts/plot_pred_vs_real.py data/consumos_long.csv results/tuning_trees.json -o pred_vs_real.png
"""

import argparse
import json
import warnings

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.panel_utils import GROUP_COLS, build_panel, load_consumption
from src.utils.plot_utils import collect_forecasts, plot_forecasts
from src.utils.tuning_utils import check_tuning_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("results", type=Path, help="json written by tune_trees.py")
    parser.add_argument("-o", "--out", type=Path, default=Path("pred_vs_real.png"))
    parser.add_argument("--h", type=int, default=1, choices=[1, 3])
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planta", default=None, help="plot one series instead of the total (needs --sku)")
    parser.add_argument("--sku", default=None)
    parser.add_argument("--csv", type=Path, default=None, help="also write the plotted values")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    df = load_consumption(args.input)
    S = build_panel(df)
    series, scope = None, "total mensual"
    if args.planta or args.sku:
        keys = df.groupby(GROUP_COLS).size().index  # same order as the panel rows
        series = keys.get_loc((args.planta, args.sku))
        scope = f"{args.planta} / {args.sku}"

    tuned = json.loads(args.results.read_text())
    check_tuning_window(tuned, S, args.n_test)
    long, summary = collect_forecasts(S, tuned, args.h, args.n_test, args.seed, series)
    print(long.pivot_table(index="mes", columns=["modelo", "variante"], values="consumo").round(0).to_string())
    print(summary.round(4).to_string(index=False))
    if args.csv:
        long.to_csv(args.csv, index=False)
    plot_forecasts(long, summary, args.h, args.out, scope)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
