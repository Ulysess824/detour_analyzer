"""
Forecast the next month of every series from the full history.

Trains LightGBM, XGBoost and Random Forest on all the months already observed (the tuned parameters of
the tuning json when present) and forecasts the month after the last one. `ml_mean` is their average,
the model used in the report. SKU forecasts are added up to planta and total (bottom-up).

The last month of the export must be complete. With --as-of YYYY-MM the data after that month is ignored,
so the forecast can be compared with the real value that is already known (a check of this script).

Next to the forecast csv it writes `forecast_<month>.manifest.json`: the commit, data, parameters and library versions the run used
(src/engine/run_manifest.py). Commit the data file before a run you want to be able to reproduce.

Usage:
    python scripts/predict_next_month.py data/consumos_long.csv
    python scripts/predict_next_month.py data/consumos_long.csv --as-of 2026-08
"""

import argparse
import warnings

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.prediction_run import run_prediction
from src.evaluation.metrics_utils import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--tuning", type=Path, default=Path("results/tuning_trees.json"))
    parser.add_argument("--as-of", default=None, help="last month to use (YYYY-MM); default: the last month of the data")
    parser.add_argument("--out", type=Path, default=None, help="default: results/forecast_<month>.csv")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)

    run = run_prediction(args.input, args.tuning, args.out, args.as_of, repo_dir=Path(__file__).resolve().parents[1])
    S, result, month = run.panel, run.result, run.month
    print(f"history {S.labels[0]} .. {S.labels[-1]} ({len(S.keys)} series)")
    print(f"forecast for {month}: {len(result)} series")

    by_planta = result.groupby("planta")[["ml_mean"]].sum().round(1).sort_values("ml_mean", ascending=False)
    print("\nforecast by planta (ml_mean, TO):")
    print(by_planta.to_string())
    print(f"\ntotal ml_mean: {result['ml_mean'].sum():,.1f}")
    print(f"saved {run.forecast_path}")
    manifest = run.manifest
    print(f"saved {run.manifest_file}  (run {manifest['run']['run_id']}, commit {manifest['code']['commit'][:8]}, data {manifest['data']['panel_fingerprint']})")

    if args.as_of:
        full = run.full
        actual = full[full["fecha"].dt.strftime("%Y-%m") == month].groupby(["planta", "sku"])["consumo"].sum().rename("real")
        joined = result.merge(actual.reset_index(), on=["planta", "sku"], how="left").fillna({"real": 0.0})
        if len(actual):
            print(f"\ncheck against the real {month} (series alive at {S.labels[-1]}):")
            for col in ("ml_mean", "lgbm", "xgb", "rf", "naive"):
                s = score(joined["real"].to_numpy(), joined[col].to_numpy())
                print(f"  {col:8s} SKU accuracy {100 * s['accuracy']:.1f}%  bias {s['bias_pct']:+.1f}%")
            print(f"  total real {joined['real'].sum():,.1f} vs ml_mean {joined['ml_mean'].sum():,.1f}")


if __name__ == "__main__":
    main()
