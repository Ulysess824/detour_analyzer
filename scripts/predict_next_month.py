"""
Forecast the next month of every series from the full history.

Trains LightGBM, XGBoost and Random Forest on all the months already observed (the tuned parameters of
the tuning json when present) and forecasts the month after the last one. `ml_mean` is their average,
the model used in the report. SKU forecasts are added up to planta and total (bottom-up).

The last month of the export must be complete. With --as-of YYYY-MM the data after that month is ignored,
so the forecast can be compared with the real value that is already known (a check of this script).

Usage:
    python scripts/predict_next_month.py data/consumos_long.csv
    python scripts/predict_next_month.py data/consumos_long.csv --as-of 2026-08
"""

import argparse
import json
import warnings

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.inference_utils import check_last_month_complete, forecast_next_month
from src.utils.metrics_utils import score
from src.utils.panel_utils import build_panel, load_consumption


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--tuning", type=Path, default=Path("results/tuning_trees.json"))
    parser.add_argument("--as-of", default=None, help="last month to use (YYYY-MM); default: the last month of the data")
    parser.add_argument("--out", type=Path, default=None, help="default: results/forecast_<month>.csv")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)

    full = load_consumption(args.input)
    df = full
    if args.as_of:
        df = full[full["fecha"] <= pd.Period(args.as_of, freq="M").end_time]
    else:
        check_last_month_complete(full)
    S = build_panel(df)
    tuned = json.loads(args.tuning.read_text()) if args.tuning.exists() else {}

    print(f"history {S.labels[0]} .. {S.labels[-1]} ({len(S.keys)} series)")
    result = forecast_next_month(S, tuned)
    month = result["mes"].iloc[0]
    print(f"forecast for {month}: {len(result)} series")

    out = args.out or Path(f"results/forecast_{month}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.round(3).to_csv(out, index=False)

    by_planta = result.groupby("planta")[["ml_mean"]].sum().round(1).sort_values("ml_mean", ascending=False)
    print("\nforecast by planta (ml_mean, TO):")
    print(by_planta.to_string())
    print(f"\ntotal ml_mean: {result['ml_mean'].sum():,.1f}")
    print(f"saved {out}")

    if args.as_of:
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
