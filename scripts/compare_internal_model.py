"""
SCAN: compare the forecast of the company's internal model with the real consumption, our model and the planner.

Months 2026-06 to 2026-09 (data/internal_model_forecast.csv). The internal model and the planner use the SKU without core width, so the
dataset series with the same type, sub-grade, gsm and width are added up. Our forecasts are those saved by run_ensembles.py (one month
ahead, `ref_h1`) and compare_horizon_strategies.py (two months ahead: `direct`, `partial`, `iterated`), all ml_mean, scored on exactly
the same SKU-months. Metrics: WAPE, accuracy = 1 - WAPE, bias_pct, MAE. Needs the cached forecasts of those two scripts.

Usage:
    python scripts/compare_internal_model.py data/consumos_long.csv
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.cache_utils import check_alignment, check_meta, panel_fingerprint
from src.utils.internal_utils import load_internal_forecast
from src.utils.member_utils import ML_MODELS
from src.utils.metrics_utils import score
from src.utils.panel_utils import build_panel, load_consumption
from src.utils.planner_utils import accuracy_table, compare_month, series_table

MODELS = ["ref_h1", "direct", "partial", "iterated", "naive"]


def by_month(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({m: {c: 100 * score(g["real"].to_numpy(), g[c].to_numpy())["accuracy"] for c in cols} for m, g in frame.groupby("mes")}).round(1)


def closer(frame: pd.DataFrame, cols: list[str], reference: str) -> None:
    for c in cols:
        wins = int(((frame["real"] - frame[c]).abs() < (frame["real"] - frame[reference]).abs()).sum())
        print(f"  {c:9s} closer to the real than {reference} in {wins} of {len(frame)} SKU-months")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--internal", type=Path, default=Path("data/internal_model_forecast.csv"))
    parser.add_argument("--planner-glob", default="data/planner_forecast_*.csv")
    parser.add_argument("--one-month", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--two-months", type=Path, default=Path("results/horizon_strategies.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/internal_model_comparison.csv"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)

    df = load_consumption(args.input)
    S = build_panel(df)
    fingerprint = {"data": panel_fingerprint(S), "n_series": len(S.keys)}
    check_meta(args.one_month, {**fingerprint, "horizon": 1}, keys=("data", "n_series", "horizon"))
    check_meta(args.two_months, fingerprint, keys=("data", "n_series"))
    one, two = pd.read_csv(args.one_month), pd.read_csv(args.two_months)
    check_alignment(one, S)
    one["p_ref_h1"] = one[[f"p_{k}" for k in ML_MODELS]].mean(axis=1)
    res = two[["series", "t", "planta_code", "y", "p_direct", "p_partial", "p_iterated", "p_naive"]].merge(one[["series", "t", "p_ref_h1"]], on=["series", "t"], how="left")
    pairs = series_table(S, df)

    planner_files = sorted(Path().glob(args.planner_glob))
    internal, unmatched = load_internal_forecast(args.internal, planner_files)
    print(f"internal model (SCAN): {internal['sku_planner'].nunique()} SKUs matched by SAP code; not in the planner files and left out: {unmatched['product_sap'].tolist()}")

    parts = []
    for month, frame in internal.groupby("mes"):
        table, missing = compare_month(frame, res, pairs, S.labels.index(month), MODELS)
        if missing:
            print(f"{month}: no match in the data: {missing}")
        parts.append(table.assign(mes=month))
    T = pd.concat(parts, ignore_index=True).rename(columns={"planner": "internal"})
    T = T[T["real"] > 0].reset_index(drop=True)
    planner = pd.concat([pd.read_csv(f)[["mes", "sku_planner", "forecast_to"]] for f in planner_files], ignore_index=True)
    planner = planner.rename(columns={"sku_planner": "sku", "forecast_to": "planner"})
    T = T.merge(planner, on=["mes", "sku"], how="left")
    T.to_csv(args.out, index=False)

    cols = ["internal", *MODELS]
    print(f"\n=== A. Internal model vs our model: {len(T)} SKU-months (rank 1 = lowest WAPE) ===")
    print(accuracy_table(T, cols).round(4))
    closer(T, MODELS, "internal")
    print("accuracy (%) by month:")
    print(by_month(T, cols))
    print(f"totals: real {T['real'].sum():,.0f} | internal {T['internal'].sum():,.0f} | direct {T['direct'].sum():,.0f} | one month ahead {T['ref_h1'].sum():,.0f}")

    both = T[T["planner"].notna()].reset_index(drop=True)
    pcols = ["planner", "internal", *MODELS]
    print(f"\n=== B. Planner, internal model and our model on the same {len(both)} SKU-months ===")
    print(accuracy_table(both, pcols).round(4))
    print("accuracy (%) by month:")
    print(by_month(both, pcols))
    closer(both, ["internal", *MODELS], "planner")
    closer(both, MODELS, "internal")
    print(f"totals: real {both['real'].sum():,.0f} | planner {both['planner'].sum():,.0f} | internal {both['internal'].sum():,.0f} | direct {both['direct'].sum():,.0f} | one month ahead {both['ref_h1'].sum():,.0f}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
