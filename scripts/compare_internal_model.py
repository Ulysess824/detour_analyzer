"""
Compare the forecast proposed by the company's internal model with the real consumption, our model and the planner.

The internal model covers many plantas and the months 2026-06 to 2026-09 (data/internal_model_forecast.csv, same SKU grain as the
planner files, no core width, so the dataset series with the same type, sub-grade, gsm and width are added up). Our forecasts are
the ones saved by run_ensembles.py (one month ahead, `ref_h1`) and by compare_horizon_strategies.py (two months ahead: `direct`,
`partial`, `iterated`), all with ml_mean, scored on exactly the same SKUs and months. The planner (SCAN only) is added on the SKUs
that have both forecasts. Metrics: WAPE, accuracy = 1 - WAPE, bias_pct, MAE.

Usage:
    python scripts/compare_internal_model.py data/consumos_long.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
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
    pd.set_option("display.max_rows", 100)

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

    internal = load_internal_forecast(args.internal)
    parts, unmatched = [], 0
    for (planta, month), frame in internal.groupby(["planta", "mes"]):
        if month not in S.labels or S.labels.index(month) not in set(res["t"]):
            continue
        table, missing = compare_month(frame, res, pairs, S.labels.index(month), MODELS)
        unmatched += len(missing)
        parts.append(table.assign(planta=planta, mes=month))
    T = pd.concat(parts, ignore_index=True).rename(columns={"planner": "internal"})
    T = T[T["real"] > 0].reset_index(drop=True)  # SKU-months with no consumption in the data cannot be scored with WAPE
    print(f"internal model: {internal['planta'].nunique()} plantas, {internal.groupby(['planta', 'sku_planner']).ngroups} SKUs, months {', '.join(sorted(internal['mes'].unique()))}")
    print(f"SKU-months: {len(internal)} in the file, {len(T)} scored ({unmatched} without a match in the data, rest with zero real consumption)")

    # the planner (SCAN only) on the same SKU-months
    planner_files = sorted(Path().glob(args.planner_glob))
    planner = pd.concat([pd.read_csv(f)[["planta", "mes", "sku_planner", "forecast_to"]] for f in planner_files], ignore_index=True)
    planner = planner.rename(columns={"sku_planner": "sku", "forecast_to": "planner"})
    T = T.merge(planner, on=["planta", "mes", "sku"], how="left")
    T.to_csv(args.out, index=False)

    def accuracies(frame: pd.DataFrame, columns: list[str]) -> dict:
        return {c: score(frame["real"].to_numpy(), frame[c].to_numpy()) for c in columns}

    cols = ["internal", *MODELS]
    print(f"\n=== All plantas: internal model vs our model ({len(T)} SKU-months, rank 1 = lowest WAPE) ===")
    print(accuracy_table(T, cols).round(4))
    for name in MODELS:
        wins = int(((T["real"] - T[name]).abs() < (T["real"] - T["internal"]).abs()).sum())
        print(f"  {name:9s} closer to the real than the internal model in {wins} of {len(T)} SKU-months")
    print(f"  totals: real {T['real'].sum():,.0f} | internal {T['internal'].sum():,.0f} ({100 * (T['internal'].sum() / T['real'].sum() - 1):+.1f}%) | direct {T['direct'].sum():,.0f} | one month ahead {T['ref_h1'].sum():,.0f}")

    print("\n=== Accuracy (%) by month ===")
    print(pd.DataFrame({m: {c: 100 * score(g["real"].to_numpy(), g[c].to_numpy())["accuracy"] for c in cols} for m, g in T.groupby("mes")}).round(1))

    print("\n=== Accuracy (%) by planta: internal / direct / one month ahead (SKU-months, share of the real volume) ===")
    rows = []
    for planta, g in T.groupby("planta"):
        a = accuracies(g, ["internal", "direct", "ref_h1"])
        rows.append({"planta": planta, "n": len(g), "volume_%": 100 * g["real"].sum() / T["real"].sum(), **{c: 100 * v["accuracy"] for c, v in a.items()}})
    print(pd.DataFrame(rows).sort_values("volume_%", ascending=False).round(1).to_string(index=False))

    both = T[T["planner"].notna()]
    if len(both):
        pcols = ["planner", "internal", *MODELS]
        print(f"\n=== SCAN: planner, internal model and our model on the same SKU-months ({len(both)}) ===")
        print(accuracy_table(both, pcols).round(4))
        print("accuracy (%) by month:")
        print(pd.DataFrame({m: {c: 100 * score(g["real"].to_numpy(), g[c].to_numpy())["accuracy"] for c in pcols} for m, g in both.groupby("mes")}).round(1))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
