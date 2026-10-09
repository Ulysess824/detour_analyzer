"""
Forecast the month after next (what a mid-month planner deadline needs) and compare strategies.

At mid-month the last complete month is `o`, and the month to forecast is `o + 2`. The model is `ml_mean`, the mean of
LightGBM, XGBoost and Random Forest; the metrics are the usual ones (WAPE, accuracy = 1 - WAPE, bias_pct, MAE) at SKU,
planta and total level, over the last --n-test months with a rolling origin.

    1. strategy      direct (one model for the two-month target) vs iterated (a one-month model applied twice)
    2. partial month direct vs direct plus the consumption of the first --cutoff-day days of the month in progress
    3. components    the three trees behind `ml_mean` (LightGBM, XGBoost, Random Forest) next to their mean

Only machine-learning models are trained. With --with-local the per-series SES, damped Holt and ARIMA are added (slow).

The reference is the one-month-ahead `ml_mean` of the report (more information: data to the end of the previous month).
When planner files exist (data/planner_forecast_<month>.csv, SCAN only), a last section scores the planner and the strategies on
the planner SKUs of those months. The combination of both forecasts is left out on purpose.

Usage:
    python scripts/compare_horizon_strategies.py data/consumos_long.csv
"""

import argparse
import json
import warnings

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.cache_utils import check_alignment, check_meta, expected_meta, panel_fingerprint, write_meta
from src.utils.evaluation_utils import score_models_by_level, test_months_of
from src.utils.feature_utils import make_frame
from src.utils.horizon_utils import HORIZON, strategy_forecasts
from src.utils.metrics_utils import score
from src.utils.panel_utils import build_panel, load_consumption
from src.utils.planner_utils import accuracy_table, compare_month, series_table
from src.utils.partial_utils import add_partial_features
from src.utils.tuning_utils import check_tuning_window

SECTIONS = {
    "1. Strategy: direct vs iterated (reference: one month ahead)": ["ref_h1", "direct", "iterated", "naive"],
    "2. Month in progress: direct vs direct + first days of the month": ["direct", "partial", "naive"],
    "3. Components of ml_mean (direct) and simple references": ["direct", "direct_lgbm", "direct_xgb", "direct_rf", "naive", "mean6", "seasonal_naive"],
    "4. Optional: global trees vs local per-series models (--with-local)": [
        "direct", "local_mean", "local_ses", "local_damped_holt", "local_arima", "naive",
    ],
}  # fmt: skip


def planner_section(S, df, res: pd.DataFrame, pattern: str, planta: str):
    r"""Score the planner and the strategies on the planner SKUs of every month that has a planner file."""
    files = sorted(Path().glob(pattern))
    if not files:
        return None
    models = [n for n in ("ref_h1", "direct", "iterated", "partial", "naive") if f"p_{n}" in res.columns]
    pairs, parts = series_table(S, df), []
    for file in files:
        planner = pd.read_csv(file)
        planner = planner[planner["planta"] == planta]
        if planner.empty:
            continue
        month = planner["mes"].iloc[0]
        if month not in S.labels or S.labels.index(month) not in set(res["t"]):
            continue
        table, missing = compare_month(planner, res, pairs, S.labels.index(month), models)
        if missing:
            print(f"{month}: planner SKUs with no match: {missing}")
        parts.append(table.assign(mes=month))
    if not parts:
        return None
    T = pd.concat(parts, ignore_index=True)
    title = f"5. Planner row: {planta} planner SKUs, months {', '.join(sorted(T['mes'].unique()))} ({len(T)} SKU-months)"
    print(f"\n=== {title} ===")
    ranking = accuracy_table(T, ["planner", *models])
    print(ranking.round(4))
    by_month = pd.DataFrame(
        {m: {c: score(g["real"].to_numpy(), g[c].to_numpy())["accuracy"] for c in ["planner", *models]} for m, g in T.groupby("mes")}
    )
    print("\naccuracy by month:")
    print((by_month * 100).round(1))
    for name in models:
        wins = int(((T["real"] - T[name]).abs() < (T["real"] - T["planner"]).abs()).sum())
        print(f"  {name:8s} closer to the real than the planner in {wins} of {len(T)} SKU-months")
    T.to_csv("results/horizon_planner_comparison.csv", index=False)
    print("saved results/horizon_planner_comparison.csv")
    return ranking.reset_index(drop=True).assign(section=title, level="sku_planner")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--tuning", type=Path, default=Path("results/tuning_trees.json"))
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--cutoff-day", type=int, default=15, help="calendar day of the month in progress at the forecast time")
    parser.add_argument("--predictions", type=Path, default=Path("results/horizon_strategies.csv"))
    parser.add_argument("--reference", type=Path, default=Path("results/member_forecasts.csv"), help="one-month-ahead forecasts (run_ensembles.py)")
    parser.add_argument("--refit", action="store_true", help="recompute the forecasts even if cached")
    parser.add_argument("--with-local", action="store_true", help="also fit SES, damped Holt and ARIMA per series (slow)")
    parser.add_argument("--planner-glob", default="data/planner_forecast_*.csv", help="planner forecasts to add as a row")
    parser.add_argument("--planta", default="SCAN", help="planta of the planner row (the planner files hold several plantas)")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    df = load_consumption(args.input)
    S = build_panel(df)
    test_months = test_months_of(S, args.n_test)
    tuned = json.loads(args.tuning.read_text()) if args.tuning.exists() else {}
    check_tuning_window(tuned, S, args.n_test)  # the tuned parameters (h=1) must not have seen the test months
    meta = {**expected_meta(S, HORIZON, args.n_test, tuned), "cutoff_day": args.cutoff_day, "local": args.with_local}
    print(f"test targets {S.labels[test_months[0]]} .. {S.labels[-1]} | last complete month = target - {HORIZON}")

    if args.predictions.exists() and not args.refit:
        check_meta(args.predictions, meta)
        res = pd.read_csv(args.predictions)
        check_alignment(res, S)
        print(f"loaded forecasts from {args.predictions}")
    else:
        frame1, frame2 = make_frame(S, 1), make_frame(S, HORIZON)
        frame2_partial = add_partial_features(frame2, S, args.cutoff_day)
        res = strategy_forecasts(S, frame1, frame2, frame2_partial, tuned, test_months, args.jobs, local=args.with_local)
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.predictions, index=False)
        write_meta(args.predictions, meta)

    if args.reference.exists():
        check_meta(args.reference, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1}, keys=("data", "n_series", "horizon"))
        ref = pd.read_csv(args.reference)
        check_alignment(ref, S)
        ref["p_ref_h1"] = ref[[f"p_{k}" for k in ("lgbm", "xgb", "rf")]].mean(axis=1)
        res = res.merge(ref[["series", "t", "p_ref_h1"]], on=["series", "t"], how="left")
    else:
        print(f"no {args.reference}: the one-month reference is skipped")

    tables = []
    for title, names in SECTIONS.items():
        models = [n for n in names if f"p_{n}" in res.columns]
        if not any(n not in ("direct", "naive") for n in models):
            continue
        print(f"\n=== {title} ===")
        for level, table in score_models_by_level(S, res, models, test_months).items():
            print(f"\n[{level}] rank 1 = lowest WAPE")
            print(table[["rank", "n", "mae", "wape", "accuracy", "bias_pct"]])
            tables.append(table.reset_index().assign(section=title, level=level))

    planner_rows = planner_section(S, df, res, args.planner_glob, args.planta)
    if planner_rows is not None:
        tables.append(planner_rows)

    shown = [c[2:] for c in ("p_ref_h1", "p_direct", "p_iterated", "p_partial", "p_naive") if c in res.columns]
    print("\n=== SKU accuracy (1 - WAPE) by target month ===")
    by_month = pd.DataFrame(
        {n: {S.labels[t]: score(g["y"].to_numpy(), g[f"p_{n}"].to_numpy())["accuracy"] for t, g in res.groupby("t")} for n in shown}
    )
    print((by_month * 100).round(1))

    out = args.predictions.with_name("horizon_strategies_scores.csv")
    pd.concat(tables).to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
