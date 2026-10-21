"""
Weekly Croston experiment for one planta and SKU (docs/plan_croston.md, branch experiment/croston). The app and the monthly
pipeline are not touched; the monthly ml_mean is read from results/member_forecasts.csv (run_ensembles.py).

Usage:
    python scripts/croston_weekly.py data/consumos_long.csv --planta SVIG --sku TSL/01/1/100gsm/2300mm --test-start 2026-01-01
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.panel_utils import build_panel, load_consumption
from src.evaluation.cache_utils import check_alignment
from src.experiments import croston_backtest as cb
from src.experiments import croston_scores as cs
from src.experiments.croston_intervals import DRAWS
from src.experiments.weekly_utils import match_planner_sku, weekly_series


def ml_mean_by_month(predictions: Path, planta: str, sku: str, df: pd.DataFrame, first_month: str) -> pd.Series:
    r"""ml_mean (mean of lgbm, xgb, rf) of the one-month-ahead cache for the target months from `first_month`, indexed by Period."""
    S = build_panel(df)
    res = pd.read_csv(predictions)
    check_alignment(res, S)
    series = S.keys.index((planta, sku))
    rows = res[res["series"] == series]
    months = pd.PeriodIndex([S.labels[t] for t in rows["t"]], freq="M")
    ml = pd.Series(rows[["p_lgbm", "p_xgb", "p_rf"]].mean(axis=1).to_numpy(), index=months)
    return ml[ml.index >= pd.Period(first_month, "M")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--planta", required=True)
    parser.add_argument("--sku", required=True, help="consumption SKU or planner name (grade/sub-grade/gsm/width)")
    parser.add_argument("--test-start", default="2026-01-01", help="first target week (Monday on or after this day)")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictions", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    df = load_consumption(args.data, validate=False)
    skus = df.loc[df["planta"] == args.planta, "sku"].unique()
    sku = args.sku if args.sku in skus else match_planner_sku(skus, args.sku)
    series = weekly_series(df, args.planta, sku)
    print(f"{args.planta} | {sku}: {len(series)} complete weeks {series.index.min().date()} .. {series.index.max().date()}, "
          f"{int((series == 0).sum())} with no consumption")

    weekly = cb.rolling_weekly(series, args.test_start, tuple(args.horizons), args.draws, args.seed)
    args.out_dir.mkdir(exist_ok=True)
    weekly.to_csv(args.out_dir / "croston_weekly_forecasts.csv", index=False)

    cs.wide_table(weekly, args.horizons[0]).to_csv(args.out_dir / "croston_weekly_ejemplo.csv")
    scores = pd.concat({h: cs.weekly_scores(weekly, h) for h in args.horizons}, names=["h"])
    intervals = pd.concat({h: cs.interval_scores(weekly, h) for h in args.horizons}, names=["h"])
    scores.to_csv(args.out_dir / "croston_weekly_scores.csv")
    intervals.to_csv(args.out_dir / "croston_weekly_intervals.csv")
    for h in args.horizons:
        print(f"\nWeekly accuracy, h={h} ({int(scores.loc[h, 'n'].iloc[0])} target weeks):\n{scores.loc[h].to_string()}")
        print(f"\n95% interval, h={h}:\n{intervals.loc[h].to_string()}")

    real_monthly = df[(df["planta"] == args.planta) & (df["sku"] == sku)].groupby(df["fecha"].dt.to_period("M"))["consumo"].sum()
    ml = ml_mean_by_month(args.predictions, args.planta, sku, df, str(pd.Timestamp(args.test_start).to_period("M")))
    real_monthly = real_monthly.reindex(ml.index, fill_value=0.0)
    monthly = cb.monthly_comparison(series, real_monthly, ml, args.draws, args.seed)
    monthly.to_csv(args.out_dir / "croston_weekly_monthly.csv", index=False)
    monthly_scores = cs.monthly_scores(monthly)
    monthly_scores.to_csv(args.out_dir / "croston_weekly_monthly_scores.csv")
    print(f"\nMonthly total, {len(monthly)} months (weekly models: forecast of the last week before the month x business days / 6):\n"
          f"{monthly_scores.to_string()}")

    if 1 in args.horizons:
        decision = cs.decision_criteria(scores.loc[1], intervals.loc[1])
        (args.out_dir / "croston_weekly_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
        print(f"\nCriteria (h=1): {json.dumps(decision)}")


if __name__ == "__main__":
    main()
