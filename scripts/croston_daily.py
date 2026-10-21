"""
Daily Croston experiment for one planta and SKU (docs/plan_croston.md, section "Versión diaria"; branch experiment/croston).
The forecast is made at the end of each working day for the next working day (h=1, flat); weeks and months are the sum of the
daily forecasts from the last working day before them. The app and the monthly pipeline are not touched; the monthly ml_mean is read
from results/member_forecasts.csv (run_ensembles.py).

Usage:
    python scripts/croston_daily.py data/consumos_long.csv --planta SVIG --sku TSL/01/1/100gsm/2300mm --test-start 2026-01-01
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.panel_utils import load_consumption
from src.experiments import croston_backtest as cb
from src.experiments import croston_scores as cs
from src.experiments import daily_utils as du
from src.experiments.croston_intervals import DRAWS
from src.experiments.weekly_utils import match_planner_sku, weekly_series

WINDOWS = (6, 24)  # moving averages in working days: one week and four weeks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--planta", required=True)
    parser.add_argument("--sku", required=True, help="consumption SKU or planner name (grade/sub-grade/gsm/width)")
    parser.add_argument("--test-start", default="2026-01-01", help="first target day")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictions", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    df = load_consumption(args.data, validate=False)
    skus = df.loc[df["planta"] == args.planta, "sku"].unique()
    sku = args.sku if args.sku in skus else match_planner_sku(skus, args.sku)
    series = du.daily_series(df, args.planta, sku)
    print(f"{args.planta} | {sku}: {len(series)} working days {series.index.min().date()} .. {series.index.max().date()}, "
          f"{int((series > 0).sum())} with consumption ({(series > 0).mean():.0%})")
    args.out_dir.mkdir(exist_ok=True)

    daily = cb.rolling_origin(series, args.test_start, (1,), args.draws, args.seed, windows=WINDOWS)
    daily.to_csv(args.out_dir / "croston_daily_forecasts.csv", index=False)
    scores, intervals = cs.weekly_scores(daily, 1, rmse=True), cs.interval_scores(daily, 1)
    scores.to_csv(args.out_dir / "croston_daily_scores.csv")
    intervals.to_csv(args.out_dir / "croston_daily_intervals.csv")
    print(f"\nDaily accuracy, h=1 ({int(scores['n'].iloc[0])} target days; secondary metric, see the plan):\n{scores.to_string()}")
    print(f"\n95% daily interval:\n{intervals.to_string()}")

    real_weekly = weekly_series(df, args.planta, sku)
    weeks = du.week_comparison(series, real_weekly, args.test_start, args.draws, args.seed, WINDOWS)
    weeks.to_csv(args.out_dir / "croston_daily_weeks.csv", index=False)
    week_scores = cs.monthly_scores(weeks)
    week_scores.to_csv(args.out_dir / "croston_daily_week_scores.csv")
    print(f"\nWeekly total, {len(weeks)} weeks (6 x the daily forecast of the Saturday before):\n{week_scores.to_string()}")

    real_monthly = df[(df["planta"] == args.planta) & (df["sku"] == sku)].groupby(df["fecha"].dt.to_period("M"))["consumo"].sum()
    ml = du.ml_mean_by_month(args.predictions, args.planta, sku, df, str(pd.Timestamp(args.test_start).to_period("M")))
    months = du.month_comparison(series, real_monthly.reindex(ml.index, fill_value=0.0), ml, args.draws, args.seed, WINDOWS)
    months.to_csv(args.out_dir / "croston_daily_months.csv", index=False)
    month_scores = cs.monthly_scores(months)
    month_scores.to_csv(args.out_dir / "croston_daily_month_scores.csv")
    print(f"\nMonthly total, {len(months)} months (working days of the month x the daily forecast of the last day before it; "
          f"ml_mean has the same information):\n{month_scores.to_string()}")

    decision = cs.daily_decision(week_scores, month_scores, intervals)
    (args.out_dir / "croston_daily_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    print(f"\nCriteria: {json.dumps(decision)}")


if __name__ == "__main__":
    main()
