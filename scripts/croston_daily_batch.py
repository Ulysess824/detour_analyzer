"""
Daily Croston experiment over several series of one planta (docs/plan_croston.md, section "Varias series"; branch experiment/croston).
Each series gets the protocol of croston_daily.py; results are pooled and compared across series. The app and the monthly pipeline
are not touched; the monthly ml_mean is read from results/member_forecasts.csv (run_ensembles.py).

Usage:
    python scripts/croston_daily_batch.py data/consumos_long.csv --planta SVIG --skus-file docs/croston_svig_skus.txt
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.panel_utils import load_consumption
from src.experiments import croston_batch as cbt
from src.experiments import croston_scores as cs
from src.experiments.croston_intervals import DRAWS
from src.experiments.daily_utils import daily_series, ml_mean_by_month
from src.experiments.weekly_utils import match_planner_sku


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--planta", required=True)
    parser.add_argument("--skus", nargs="+", default=None, help="planner names or consumption SKUs")
    parser.add_argument("--skus-file", type=Path, default=None, help="one SKU per line (blank lines and # comments are ignored)")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictions", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    names = list(args.skus or [])
    if args.skus_file:
        names += [ln.strip() for ln in args.skus_file.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if not names:
        raise SystemExit("Give --skus or --skus-file.")

    df = load_consumption(args.data, validate=False)
    available = df.loc[df["planta"] == args.planta, "sku"].unique()
    first_month = str(pd.Timestamp(args.test_start).to_period("M"))
    tables = {"daily": [], "weeks": [], "months": []}
    profile = []
    for name in names:
        sku = name if name in available else match_planner_sku(available, name, ignore_substitutes=True)
        ml = ml_mean_by_month(args.predictions, args.planta, sku, df, first_month)
        result = cbt.run_series(df, args.planta, sku, ml, args.test_start, args.draws, args.seed)
        for k in tables:
            tables[k].append(result[k])
        s = daily_series(df, args.planta, sku)
        test = s[s.index >= args.test_start]
        profile.append({"planner_name": name, "sku": sku, "working_days": len(s), "days_with_consumption": int((s > 0).sum()),
                        "share_with_consumption": round(float((s > 0).mean()), 3), "to_in_test": round(float(test.sum()), 1),
                        "months_with_ml_mean": len(result["months"])})
        print(f"[ok] {name} -> {sku}: {profile[-1]['share_with_consumption']:.0%} of the days with consumption, "
              f"{profile[-1]['to_in_test']} TO in the test, {profile[-1]['months_with_ml_mean']} months with ml_mean", flush=True)

    daily, weeks, months = (pd.concat(tables[k], ignore_index=True) for k in ("daily", "weeks", "months"))
    out = args.out_dir
    out.mkdir(exist_ok=True)
    pd.DataFrame(profile).to_csv(out / "croston_batch_series.csv", index=False)
    weeks.to_csv(out / "croston_batch_weeks.csv", index=False)
    months.to_csv(out / "croston_batch_months.csv", index=False)

    daily_scores, intervals = cs.weekly_scores(daily, 1, rmse=True), cs.interval_scores(daily, 1)
    week_scores, month_scores = cs.monthly_scores(weeks), cs.monthly_scores(months)
    week_series, month_series = cbt.per_series_accuracy(weeks), cbt.per_series_accuracy(months)
    daily_scores.to_csv(out / "croston_batch_daily_scores.csv")
    intervals.to_csv(out / "croston_batch_intervals.csv")
    week_scores.to_csv(out / "croston_batch_week_scores.csv")
    month_scores.to_csv(out / "croston_batch_month_scores.csv")
    week_series.to_csv(out / "croston_batch_week_per_series.csv")
    month_series.to_csv(out / "croston_batch_month_per_series.csv")

    print(f"\nPooled weekly accuracy ({int(week_scores['n'].iloc[0])} series-weeks):\n{week_scores.to_string()}")
    print(f"\nPooled monthly accuracy ({int(month_scores['n'].iloc[0])} series-months, ml_mean has the same information):\n{month_scores.to_string()}")
    print(f"\nPooled daily ({int(daily_scores['n'].iloc[0])} series-days; secondary):\n{daily_scores.to_string()}")
    print(f"\nPooled 95% daily interval:\n{intervals.to_string()}")
    print(f"\nMonthly accuracy per series:\n{month_series.to_string()}")

    comparisons = []
    for model in cs.CROSTON_MODELS:
        for versus in ("ml_mean", "ses"):
            wins = cbt.win_count(month_series, model, versus)
            boot = cbt.bootstrap_accuracy_diff(months, model, versus, seed=args.seed)
            comparisons.append({"model": model, "versus": versus, "series_won": wins, "series": len(month_series),
                                "sign_test_p": round(cbt.sign_test_p(wins, len(month_series)), 3), **{k: round(v, 4) for k, v in boot.items() if k != "series"}})
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(out / "croston_batch_comparisons.csv", index=False)
    print(f"\nMonthly: Croston models against ml_mean and ses (accuracy difference pooled, 90% bootstrap over series):\n{comparisons.to_string()}")

    decision = cbt.batch_decision(week_scores, month_scores, month_series, intervals)
    (out / "croston_batch_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    print(f"\nCriteria: {json.dumps(decision)}")


if __name__ == "__main__":
    main()
