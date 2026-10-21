"""
Croston as a probability of consumption and a size (docs/plan_croston.md, section "Probabilidad y tamaño"; branch experiment/croston).
Daily rolling origin over the series of one planta, Monday to Friday; each day uses only the history before it. The app and the monthly
pipeline are not touched.

Usage:
    python scripts/croston_probability.py data/consumos_long.csv --planta SVIG --skus-file docs/croston_svig_skus.txt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.panel_utils import load_consumption
from src.experiments import croston_probability as cp
from src.experiments.daily_utils import daily_series
from src.experiments.weekday_profile import SHRINK
from src.experiments.weekly_utils import match_planner_sku


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--planta", required=True)
    parser.add_argument("--skus", nargs="+", default=None, help="planner names or consumption SKUs")
    parser.add_argument("--skus-file", type=Path, default=None, help="one SKU per line (blank lines and # comments are ignored)")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--weekdays", type=int, choices=(5, 6), default=5)
    parser.add_argument("--shrink", type=float, default=SHRINK)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    names = list(args.skus or [])
    if args.skus_file:
        names += [ln.strip() for ln in args.skus_file.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if not names:
        raise SystemExit("Give --skus or --skus-file.")
    df = load_consumption(args.data, validate=False)
    available = df.loc[df["planta"] == args.planta, "sku"].unique()
    tables = []
    for name in names:
        sku = name if name in available else match_planner_sku(available, name, ignore_substitutes=True)
        table = cp.rolling_probabilities(daily_series(df, args.planta, sku, args.weekdays), args.test_start, args.weekdays, args.shrink)
        table.insert(0, "series", f"{args.planta}|{sku}")
        tables.append(table)
        print(f"[ok] {name}: {len(table)} days, {int(table['has'].sum())} with consumption", flush=True)
    days = pd.concat(tables, ignore_index=True)

    out = args.out_dir
    out.mkdir(exist_ok=True)
    scores, sizes = cp.probability_scores(days), cp.size_scores(days)
    calib = pd.concat({m: cp.calibration(days, m) for m in cp.PROB_MODELS}, names=["model"]).reset_index(level=0)
    per_series = pd.DataFrame({s: cp.probability_scores(g)["brier_skill"] for s, g in days.groupby("series")}).T
    days.round(4).to_csv(out / "croston_prob_days.csv", index=False)
    scores.round(4).to_csv(out / "croston_prob_scores.csv")
    sizes.to_csv(out / "croston_prob_sizes.csv")
    calib.round(4).to_csv(out / "croston_prob_calibration.csv", index=False)
    per_series.round(4).to_csv(out / "croston_prob_per_series.csv")
    print(f"\nProbability of consumption ({len(days)} series-days, share with consumption {days['has'].mean():.1%}):\n{scores.round(4).to_string()}")
    print(f"\nSize on the {int(days['has'].sum())} days with consumption (TO):\n{sizes.to_string()}")
    print(f"\nCalibration of croston_dow:\n{calib[calib['model'] == 'croston_dow'].round(3).to_string(index=False)}")
    print(f"\nBrier skill against the frequency, per series:\n{per_series.round(3).to_string()}")


if __name__ == "__main__":
    main()
