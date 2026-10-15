"""
ml_mean fed with the month in progress completed by the mean of its first days (nowcast input), against the strategies already measured.

Target month t is forecast at mid-month t-1: last complete month t-2 (real), first 15 days of t-1 (real) and the rest of t-1 filled with the
mean of those days (src/utils/nowcast_input_utils.py); the one-month-ahead ml_mean (tuned parameters of results/tuning_trees.json) is applied
to that completed month. Compared with the caches of compare_horizon_strategies.py (direct, direct + first days, iterated, naive) and
run_ensembles.py (one-month reference, with the whole real month t-1), on the same series (alive at t-2).

Usage:
    python scripts/nowcast_input_ml_mean.py data/consumos_long.csv --months 2026-06 2026-07 2026-08 2026-09
    python scripts/nowcast_input_ml_mean.py data/consumos_long.csv --last-months 12
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.cache_utils import check_alignment, check_meta, expected_meta, panel_fingerprint
from src.utils.evaluation_utils import score_models_by_level
from src.utils.feature_utils import make_frame
from src.utils.horizon_utils import HORIZON
from src.utils.nowcast_input_utils import nowcast_input_forecast
from src.utils.panel_utils import build_panel, load_consumption
from src.utils.tuning_utils import check_tuning_window

NAMES = {
    "nowcast": "A. Mes completado con la media",
    "direct": "B. Directa (t-2)",
    "partial": "C. Directa + 15 días de t-1",
    "ref_h1": "D. Referencia (mes t-1 real completo)",
    "iterated": "F. Iterada",
    "naive": "E. Naive",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--months", nargs="+", default=None, help="target months (YYYY-MM)")
    parser.add_argument("--last-months", type=int, default=0, help="the last N months of the data instead of --months")
    parser.add_argument("--cutoff-day", type=int, default=15)
    parser.add_argument("--tuning", type=Path, default=Path("results/tuning_trees.json"))
    parser.add_argument("--n-test", type=int, default=12, help="test window of the caches")
    parser.add_argument("--strategies", type=Path, default=Path("results/horizon_strategies.csv"))
    parser.add_argument("--reference", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--out", type=Path, default=None, help="csv with the forecasts of every method (series x month)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:,.1f}")

    S = build_panel(load_consumption(args.input))
    tuned = json.loads(args.tuning.read_text()) if args.tuning.exists() else {}
    check_tuning_window(tuned, S, args.n_test)
    meta = {**expected_meta(S, HORIZON, args.n_test, tuned), "cutoff_day": args.cutoff_day}
    check_meta(args.strategies, meta)
    check_meta(args.reference, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1}, keys=("data", "n_series", "horizon"))
    strat, ref = pd.read_csv(args.strategies), pd.read_csv(args.reference)
    check_alignment(strat, S)
    check_alignment(ref, S)
    ref["p_ref_h1"] = ref[["p_lgbm", "p_xgb", "p_rf"]].mean(axis=1)

    if args.last_months:
        targets = list(range(len(S.labels) - args.last_months, len(S.labels)))
    else:
        targets = [S.labels.index(m) for m in (args.months or ["2026-06", "2026-07", "2026-08", "2026-09"])]
    frame1 = make_frame(S, 1)

    parts = []
    for t in targets:
        origin = t - HORIZON
        a = nowcast_input_forecast(S, frame1, origin, tuned, args.cutoff_day).rename(columns={"ml_mean": "p_nowcast", "lgbm": "p_nowcast_lgbm", "xgb": "p_nowcast_xgb", "rf": "p_nowcast_rf"})
        base = strat[strat["t"] == t].rename(columns={"p_direct": "p_direct", "p_partial": "p_partial", "p_iterated": "p_iterated", "p_naive": "p_naive"})
        base = base.merge(ref[ref["t"] == t][["series", "p_ref_h1"]], on="series", how="left").rename(columns={"p_ref_h1": "p_ref_h1"})
        block = base.merge(a, on="series", how="left")
        parts.append(block)
        print(f"target {S.labels[t]} (origin {S.labels[origin]}): {len(block):,} series", flush=True)
    res = pd.concat(parts, ignore_index=True)
    models = [k for k in NAMES if f"p_{k}" in res.columns]
    res = res.dropna(subset=[f"p_{k}" for k in models]).reset_index(drop=True)  # the same series for every method
    print(f"series-months compared: {len(res):,}\n")

    def table(months: list[int]) -> pd.DataFrame:
        sub = res[res["t"].isin(months)]
        t = score_models_by_level(S, sub, models, months)
        out = pd.DataFrame({lv: t[lv]["accuracy"] * 100 for lv in ("sku", "planta", "total")})
        out["sesgo_total_%"] = -t["total"]["bias_pct"]  # forecast / real - 1 (the project's bias_pct is real - forecast)
        return out.reindex(models).rename(index=NAMES)

    print("POOLED over the target months: accuracy %, bias of the total (forecast / real - 1, %)")
    print(table(targets).to_string())
    monthly = pd.DataFrame({S.labels[t]: table([t])["sku"] for t in targets}).T
    print("\nSKU accuracy % by target month")
    print(monthly.to_string())
    print("\nmean:", monthly.mean().round(1).to_dict())
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
