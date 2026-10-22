"""
Compare a saved forecast with the real consumption once its month has closed, and store the accuracy.

Reads `results/forecast_<month>[_<strategy>].csv` and its manifest, checks that the data covers the month completely, scores every model at SKU,
planta and total level, prints the result and the alerts (thresholds in src/evaluation/tracking_utils.py: one set for the one-month forecast and a lower
one for the strategies that forecast two months ahead), and stores the rows, with the strategy and cutoff day of the forecast, in the table `seguimiento`
of the database and in `results/seguimiento.csv` (kept in git, so a rebuilt database gets the history back). Running it again for the same run and
month replaces the rows. With --backtest it fills the history with the 12 months of the one-month-ahead backtest (run `backtest-h1`) and, when
results/horizon_strategies.csv exists, those of the two-month strategies (`backtest-directa`, `backtest-parcial`, `backtest-iterada`).
Exit code 2 when an alert fires (the alerts of the two-month backtests are shown but do not count).

Usage:
    python scripts/track_forecast.py results/forecast_2026-10.csv --data data/consumos_long.csv
    python scripts/track_forecast.py --backtest --data data/consumos_long.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.panel_utils import build_panel, load_consumption
from src.engine.run_manifest import manifest_path
from src.evaluation.tracking_run import save_tracking, track_backtest, track_horizon_backtests, track_saved_forecast
from src.evaluation.tracking_utils import THRESHOLDS


def show(scores: pd.DataFrame, alerts: pd.DataFrame, strategy: str = "h1", cutoff: int | None = None) -> None:
    month = scores["mes"].iloc[0]
    table = scores[(scores["planta"] == "") & scores["nivel"].isin(["sku", "planta", "total"])].pivot(index="modelo", columns="nivel", values="acierto")
    table = (table[["sku", "planta", "total"]] * 100).reindex([m for m in ("ml_mean", "lgbm", "xgb", "rf", "naive") if m in table.index]).round(1)
    print(f"\n{month}: accuracy % (1 - WAPE), strategy {strategy}" + (f", cutoff day {int(cutoff)}" if cutoff else ""))
    print(table.to_string())
    ml = scores[(scores["modelo"] == "ml_mean") & (scores["nivel"] == "total")].iloc[0]
    print(f"total: real {ml['real_TO']:,.1f} TO, ml_mean {ml['pronostico_TO']:,.1f} TO, bias_pct {ml['bias_pct']:+.1f} (positive = forecast too low)")
    extra = scores[scores["nivel"] == "sin_pronostico"].iloc[0]
    if extra["n"]:
        print(f"series without forecast: {int(extra['n'])} with {extra['real_TO']:,.1f} TO in the real")
    for _, a in alerts.iterrows():
        where = f" {a['planta']}" if a["planta"] else ""
        print(f"{'ALERT' if a['tipo'] == 'alerta' else 'review'} {a['codigo']}{where}: {a['valor'] * 100:.1f}% (threshold {a['umbral'] * 100:.1f}%) - {a['texto']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("forecast", nargs="?", type=Path, help="results/forecast_<month>.csv (not needed with --backtest)")
    parser.add_argument("--data", type=Path, default=Path("data/consumos_long.csv"))
    parser.add_argument("--db", type=Path, default=Path("data/detour.db"))
    parser.add_argument("--csv", type=Path, default=Path("results/seguimiento.csv"))
    parser.add_argument("--backtest", action="store_true", help="store the 12 backtest months of results/member_forecasts.csv (and of the two-month strategies)")
    parser.add_argument("--reference", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--horizon-reference", type=Path, default=Path("results/horizon_strategies.csv"), help="two-month backtest cache (compare_horizon_strategies.py)")
    for key, value in THRESHOLDS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=None, help=f"threshold for every strategy (default: by strategy; one month ahead {value})")
    args = parser.parse_args()
    if not args.backtest and args.forecast is None:
        parser.error("give the forecast csv, or --backtest")
    thresholds = {key: getattr(args, key) for key in THRESHOLDS if getattr(args, key) is not None} or None

    df = load_consumption(args.data)
    S = build_panel(df)
    if args.backtest:
        repo = Path(__file__).resolve().parents[1]
        runs = track_backtest(S, args.reference, thresholds, repo_dir=repo)
        horizon = track_horizon_backtests(S, args.horizon_reference, thresholds, repo_dir=repo) if args.horizon_reference.exists() else []
        if not horizon:
            print(f"no {args.horizon_reference}: the two-month backtests are skipped")
        for group in (runs, horizon):
            for run_id in dict.fromkeys(run.run_id for run in group):
                rows = pd.concat([run.rows for run in group if run.run_id == run_id], ignore_index=True)
                ml = rows[(rows["modelo"] == "ml_mean") & (rows["planta"] == "") & rows["nivel"].isin(["sku", "planta", "total"])]
                print(f"\n{run_id}")
                print((ml.pivot(index="mes", columns="nivel", values="acierto")[["sku", "planta", "total"]] * 100).round(1).to_string())
                print("mean:", (ml.groupby("nivel")["acierto"].mean() * 100).round(1).to_dict(), "| alerts fired:", sum(run.fired for run in group if run.run_id == run_id))
        fired = sum(run.fired for run in runs)  # the alerts of the two-month backtests are a reference, not a failure
        runs = runs + horizon
    else:
        run = track_saved_forecast(args.forecast, df, S, thresholds)
        runs = [run]
        if not run.manifest_found:
            print(f"warning: {manifest_path(args.forecast)} not found; the run is identified by the hash of the csv")
        show(run.scores, run.alerts, run.strategy, run.cutoff)
        fired = run.fired

    written = save_tracking(runs, args.csv, args.db)
    print(f"\nsaved {written} rows to {args.csv} and to {args.db}")
    if fired:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
