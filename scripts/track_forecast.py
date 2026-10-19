"""
Compare a saved forecast with the real consumption once its month has closed, and store the accuracy.

Reads `results/forecast_<month>.csv` and its manifest, checks that the data covers the month completely, scores every model at SKU, planta and
total level, prints the result and the alerts (thresholds in src/evaluation/tracking_utils.py), and stores the rows in the table `seguimiento` of
the database and in `results/seguimiento.csv` (kept in git, so a rebuilt database gets the history back). Running it again for the same run and
month replaces the rows. With --backtest it fills the history with the 12 months of the one-month-ahead backtest (run `backtest-h1`).
Exit code 2 when an alert fires.

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
from src.evaluation.tracking_run import save_tracking, track_backtest, track_saved_forecast
from src.evaluation.tracking_utils import THRESHOLDS


def show(scores: pd.DataFrame, alerts: pd.DataFrame) -> None:
    month = scores["mes"].iloc[0]
    table = scores[(scores["planta"] == "") & scores["nivel"].isin(["sku", "planta", "total"])].pivot(index="modelo", columns="nivel", values="acierto")
    table = (table[["sku", "planta", "total"]] * 100).reindex([m for m in ("ml_mean", "lgbm", "xgb", "rf", "naive") if m in table.index]).round(1)
    print(f"\n{month}: accuracy % (1 - WAPE)")
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
    parser.add_argument("--backtest", action="store_true", help="store the 12 backtest months of results/member_forecasts.csv")
    parser.add_argument("--reference", type=Path, default=Path("results/member_forecasts.csv"))
    for key, value in THRESHOLDS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=value, help=f"threshold (default {value})")
    args = parser.parse_args()
    if not args.backtest and args.forecast is None:
        parser.error("give the forecast csv, or --backtest")
    thresholds = {key: getattr(args, key) for key in THRESHOLDS}

    df = load_consumption(args.data)
    S = build_panel(df)
    if args.backtest:
        runs = track_backtest(S, args.reference, thresholds, repo_dir=Path(__file__).resolve().parents[1])
        rows = pd.concat([run.rows for run in runs], ignore_index=True)
        ml = rows[(rows["modelo"] == "ml_mean") & (rows["planta"] == "") & rows["nivel"].isin(["sku", "planta", "total"])]
        print((ml.pivot(index="mes", columns="nivel", values="acierto")[["sku", "planta", "total"]] * 100).round(1).to_string())
        print("mean:", (ml.groupby("nivel")["acierto"].mean() * 100).round(1).to_dict(), "| alerts fired:", sum(run.fired for run in runs))
    else:
        run = track_saved_forecast(args.forecast, df, S, thresholds)
        runs = [run]
        if not run.manifest_found:
            print(f"warning: {manifest_path(args.forecast)} not found; the run is identified by the hash of the csv")
        show(run.scores, run.alerts)

    written = save_tracking(runs, args.csv, args.db)
    print(f"\nsaved {written} rows to {args.csv} and to {args.db}")
    if sum(run.fired for run in runs):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
