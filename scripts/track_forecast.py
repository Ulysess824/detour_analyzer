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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.db_utils import connect, create_schema, replace_tracking, upsert_run
from src.engine.fingerprints import file_sha256, panel_fingerprint
from src.engine.panel_utils import build_panel, load_consumption
from src.engine.run_manifest import git_info, manifest_path
from src.evaluation.cache_utils import check_alignment, check_meta
from src.evaluation.tracking_utils import THRESHOLDS, backtest_forecasts, check_month_closed, find_alerts, month_scores, tracking_rows, write_tracking_csv

BACKTEST_RUN = "backtest-h1"


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
        cache = pd.read_csv(args.reference)
        check_meta(args.reference, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1}, keys=("data", "n_series", "horizon"))
        check_alignment(cache, S)
        jobs = {BACKTEST_RUN: backtest_forecasts(S, cache)}
        manifests = {BACKTEST_RUN: {"run": {"run_id": BACKTEST_RUN, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "forecast_month": "backtest"},
                                    "code": git_info(Path(__file__).resolve().parents[1]), "data": {"panel_fingerprint": panel_fingerprint(S)}}}
    else:
        forecast = pd.read_csv(args.forecast)
        month = str(forecast["mes"].iloc[0])
        check_month_closed(df, month)
        path = manifest_path(args.forecast)
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        run_id = manifest["run"]["run_id"] if manifest else f"sin-manifiesto-{file_sha256(args.forecast)[:8]}"
        jobs = {run_id: {month: forecast}}
        manifests = {run_id: manifest or {"run": {"run_id": run_id, "forecast_month": month}, "output": {"csv_sha256": file_sha256(args.forecast)}}}
        if manifest is None:
            print(f"warning: {path} not found; the run is identified by the hash of the csv")

    fired, all_rows = 0, []
    for run_id, months in jobs.items():
        for month, forecast in months.items():
            scores = month_scores(S, forecast, month)
            alerts = find_alerts(scores, thresholds)
            fired += int((alerts["tipo"] == "alerta").sum())
            all_rows.append(tracking_rows(run_id, scores, alerts))
            if not args.backtest:
                show(scores, alerts)
    rows = pd.concat(all_rows, ignore_index=True)
    if args.backtest:
        ml = rows[(rows["modelo"] == "ml_mean") & (rows["planta"] == "") & rows["nivel"].isin(["sku", "planta", "total"])]
        print((ml.pivot(index="mes", columns="nivel", values="acierto")[["sku", "planta", "total"]] * 100).round(1).to_string())
        print("mean:", (ml.groupby("nivel")["acierto"].mean() * 100).round(1).to_dict(), "| alerts fired:", fired)

    write_tracking_csv(args.csv, rows)
    con = connect(args.db)
    create_schema(con)
    for manifest in manifests.values():
        upsert_run(con, manifest)
    replace_tracking(con, rows)
    con.close()
    print(f"\nsaved {len(rows)} rows to {args.csv} and to {args.db}")
    if fired:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
