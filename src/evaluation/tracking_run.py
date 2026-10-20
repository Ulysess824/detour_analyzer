"""
Tracking runs: score a saved forecast (or the backtest) against the real and store the result.

The logic of scripts/track_forecast.py, so that the desktop app and the command line do the same thing. Scoring and alerts are in
tracking_utils; this module reads the forecast and its manifest, checks that the month has closed and writes the history (csv in git and database).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.engine.db_utils import connect, create_schema, replace_tracking, upsert_run
from src.engine.fingerprints import file_sha256, panel_fingerprint
from src.engine.panel_utils import Panel
from src.engine.run_manifest import git_info, manifest_path
from src.evaluation.cache_utils import check_alignment, check_meta
from src.evaluation.tracking_utils import (
    backtest_forecasts, check_forecast_matches, check_month_closed, find_alerts, month_scores, tracking_rows, write_tracking_csv,
)

BACKTEST_RUN = "backtest-h1"


@dataclass
class TrackingRun:
    run_id: str
    month: str
    scores: pd.DataFrame
    alerts: pd.DataFrame
    rows: pd.DataFrame  # the scores as stored
    manifest: dict  # what the `corridas` table keeps of the run
    manifest_found: bool = True

    @property
    def fired(self) -> int:
        r"""Number of alerts (the informational plantas to review do not count)."""
        return int((self.alerts["tipo"] == "alerta").sum())


def track_saved_forecast(forecast_path: str | Path, df: pd.DataFrame, S: Panel, thresholds: dict | None = None) -> TrackingRun:
    r"""
    Score a saved forecast against the real of its month; stops (SystemExit) unless the data covers the month completely.

    `df` is the consumption table and `S` the panel built from it. The run is identified by the manifest next to the forecast, or by the hash of the
    csv when there is none (`manifest_found` is False).
    """
    forecast_path = Path(forecast_path)
    forecast = pd.read_csv(forecast_path)
    month = str(forecast["mes"].iloc[0])
    check_month_closed(df, month)
    check_forecast_matches(S, forecast, month)
    path = manifest_path(forecast_path)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    run_id = manifest["run"]["run_id"] if manifest else f"sin-manifiesto-{file_sha256(forecast_path)[:8]}"
    scores = month_scores(S, forecast, month)
    alerts = find_alerts(scores, thresholds)
    kept = manifest or {"run": {"run_id": run_id, "forecast_month": month}, "output": {"csv_sha256": file_sha256(forecast_path)}}
    return TrackingRun(run_id, month, scores, alerts, tracking_rows(run_id, scores, alerts), kept, manifest is not None)


def track_backtest(S: Panel, reference_path: str | Path = "results/member_forecasts.csv", thresholds: dict | None = None,
                   repo_dir: str | Path = ".") -> list[TrackingRun]:
    r"""Score the 12 test months of the one-month-ahead backtest cache (run `backtest-h1`); the cache is checked against the data first."""
    reference_path = Path(reference_path)
    cache = pd.read_csv(reference_path)
    check_meta(reference_path, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1}, keys=("data", "n_series", "horizon"))
    check_alignment(cache, S)
    manifest = {"run": {"run_id": BACKTEST_RUN, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "forecast_month": "backtest"},
                "code": git_info(repo_dir), "data": {"panel_fingerprint": panel_fingerprint(S)}}
    runs = []
    for month, forecast in backtest_forecasts(S, cache).items():
        scores = month_scores(S, forecast, month)
        alerts = find_alerts(scores, thresholds)
        runs.append(TrackingRun(BACKTEST_RUN, month, scores, alerts, tracking_rows(BACKTEST_RUN, scores, alerts), manifest))
    return runs


def save_tracking(runs: list[TrackingRun], csv_path: str | Path = "results/seguimiento.csv", db_path: str | Path | None = "data/detour.db") -> int:
    r"""Store the rows of the runs in the history csv and, when `db_path` is given, in the database (runs and tracking); returns the rows written."""
    rows = pd.concat([run.rows for run in runs], ignore_index=True)
    write_tracking_csv(csv_path, rows)
    if db_path is not None:
        con = connect(db_path)
        create_schema(con)
        for manifest in {run.run_id: run.manifest for run in runs}.values():
            upsert_run(con, manifest)
        replace_tracking(con, rows)
        con.close()
    return len(rows)
