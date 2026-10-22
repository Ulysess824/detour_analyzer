"""
Tracking runs: score a saved forecast (or the backtest) against the real and store the result.

The logic of scripts/track_forecast.py, so that the desktop app and the command line do the same thing. Scoring and alerts are in
tracking_utils; this module reads the forecast and its manifest, checks that the month has closed and writes the history (csv in git and database).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.engine.db_utils import connect, create_schema, replace_tracking, upsert_run
from src.engine.fingerprints import file_sha256, panel_fingerprint
from src.engine.panel_utils import Panel
from src.engine.run_manifest import git_info, manifest_path
from src.evaluation.cache_utils import check_alignment, check_meta, meta_path
from src.evaluation.tracking_utils import (
    backtest_forecasts, check_forecast_matches, check_month_closed, find_alerts, horizon_backtest_forecasts, month_scores, thresholds_for, tracking_rows,
    write_tracking_csv,
)

BACKTEST_RUN = "backtest-h1"
BACKTEST_PREFIX = "backtest-"
HORIZON_BACKTESTS = [("directa", "p_direct", True), ("parcial", "p_partial", False), ("iterada", "p_iterated", False)]  # strategy, cache column, keeps the members
FORECAST_FILE = re.compile(r"^forecast_\d{4}-\d{2}(?:_([a-z_]+))?$")


def is_backtest(run_id: str) -> bool:
    r"""Whether a run of the history is a historical test (`backtest-<strategy>`) and not a forecast that was really made."""
    return str(run_id).startswith(BACKTEST_PREFIX)


def strategy_of(forecast_path: str | Path, manifest: dict | None) -> tuple[str, int | None]:
    r"""The strategy and the cutoff day of a saved forecast: from its manifest, else from the file name, else the one-month forecast."""
    record = (manifest or {}).get("strategy")
    if record and record.get("used"):
        return record["used"], record.get("cutoff_day")
    match = FORECAST_FILE.match(Path(forecast_path).stem)
    return (match.group(1) if match and match.group(1) else "h1"), None


@dataclass
class TrackingRun:
    run_id: str
    month: str
    scores: pd.DataFrame
    alerts: pd.DataFrame
    rows: pd.DataFrame  # the scores as stored
    manifest: dict  # what the `corridas` table keeps of the run
    manifest_found: bool = True
    strategy: str = "h1"
    cutoff: int | None = None

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
    strategy, cutoff = strategy_of(forecast_path, manifest)
    scores = month_scores(S, forecast, month)
    alerts = find_alerts(scores, thresholds or thresholds_for(strategy))
    kept = manifest or {"run": {"run_id": run_id, "forecast_month": month}, "output": {"csv_sha256": file_sha256(forecast_path)}, "strategy": {"used": strategy, "cutoff_day": cutoff}}
    return TrackingRun(run_id, month, scores, alerts, tracking_rows(run_id, scores, alerts, strategy=strategy, cutoff=cutoff), kept, manifest is not None, strategy, cutoff)


def track_backtest(S: Panel, reference_path: str | Path = "results/member_forecasts.csv", thresholds: dict | None = None,
                   repo_dir: str | Path = ".") -> list[TrackingRun]:
    r"""Score the 12 test months of the one-month-ahead backtest cache (run `backtest-h1`); the cache is checked against the data first."""
    reference_path = Path(reference_path)
    cache = pd.read_csv(reference_path)
    check_meta(reference_path, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1}, keys=("data", "n_series", "horizon"))
    check_alignment(cache, S)
    manifest = {"run": {"run_id": BACKTEST_RUN, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "forecast_month": "backtest"},
                "code": git_info(repo_dir), "data": {"panel_fingerprint": panel_fingerprint(S)}, "strategy": {"used": "h1", "cutoff_day": None}}
    runs = []
    for month, forecast in backtest_forecasts(S, cache).items():
        scores = month_scores(S, forecast, month)
        alerts = find_alerts(scores, thresholds or thresholds_for("h1"))
        runs.append(TrackingRun(BACKTEST_RUN, month, scores, alerts, tracking_rows(BACKTEST_RUN, scores, alerts, strategy="h1"), manifest))
    return runs


def track_horizon_backtests(S: Panel, reference_path: str | Path = "results/horizon_strategies.csv", thresholds: dict | None = None,
                            repo_dir: str | Path = ".") -> list[TrackingRun]:
    r"""
    Score the 12 test months of the two-month backtest cache for `directa`, `parcial` (with the cutoff day the cache was made with) and `iterada`
    (runs `backtest-directa`, `backtest-parcial`, `backtest-iterada`); the cache is checked against the data first. `parcial` and `iterada` keep only
    `ml_mean` and the naive in the cache, so only those two are scored for them.
    """
    reference_path = Path(reference_path)
    cache = pd.read_csv(reference_path)
    check_meta(reference_path, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 2}, keys=("data", "n_series", "horizon"))
    check_alignment(cache, S)
    cutoff = json.loads(meta_path(reference_path).read_text(encoding="utf-8")).get("cutoff_day")
    code, runs = git_info(repo_dir), []
    for strategy, column, members in HORIZON_BACKTESTS:
        run_id, used_cutoff = f"{BACKTEST_PREFIX}{strategy}", (cutoff if strategy == "parcial" else None)
        manifest = {"run": {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "forecast_month": "backtest"},
                    "code": code, "data": {"panel_fingerprint": panel_fingerprint(S)}, "strategy": {"used": strategy, "cutoff_day": used_cutoff}}
        models = ["ml_mean", "naive"] + (["lgbm", "xgb", "rf"] if members else [])
        for month, forecast in horizon_backtest_forecasts(S, cache, column, members).items():
            scores = month_scores(S, forecast, month, models)
            alerts = find_alerts(scores, thresholds or thresholds_for(strategy))
            runs.append(TrackingRun(run_id, month, scores, alerts, tracking_rows(run_id, scores, alerts, strategy=strategy, cutoff=used_cutoff), manifest, True, strategy, used_cutoff))
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
