"""
One production forecast run: load and check the data, forecast with the strategy that fits the state of the data, save the csv and its manifest.

When the last month of the data is complete the run forecasts the next month (strategy `h1`). When it is in progress (the planner's
mid-month deadline) the strategy is chosen from the day of its last record (src/engine/month_state.py) and the run forecasts the month after
the one in progress.

The logic of scripts/predict_next_month.py, so that the desktop app and the command line do exactly the same thing. `progress(message, fraction)`
is called at each step (fraction from 0 to 1) so a screen can show a progress bar.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.engine.data_checks import Report, validate_consumption
from src.engine.month_state import MonthState, StrategyPlan, detect_month_state, resolve_strategy
from src.engine.panel_utils import Panel, build_panel, load_consumption
from src.engine.run_manifest import build_manifest, git_info, manifest_path, write_manifest
from src.engine.strategies import forecast_with_strategy

Progress = Callable[[str, float], None]
FORECAST_COLUMNS = ["planta", "sku", "mes", "lgbm", "xgb", "rf", "ml_mean", "naive"]  # the csv keeps these for every strategy


@dataclass
class PredictionRun:
    result: pd.DataFrame  # planta, sku, mes, lgbm, xgb, rf, ml_mean, naive
    manifest: dict
    forecast_path: Path
    manifest_file: Path
    validation: Report
    panel: Panel  # built from the data used (after `as_of`)
    full: pd.DataFrame  # the whole table that was read (before `as_of`)
    plan: StrategyPlan | None = None  # the strategy that ran, its cutoff day and the notes to show
    state: MonthState | None = None  # complete or in progress, read from the last record

    @property
    def month(self) -> str:
        return str(self.result["mes"].iloc[0])


def strategy_record(requested: str, plan: StrategyPlan, state: MonthState) -> dict:
    r"""What the manifest keeps about the strategy: the choice, the cutoff, the months and the state of the data it read."""
    return {
        "requested": requested,
        "used": plan.strategy,
        "cutoff_day": plan.cutoff_day,
        "origin_month": plan.origin,
        "target_month": plan.target,
        "estimated_months": [state.month] if plan.strategy == "nowcast_media" else [],
        "notes": plan.notes,
        "month_state": {
            "status": state.status,
            "month": state.month,
            "last_date": str(state.last_date.date()),
            "days_seen": state.days_seen,
            "days_left": state.days_left,
        },
    }


def run_prediction(
    input_path: str | Path,
    tuning_path: str | Path | None = Path("results/tuning_trees.json"),
    out_path: str | Path | None = None,
    as_of: str | None = None,
    repo_dir: str | Path = ".",
    progress: Progress | None = None,
    strategy: str = "automatica",
    cutoff_day: int | None = None,
) -> PredictionRun:
    r"""
    Forecast with `strategy` (see `resolve_strategy`) and save `forecast_<month>.csv` and its manifest.

    Last month complete: the month after it. Last month in progress: the month after the one in progress, using its first `cutoff_day` days
    (default: up to the last record). Stops (SystemExit with a message) when the input checks find errors or the settings do not fit the data;
    `as_of` (YYYY-MM) ignores the data after that month. `tuning_path` is the json with the tuned parameters; without it the baseline parameters
    are used. A forecast that is not `h1` is saved as `forecast_<month>_<strategy>.csv` so it does not replace the one-month forecast of that month.
    """
    report = progress or (lambda message, fraction: None)
    started, clock = datetime.now(timezone.utc), time.perf_counter()
    git = git_info(repo_dir)  # before the run writes anything, so its own outputs do not count as changes

    report("Leyendo y revisando los datos", 0.05)
    full = load_consumption(input_path, validate=False)
    validation = validate_consumption(full)  # stops on errors, prints the warnings
    df = full
    if as_of:
        df = full[full["fecha"] <= pd.Period(as_of, freq="M").end_time]
    state = detect_month_state(df)
    try:
        plan = resolve_strategy(state, strategy, cutoff_day)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    S = build_panel(df)
    tuning = Path(tuning_path) if tuning_path else None
    tuned = json.loads(tuning.read_text()) if tuning is not None and tuning.exists() else {}

    report("Entrenando los modelos y pronosticando", 0.25)
    result = forecast_with_strategy(S, tuned, plan.strategy, S.labels.index(plan.origin), plan.cutoff_day)[FORECAST_COLUMNS]

    report("Guardando el pronóstico", 0.9)
    suffix = "" if plan.strategy == "h1" else f"_{plan.strategy}"
    out = Path(out_path) if out_path else Path(f"results/forecast_{result['mes'].iloc[0]}{suffix}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.round(3).to_csv(out, index=False)
    manifest = build_manifest(
        S=S, df=df, input_path=input_path, tuned=tuned, tuning_path=tuning, result=result, forecast_path=out, as_of=as_of,
        started_at=started, seconds=time.perf_counter() - clock, git=git, validation=validation, rows_in_file=len(full),
        strategy=strategy_record(strategy, plan, state),
    )
    write_manifest(manifest_path(out), manifest)
    report("Listo", 1.0)
    return PredictionRun(result, manifest, out, manifest_path(out), validation, S, full, plan, state)
