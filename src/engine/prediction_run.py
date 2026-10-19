"""
One production forecast run: load and check the data, forecast the next month, save the csv and its manifest.

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
from src.engine.inference_utils import check_last_month_complete, forecast_next_month
from src.engine.panel_utils import Panel, build_panel, load_consumption
from src.engine.run_manifest import build_manifest, git_info, manifest_path, write_manifest

Progress = Callable[[str, float], None]


@dataclass
class PredictionRun:
    result: pd.DataFrame  # planta, sku, mes, lgbm, xgb, rf, ml_mean, naive
    manifest: dict
    forecast_path: Path
    manifest_file: Path
    validation: Report
    panel: Panel  # built from the data used (after `as_of`)
    full: pd.DataFrame  # the whole table that was read (before `as_of`)

    @property
    def month(self) -> str:
        return str(self.result["mes"].iloc[0])


def run_prediction(
    input_path: str | Path,
    tuning_path: str | Path | None = Path("results/tuning_trees.json"),
    out_path: str | Path | None = None,
    as_of: str | None = None,
    repo_dir: str | Path = ".",
    progress: Progress | None = None,
) -> PredictionRun:
    r"""
    Forecast the month after the last one in the data and save `forecast_<month>.csv` and its manifest.

    Stops (SystemExit with a message) when the input checks find errors or the last month of the data is incomplete; `as_of` (YYYY-MM) ignores the
    data after that month. `tuning_path` is the json with the tuned parameters; without it the baseline parameters are used.
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
    else:
        check_last_month_complete(full)
    S = build_panel(df)
    tuning = Path(tuning_path) if tuning_path else None
    tuned = json.loads(tuning.read_text()) if tuning is not None and tuning.exists() else {}

    report("Entrenando los modelos y pronosticando", 0.25)
    result = forecast_next_month(S, tuned)

    report("Guardando el pronóstico", 0.9)
    out = Path(out_path) if out_path else Path(f"results/forecast_{result['mes'].iloc[0]}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.round(3).to_csv(out, index=False)
    manifest = build_manifest(
        S=S, df=df, input_path=input_path, tuned=tuned, tuning_path=tuning, result=result, forecast_path=out, as_of=as_of,
        started_at=started, seconds=time.perf_counter() - clock, git=git, validation=validation, rows_in_file=len(full),
    )
    write_manifest(manifest_path(out), manifest)
    report("Listo", 1.0)
    return PredictionRun(result, manifest, out, manifest_path(out), validation, S, full)
