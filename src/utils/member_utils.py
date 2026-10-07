"""Forecasts of every member model of the ensembles, for each test month."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.econometric_utils import ECONOMETRIC_MODELS, econometric_forecasts
from src.utils.tree_utils import TREE_BASE, fit_predict_trees

ML_MODELS = ["lgbm", "xgb", "rf"]
# Simple statistical models already stored in the frame (p_<name> columns) plus the three fitted per series.
SIMPLE_MODELS = ["naive", "mean3", "mean6", "mean12", "perday6", "seasonal_naive", "mean6_x_planta"]
ECON_MODELS = SIMPLE_MODELS + ECONOMETRIC_MODELS
ENSEMBLE_GROUPS = {"ml": ML_MODELS, "econ": ECON_MODELS, "ml_econ": ML_MODELS + ECON_MODELS}


def ml_forecast(
    kind: str, train: pd.DataFrame, rows: pd.DataFrame, tuned: dict, features: list[str] | None = None
) -> np.ndarray:
    r"""One machine-learning member, with the Optuna-tuned parameters when available (the baseline otherwise)."""
    params = tuned.get(kind, {}).get("best_params", TREE_BASE[kind])
    return fit_predict_trees(kind, params, train, rows, seed=0, features=features)


def collect_member_forecasts(
    S, frame: pd.DataFrame, test_months: list[int], horizon: int, tuned: dict, n_jobs: int
) -> pd.DataFrame:
    r"""One row per (series, test month) with the target `y` and a column p_<member> for each member."""
    parts = []
    for t in test_months:
        origin = t - horizon
        train = frame[frame["t"] <= origin]
        rows = frame[frame["o"] == origin]
        block = rows[["series", "o", "t", "planta_code", "y"]].copy()
        for kind in ML_MODELS:
            block[f"p_{kind}"] = ml_forecast(kind, train, rows, tuned)
        for name in SIMPLE_MODELS:
            block[f"p_{name}"] = rows[f"p_{name}"]
        econ = econometric_forecasts(S, rows, origin, horizon, n_jobs)
        for name in ECONOMETRIC_MODELS:
            block[f"p_{name}"] = econ[name]
        parts.append(block)
        print(f"    month {S.labels[t]} done", flush=True)
    return pd.concat(parts, ignore_index=True)
