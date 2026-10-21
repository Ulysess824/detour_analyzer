"""Forecasts of the ensemble members (LightGBM, XGBoost, Random Forest) and of the naive reference, for each test month."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.tree_utils import TREE_BASE, fit_predict_trees

ML_MODELS = ["lgbm", "xgb", "rf"]
SEED = 0  # seed of every tree model in the forecasts (recorded in the run manifest)
# The only reference outside the ensemble: last month's value, already stored in the frame (p_naive column).
SIMPLE_MODELS = ["naive"]
ENSEMBLE_GROUPS = {"ml": ML_MODELS}


def ml_forecast(
    kind: str, train: pd.DataFrame, rows: pd.DataFrame, tuned: dict, features: list[str] | None = None
) -> np.ndarray:
    r"""One machine-learning member, with the Optuna-tuned parameters when available (the baseline otherwise)."""
    params = tuned.get(kind, {}).get("best_params", TREE_BASE[kind])
    return fit_predict_trees(kind, params, train, rows, seed=SEED, features=features)


def ml_forecasts(train: pd.DataFrame, rows: pd.DataFrame, tuned: dict, features: list[str] | None = None) -> dict:
    r"""Forecast of each machine-learning model plus their mean (`ml_mean`) for `rows`."""
    out = {kind: ml_forecast(kind, train, rows, tuned, features) for kind in ML_MODELS}
    out["ml_mean"] = np.mean([out[kind] for kind in ML_MODELS], axis=0)
    return out


def collect_member_forecasts(S, frame: pd.DataFrame, test_months: list[int], horizon: int, tuned: dict) -> pd.DataFrame:
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
        parts.append(block)
        print(f"    month {S.labels[t]} done", flush=True)
    return pd.concat(parts, ignore_index=True)
