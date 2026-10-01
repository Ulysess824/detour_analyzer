"""Self-contained HTML viewer of the base vs Optuna comparison."""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "templates" / "tuning_viewer.html"

MODELS = [("lgbm", "LightGBM"), ("xgb", "XGBoost"), ("rf", "Random Forest")]

# Library defaults for parameters the baseline leaves unset, so the tables can show them.
DEFAULTS = {
    "lgbm": {"reg_alpha": 0, "reg_lambda": 0},
    "xgb": {"gamma": 0, "reg_alpha": 0, "reg_lambda": 1},
    "rf": {},
}


def build_data(results: dict, test_start: str, tune_window: str, test_window: str) -> dict:
    r"""The json embedded in the page: one entry per model plus the page metadata."""
    models = []
    for key, label in MODELS:
        r = results[key]
        models.append(
            {
                "key": key,
                "label": label,
                "n_trials": r["n_trials"],
                "best_trial": r["best_trial"],
                "minutes": r["minutes"],
                "val_base": r["val_base"],
                "val_best": r["val_best"],
                "val_base_folds": r["val_base_folds"],
                "val_best_folds": r["val_best_folds"],
                "fold_rows": r["fold_train_test_rows"],
                "params_base": dict(r["base_params"]),
                "params_opt": r["best_params"],
                "defaults": DEFAULTS[key],
                "importance": r["importance"],
                "nseeds": len(r["h1"]["base"]["sku_acc"]),
                "h": {
                    "1": {"base": r["h1"]["base"], "tuned": r["h1"]["tuned"], "naive_sku_acc": r["h1"]["naive_sku_acc"]},
                    "3": {"base": r["h3"]["base"], "tuned": r["h3"]["tuned"], "naive_sku_acc": r["h3"]["naive_sku_acc"]},
                },
            }
        )
    trials = "/".join(str(m["n_trials"]) for m in models)
    return {
        "meta": {
            "test_start": test_start,
            "tune_window": tune_window,
            "test_window": test_window,
            "folds": ["2024-10 a 2024-12", "2025-01 a 2025-03", "2025-04 a 2025-06"],
            "subtitle": (
                f"Ajuste bayesiano (Optuna, TPE) de LightGBM, XGBoost y Random Forest sobre el pronóstico mensual "
                f"por SKU, con {trials} ensayos. Se compara contra los hiperparámetros que se usaban antes, en los "
                f"12 meses de test."
            ),
        },
        "models": models,
    }


def render_viewer(data: dict) -> str:
    r"""Insert the data into the html template."""
    return TEMPLATE_PATH.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data))
