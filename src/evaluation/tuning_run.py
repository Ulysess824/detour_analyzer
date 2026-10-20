"""
One tuning run: Optuna search of each tree model on the tuning window, then the rolling test of the baseline against the tuned parameters.

The logic of scripts/tune_trees.py, so that the desktop app and the command line do the same thing. Protocol (chronological; the test months are
never used to choose anything): the last `n_test` months are the test, everything before is the tuning window with three walk-forward folds, the
baseline parameters are the first trial, and the search is at h=1 (h=3 is only checked with those parameters).
"""

from __future__ import annotations

import time
from typing import Callable

from src.engine.feature_utils import make_frame
from src.engine.panel_utils import Panel
from src.engine.tree_utils import TREE_BASE, fit_predict_trees
from src.evaluation.search_space_utils import TREE_BASELINE_TRIAL, suggest_tree_params
from src.evaluation.tuning_utils import compare_base_and_tuned, run_study, study_summary, tuning_meta

Progress = Callable[[str, float], None]
MODELS = ["lgbm", "xgb", "rf"]
NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest"}


def tree_predictor(kind: str, params: dict):
    r"""A predict(train, rows, seed) function for one tree model and one set of parameters."""
    return lambda train, rows, seed: fit_predict_trees(kind, params, train, rows, seed)


def run_tuning(
    S: Panel,
    models: list[str] = MODELS,
    trials: dict[str, int] | None = None,
    n_test: int = 12,
    seeds: list[int] = (0, 1, 2),
    rf_seeds: list[int] = (0, 1),
    sampler_seed: int = 42,
    storage: str | None = None,
    progress: Progress | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    r"""
    Tune each model in `models` and return the results (the json written by tune_trees.py): the best and base parameters, the validation scores, the
    importance of each parameter, the test results of the baseline and of the tuned configuration at h=1 and h=3 and the windows used (`meta`).
    """
    report = progress or (lambda message, fraction: None)
    trials = trials or {"lgbm": 100, "xgb": 100, "rf": 40}
    n_months = S.monthly.shape[1]
    test_months = list(range(n_months - n_test, n_months))
    tune_end = n_months - n_test - 1
    log(f"tuning window: {S.labels[0]} .. {S.labels[tune_end]} | test: {S.labels[test_months[0]]} .. {S.labels[-1]}")
    frames = {h: make_frame(S, h) for h in (1, 3)}
    meta = tuning_meta(S, n_test)

    results = {}
    for i, kind in enumerate(models):
        report(f"Ajustando {NAMES[kind]}", i / len(models))
        log(f"\n=== {kind}: {trials[kind]} trials ===")
        start = time.time()
        study, fold_sizes = run_study(
            kind, lambda params, k=kind: tree_predictor(k, params), lambda trial, k=kind: suggest_tree_params(k, trial),
            TREE_BASELINE_TRIAL[kind], frames[1], 1, tune_end, trials[kind], sampler_seed, storage, print_every=10,
            fingerprint=f"{meta['data']}:{n_test}",
            on_trial=lambda done, total, i=i, k=kind: report(f"Ajustando {NAMES[k]}: prueba {done} de {total}", (i + 0.7 * done / total) / len(models)),
        )  # fmt: skip
        minutes = (time.time() - start) / 60
        results[kind] = study_summary(study, fold_sizes, minutes, TREE_BASE[kind])
        results[kind]["meta"] = meta
        log(f"    validation WAPE: baseline {study.trials[0].value:.4f} -> best {study.best_value:.4f} ({minutes:.1f} min)")

        report(f"Comprobando {NAMES[kind]} en los meses de prueba", (i + 0.7) / len(models))
        results[kind].update(
            compare_base_and_tuned(
                tree_predictor(kind, TREE_BASE[kind]), tree_predictor(kind, study.best_trial.params), S, frames, test_months,
                list(rf_seeds if kind == "rf" else seeds),
            )
        )
    report("Listo", 1.0)
    return results
