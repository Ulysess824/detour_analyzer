"""
Bayesian tuning (Optuna, TPE sampler) of LightGBM, XGBoost and Random Forest on the monthly
forecasting frame, compared against the hyperparameters used before tuning.

Protocol (chronological; the test months are never used to choose anything):
    - The last --n-test months are the test set; everything before is the tuning window.
    - Inside the tuning window there are three walk-forward folds. Each validates on the next
      3 months and trains only on rows whose target is already known at its first origin. A
      trial's score is the mean SKU-month WAPE over the folds.
    - The baseline parameters are enqueued as the first trial.
    - After the search the best parameters are frozen and every test month is forecast with a
      rolling refit, for the baseline and the tuned configuration, over several seeds.
    - Tuning is done at h=1; h=3 is only checked with the h=1 parameters.

Usage:
    python scripts/tune_trees.py data/consumos_long.csv --trials 100 100 40 --out results/tuning_trees.json
"""

import argparse
import json
import time
import warnings

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.feature_utils import make_frame
from src.utils.panel_utils import build_panel, load_consumption
from src.utils.search_space_utils import TREE_BASELINE_TRIAL, suggest_tree_params
from src.utils.tree_utils import TREE_BASE, fit_predict_trees
from src.utils.tuning_utils import compare_base_and_tuned, print_comparison, run_study, study_summary, to_jsonable


def tree_predictor(kind: str, params: dict):
    r"""A predict(train, rows, seed) function for one tree model and one set of parameters."""
    return lambda train, rows, seed: fit_predict_trees(kind, params, train, rows, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--models", nargs="+", default=["lgbm", "xgb", "rf"], choices=["lgbm", "xgb", "rf"])
    parser.add_argument("--trials", nargs="+", type=int, default=[100, 100, 40], help="one value per model, in order")
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rf-seeds", nargs="+", type=int, default=[0, 1], help="fewer seeds: a forest fit is slow")
    parser.add_argument("--seed", type=int, default=42, help="seed of the TPE sampler")
    parser.add_argument("--storage", type=str, default=None, help="e.g. sqlite:///optuna.db, to resume a search")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    S = build_panel(load_consumption(args.input))
    n_months = S.monthly.shape[1]
    test_months = list(range(n_months - args.n_test, n_months))
    tune_end = n_months - args.n_test - 1
    print(f"tuning window: {S.labels[0]} .. {S.labels[tune_end]} | test: {S.labels[test_months[0]]} .. {S.labels[-1]}")
    frames = {h: make_frame(S, h) for h in (1, 3)}
    trials = dict(zip(args.models, args.trials))

    results = {}
    for kind in args.models:
        print(f"\n=== {kind}: {trials[kind]} trials ===", flush=True)
        start = time.time()
        study, fold_sizes = run_study(
            kind, lambda params, k=kind: tree_predictor(k, params), lambda trial, k=kind: suggest_tree_params(k, trial),
            TREE_BASELINE_TRIAL[kind], frames[1], 1, tune_end, trials[kind], args.seed, args.storage, print_every=10,
        )  # fmt: skip
        minutes = (time.time() - start) / 60
        results[kind] = study_summary(study, fold_sizes, minutes, TREE_BASE[kind])
        print(f"    validation WAPE: baseline {study.trials[0].value:.4f} -> best {study.best_value:.4f} ({minutes:.1f} min)", flush=True)

        seeds = args.rf_seeds if kind == "rf" else args.seeds
        results[kind].update(
            compare_base_and_tuned(
                tree_predictor(kind, TREE_BASE[kind]), tree_predictor(kind, study.best_trial.params), S, frames, test_months, seeds
            )
        )

    print_comparison(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(to_jsonable(results), indent=1))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
