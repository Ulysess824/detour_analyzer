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
import warnings

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.panel_utils import build_panel, load_consumption
from src.evaluation.tuning_run import run_tuning
from src.evaluation.tuning_utils import print_comparison, to_jsonable


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
    results = run_tuning(
        S, args.models, dict(zip(args.models, args.trials)), args.n_test, args.seeds, args.rf_seeds, args.seed, args.storage, log=lambda m: print(m, flush=True)
    )
    print_comparison(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(to_jsonable(results), indent=1))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
