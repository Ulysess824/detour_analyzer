"""
Bayesian tuning (Optuna, TPE sampler) of the daily-then-aggregate DNN and LSTM, compared against
the configuration used before tuning. Same protocol as tune_trees.py:

    - The last --n-test months are the test set; everything before is the tuning window.
    - Three walk-forward folds inside the tuning window; a trial's score is the mean SKU-month
      WAPE of the monthly sums of the daily forecasts (one seed per trial).
    - The baseline (the paper's architecture with the Tweedie loss) is enqueued as trial 0.
    - After the search the best parameters are frozen and every test month is forecast with a
      rolling refit, for the baseline and the tuned configuration, over several seeds. The mean
      over seeds is reported (not the best seed) together with the seed range.
    - Tuning is done at h=1; h=3 is only checked with the h=1 parameters.

Also stores the seed-mean forecasts of the monthly total and of one series (--planta, --sku).

Usage:
    python scripts/tune_neural.py data/consumos_long.csv --models dnn --trials 60 --out results/tuning_dnn.json
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
from src.utils.neural_utils import NEURAL_BASE, fit_predict_neural
from src.utils.panel_utils import GROUP_COLS, build_panel, load_consumption
from src.utils.search_space_utils import NEURAL_BASELINE_TRIAL, suggest_neural_params
from src.utils.tuning_utils import (
    compare_base_and_tuned, print_comparison, run_study, seed_mean_traces, study_summary, to_jsonable,
)  # fmt: skip


def neural_predictor(S, kind: str, params: dict):
    r"""A predict(train, rows, seed) function for one network and one set of parameters."""
    return lambda train, rows, seed: fit_predict_neural(S, kind, params, train, rows, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--models", nargs="+", default=["dnn", "lstm"], choices=["dnn", "lstm"])
    parser.add_argument("--trials", nargs="+", type=int, default=[60, 40], help="one value per model, in order")
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=42, help="seed of the TPE sampler")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--planta", default="SCAN")
    parser.add_argument("--sku", default="TSL/01/80gsm/2450mm/1200-1450")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    import torch

    torch.set_num_threads(args.threads)

    df = load_consumption(args.input)
    S = build_panel(df)
    series = df.groupby(GROUP_COLS).size().index.get_loc((args.planta, args.sku))
    n_months = S.monthly.shape[1]
    test_months = list(range(n_months - args.n_test, n_months))
    tune_end = n_months - args.n_test - 1
    print(f"tuning window: {S.labels[0]} .. {S.labels[tune_end]} | test: {S.labels[test_months[0]]} .. {S.labels[-1]}", flush=True)
    frames = {h: make_frame(S, h) for h in (1, 3)}
    trials = dict(zip(args.models, args.trials))

    results = {}
    for kind in args.models:
        print(f"\n=== {kind}: {trials[kind]} trials ===", flush=True)
        start = time.time()
        study, fold_sizes = run_study(
            kind, lambda params, k=kind: neural_predictor(S, k, params), lambda trial, k=kind: suggest_neural_params(k, trial),
            NEURAL_BASELINE_TRIAL[kind], frames[1], 1, tune_end, trials[kind], args.seed, args.storage,
        )  # fmt: skip
        minutes = (time.time() - start) / 60
        results[kind] = study_summary(study, fold_sizes, minutes, NEURAL_BASE[kind])
        results[kind].update(seeds=args.seeds, series=f"{args.planta} / {args.sku}", months=[S.labels[t] for t in test_months])
        print(f"    validation WAPE: baseline {study.trials[0].value:.4f} -> best {study.best_value:.4f} ({minutes:.1f} min)", flush=True)

        comparison = compare_base_and_tuned(
            neural_predictor(S, kind, NEURAL_BASE[kind]), neural_predictor(S, kind, study.best_trial.params),
            S, frames, test_months, args.seeds,
        )  # fmt: skip
        for horizon_key, res in comparison.items():
            res["real_total"] = [float(S.total_monthly[t]) for t in test_months]
            res["real_series"] = [float(S.monthly[series, t]) for t in test_months]
            for name, runs in res["_runs"].items():
                total, one = seed_mean_traces(runs, test_months, series)
                res[f"{name}_total"], res[f"{name}_series"] = total, one
                res[f"{name}_pred_std"] = float(sum(r["p"].std() for r in runs) / len(runs))
        results[kind].update(comparison)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(to_jsonable(results), indent=1))  # saved after each model

    print_comparison(results)


if __name__ == "__main__":
    main()
