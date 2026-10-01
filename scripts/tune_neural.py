"""
Bayesian tuning (Optuna, TPE sampler) of the daily-then-aggregate DNN and LSTM, compared against the
configuration used so far. Same protocol as tune_trees.py:

    - The last --n-test months are the test set; everything before is the tuning window.
    - Three walk-forward folds inside the tuning window; a trial's score is the mean SKU-month WAPE
      of the monthly sums of the daily forecasts.
    - The baseline (paper configuration, Tweedie loss) is enqueued as trial 0.
    - After the search, the best parameters are frozen and every test month is forecast with a
      rolling refit, for the baseline and the tuned configuration, over several seeds. The mean over
      seeds is reported (not the best seed), with the seed range.
    - Tuning is done at h=1 with one seed per trial; h=3 is only checked with the h=1 parameters.

Requires: torch, optuna.

Usage:
    python scripts/tune_neural.py data/consumos_long.csv --models dnn --trials 60 --out results/tuning_dnn.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_monthly as fm
import neural_daily as nd
import tune_trees as tt

FOLD_MONTHS, N_FOLDS = tt.FOLD_MONTHS, tt.N_FOLDS
BATCHES = [256, 512, 1024, 2048]
LOOK_BACKS = [6, 8, 12]

ENQUEUE = {
    "dnn": {**nd.BASE["dnn"], "weight_decay": 1e-6},
    "lstm": {**nd.BASE["lstm"], "weight_decay": 1e-6},
}


def suggest(kind: str, trial) -> dict:
    p = dict(
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", BATCHES),
        tweedie_p=trial.suggest_float("tweedie_p", 1.05, 1.9),
    )
    if kind == "dnn":
        p["n_layers"] = trial.suggest_int("n_layers", 1, 3)
        p["units"] = trial.suggest_int("units", 8, 128, log=True)
    else:
        p["hidden"] = trial.suggest_int("hidden", 8, 64, log=True)
        p["units"] = trial.suggest_int("units", 8, 64, log=True)
        p["look_back"] = trial.suggest_categorical("look_back", LOOK_BACKS)
    return p


def tune(S, kind, frame, h, tune_end, n_trials, seed, storage):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frame = frame[frame["t"] <= tune_end]
    starts = [tune_end - FOLD_MONTHS + 1 - FOLD_MONTHS * (N_FOLDS - 1 - k) for k in range(N_FOLDS)]
    folds = [(frame[frame["t"] <= a - h], frame[(frame["t"] >= a) & (frame["t"] < a + FOLD_MONTHS)]) for a in starts]

    def objective(trial) -> float:
        params = suggest(kind, trial)
        wapes = []
        for train, val in folds:
            pred = nd.fit_predict(S, kind, params, train, val, seed=0)
            wapes.append(float(np.abs(val["y"].to_numpy() - pred).sum() / val["y"].sum()))
        trial.set_user_attr("fold_wape", [round(w, 4) for w in wapes])
        return float(np.mean(wapes))

    study = optuna.create_study(
        study_name=f"{kind}_h{h}", direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10), storage=storage, load_if_exists=True,
    )
    finished = sum(t.state.is_finished() for t in study.trials)
    if len(study.trials) == 0:
        study.enqueue_trial(ENQUEUE[kind])

    def progress(st, tr) -> None:
        print(f"    {kind}: trial {tr.number + 1}/{n_trials}  value {tr.value:.4f}  best {st.best_value:.4f}", flush=True)

    study.optimize(objective, n_trials=max(n_trials - finished, 0), callbacks=[progress])
    return study, [(len(t), len(v)) for t, v in folds]


def test_run(S, kind, params, frame, h, test_months, seeds):
    out = []
    for seed in seeds:
        parts = []
        for t in test_months:
            origin = t - h
            train = frame[frame["t"] <= origin]
            rows = frame[frame["o"] == origin]
            block = rows[["series", "o", "t", "planta_code", "y", "p_naive"]].copy()
            block["p"] = nd.fit_predict(S, kind, params, train, rows, seed)
            parts.append(block)
        res = pd.concat(parts, ignore_index=True)
        res["p_rec"] = fm._reconcile(res, S, h, test_months, "p")
        out.append(res)
    return out


def trace(runs, S, test_months, series):
    r"""Seed-mean monthly forecasts of the total and of one series, for plotting."""
    tot = np.mean([[r.loc[r["t"] == t, "p"].sum() for t in test_months] for r in runs], axis=0)
    one = None
    if series is not None:
        one = np.mean([[r.loc[(r["t"] == t) & (r["series"] == series), "p"].sum() for t in test_months] for r in runs], axis=0)
    return tot, one


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("--models", nargs="+", default=["dnn", "lstm"], choices=["dnn", "lstm"])
    ap.add_argument("--trials", nargs="+", type=int, default=[60, 40])
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=42, help="seed of the TPE sampler")
    ap.add_argument("--storage", type=str, default=None)
    ap.add_argument("--planta", default="SCAN")
    ap.add_argument("--sku", default="TSL/01/80gsm/2450mm/1200-1450")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch

    torch.set_num_threads(a.threads)

    df = pd.read_csv(a.input, parse_dates=["fecha"])
    S = fm.build_panel(df)
    series = df.groupby(fm.GROUP_COLS).size().index.get_loc((a.planta, a.sku))
    n = S.Y.shape[1]
    test_months = list(range(n - a.n_test, n))
    tune_end = n - a.n_test - 1
    print(f"tuning window: {S.labels[0]} .. {S.labels[tune_end]} | test: {S.labels[test_months[0]]} .. {S.labels[-1]}", flush=True)
    frames = {h: fm.make_frame(S, h) for h in (1, 3)}
    trials = dict(zip(a.models, a.trials))

    results = {}
    for kind in a.models:
        print(f"\n=== {kind}: {trials[kind]} trials ===", flush=True)
        t0 = time.time()
        study, fold_sizes = tune(S, kind, frames[1], 1, tune_end, trials[kind], a.seed, a.storage)
        minutes = (time.time() - t0) / 60
        best, base_trial = study.best_trial, study.trials[0]
        try:
            import optuna

            imp = optuna.importance.get_param_importances(study)
        except Exception:
            imp = {}
        results[kind] = {
            "best_params": best.params, "base_params": nd.BASE[kind],
            "val_base": base_trial.value, "val_best": best.value,
            "val_best_folds": best.user_attrs["fold_wape"], "val_base_folds": base_trial.user_attrs["fold_wape"],
            "best_trial": best.number, "n_trials": len(study.trials), "fold_train_test_rows": fold_sizes,
            "minutes": round(minutes, 1), "importance": {k: round(v, 3) for k, v in imp.items()},
            "seeds": a.seeds, "series": f"{a.planta} / {a.sku}",
            "months": [str(S.labels[t])[:7] for t in test_months],
        }
        print(f"    validation WAPE: baseline {base_trial.value:.4f} -> best {best.value:.4f} (trial {best.number}, {minutes:.1f} min)", flush=True)
        for h in (1, 3):
            base_runs = test_run(S, kind, nd.BASE[kind], frames[h], h, test_months, a.seeds)
            tuned_runs = test_run(S, kind, best.params, frames[h], h, test_months, a.seeds)
            res = {"base": tt.summarise(base_runs, S, test_months), "tuned": tt.summarise(tuned_runs, S, test_months)}
            res["naive_sku_acc"] = fm.score(base_runs[0]["y"].to_numpy(), base_runs[0]["p_naive"].to_numpy())["accuracy"]
            res["real_total"] = [float(S.T[t]) for t in test_months]
            res["real_series"] = [float(S.Y[series, t]) for t in test_months]
            for name, runs in (("base", base_runs), ("tuned", tuned_runs)):
                tot, one = trace(runs, S, test_months, series)
                res[f"{name}_total"], res[f"{name}_series"] = tot.tolist(), one.tolist()
                res[f"{name}_pred_std"] = float(np.mean([r["p"].std() for r in runs]))
            results[kind][f"h{h}"] = res
            print(f"    test h={h} done", flush=True)

        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o

        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(clean(results), indent=1))

    pd.set_option("display.width", 200)
    for kind, r in results.items():
        print(f"\n################ {kind} ################")
        print(f"validation WAPE: baseline {r['val_base']:.4f} | best {r['val_best']:.4f} (trials {r['n_trials']})")
        print("params baseline:", r["base_params"])
        print("params optuna  :", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r["best_params"].items()})
        for h in (1, 3):
            b, t = r[f"h{h}"]["base"], r[f"h{h}"]["tuned"]
            rows = [{"metric": nm, "baseline": b[k].mean(), "optuna": t[k].mean(), "delta_pts": (t[k].mean() - b[k].mean()) * 100,
                     "base_range": f"{b[k].min():.4f}-{b[k].max():.4f}", "optuna_range": f"{t[k].min():.4f}-{t[k].max():.4f}"}
                    for nm, k in [("SKU accuracy", "sku_acc"), ("planta", "planta_acc"), ("total", "total_acc"),
                                  ("SKU reconciled", "sku_acc_rec")]]
            print(f"\ntest h={h} (naive SKU accuracy {r[f'h{h}']['naive_sku_acc']:.4f}; pred std base {r[f'h{h}']['base_pred_std']:.2f}, optuna {r[f'h{h}']['tuned_pred_std']:.2f}):")
            print(pd.DataFrame(rows).set_index("metric").round(4))


if __name__ == "__main__":
    main()
