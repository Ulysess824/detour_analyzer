"""
Bayesian tuning (Optuna, TPE sampler) of the tree ensembles on the monthly-direct pipeline:
LightGBM, XGBoost and Random Forest, compared against the hyperparameters used so far.

Protocol (chronological, the test months are never used to choose anything):
    - The last --n-test months are the test set. Everything before them is the tuning window.
    - Inside the tuning window, three walk-forward folds: each validates on the next
      3 months and trains only on rows whose target is already known at the first validation
      origin. A trial's score is the mean SKU-month WAPE over the three folds.
    - The current (baseline) hyperparameters are enqueued as the first trial, so the study
      can only end at least as good as the baseline on validation.
    - After the search the best parameters are frozen and every test month is forecast with
      a rolling refit (fit on what was known at that origin), for the baseline and the tuned
      configuration alike, over several seeds because subsampling makes the fit stochastic.
    - Tuning is done at h=1; h=3 is only checked with the h=1 parameters.

Requires: optuna, lightgbm, xgboost, scikit-learn.

Usage:
    python scripts/tune_trees.py data/consumos_long.csv --trials 100 100 40 --out tuning.json
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

FEATURES = fm.SERIES_FEATURES
MAX_TREES = 1_500
PATIENCE = 50
RF_TREES = 200
FOLD_MONTHS = 3
N_FOLDS = 3

BASE = {
    "lgbm": dict(
        tweedie_variance_power=1.2, learning_rate=0.05, num_leaves=15, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8,
    ),
    "xgb": dict(
        tweedie_variance_power=1.2, learning_rate=0.05, max_depth=6, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8,
    ),
    "rf": dict(criterion="squared_error", max_features=1.0, min_samples_leaf=1, max_samples=None),
}

# Baseline values as trial parameters. Defaults that the baseline leaves unset (zero
# regularisation for LightGBM, lambda=1 for XGBoost) are entered at their closest value.
ENQUEUE = {
    "lgbm": {**BASE["lgbm"], "reg_alpha": 1e-8, "reg_lambda": 1e-8},
    "xgb": {**BASE["xgb"], "gamma": 1e-8, "reg_alpha": 1e-8, "reg_lambda": 1.0},
    "rf": {**BASE["rf"], "max_samples": 1.0},
}


def suggest(kind: str, trial) -> dict:
    r"""Search space per model, on log scales where the parameter spans orders of magnitude."""
    if kind == "lgbm":
        return dict(
            tweedie_variance_power=trial.suggest_float("tweedie_variance_power", 1.05, 1.9),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
    if kind == "xgb":
        return dict(
            tweedie_variance_power=trial.suggest_float("tweedie_variance_power", 1.05, 1.9),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 200.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
    return dict(
        criterion=trial.suggest_categorical("criterion", ["squared_error", "poisson"]),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 60, log=True),
        max_samples=trial.suggest_float("max_samples", 0.3, 1.0),
    )


def fit_predict(kind: str, params: dict, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    r"""
    Fit on train and predict test.

    *   LightGBM and XGBoost use the same procedure as forecast_monthly.py: early stopping on
        the last two target months of train picks the number of trees, then the model is
        refit on all of train with that number.
    *   Random Forest has no early stopping; missing values are filled with -1 because trees
        need a sentinel, and OOB scores are not used since bootstrapping mixes the time order.
    *
    """
    y = train["y"]
    tail = (train["t"] >= train["t"].max() - 1).to_numpy()

    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=RF_TREES, n_jobs=-1, random_state=seed, **params)
        model.fit(train[FEATURES].fillna(-1.0), y)
        return np.clip(model.predict(test[FEATURES].fillna(-1.0)), 0.0, None).astype(float)

    if kind == "lgbm":
        import lightgbm as lgb

        p = dict(objective="tweedie", subsample_freq=1, random_state=seed, verbose=-1, **params)
        probe = lgb.LGBMRegressor(n_estimators=MAX_TREES, **p)
        probe.fit(
            train.loc[~tail, FEATURES],
            y[~tail],
            eval_X=train.loc[tail, FEATURES],
            eval_y=y[tail],
            eval_metric="mae",
            categorical_feature=["planta_code"],
            callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
        )
        final = lgb.LGBMRegressor(n_estimators=max(probe.best_iteration_, 20), **p)
        final.fit(train[FEATURES], y, categorical_feature=["planta_code"])
        return np.clip(final.predict(test[FEATURES]), 0.0, None).astype(float)

    import xgboost as xgb

    p = dict(objective="reg:tweedie", tree_method="hist", random_state=seed, verbosity=0, **params)
    probe = xgb.XGBRegressor(n_estimators=MAX_TREES, early_stopping_rounds=PATIENCE, eval_metric="mae", **p)
    probe.fit(train.loc[~tail, FEATURES], y[~tail], eval_set=[(train.loc[tail, FEATURES], y[tail])], verbose=False)
    final = xgb.XGBRegressor(n_estimators=max(int(probe.best_iteration) + 1, 20), **p)
    final.fit(train[FEATURES], y, verbose=False)
    return np.clip(final.predict(test[FEATURES]), 0.0, None).astype(float)


def tune(kind: str, frame: pd.DataFrame, h: int, tune_end: int, n_trials: int, seed: int, storage: str | None):
    r"""Run the TPE search on the tuning window; return the finished study."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frame = frame[frame["t"] <= tune_end]
    starts = [tune_end - FOLD_MONTHS + 1 - FOLD_MONTHS * (N_FOLDS - 1 - k) for k in range(N_FOLDS)]
    folds = []
    for a in starts:
        train = frame[frame["t"] <= a - h]
        val = frame[(frame["t"] >= a) & (frame["t"] < a + FOLD_MONTHS)]
        folds.append((train, val))

    def objective(trial) -> float:
        params = suggest(kind, trial)
        wapes = []
        for train, val in folds:
            pred = fit_predict(kind, params, train, val, seed=0)
            wapes.append(float(np.abs(val["y"].to_numpy() - pred).sum() / val["y"].sum()))
        trial.set_user_attr("fold_wape", [round(w, 4) for w in wapes])
        return float(np.mean(wapes))

    study = optuna.create_study(
        study_name=f"{kind}_h{h}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10),
        storage=storage,
        load_if_exists=True,
    )
    finished = sum(t.state.is_finished() for t in study.trials)
    if len(study.trials) == 0:
        study.enqueue_trial(ENQUEUE[kind])

    def progress(st, tr) -> None:
        if (tr.number + 1) % 10 == 0:
            print(f"    {kind}: trial {tr.number + 1}/{n_trials}  best validation WAPE {st.best_value:.4f}", flush=True)

    remaining = max(n_trials - finished, 0)
    study.optimize(objective, n_trials=remaining, callbacks=[progress])
    return study, [(len(t), len(v)) for t, v in folds]


def _level_scores(res: pd.DataFrame, col: str, S: fm.Panel, test_months: list[int]) -> dict:
    act_p, pred_p, act_t, pred_t = [], [], [], []
    for t in test_months:
        blk = res[res["t"] == t]
        act_p.append(S.P[:, t])
        pred_p.append(blk.groupby("planta_code")[col].sum().reindex(range(len(S.plantas))).fillna(0.0).to_numpy())
        act_t.append(S.T[t])
        pred_t.append(blk[col].sum())
    return {
        "sku": fm.score(res["y"].to_numpy(), res[col].to_numpy()),
        "planta": fm.score(np.concatenate(act_p), np.concatenate(pred_p)),
        "total": fm.score(np.array(act_t), np.array(pred_t)),
    }


def test_run(kind, params, frame, S, h, test_months, seeds):
    r"""Rolling refit over the test months; returns one result frame per seed."""
    out = []
    for seed in seeds:
        parts = []
        for t in test_months:
            origin = t - h
            train = frame[frame["t"] <= origin]
            block = frame[frame["o"] == origin][["series", "o", "t", "planta_code", "y", "p_naive"]].copy()
            block["p"] = fit_predict(kind, params, train, frame[frame["o"] == origin], seed)
            parts.append(block)
        res = pd.concat(parts, ignore_index=True)
        res["p_rec"] = fm._reconcile(res, S, h, test_months, "p")
        out.append(res)
    return out


def summarise(runs: list[pd.DataFrame], S, test_months) -> dict:
    per_seed = [_level_scores(r, "p", S, test_months) for r in runs]
    per_seed_rec = [_level_scores(r, "p_rec", S, test_months) for r in runs]
    month = np.array(
        [[fm.score(r.loc[r["t"] == t, "y"].to_numpy(), r.loc[r["t"] == t, "p"].to_numpy())["wape"] for t in test_months] for r in runs]
    ).mean(axis=0)
    acc = lambda lst, lvl: np.array([s[lvl]["accuracy"] for s in lst])  # noqa: E731
    return {
        "sku_acc": acc(per_seed, "sku"),
        "sku_mae": np.array([s["sku"]["mae"] for s in per_seed]),
        "planta_acc": acc(per_seed, "planta"),
        "total_acc": acc(per_seed, "total"),
        "sku_acc_rec": acc(per_seed_rec, "sku"),
        "planta_acc_rec": acc(per_seed_rec, "planta"),
        "total_acc_rec": acc(per_seed_rec, "total"),
        "month_wape": month,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--models", nargs="+", default=["lgbm", "xgb", "rf"], choices=["lgbm", "xgb", "rf"])
    parser.add_argument("--trials", nargs="+", type=int, default=[100, 100, 40])
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rf-seeds", nargs="+", type=int, default=[0, 1], help="fewer seeds: a forest fit is slow")
    parser.add_argument("--seed", type=int, default=42, help="seed of the TPE sampler")
    parser.add_argument("--storage", type=str, default=None, help="e.g. sqlite:///optuna.db, to resume a search")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    S = fm.build_panel(pd.read_csv(args.input, parse_dates=["fecha"]))
    n_months = S.Y.shape[1]
    test_months = list(range(n_months - args.n_test, n_months))
    tune_end = n_months - args.n_test - 1
    print(f"tuning window: {S.labels[0]} .. {S.labels[tune_end]} | test: {S.labels[test_months[0]]} .. {S.labels[-1]}")
    frames = {h: fm.make_frame(S, h) for h in (1, 3)}
    trials = dict(zip(["lgbm", "xgb", "rf"], [None] * 3))
    trials.update(dict(zip(args.models, args.trials)))

    results: dict = {}
    for kind in args.models:
        print(f"\n=== {kind}: {trials[kind]} trials ===", flush=True)
        t0 = time.time()
        study, fold_sizes = tune(kind, frames[1], 1, tune_end, trials[kind], args.seed, args.storage)
        minutes = (time.time() - t0) / 60
        best = study.best_trial
        base_trial = study.trials[0]
        try:
            import optuna

            imp = optuna.importance.get_param_importances(study)
        except Exception:
            imp = {}
        results[kind] = {
            "best_params": best.params,
            "base_params": BASE[kind],
            "val_base": base_trial.value,
            "val_best": best.value,
            "val_best_folds": best.user_attrs["fold_wape"],
            "val_base_folds": base_trial.user_attrs["fold_wape"],
            "best_trial": best.number,
            "n_trials": len(study.trials),
            "fold_train_test_rows": fold_sizes,
            "minutes": round(minutes, 1),
            "importance": {k: round(v, 3) for k, v in imp.items()},
        }
        print(f"    validation WAPE: baseline {base_trial.value:.4f} -> best {best.value:.4f} (trial {best.number}, {minutes:.1f} min)", flush=True)

        for h in (1, 3):
            seeds = args.rf_seeds if kind == "rf" else args.seeds
            base_runs = test_run(kind, BASE[kind], frames[h], S, h, test_months, seeds)
            tuned_runs = test_run(kind, best.params, frames[h], S, h, test_months, seeds)
            results[kind][f"h{h}"] = {"base": summarise(base_runs, S, test_months), "tuned": summarise(tuned_runs, S, test_months)}
            naive = base_runs[0]
            results[kind][f"h{h}"]["naive_sku_acc"] = fm.score(naive["y"].to_numpy(), naive["p_naive"].to_numpy())["accuracy"]
            print(f"    test h={h} done", flush=True)

    for kind, r in results.items():
        print(f"\n################ {kind} ################")
        print(f"validation WAPE (mean of {N_FOLDS} folds): baseline {r['val_base']:.4f}  |  best {r['val_best']:.4f}  (trials: {r['n_trials']})")
        print("params baseline:", r["base_params"])
        print("params optuna  :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["best_params"].items()})
        print("importance     :", r["importance"])
        for h in (1, 3):
            b, t = r[f"h{h}"]["base"], r[f"h{h}"]["tuned"]
            rows = []
            for name, key in [("SKU accuracy", "sku_acc"), ("planta (bottom-up)", "planta_acc"), ("total (bottom-up)", "total_acc"),
                              ("SKU accuracy reconciled", "sku_acc_rec"), ("planta reconciled", "planta_acc_rec"), ("total reconciled", "total_acc_rec")]:
                rows.append({"metric": name, "baseline": b[key].mean(), "optuna": t[key].mean(),
                             "delta_pts": (t[key].mean() - b[key].mean()) * 100,
                             "base_seed_range": f"{b[key].min():.4f}-{b[key].max():.4f}", "optuna_seed_range": f"{t[key].min():.4f}-{t[key].max():.4f}"})
            print(f"\ntest h={h} (naive SKU accuracy {r[f'h{h}']['naive_sku_acc']:.4f}):")
            print(pd.DataFrame(rows).set_index("metric").round(4))
            wins = int((t["month_wape"] < b["month_wape"]).sum())
            diff = (b["month_wape"] - t["month_wape"]) * 100
            print(f"months where optuna has lower SKU WAPE: {wins} of {len(test_months)} | mean gain {diff.mean():+.2f} pts (sd {diff.std():.2f})")

    if args.out:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (np.ndarray, list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o

        args.out.write_text(json.dumps(clean(results), indent=1))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
