"""Optuna search with chronological folds, and the rolling test used to compare base and tuned models."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.utils.evaluation_utils import reconcile
from src.utils.metrics_utils import score
from src.utils.cache_utils import panel_fingerprint
from src.utils.panel_utils import Panel

FOLD_MONTHS = 3
N_FOLDS = 3

# predict(train, rows, seed) -> monthly forecast of each row of `rows`
Predictor = Callable[[pd.DataFrame, pd.DataFrame, int], np.ndarray]


def fold_starts(tune_end: int) -> list[int]:
    r"""First validation month of each fold; the last fold ends at `tune_end`."""
    return [tune_end - FOLD_MONTHS + 1 - FOLD_MONTHS * (N_FOLDS - 1 - k) for k in range(N_FOLDS)]


def make_folds(frame: pd.DataFrame, horizon: int, tune_end: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    r"""
    Walk-forward folds inside the tuning window.

    Each fold validates on the next FOLD_MONTHS target months and trains only on rows whose
    target month is already known `horizon` months before the first validation month.
    """
    frame = frame[frame["t"] <= tune_end]
    folds = []
    for first_val in fold_starts(tune_end):
        train = frame[frame["t"] <= first_val - horizon]
        val = frame[(frame["t"] >= first_val) & (frame["t"] < first_val + FOLD_MONTHS)]
        folds.append((train, val))
    return folds


def run_study(
    name: str,
    make_predictor: Callable[[dict], Predictor],
    suggest: Callable,
    baseline_trial: dict,
    frame: pd.DataFrame,
    horizon: int,
    tune_end: int,
    n_trials: int,
    sampler_seed: int,
    storage: str | None,
    print_every: int = 1,
    fingerprint: str | None = None,
):
    r"""
    TPE search of the hyperparameters; a trial's score is the mean SKU-month WAPE over the folds.

    The baseline parameters are enqueued as trial 0, so the study can only end at least as
    good as the baseline on validation. Returns (study, [(train rows, validation rows) per fold]).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = make_folds(frame, horizon, tune_end)

    def objective(trial) -> float:
        predict = make_predictor(suggest(trial))
        wapes = []
        for train, val in folds:
            pred = predict(train, val, 0)
            wapes.append(float(np.abs(val["y"].to_numpy() - pred).sum() / val["y"].sum()))
        trial.set_user_attr("fold_wape", [round(w, 4) for w in wapes])
        return float(np.mean(wapes))

    study = optuna.create_study(
        study_name=f"{name}_h{horizon}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=10),
        storage=storage,
        load_if_exists=True,
    )
    # A study kept in `storage` was fitted to some data; resuming it with other data would mix both.
    stored = study.user_attrs.get("fingerprint")
    if fingerprint is not None and stored is not None and stored != fingerprint:
        raise SystemExit(f"The stored study {study.study_name!r} was fitted to other data or settings; use another --storage file.")
    if fingerprint is not None:
        study.set_user_attr("fingerprint", fingerprint)
    finished = sum(t.state.is_finished() for t in study.trials)  # count before enqueueing
    if len(study.trials) == 0:
        study.enqueue_trial(baseline_trial)

    def progress(st, trial) -> None:
        if (trial.number + 1) % print_every == 0:
            print(f"    {name}: trial {trial.number + 1}/{n_trials}  value {trial.value:.4f}  best {st.best_value:.4f}", flush=True)

    study.optimize(objective, n_trials=max(n_trials - finished, 0), callbacks=[progress])
    return study, [(len(train), len(val)) for train, val in folds]


def tuning_meta(S: Panel, n_test: int) -> dict:
    r"""Windows of a tuning run, saved with its results so later steps can check them."""
    n_months = S.monthly.shape[1]
    tune_end = n_months - n_test - 1
    return {
        "data": panel_fingerprint(S),
        "n_test": n_test,
        "tune_window": f"{S.labels[0]} a {S.labels[tune_end]}",
        "tune_end": S.labels[tune_end],
        "test_window": f"{S.labels[n_months - n_test]} a {S.labels[-1]}",
        "folds": [f"{S.labels[a]} a {S.labels[a + FOLD_MONTHS - 1]}" for a in fold_starts(tune_end)],
    }


def check_tuning_window(tuned: dict, S: Panel, n_test: int) -> None:
    r"""
    Stop if any tuned parameters were chosen using months that are test months now.

    Reusing them would put the test into the hyperparameter choice. The tuning window of each
    model must end before the first test month.
    """
    first_test = S.labels[S.monthly.shape[1] - n_test]
    for kind, result in tuned.items():
        meta = result.get("meta")
        if meta is None:
            raise SystemExit(f"The tuning of {kind} has no 'meta'; its tuning window is unknown. Run tune_trees.py again.")
        if meta["tune_end"] >= first_test:
            raise SystemExit(
                f"The tuning of {kind} used data up to {meta['tune_end']}, but the test starts in {first_test}: "
                "the test would leak into the parameters. Run tune_trees.py again with this test window."
            )


def study_summary(study, fold_sizes, minutes: float, base_params: dict) -> dict:
    r"""The part of the results json that describes the search itself."""
    best, first = study.best_trial, study.trials[0]
    try:
        import optuna

        importance = optuna.importance.get_param_importances(study)
    except Exception:
        importance = {}
    return {
        "best_params": best.params,
        "base_params": base_params,
        "val_base": first.value,
        "val_best": best.value,
        "val_best_folds": best.user_attrs["fold_wape"],
        "val_base_folds": first.user_attrs["fold_wape"],
        "best_trial": best.number,
        "n_trials": len(study.trials),
        "fold_train_test_rows": fold_sizes,
        "minutes": round(minutes, 1),
        "importance": {k: round(v, 3) for k, v in importance.items()},
    }


def rolling_test(predict: Predictor, S: Panel, frame: pd.DataFrame, horizon: int, test_months: list[int], seeds) -> list[pd.DataFrame]:
    r"""Refit before every test month, once per seed; returns one result frame per seed."""
    runs = []
    for seed in seeds:
        parts = []
        for t in test_months:
            origin = t - horizon
            train = frame[frame["t"] <= origin]
            rows = frame[frame["o"] == origin]
            block = rows[["series", "o", "t", "planta_code", "y", "p_naive"]].copy()
            block["p"] = predict(train, rows, seed)
            parts.append(block)
        res = pd.concat(parts, ignore_index=True)
        res["p_rec"] = reconcile(res, S, horizon, test_months, "p")
        runs.append(res)
    return runs


def _level_scores(res: pd.DataFrame, col: str, S: Panel, test_months: list[int]) -> dict:
    r"""Score at SKU level and, summing the SKU forecasts, at planta and total level."""
    actual_p, pred_p, actual_t, pred_t = [], [], [], []
    for t in test_months:
        block = res[res["t"] == t]
        actual_p.append(S.planta_monthly[:, t])
        sums = block.groupby("planta_code")[col].sum()
        pred_p.append(sums.reindex(range(len(S.plantas))).fillna(0.0).to_numpy())
        actual_t.append(S.total_monthly[t])
        pred_t.append(block[col].sum())
    return {
        "sku": score(res["y"].to_numpy(), res[col].to_numpy()),
        "planta": score(np.concatenate(actual_p), np.concatenate(pred_p)),
        "total": score(np.array(actual_t), np.array(pred_t)),
    }


def summarise_runs(runs: list[pd.DataFrame], S: Panel, test_months: list[int]) -> dict:
    r"""Accuracy at each level for every seed, and the SKU WAPE of every test month (seed mean)."""
    plain = [_level_scores(r, "p", S, test_months) for r in runs]
    reconciled = [_level_scores(r, "p_rec", S, test_months) for r in runs]

    def accuracy(scores, level):
        return np.array([s[level]["accuracy"] for s in scores])

    month_wape = np.array(
        [[score(r.loc[r["t"] == t, "y"].to_numpy(), r.loc[r["t"] == t, "p"].to_numpy())["wape"] for t in test_months] for r in runs]
    ).mean(axis=0)
    return {
        "sku_acc": accuracy(plain, "sku"),
        "sku_mae": np.array([s["sku"]["mae"] for s in plain]),
        "planta_acc": accuracy(plain, "planta"),
        "total_acc": accuracy(plain, "total"),
        "sku_acc_rec": accuracy(reconciled, "sku"),
        "planta_acc_rec": accuracy(reconciled, "planta"),
        "total_acc_rec": accuracy(reconciled, "total"),
        "month_wape": month_wape,
    }


def compare_base_and_tuned(
    base_predict: Predictor, tuned_predict: Predictor, S: Panel, frames: dict, test_months: list[int], seeds
) -> dict:
    r"""Test results (h=1 and h=3) of the baseline and the tuned configuration, plus the naive accuracy."""
    results = {}
    for horizon in (1, 3):
        base_runs = rolling_test(base_predict, S, frames[horizon], horizon, test_months, seeds)
        tuned_runs = rolling_test(tuned_predict, S, frames[horizon], horizon, test_months, seeds)
        naive = base_runs[0]
        results[f"h{horizon}"] = {
            "base": summarise_runs(base_runs, S, test_months),
            "tuned": summarise_runs(tuned_runs, S, test_months),
            "naive_sku_acc": score(naive["y"].to_numpy(), naive["p_naive"].to_numpy())["accuracy"],
            "_runs": {"base": base_runs, "tuned": tuned_runs},
        }
        print(f"    test h={horizon} done", flush=True)
    return results


def seed_mean_traces(runs: list[pd.DataFrame], test_months: list[int], series: int | None) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Seed-mean monthly forecast of the total, and of one series, for plotting."""
    total = np.mean([[r.loc[r["t"] == t, "p"].sum() for t in test_months] for r in runs], axis=0)
    if series is None:
        return total, None
    one = [[r.loc[(r["t"] == t) & (r["series"] == series), "p"].sum() for t in test_months] for r in runs]
    return total, np.mean(one, axis=0)


def to_jsonable(obj):
    r"""Convert numpy values and arrays inside nested dicts and lists, so they can be written as json."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def print_comparison(results: dict) -> None:
    r"""Print, for each model and horizon, the baseline against the tuned configuration."""
    pd.set_option("display.width", 220)
    metrics = [
        ("SKU accuracy", "sku_acc"), ("planta (bottom-up)", "planta_acc"), ("total (bottom-up)", "total_acc"),
        ("SKU accuracy reconciled", "sku_acc_rec"), ("planta reconciled", "planta_acc_rec"), ("total reconciled", "total_acc_rec"),
    ]  # fmt: skip
    for kind, r in results.items():
        print(f"\n################ {kind} ################")
        print(f"validation WAPE (mean of {N_FOLDS} folds): baseline {r['val_base']:.4f}  |  best {r['val_best']:.4f}  (trials: {r['n_trials']})")
        print("params baseline:", r["base_params"])
        print("params optuna  :", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r["best_params"].items()})
        print("importance     :", r["importance"])
        for h in (1, 3):
            base, tuned = r[f"h{h}"]["base"], r[f"h{h}"]["tuned"]
            rows = []
            for name, key in metrics:
                rows.append(
                    {
                        "metric": name,
                        "baseline": np.mean(base[key]),
                        "optuna": np.mean(tuned[key]),
                        "delta_pts": (np.mean(tuned[key]) - np.mean(base[key])) * 100,
                        "base_seed_range": f"{np.min(base[key]):.4f}-{np.max(base[key]):.4f}",
                        "optuna_seed_range": f"{np.min(tuned[key]):.4f}-{np.max(tuned[key]):.4f}",
                    }
                )
            print(f"\ntest h={h} (naive SKU accuracy {r[f'h{h}']['naive_sku_acc']:.4f}):")
            print(pd.DataFrame(rows).set_index("metric").round(4))
            wins = int((np.array(tuned["month_wape"]) < np.array(base["month_wape"])).sum())
            diff = (np.array(base["month_wape"]) - np.array(tuned["month_wape"])) * 100
            print(f"months where optuna has lower SKU WAPE: {wins} of {len(diff)} | mean gain {diff.mean():+.2f} pts (sd {diff.std():.2f})")
