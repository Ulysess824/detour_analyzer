"""Optuna search spaces and the baseline trial of every tuned model."""

from __future__ import annotations

from src.utils.neural_utils import NEURAL_BASE
from src.utils.tree_utils import TREE_BASE

NEURAL_BATCH_SIZES = [256, 512, 1024, 2048]
NEURAL_LOOK_BACKS = [6, 8, 12]

# The baseline is enqueued as trial 0. Parameters the baseline leaves unset (library defaults)
# are entered at their closest value inside the search range.
TREE_BASELINE_TRIAL = {
    "lgbm": {**TREE_BASE["lgbm"], "reg_alpha": 1e-8, "reg_lambda": 1e-8},
    "xgb": {**TREE_BASE["xgb"], "gamma": 1e-8, "reg_alpha": 1e-8, "reg_lambda": 1.0},
    "rf": {**TREE_BASE["rf"], "max_samples": 1.0},
}
NEURAL_BASELINE_TRIAL = {
    "dnn": {**NEURAL_BASE["dnn"], "weight_decay": 1e-6},
    "lstm": {**NEURAL_BASE["lstm"], "weight_decay": 1e-6},
}


def suggest_tree_params(kind: str, trial) -> dict:
    r"""Search space of a tree model; log scales for parameters that span orders of magnitude."""
    if kind == "rf":
        return dict(
            criterion=trial.suggest_categorical("criterion", ["squared_error", "poisson"]),
            max_features=trial.suggest_float("max_features", 0.2, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 60, log=True),
            max_samples=trial.suggest_float("max_samples", 0.3, 1.0),
        )

    # The order of the suggest calls matters: with a fixed seed it fixes the sampled values.
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
    return dict(  # xgb
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


def suggest_neural_params(kind: str, trial) -> dict:
    r"""Search space of the DNN and the LSTM."""
    params = dict(
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", NEURAL_BATCH_SIZES),
        tweedie_p=trial.suggest_float("tweedie_p", 1.05, 1.9),
    )
    if kind == "dnn":
        params["n_layers"] = trial.suggest_int("n_layers", 1, 3)
        params["units"] = trial.suggest_int("units", 8, 128, log=True)
    else:
        params["hidden"] = trial.suggest_int("hidden", 8, 64, log=True)
        params["units"] = trial.suggest_int("units", 8, 64, log=True)
        params["look_back"] = trial.suggest_categorical("look_back", NEURAL_LOOK_BACKS)
    return params
