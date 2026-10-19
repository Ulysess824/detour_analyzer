"""Optuna search spaces: every sampled value stays in its documented range, trains a real model, and the baseline trial fits inside the space."""

import numpy as np
import optuna
import pytest

from src.engine.feature_utils import make_frame
from src.engine.tree_utils import fit_predict_trees
from src.evaluation.search_space_utils import TREE_BASELINE_TRIAL, suggest_tree_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

# the documented ranges: (low, high); None for categorical choices
BOUNDS = {
    "lgbm": {"tweedie_variance_power": (1.05, 1.9), "learning_rate": (0.02, 0.2), "num_leaves": (4, 128), "min_child_samples": (5, 200),
             "subsample": (0.5, 1.0), "colsample_bytree": (0.4, 1.0), "reg_alpha": (1e-8, 10.0), "reg_lambda": (1e-8, 10.0)},
    "xgb": {"tweedie_variance_power": (1.05, 1.9), "learning_rate": (0.02, 0.2), "max_depth": (3, 10), "min_child_weight": (1.0, 200.0),
            "gamma": (1e-8, 5.0), "subsample": (0.5, 1.0), "colsample_bytree": (0.4, 1.0), "reg_alpha": (1e-8, 10.0), "reg_lambda": (1e-8, 10.0)},
    "rf": {"max_features": (0.2, 1.0), "min_samples_leaf": (1, 60), "max_samples": (0.3, 1.0)},
}


def sample(kind, n_trials=30, enqueue=None):
    drawn = []

    def objective(trial):
        drawn.append(suggest_tree_params(kind, trial))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    if enqueue is not None:
        study.enqueue_trial(enqueue)
    study.optimize(objective, n_trials=n_trials)
    return drawn


@pytest.mark.parametrize("kind", ["lgbm", "xgb", "rf"])
def test_every_sampled_value_is_inside_its_range(kind):
    drawn = sample(kind)
    assert set(drawn[0]) >= set(BOUNDS[kind])
    for params in drawn:
        for name, (low, high) in BOUNDS[kind].items():
            assert low <= params[name] <= high, (kind, name, params[name])
    if kind == "rf":
        assert {p["criterion"] for p in drawn} <= {"squared_error", "poisson"}
    else:
        assert all(isinstance(p["num_leaves" if kind == "lgbm" else "max_depth"], int) for p in drawn)


@pytest.mark.parametrize("kind", ["lgbm", "xgb", "rf"])
def test_a_sampled_configuration_trains_a_model(toy_panel, kind):
    S = toy_panel
    n = S.monthly.shape[1]
    frame = make_frame(S, 1)
    train, rows = frame[frame["t"] <= n - 2], frame[frame["o"] == n - 2]
    params = sample(kind, n_trials=3)[-1]
    forecast = fit_predict_trees(kind, params, train, rows, seed=0)
    assert len(forecast) == len(rows) and np.isfinite(forecast).all() and (forecast >= 0).all()


@pytest.mark.parametrize("kind", ["lgbm", "xgb", "rf"])
def test_the_baseline_trial_lies_inside_the_search_space(kind):
    baseline = TREE_BASELINE_TRIAL[kind]
    suggested = {k: v for k, v in baseline.items() if k in BOUNDS[kind]}
    assert set(suggested) == set(BOUNDS[kind])  # every searched parameter has a baseline value
    first = sample(kind, n_trials=1, enqueue=suggested)[0]
    for name, value in suggested.items():
        assert first[name] == pytest.approx(value), (kind, name)  # enqueued values are used as they are
        low, high = BOUNDS[kind][name]
        assert low <= value <= high
