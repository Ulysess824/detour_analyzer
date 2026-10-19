"""Rolling test of the tuned models: refit with known months only, summaries per seed, json conversion and the printed comparison."""

import json

import numpy as np
import optuna
import pytest

from src.engine.feature_utils import make_frame
from src.evaluation.metrics_utils import score
from src.evaluation.tuning_utils import compare_base_and_tuned, print_comparison, rolling_test, study_summary, summarise_runs, to_jsonable

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture(scope="module")
def setup(toy_panel):
    S = toy_panel
    n = S.monthly.shape[1]
    return S, {1: make_frame(S, 1), 3: make_frame(S, 3)}, [n - 2, n - 1]


def perfect(train, rows, seed):
    return rows["y"].to_numpy(dtype=float)  # cheats on purpose: the scores are then known exactly


def scaled(train, rows, seed):
    return rows["y"].to_numpy(dtype=float) * (1.0 + 0.1 * seed)


def naive(train, rows, seed):
    return rows["p_naive"].to_numpy(dtype=float)


def test_rolling_test_refits_before_every_month_with_known_months_only(setup):
    S, frames, months = setup
    calls = []

    def spy(train, rows, seed):
        calls.append((int(train["t"].max()), int(rows["o"].iloc[0]), seed))
        return rows["p_naive"].to_numpy(dtype=float)

    runs = rolling_test(spy, S, frames[1], 1, months, seeds=[0, 1])
    assert len(runs) == 2 and len(calls) == 2 * len(months)  # one refit per seed and test month
    assert all(last_train <= origin for last_train, origin, _ in calls)  # nothing at or after the target month
    assert {seed for *_, seed in calls} == {0, 1}
    for run in runs:
        assert set(run["t"]) == set(months) and {"p", "p_rec", "p_naive", "y"} <= set(run.columns)
        assert np.allclose(run["p"], run["p_naive"]) and np.isfinite(run["p_rec"]).all() and (run["p_rec"] >= 0).all()


def test_rolling_test_at_horizon_three_trains_further_back(setup):
    S, frames, months = setup
    seen = []
    rolling_test(lambda train, rows, seed: seen.append(int(train["t"].max())) or rows["p_naive"].to_numpy(dtype=float), S, frames[3], 3, months, seeds=[0])
    assert all(last <= t - 3 for last, t in zip(seen, months))


def test_summaries_match_a_hand_calculation(setup):
    S, frames, months = setup
    runs = rolling_test(scaled, S, frames[1], 1, months, seeds=[0, 1, 2])  # forecast = y, 1.1 y, 1.2 y
    out = summarise_runs(runs, S, months)
    assert np.allclose(out["sku_acc"], [1.0, 0.9, 0.8]) and np.allclose(out["planta_acc"], [1.0, 0.9, 0.8]) and np.allclose(out["total_acc"], [1.0, 0.9, 0.8])
    assert np.allclose(out["month_wape"], [0.1, 0.1])  # mean over the seeds of 0, 0.1 and 0.2
    y = runs[0]["y"].to_numpy()
    assert np.allclose(out["sku_mae"], [0.0, 0.1 * y.mean(), 0.2 * y.mean()])
    assert set(out) == {"sku_acc", "sku_mae", "planta_acc", "total_acc", "sku_acc_rec", "planta_acc_rec", "total_acc_rec", "month_wape"}
    assert all(np.isfinite(out[k]).all() for k in out)


def test_compare_base_and_tuned_runs_both_horizons_and_keeps_the_runs(setup, capsys):
    S, frames, months = setup
    results = compare_base_and_tuned(naive, perfect, S, frames, months, seeds=[0, 1])
    assert set(results) == {"h1", "h3"}
    for horizon, key in ((1, "h1"), (3, "h3")):
        r = results[key]
        assert np.allclose(r["tuned"]["sku_acc"], 1.0) and (r["base"]["sku_acc"] < 1.0).all()
        first = r["_runs"]["base"][0]
        assert r["naive_sku_acc"] == pytest.approx(score(first["y"].to_numpy(), first["p_naive"].to_numpy())["accuracy"])
        assert len(r["_runs"]["base"]) == len(r["_runs"]["tuned"]) == 2
        assert set(first["t"]) == set(months)
    assert "test h=1 done" in capsys.readouterr().out


def test_to_jsonable_converts_numpy_and_drops_private_keys():
    obj = {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, 2.0]), "d": (np.float32(0.5), [np.int32(2)]), "_runs": object(), "e": {"_hidden": 1, "f": "x"}}
    out = to_jsonable(obj)
    assert out == {"a": 1.5, "b": 3, "c": [1.0, 2.0], "d": [0.5, [2]], "e": {"f": "x"}}
    assert json.loads(json.dumps(out)) == out
    assert type(out["a"]) is float and type(out["b"]) is int


def test_the_results_of_a_comparison_can_be_written_as_json(setup):
    S, frames, months = setup
    results = compare_base_and_tuned(naive, perfect, S, frames, months, seeds=[0])
    text = json.dumps(to_jsonable(results))
    back = json.loads(text)
    assert "_runs" not in back["h1"] and back["h1"]["tuned"]["sku_acc"] == [1.0]


def test_study_summary_describes_the_search():
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        trial.suggest_float("z", 0.0, 1.0)
        trial.set_user_attr("fold_wape", [x, x, x])
        return x

    study.enqueue_trial({"x": 0.9, "z": 0.1})
    study.optimize(objective, n_trials=6)
    out = study_summary(study, [(10, 5), (12, 5), (14, 5)], minutes=1.234, base_params={"x": 0.9})
    assert out["n_trials"] == 6 and out["val_base"] == pytest.approx(0.9) and out["val_best"] == study.best_value <= out["val_base"]
    assert out["best_params"] == study.best_params and out["best_trial"] == study.best_trial.number
    assert out["val_base_folds"] == [0.9] * 3 and out["minutes"] == 1.2 and out["base_params"] == {"x": 0.9}
    assert set(out["importance"]) <= {"x", "z"} and out["fold_train_test_rows"] == [(10, 5), (12, 5), (14, 5)]


def test_print_comparison_shows_every_metric_and_the_months_won(setup, capsys):
    S, frames, months = setup
    compared = compare_base_and_tuned(naive, perfect, S, frames, months, seeds=[0, 1])
    capsys.readouterr()
    result = {
        "val_base": 0.4, "val_best": 0.3, "n_trials": 5, "base_params": {"a": 1}, "best_params": {"a": 2.123456789}, "importance": {"a": 1.0},
        "h1": compared["h1"], "h3": compared["h3"],
    }
    print_comparison({"lgbm": result})
    shown = capsys.readouterr().out
    assert "lgbm" in shown and "SKU accuracy" in shown and "total reconciled" in shown and "test h=3" in shown
    assert "months where optuna has lower SKU WAPE: 2 of 2" in shown and "2.12346" in shown
