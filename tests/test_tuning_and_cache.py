"""Tuning folds and the guards around stored results."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.utils.cache_utils import check_alignment, check_meta, expected_meta, panel_fingerprint, write_meta
from src.utils.feature_utils import make_frame
from src.utils.panel_utils import build_panel
from src.utils.tuning_utils import FOLD_MONTHS, N_FOLDS, check_tuning_window, fold_starts, make_folds, run_study, tuning_meta
from tests.conftest import make_consumption


@pytest.mark.parametrize("horizon", [1, 3])
def test_folds_never_train_on_months_that_are_validated_or_in_the_test(toy_panel, horizon):
    S = toy_panel
    n, n_test = S.monthly.shape[1], 6
    tune_end = n - n_test - 1
    folds = make_folds(make_frame(S, horizon), horizon, tune_end)
    assert len(folds) == N_FOLDS
    for train, val in folds:
        assert train["t"].max() < val["t"].min()
        assert val["t"].max() <= tune_end
        assert val["t"].nunique() == FOLD_MONTHS
    assert fold_starts(tune_end)[-1] + FOLD_MONTHS - 1 == tune_end


def test_tuning_meta_windows(toy_panel):
    S = toy_panel
    meta = tuning_meta(S, 6)
    n = S.monthly.shape[1]
    assert meta["tune_end"] == S.labels[n - 7]
    assert meta["test_window"] == f"{S.labels[n - 6]} a {S.labels[-1]}"
    assert len(meta["folds"]) == N_FOLDS


def test_tuning_that_overlaps_the_test_is_rejected(toy_panel):
    S = toy_panel
    tuned = {"lgbm": {"meta": tuning_meta(S, 6)}}
    check_tuning_window(tuned, S, 6)  # same window: fine
    check_tuning_window(tuned, S, 4)  # a shorter test starts later than the tuning ends: fine too
    with pytest.raises(SystemExit, match="leak"):
        check_tuning_window(tuned, S, 10)  # the test now starts inside the tuning window
    with pytest.raises(SystemExit, match="no 'meta'"):
        check_tuning_window({"lgbm": {}}, S, 6)


def test_fingerprint_changes_with_a_new_sku_or_new_values_but_not_with_a_rebuild(toy_df):
    base = panel_fingerprint(build_panel(toy_df))
    assert base == panel_fingerprint(build_panel(toy_df.copy()))
    assert base != panel_fingerprint(build_panel(make_consumption(extra_sku=True)))
    changed = toy_df.copy()
    changed.loc[0, "consumo"] += 1
    assert base != panel_fingerprint(build_panel(changed))


def test_cache_made_for_other_data_is_rejected(toy_panel, tmp_path):
    cache = tmp_path / "member_forecasts.csv"
    meta = expected_meta(toy_panel, 1, 6, {})
    with pytest.raises(SystemExit, match="no member_forecasts.meta.json"):
        check_meta(cache, meta)
    write_meta(cache, meta)
    check_meta(cache, meta)
    other = expected_meta(build_panel(make_consumption(extra_sku=True)), 1, 6, {})
    with pytest.raises(SystemExit, match="out of date"):
        check_meta(cache, other)
    with pytest.raises(SystemExit, match="horizon"):
        check_meta(cache, {**meta, "horizon": 3})
    check_meta(cache, {**meta, "horizon": 3}, keys=("data", "n_series"))  # only the listed keys are compared


def test_alignment_detects_a_shifted_series_index(toy_df, toy_panel):
    S = toy_panel
    frame = make_frame(S, 1)
    res = frame[frame["t"] >= 25][["series", "t", "y", "planta_code"]].reset_index(drop=True)
    check_alignment(res, S)
    shifted = build_panel(make_consumption(extra_sku=True))  # one SKU added at the front
    with pytest.raises(SystemExit, match="differ"):
        check_alignment(res, shifted)


def test_study_stored_for_other_data_is_not_resumed(toy_panel, tmp_path):
    S = toy_panel
    frame = make_frame(S, 1)
    storage = f"sqlite:///{tmp_path / 'study.db'}"

    def predictor(params):
        return lambda train, rows, seed: np.full(len(rows), params["value"])

    def suggest(trial):
        return {"value": trial.suggest_float("value", 0.0, 10.0)}

    args = dict(name="t", make_predictor=predictor, suggest=suggest, baseline_trial={"value": 1.0}, frame=frame, horizon=1,
                tune_end=S.monthly.shape[1] - 7, n_trials=2, sampler_seed=0, storage=storage)  # fmt: skip
    run_study(**args, fingerprint="data-A:6")
    run_study(**{**args, "n_trials": 3}, fingerprint="data-A:6")  # same data: resumes
    with pytest.raises(SystemExit, match="another --storage"):
        run_study(**args, fingerprint="data-B:6")
