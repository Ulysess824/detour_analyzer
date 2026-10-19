"""Member forecasts of the rolling origin: one row per series and test month, and nothing from after the origin reaches a forecast."""

import numpy as np
import pytest

from src.engine.feature_utils import make_frame
from src.engine.member_utils import ML_MODELS, collect_member_forecasts

HORIZON = 1


@pytest.fixture(scope="module")
def setup(toy_panel):
    n = toy_panel.monthly.shape[1]
    return toy_panel, make_frame(toy_panel, HORIZON), [n - 3, n - 2]


@pytest.fixture(scope="module")
def forecasts(setup):
    S, frame, months = setup
    return collect_member_forecasts(S, frame, months, HORIZON, {})


def test_one_row_per_series_alive_at_the_origin_and_test_month(setup, forecasts):
    S, frame, months = setup
    assert set(forecasts.columns) == {"series", "o", "t", "planta_code", "y", "p_lgbm", "p_xgb", "p_rf", "p_naive"}
    for t in months:
        origin = t - HORIZON
        alive = np.where(~np.isnan(S.monthly[:, origin]))[0]
        block = forecasts[forecasts["t"] == t]
        assert sorted(block["series"]) == sorted(alive) and (block["o"] == origin).all()
        assert np.allclose(block["y"].to_numpy(), np.nan_to_num(S.monthly[block["series"].to_numpy(), t]))


def test_forecasts_are_finite_and_not_negative_and_naive_is_last_month(setup, forecasts):
    S, _, _ = setup
    values = forecasts[[f"p_{k}" for k in ML_MODELS]].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all()
    last_month = S.monthly[forecasts["series"].to_numpy(), forecasts["o"].to_numpy()]
    assert np.allclose(forecasts["p_naive"].to_numpy(), last_month)


def test_a_forecast_does_not_depend_on_months_after_its_origin(setup, forecasts):
    S, frame, months = setup
    origin = months[0] - HORIZON
    future = frame.copy()
    future.loc[future["t"] > origin, "y"] *= 7.0  # the target and everything later: unknown at the origin
    other = collect_member_forecasts(S, future, months[:1], HORIZON, {})
    first = forecasts[forecasts["t"] == months[0]].reset_index(drop=True)
    assert not np.allclose(other["y"], first["y"])  # the perturbation is real
    for kind in ML_MODELS:
        assert np.allclose(other[f"p_{kind}"].to_numpy(), first[f"p_{kind}"].to_numpy())


def test_the_same_inputs_give_the_same_forecasts(setup, forecasts):
    S, frame, months = setup
    again = collect_member_forecasts(S, frame, months, HORIZON, {})
    for kind in ML_MODELS:
        assert np.array_equal(again[f"p_{kind}"].to_numpy(), forecasts[f"p_{kind}"].to_numpy())


def test_tuned_parameters_change_only_their_own_model(setup, forecasts):
    S, frame, months = setup
    tuned = {"lgbm": {"best_params": {"learning_rate": 0.2, "num_leaves": 4}}}
    other = collect_member_forecasts(S, frame, months[:1], HORIZON, tuned)
    first = forecasts[forecasts["t"] == months[0]].reset_index(drop=True)
    assert not np.allclose(other["p_lgbm"], first["p_lgbm"])
    assert np.allclose(other["p_xgb"], first["p_xgb"]) and np.allclose(other["p_rf"], first["p_rf"])
