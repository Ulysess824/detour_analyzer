"""Scoring of the rolling-origin forecasts: test months, bottom-up sums and the ranked tables at SKU, planta and total level."""

import numpy as np
import pandas as pd
import pytest

from src.engine.feature_utils import make_frame
from src.evaluation.evaluation_utils import level_actual_and_bottom_up, score_models_by_level, test_months_of as last_months_of


@pytest.fixture(scope="module")
def setup(toy_panel):
    S = toy_panel
    n = S.monthly.shape[1]
    months = [n - 2, n - 1]
    frame = make_frame(S, 1)
    res = frame[frame["t"].isin(months)][["series", "o", "t", "planta_code", "y"]].reset_index(drop=True)
    res["p_perfect"] = res["y"]
    res["p_high"] = res["y"] * 1.1
    res["p_zero"] = 0.0
    return S, res, months


def test_the_test_months_are_the_last_ones(toy_panel):
    n = toy_panel.monthly.shape[1]
    assert last_months_of(toy_panel, 3) == [n - 3, n - 2, n - 1]
    assert last_months_of(toy_panel, 1) == [n - 1]


def test_every_series_is_alive_in_the_test_months_so_the_bottom_up_is_complete(setup):
    S, res, months = setup
    for t in months:
        assert res[res["t"] == t]["series"].nunique() == len(S.keys)


def test_actual_and_bottom_up_at_planta_and_total_level(setup):
    S, res, months = setup
    t = months[0]
    actual, bottom = level_actual_and_bottom_up(S, res, t, "planta", ["perfect", "high"])
    assert np.allclose(actual, S.planta_monthly[:, t]) and len(actual) == len(S.plantas)
    block = res[res["t"] == t]
    assert np.allclose(bottom["perfect"], block.groupby("planta_code")["p_perfect"].sum().to_numpy())
    assert np.allclose(bottom["high"], 1.1 * actual)
    actual_t, bottom_t = level_actual_and_bottom_up(S, res, t, "total", ["perfect"])
    assert actual_t.shape == (1,) and np.isclose(actual_t[0], S.total_monthly[t]) and np.isclose(bottom_t["perfect"][0], block["p_perfect"].sum())
    assert np.isclose(bottom["perfect"].sum(), bottom_t["perfect"][0])  # planta sums add up to the total


def test_a_planta_without_rows_has_a_zero_bottom_up(setup):
    S, res, months = setup
    t = months[0]
    missing = res[~((res["t"] == t) & (res["planta_code"] == 0))]
    _, bottom = level_actual_and_bottom_up(S, missing, t, "planta", ["perfect"])
    assert bottom["perfect"][0] == 0.0 and (bottom["perfect"][1:] > 0).all()


def test_ranked_tables_at_the_three_levels(setup):
    S, res, months = setup
    tables = score_models_by_level(S, res, ["zero", "high", "perfect"], months)
    assert set(tables) == {"sku", "planta", "total"}
    for level, table in tables.items():
        assert list(table.index) == ["perfect", "high", "zero"] and list(table["rank"]) == [1, 2, 3]
        assert table.loc["perfect", "accuracy"] == pytest.approx(1.0)
        assert table.loc["high", "accuracy"] == pytest.approx(0.9, abs=1e-4)
        assert table.loc["zero", "accuracy"] == pytest.approx(0.0, abs=1e-4)
        assert table.loc["high", "bias_pct"] == pytest.approx(-10.0, abs=1e-3)  # forecast too high: negative bias


def test_equal_models_share_a_rank(setup):
    S, res, months = setup
    res = res.assign(p_copy=res["p_high"])
    table = score_models_by_level(S, res, ["high", "copy", "perfect"], months)["sku"]
    assert table.loc["high", "rank"] == table.loc["copy", "rank"] == 2 and table.loc["perfect", "rank"] == 1


def test_aggregating_errors_that_cancel_helps_the_upper_levels(setup):
    S, res, months = setup
    sign = np.where(res["series"].to_numpy() % 2 == 0, 1.0, -1.0)
    res = res.assign(p_noise=res["y"] + sign * 0.5 * res["y"].where(res["y"] > 0, 0))
    tables = score_models_by_level(S, res, ["noise"], months)
    assert tables["sku"].loc["noise", "accuracy"] < tables["planta"].loc["noise", "accuracy"] < tables["total"].loc["noise", "accuracy"] + 1e-9
