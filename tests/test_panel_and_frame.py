"""The panel adds up and the feature frame never looks past its origin."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.utils.daily_utils import expand_to_days
from src.utils.feature_utils import SERIES_FEATURES, make_frame


def test_daily_matrix_adds_up_to_monthly_totals(toy_panel):
    S = toy_panel
    by_month = np.add.reduceat(S.daily, S.month_first_date, axis=1)
    assert np.allclose(by_month, np.nan_to_num(S.monthly, nan=0.0), atol=0.05)


def test_series_is_nan_before_its_first_month_and_not_after(toy_panel):
    S = toy_panel
    for i in range(S.monthly.shape[0]):
        first = S.first_month[i]
        assert np.isnan(S.monthly[i, :first]).all()
        assert not np.isnan(S.monthly[i, first:]).any()


def test_planta_and_total_are_sums_of_series(toy_panel):
    S = toy_panel
    assert np.allclose(S.planta_monthly.sum(axis=0), S.total_monthly)
    assert np.allclose(S.total_monthly, np.nansum(S.monthly, axis=0))


def test_only_series_alive_at_the_origin_have_rows(toy_panel):
    S = toy_panel
    frame = make_frame(S, 1)
    for origin, block in frame.groupby("o"):
        assert set(block["series"]) == set(np.flatnonzero(S.first_month <= origin))


@pytest.mark.parametrize("horizon", [1, 3])
def test_features_do_not_depend_on_data_after_the_origin(toy_panel, horizon):
    S = toy_panel
    n = S.monthly.shape[1]
    reference = make_frame(S, horizon)
    rng = np.random.default_rng(0)
    for origin in (12, 20):
        S2 = copy.deepcopy(S)
        after = slice(origin + 1, n)
        noise = rng.uniform(0, 1e4, S2.monthly[:, after].shape)
        S2.monthly[:, after] = np.where(np.isnan(S2.monthly[:, after]), np.nan, noise)
        S2.active_days[:, after] = np.where(np.isnan(S2.active_days[:, after]), np.nan, 7.0)
        zero_filled = np.nan_to_num(S2.monthly, nan=0.0)
        S2.planta_monthly[:] = 0
        np.add.at(S2.planta_monthly, S2.planta_code, zero_filled)
        S2.total_monthly = zero_filled.sum(axis=0)

        changed = make_frame(S2, horizon)
        a, b = reference[reference["o"] == origin], changed[changed["o"] == origin]
        columns = SERIES_FEATURES + [c for c in a.columns if c.startswith("p_")]
        for column in columns:
            assert np.allclose(a[column].to_numpy(float), b[column].to_numpy(float), equal_nan=True), column
        assert not np.allclose(a["y"].to_numpy(float), b["y"].to_numpy(float), equal_nan=True)  # the target did change


def test_target_month_is_origin_plus_horizon(toy_panel):
    frame = make_frame(toy_panel, 3)
    assert ((frame["t"] - frame["o"]) == 3).all()
    S = toy_panel
    assert np.allclose(frame["y"], S.monthly[frame["series"], frame["t"]], equal_nan=True)


def test_daily_expansion_adds_up_to_the_monthly_target(toy_panel):
    S = toy_panel
    rows = make_frame(S, 1).query("o == 20").reset_index(drop=True)
    X, y, position = expand_to_days(S, rows, SERIES_FEATURES, None, 0)
    per_row = pd.Series(y).groupby(position).sum().reindex(range(len(rows))).to_numpy()
    assert np.allclose(per_row, rows["y"].to_numpy(), atol=1e-3)
    assert (pd.Series(position).value_counts().sort_index().to_numpy() == S.calendar_days[rows["t"].to_numpy()]).all()
