"""Production strategies: they reproduce the backtest code on the same information and read nothing after the cutoff."""

import numpy as np
import pandas as pd
import pytest

from src.engine.feature_utils import make_frame
from src.engine.inference_utils import forecast_next_month
from src.engine.panel_utils import build_panel
from src.engine.month_in_progress import complete_calendar
from src.engine.strategies import forecast_with_strategy, iterated_forecast, nowcast_input_forecast

MONTH = 27  # 2025-04 in the toy table: the month in progress; the last complete month is 26
ORIGIN = MONTH - 1
COLUMNS = ["planta", "sku", "mes", "lgbm", "xgb", "rf", "ml_mean", "naive", "mes_origen", "estrategia"]


def _in_progress(toy_df, day):
    r"""Table as exported on day `day` of the month in progress."""
    last = pd.Timestamp("2023-01-01") + pd.DateOffset(months=MONTH, days=day - 1)
    return toy_df[toy_df["fecha"] <= last]


def _by_key(table, column="ml_mean"):
    return table.set_index(["planta", "sku"])[column].sort_index()


@pytest.fixture(scope="module")
def full(toy_panel):
    return toy_panel


@pytest.fixture(scope="module")
def partial_panel(toy_df):
    return build_panel(_in_progress(toy_df, 20))


def test_the_toy_panel_is_set_up_as_expected(partial_panel):
    assert partial_panel.monthly.shape[1] == MONTH + 1 and partial_panel.labels[MONTH] == "2025-04"


def test_the_calendar_of_a_month_in_progress_is_completed_like_the_full_one(full, partial_panel):
    done = complete_calendar(partial_panel)
    assert done.dates[-1] == full.dates[done.dates.size - 1] and done.dates.size <= full.dates.size
    assert done.calendar_days[MONTH] == full.calendar_days[MONTH] and done.business_days[MONTH] == full.business_days[MONTH]
    assert done.daily.shape == (partial_panel.daily.shape[0], done.dates.size) and (done.daily[:, partial_panel.daily.shape[1] :] == 0).all()
    assert np.array_equal(done.daily[:, : partial_panel.daily.shape[1]], partial_panel.daily)
    assert complete_calendar(full) is full  # a complete calendar is left as it is


def test_nowcast_media_reproduces_the_experiment_and_ignores_later_days(toy_df, full, partial_panel):
    day = 12
    got = forecast_with_strategy(partial_panel, {}, "nowcast_media", ORIGIN, day)
    expected = nowcast_input_forecast(full, make_frame(full, 1), ORIGIN, {}, day)
    assert np.allclose(np.sort(got["ml_mean"].to_numpy()), np.sort(expected["ml_mean"].to_numpy()))
    other = forecast_with_strategy(build_panel(_in_progress(toy_df, 25)), {}, "nowcast_media", ORIGIN, day)
    assert np.allclose(_by_key(got), _by_key(other))


def test_iterated_reproduces_the_backtest_forecast(full, partial_panel):
    got = forecast_with_strategy(partial_panel, {}, "iterada", ORIGIN)
    expected = iterated_forecast(full, make_frame(full, 1), ORIGIN, {})
    assert np.allclose(np.sort(got["ml_mean"].to_numpy()), np.sort(expected.to_numpy()))
    assert (got["mes"] == "2025-05").all()


def test_one_month_forecast_of_a_complete_panel_is_the_existing_forecast(full):
    n = full.monthly.shape[1]
    got = forecast_with_strategy(full, {}, "h1", n - 1)
    base = forecast_next_month(full, {})
    assert list(got.columns) == COLUMNS and (got["estrategia"] == "h1").all()
    assert got[list(base.columns)].equals(base)


def test_one_month_forecast_with_the_month_in_progress_targets_that_month(partial_panel):
    got = forecast_with_strategy(partial_panel, {}, "h1", ORIGIN)
    assert (got["mes"] == "2025-04").all() and (got["mes_origen"] == "2025-03").all()


def test_the_nowcast_needs_the_month_in_progress_and_a_cutoff(full, partial_panel):
    with pytest.raises(ValueError):
        forecast_with_strategy(partial_panel, {}, "nowcast_media", ORIGIN)
    with pytest.raises(ValueError):
        forecast_with_strategy(full, {}, "nowcast_media", full.monthly.shape[1] - 1, 15)
    with pytest.raises(ValueError, match="unknown strategy"):
        forecast_with_strategy(partial_panel, {}, "otra", ORIGIN)
