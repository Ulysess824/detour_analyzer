"""Weekly series: complete weeks only, zeros after the first consumption, sums equal the daily data, SKU matching."""

import pandas as pd
import pytest

from src.experiments.weekly_utils import complete_weeks, match_planner_sku, weekly_series


def test_complete_weeks_drop_partial_first_and_last_week():
    # Tue 2024-01-02 .. Wed 2024-01-17: the week of Mon 01-01 starts before the data, the week of Mon 01-15 ends after it
    mondays = complete_weeks(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-17"))
    assert list(mondays) == [pd.Timestamp("2024-01-08")]


def test_complete_weeks_keep_a_week_that_ends_exactly_on_the_last_day():
    mondays = complete_weeks(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-21"))  # Monday .. Sunday
    assert list(mondays) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")]


def _master():
    days = pd.date_range("2024-01-02", "2024-02-07")  # Tue .. Wed
    other = pd.DataFrame({"planta": "P", "sku": "OTHER", "fecha": days, "consumo": 1.0})  # sets the date range of the master
    target = pd.DataFrame({"planta": "P", "sku": "A/01/100gsm/2300mm/1200-1400",
                           "fecha": pd.to_datetime(["2024-01-17", "2024-01-18", "2024-01-31", "2024-02-06"]),
                           "consumo": [2.0, 3.0, 4.0, 9.0]})
    return pd.concat([other, target], ignore_index=True)


def test_weekly_series_starts_at_first_consumption_and_fills_zero_weeks():
    s = weekly_series(_master(), "P", "A/01/100gsm/2300mm/1200-1400")
    # complete weeks: 01-08, 01-15, 01-22, 01-29 (the week of 02-05 ends 02-11 > last day 02-07)
    assert list(s.index) == [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-22"), pd.Timestamp("2024-01-29")]
    assert list(s.values) == [5.0, 0.0, 4.0]


def test_weekly_series_drops_the_partial_last_week_and_matches_daily_sums_elsewhere():
    master = _master()
    s = weekly_series(master, "P", "OTHER")
    days = master[(master["sku"] == "OTHER") & (master["fecha"] >= "2024-01-08") & (master["fecha"] <= "2024-02-04")]
    assert s.sum() == days["consumo"].sum()
    assert (s == 7.0).all()  # a full week of 1.0 per day, Sundays included


def test_weekly_series_unknown_series_stops():
    with pytest.raises(ValueError):
        weekly_series(_master(), "P", "NOPE")


def test_match_planner_sku_ignores_core_width_and_extra_part():
    skus = ["TSL/01/100gsm/2300mm/1200-1400", "TSL/01/100gsm/2200mm/1200-1400", "TSL/01/125gsm/2300mm/1200-1450", "KS/01/100gsm/2300mm/1200-1400"]
    assert match_planner_sku(skus, "TSL/01/1/100gsm/2300mm") == "TSL/01/100gsm/2300mm/1200-1400"
    assert match_planner_sku(skus, "TSL/01/100gsm/2300mm") == "TSL/01/100gsm/2300mm/1200-1400"


def test_match_planner_sku_needs_exactly_one_match():
    skus = ["TSL/01/100gsm/2300mm/1200-1400", "TSL/02/100gsm/2300mm/1200-1400"]
    with pytest.raises(ValueError):
        match_planner_sku(skus, "TSL/01/100gsm/2300mm")
    with pytest.raises(ValueError):
        match_planner_sku(skus, "ZZZ/01/100gsm/2300mm")
    with pytest.raises(ValueError):
        match_planner_sku(skus, "TSL/01")
