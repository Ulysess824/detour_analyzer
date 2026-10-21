"""Daily series and its week and month aggregates: Sunday folding, zero fill, origins one working day before the period, scaling."""

import numpy as np
import pandas as pd
import pytest

from src.experiments import daily_utils as du


def _master(rows, last="2024-03-31"):
    """Rows of one SKU plus a filler series that sets the date range of the master."""
    other = pd.DataFrame({"planta": "P", "sku": "OTHER", "fecha": pd.to_datetime(["2024-01-02", last]), "consumo": 1.0})
    target = pd.DataFrame([{"planta": "P", "sku": "A", "fecha": pd.Timestamp(d), "consumo": v} for d, v in rows])
    return pd.concat([other, target], ignore_index=True)


def test_daily_series_zero_fills_working_days_and_has_no_sundays():
    s = du.daily_series(_master([("2024-01-03", 2.0), ("2024-01-06", 3.0)]), "P", "A")
    assert s.index[0] == pd.Timestamp("2024-01-03") and s.index[-1] == pd.Timestamp("2024-03-30")  # last Saturday of the data
    assert (s.index.dayofweek != 6).all()
    assert s["2024-01-04"] == 0.0 and s["2024-01-06"] == 3.0
    assert s.sum() == 5.0


def test_sunday_consumption_goes_to_the_next_monday_and_totals_are_kept():
    s = du.daily_series(_master([("2024-01-05", 1.0), ("2024-01-07", 4.0), ("2024-01-08", 2.0)]), "P", "A")  # Fri, Sun, Mon
    assert s["2024-01-08"] == 6.0 and s["2024-01-05"] == 1.0
    assert s.sum() == 7.0


def test_a_sunday_on_the_last_day_of_the_data_is_dropped():
    s = du.daily_series(_master([("2024-01-03", 1.0), ("2024-03-31", 5.0)], last="2024-03-31"), "P", "A")  # 2024-03-31 is a Sunday
    assert s.index[-1] == pd.Timestamp("2024-03-30")
    assert s.sum() == 1.0


def test_first_consumption_on_a_sunday_starts_the_series_on_monday():
    s = du.daily_series(_master([("2024-01-07", 2.0)]), "P", "A")
    assert s.index[0] == pd.Timestamp("2024-01-08") and s.iloc[0] == 2.0


def test_unknown_series_stops():
    with pytest.raises(ValueError):
        du.daily_series(_master([("2024-01-03", 1.0)]), "P", "NOPE")


DAYS = pd.bdate_range("2025-07-07", "2026-03-28", freq="C", weekmask="Mon Tue Wed Thu Fri Sat")


def _constant():
    return pd.Series(np.full(len(DAYS), 2.0), index=DAYS)


def test_week_comparison_forecasts_six_days_from_the_saturday_before():
    weeks = pd.date_range("2025-12-01", "2026-03-23", freq="7D")  # Mondays
    real = pd.Series(12.0, index=weeks)
    out = du.week_comparison(_constant(), real, "2026-01-05", draws=100, seed=1)
    assert out["week"].min() == pd.Timestamp("2026-01-05")
    assert (out["origin"] == out["week"] - pd.Timedelta(days=2)).all()  # Saturday before the Monday
    assert out["naive"].eq(12.0).all() and out["mean6"].eq(12.0).all() and out["ses"].eq(12.0).all()
    assert (out["real"] == 12.0).all()


def test_month_comparison_scales_by_the_working_days_of_the_month_from_the_last_day_before_it():
    months = pd.PeriodIndex(["2026-01", "2026-02"], freq="M")
    out = du.month_comparison(_constant(), pd.Series([60.0, 50.0], index=months), pd.Series([55.0, 45.0], index=months), draws=100, seed=1)
    assert list(out["origin"]) == [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-31")]
    assert out.loc[0, "naive"] == pytest.approx(2.0 * 27)  # January 2026: 27 non-Sunday days
    assert out.loc[1, "mean24"] == pytest.approx(2.0 * 24)  # February 2026: 24
    assert list(out["ml_mean"]) == [55.0, 45.0]


def test_comparisons_skip_origins_with_too_little_history():
    short = _constant().iloc[:20]
    months = pd.PeriodIndex(["2025-07"], freq="M")
    assert du.month_comparison(short, pd.Series([1.0], index=months), pd.Series([1.0], index=months), draws=50).empty
    assert du.week_comparison(short, pd.Series([1.0], index=pd.to_datetime(["2025-07-21"])), "2025-07-01", draws=50).empty


def test_ml_mean_by_month_reads_the_cache_and_checks_it_against_the_panel(tmp_path):
    from src.engine.panel_utils import build_panel

    days = pd.date_range("2024-01-02", "2024-06-29")
    df = pd.concat([pd.DataFrame({"planta": "P", "sku": s, "fecha": days, "consumo": v}) for s, v in (("A", 1.0), ("B", 2.0))], ignore_index=True)
    S = build_panel(df)
    series = S.keys.index(("P", "A"))
    rows = [{"series": series, "o": t - 1, "t": t, "planta_code": S.planta_code[series], "y": S.monthly[series, t],
             "p_lgbm": 10.0 + t, "p_xgb": 20.0 + t, "p_rf": 30.0 + t, "p_naive": 0.0} for t in (3, 4, 5)]
    path = tmp_path / "cache.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    ml = du.ml_mean_by_month(path, "P", "A", df, "2024-05")
    assert list(ml.index) == [pd.Period("2024-05", "M"), pd.Period("2024-06", "M")]  # t = 4, 5; earlier months dropped
    assert list(ml.values) == [24.0, 25.0]  # mean of 14, 24, 34 and 15, 25, 35

    broken = pd.DataFrame(rows)
    broken["y"] += 1.0  # actuals that do not match the panel: the cache is refused
    broken.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        du.ml_mean_by_month(path, "P", "A", df, "2024-05")


def test_monday_to_friday_series_folds_the_weekend_into_the_next_monday():
    rows = [("2024-01-05", 1.0), ("2024-01-06", 4.0), ("2024-01-07", 2.0), ("2024-01-08", 3.0)]  # Fri, Sat, Sun, Mon
    s = du.daily_series(_master(rows), "P", "A", weekdays=5)
    assert (s.index.dayofweek < 5).all()
    assert s["2024-01-05"] == 1.0 and s["2024-01-08"] == 9.0  # 3 + Saturday 4 + Sunday 2
    assert s.sum() == 10.0
    assert s.index[-1] == pd.Timestamp("2024-03-29")  # Friday: the last Saturday and Sunday of the data have no Monday


def test_default_calendar_is_still_monday_to_saturday():
    s = du.daily_series(_master([("2024-01-06", 3.0)]), "P", "A")
    assert s["2024-01-06"] == 3.0 and (s.index.dayofweek < 6).all()


def test_month_comparison_with_a_profile_adds_dow_totals_that_change_only_through_the_weekday_mix():
    days = pd.bdate_range("2025-07-07", "2026-03-27")  # Monday to Friday
    values = np.where(days.dayofweek == 4, 0.0, 5.0)  # nothing on Fridays
    s = pd.Series(values, index=days)
    months = pd.PeriodIndex(["2026-01", "2026-02"], freq="M")
    out = du.month_comparison(s, pd.Series([100.0, 90.0], index=months), pd.Series([95.0, 85.0], index=months), draws=100, seed=1,
                              windows=(5, 20), weekdays=5, shrink=0.0)
    assert "naive_dow" in out.columns and "croston_dow" in out.columns
    # January 2026 has 22 working days: 4 Fridays (2, 9, 16, 23, 30 -> 5) so 5 Fridays and 17 other days
    jan_fridays = 5
    assert out.loc[0, "mean5_dow"] != out.loc[0, "mean5"]  # five Fridays: the weekday mix is not the average week
    assert out.loc[0, "mean5_dow"] / out.loc[0, "mean5"] == pytest.approx((22 - jan_fridays) * 1.25 / 22, rel=1e-6)
    assert out.loc[1, "mean5_dow"] / out.loc[1, "mean5"] == pytest.approx((20 - 4) * 1.25 / 20, rel=1e-6)  # February 2026: 4 Fridays in 20 days
