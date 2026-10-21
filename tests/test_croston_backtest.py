"""Weekly backtest: no look-ahead, origins and targets line up, monthly scaling, deterministic."""

import numpy as np
import pandas as pd
import pytest

from src.experiments import croston_backtest as cb

WEEKS = pd.date_range("2025-07-07", periods=40, freq="7D")


def _series(seed=0, n=40):
    rng = np.random.default_rng(seed)
    y = rng.gamma(3.0, 2.0, n)
    y[[5, 11, 20]] = 0.0
    return pd.Series(y, index=WEEKS[:n])


def test_origin_forecasts_include_every_model_and_intervals_only_where_defined():
    rows = cb.origin_forecasts(_series().to_numpy()[:30], 1, draws=300, seed=1)
    table = pd.DataFrame(rows).set_index("model")
    assert {"naive", "mean4", "mean8", "ses", "croston", "sb0", "sb1", "modified", "modified_log", "croston_model", "log_croston_model",
            "croston_a0.1", "croston_a0.5", "sb0_a0.5", "sb1_a0.5"} <= set(table.index)
    assert table.loc[list(cb.MODELS_WITH_INTERVAL), ["lower", "upper"]].notna().all().all()
    assert table.drop(index=list(cb.MODELS_WITH_INTERVAL))[["lower", "upper"]].isna().all().all()
    assert (table["lower"].dropna() <= table.loc[table["lower"].notna(), "upper"]).all()


def test_series_with_too_few_demands_only_gets_the_simple_models():
    rows = cb.origin_forecasts(np.array([0.0, 0.0, 4.0, 0.0, 6.0]), 1, draws=100, seed=1)
    assert {r["model"] for r in rows} == {"naive", "mean4", "mean8", "ses"}


def test_rolling_weekly_origin_is_h_weeks_before_the_target_and_uses_only_the_past():
    s = _series()
    table = cb.rolling_weekly(s, test_start=str(WEEKS[30].date()), horizons=(1, 4), draws=200, seed=2)
    assert (table["target"] - table["origin"] == pd.to_timedelta(7 * table["h"], unit="D")).all()
    assert table["target"].min() == WEEKS[30]
    # the forecast of a target does not change when every value from the target on is replaced
    changed = s.copy()
    changed.iloc[30:] = 1e6
    again = cb.rolling_weekly(changed, test_start=str(WEEKS[30].date()), horizons=(1,), draws=200, seed=2)
    first = table[(table["target"] == WEEKS[30]) & (table["h"] == 1)].set_index("model")["forecast"]
    assert again[again["target"] == WEEKS[30]].set_index("model")["forecast"].equals(first)
    assert (table["real"] == s.reindex(table["target"]).to_numpy()).all()


def test_rolling_weekly_is_deterministic():
    s = _series()
    a = cb.rolling_weekly(s, str(WEEKS[34].date()), horizons=(1,), draws=200, seed=3)
    b = cb.rolling_weekly(s, str(WEEKS[34].date()), horizons=(1,), draws=200, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_month_business_days_excludes_sundays():
    assert cb.month_business_days(pd.Period("2026-02")) == 24  # 28 days, 4 Sundays
    assert cb.month_business_days(pd.Period("2026-01")) == 27  # 31 days, 4 Sundays (4, 11, 18, 25)


def test_monthly_comparison_scales_the_flat_forecast_and_uses_the_last_week_before_the_month():
    s = pd.Series(np.full(40, 6.0), index=WEEKS)  # constant: every model forecasts 6 per week
    months = pd.PeriodIndex(["2026-01", "2026-02"], freq="M")
    out = cb.monthly_comparison(s, pd.Series([100.0, 90.0], index=months), pd.Series([95.0, 85.0], index=months), draws=100, seed=1)
    assert list(out["month"]) == ["2026-01", "2026-02"]
    assert out.loc[0, "origin"] == pd.Timestamp("2025-12-22")  # the week of 12-29 ends on 2026-01-04, inside the month
    assert out.loc[1, "origin"] == pd.Timestamp("2026-01-19")  # the week of 01-26 ends on 02-01, inside the month
    assert out.loc[0, "naive"] == pytest.approx(6.0 * 27 / 6)
    assert out.loc[1, "mean4"] == pytest.approx(6.0 * 24 / 6)
    assert list(out["ml_mean"]) == [95.0, 85.0]


def test_monthly_comparison_skips_months_without_enough_history():
    s = pd.Series(np.full(5, 6.0), index=WEEKS[:5])
    months = pd.PeriodIndex(["2025-07", "2026-01"], freq="M")
    out = cb.monthly_comparison(s, pd.Series([1.0, 1.0], index=months), pd.Series([1.0, 1.0], index=months), draws=50, seed=1)
    assert list(out["month"]) == ["2026-01"]
