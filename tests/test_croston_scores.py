"""Scores of the weekly Croston backtest: common weeks, coverage by hand, monthly table and the pre-set criteria."""

import pandas as pd
import pytest

from src.experiments import croston_scores as cs

T = pd.date_range("2026-01-05", periods=4, freq="7D")


def _weekly():
    rows = []
    real = [10.0, 0.0, 5.0, 5.0]
    forecasts = {"naive": [8.0, 10.0, 0.0, 5.0], "mean4": [9.0, 9.0, 9.0, 9.0], "croston": [10.0, 4.0, 5.0, 5.0],
                 "modified": [10.0, 4.0, 5.0, 5.0]}
    for model, f in forecasts.items():
        for i, t in enumerate(T):
            if model == "croston" and i == 0:
                continue  # a model without a forecast in the first week: the table is cut to the weeks every model has
            lower, upper = (f[i] - 3, f[i] + 3) if model == "modified" else (float("nan"), float("nan"))
            rows.append({"origin": t - pd.Timedelta(days=7), "target": t, "h": 1, "model": model, "forecast": f[i],
                         "lower": lower, "upper": upper, "real": real[i]})
    return pd.DataFrame(rows)


def test_weekly_scores_use_only_the_weeks_every_model_forecasts():
    table = cs.weekly_scores(_weekly(), 1)
    assert set(table["n"]) == {3}
    # naive over weeks 2..4: |0-10| + |5-0| + |5-5| = 15 on a real total of 10
    assert table.loc["naive", "wape"] == pytest.approx(1.5, abs=1e-4)
    assert table.loc["croston", "wape"] == pytest.approx(4 / 10, abs=1e-4)
    assert table.loc["croston", "accuracy"] == pytest.approx(0.6, abs=1e-4)
    assert table.index[0] in ("croston", "modified")  # ranked by WAPE


def test_interval_scores_coverage_and_width_by_hand():
    out = cs.interval_scores(_weekly(), 1)
    # modified: forecasts 10,4,5,5 with +-3 -> [7,13] [1,7] [2,8] [2,8]; real 10,0,5,5 -> inside 3 of 4
    assert out.loc["modified", "coverage"] == pytest.approx(0.75)
    assert out.loc["modified", "mean_width"] == pytest.approx(6.0)
    assert out.loc["modified", "weeks_with_zero_lower"] == 0
    assert list(out.index) == ["modified"]


def test_monthly_scores_rank_the_forecast_columns_against_real():
    monthly = pd.DataFrame({"month": ["2026-01", "2026-02"], "origin": T[:2], "real": [10.0, 20.0], "ml_mean": [9.0, 21.0], "naive": [5.0, 5.0]})
    out = cs.monthly_scores(monthly)
    assert out.loc["ml_mean", "wape"] == pytest.approx(2 / 30, abs=1e-4)
    assert out.loc["naive", "wape"] == pytest.approx(20 / 30, abs=1e-4)
    assert out.index[0] == "ml_mean"


def _scores(accuracy):
    return pd.DataFrame({"accuracy": accuracy})


def test_criteria_need_a_croston_model_above_naive_and_mean4_with_coverage_in_range():
    scores = _scores({"naive": 0.30, "mean4": 0.40, "croston": 0.45, "modified": 0.50, "sb0": 0.35})
    intervals = pd.DataFrame({"coverage": {"modified": 0.95, "croston_model": 0.50}})
    d = cs.decision_criteria(scores, intervals)
    assert d["bar_accuracy"] == 0.40
    assert d["beat_naive_and_mean4"] == ["croston", "modified"]
    assert d["with_coverage_in_range"] == ["modified"]
    assert d["criterion_1"] and d["criterion_2"] and d["meets_preset_criteria"]


def test_criteria_fail_when_no_croston_model_beats_the_references_or_covers():
    none_beats = cs.decision_criteria(_scores({"naive": 0.4, "mean4": 0.5, "croston": 0.45}), pd.DataFrame({"coverage": {}}))
    assert not none_beats["criterion_1"] and not none_beats["meets_preset_criteria"]
    low_coverage = cs.decision_criteria(_scores({"naive": 0.2, "mean4": 0.3, "modified": 0.5}), pd.DataFrame({"coverage": {"modified": 0.6}}))
    assert low_coverage["criterion_1"] and not low_coverage["criterion_2"] and not low_coverage["meets_preset_criteria"]


def test_references_do_not_count_as_croston_models():
    assert not set(cs.CROSTON_MODELS) & set(cs.REFERENCES)
    assert not any(m.startswith("croston_a") or m.endswith("_a0.5") for m in cs.CROSTON_MODELS)


def test_wide_table_has_one_row_per_week_with_real_forecasts_and_limits():
    wide = cs.wide_table(_weekly(), 1)
    assert list(wide.index) == list(T)
    assert list(wide["real"]) == [10.0, 0.0, 5.0, 5.0]
    assert wide.loc[T[1], "naive"] == 10.0
    assert wide.loc[T[0], "modified_lower"] == 7.0 and wide.loc[T[0], "modified_upper"] == 13.0
    assert "croston_model_lower" not in wide.columns  # no interval for models not in the table
    assert pd.isna(wide.loc[T[0], "croston"])  # the week the model skipped


def test_rmse_column_only_when_asked():
    assert "rmse" not in cs.weekly_scores(_weekly(), 1).columns
    out = cs.weekly_scores(_weekly(), 1, rmse=True)
    assert out.loc["mean4", "rmse"] == pytest.approx(((81 + 16 + 16) / 3) ** 0.5, abs=1e-4)  # weeks 2..4: real 0, 5, 5 against 9


def test_decision_criteria_with_other_references():
    scores = _scores({"naive": 0.3, "mean6": 0.45, "croston": 0.5})
    assert cs.decision_criteria(scores, pd.DataFrame({"coverage": {"croston": 0.9}}), references=("naive", "mean6"))["bar_accuracy"] == 0.45


def test_daily_decision_needs_the_same_model_in_all_three_criteria():
    week = _scores({"naive": 0.3, "mean6": 0.4, "mean24": 0.45, "ses": 0.44, "croston": 0.5, "modified": 0.5, "sb0": 0.2})
    month = _scores({"ml_mean": 0.7, "croston": 0.75, "modified": 0.6, "sb0": 0.9})
    cover = pd.DataFrame({"coverage": {"croston": 0.9, "modified": 0.9, "croston_model": 0.9}})
    d = cs.daily_decision(week, month, cover)
    assert d["week_bar_accuracy"] == 0.45
    assert d["criterion_1"] == ["croston", "modified"] and d["criterion_2"] == ["croston", "sb0"] and d["criterion_3"] == ["croston", "modified", "croston_model"]
    assert d["meets_all"] == ["croston"] and d["adopted"]
    none = cs.daily_decision(week, month, pd.DataFrame({"coverage": {"croston": 0.5}}))
    assert none["meets_all"] == [] and not none["adopted"]
