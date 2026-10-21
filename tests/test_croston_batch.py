"""Croston over several series: pooled and per-series accuracy, sign test, bootstrap over series, criteria, substitute SKUs."""

import numpy as np
import pandas as pd
import pytest

from src.experiments import croston_batch as cbt
from src.experiments import croston_scores as cs
from src.experiments.weekly_utils import match_planner_sku


def _table():
    # series A real 10 and 10; series B real 2 and 2. Model f is exact on A and off by 2 on B; g is off by 1 everywhere.
    return pd.DataFrame({"series": ["A", "A", "B", "B"], "month": ["m1", "m2", "m1", "m2"], "real": [10.0, 10.0, 2.0, 2.0],
                         "f": [10.0, 10.0, 0.0, 0.0], "g": [9.0, 11.0, 3.0, 1.0], "ml_mean": [10.0, 8.0, 2.0, 2.0]})


def test_per_series_accuracy_by_hand():
    out = cbt.per_series_accuracy(_table())
    assert out.loc["A", "f"] == 1.0 and out.loc["B", "f"] == 0.0
    assert out.loc["A", "g"] == pytest.approx(0.9) and out.loc["B", "g"] == pytest.approx(0.5)


def test_per_series_accuracy_leaves_out_rows_with_a_missing_forecast():
    t = _table()
    t.loc[0, "f"] = np.nan
    assert cbt.per_series_accuracy(t).loc["A", "f"] == 1.0  # month 1 of A is dropped for every model; month 2 is exact
    assert cbt.per_series_accuracy(t).loc["A", "g"] == pytest.approx(0.9)  # A: month 2 only, real 10 against 11 -> 0.9


def test_win_count_and_sign_test():
    ps = cbt.per_series_accuracy(_table())
    assert cbt.win_count(ps, "f", "g") == 1 and cbt.win_count(ps, "g", "f") == 1
    assert cbt.sign_test_p(1, 2) == 1.0
    assert cbt.sign_test_p(10, 10) == pytest.approx(2 / 1024)
    assert cbt.sign_test_p(8, 15) == pytest.approx(1.0)  # 8 of 15 is not different from a coin


def test_pooled_scores_are_dominated_by_large_series_and_weekly_scores_handle_several_series():
    pooled = cs.monthly_scores(_table())
    assert pooled.loc["f", "wape"] == pytest.approx(4 / 24, abs=1e-4)
    assert pooled.loc["g", "wape"] == pytest.approx(4 / 24, abs=1e-4)
    dates = pd.to_datetime(["2026-01-05", "2026-01-12"])
    rows = [{"series": s, "target": t, "h": 1, "model": m, "forecast": 1.0, "lower": np.nan, "upper": np.nan, "real": 2.0}
            for s in ("A", "B") for t in dates for m in ("naive", "ses")]
    assert set(cs.weekly_scores(pd.DataFrame(rows), 1)["n"]) == {4}  # 2 series x 2 dates, not 2 dates


def test_bootstrap_difference_resamples_series_and_signs_correctly():
    t = pd.DataFrame({"series": list("AABBCC"), "real": [10.0] * 6, "a": [10.0] * 6, "b": [5.0] * 6})
    out = cbt.bootstrap_accuracy_diff(t, "a", "b", n_boot=500, seed=1)
    assert out["diff"] == pytest.approx(0.5) and out["low90"] == pytest.approx(0.5) and out["high90"] == pytest.approx(0.5)  # a is better everywhere
    assert out["series"] == 3
    assert cbt.bootstrap_accuracy_diff(t, "b", "a", n_boot=500, seed=1)["diff"] == pytest.approx(-0.5)
    assert cbt.bootstrap_accuracy_diff(t, "a", "b", n_boot=500, seed=1) == out  # deterministic


def test_batch_decision_needs_the_same_model_in_the_three_criteria():
    week = pd.DataFrame({"accuracy": {"naive": 0.3, "mean6": 0.4, "mean24": 0.45, "ses": 0.44, "croston": 0.5, "modified": 0.5}})
    month = pd.DataFrame({"accuracy": {"ml_mean": 0.7, "croston": 0.75, "modified": 0.75}})
    per_series = pd.DataFrame({"ml_mean": [0.7, 0.7, 0.7, 0.7], "croston": [0.9, 0.9, 0.9, 0.1], "modified": [0.9, 0.1, 0.1, 0.1]})
    cover = pd.DataFrame({"coverage": {"croston": 0.9, "modified": 0.9}})
    d = cbt.batch_decision(week, month, per_series, cover)
    assert d["criterion_1"] == ["croston", "modified"]
    assert d["criterion_2"] == ["croston"]  # modified wins in 1 of 4 series, less than half
    assert d["meets_all"] == ["croston"] and d["adopted"]
    assert not cbt.batch_decision(week, month, per_series, pd.DataFrame({"coverage": {"croston": 0.5}}))["adopted"]


def test_match_planner_sku_can_ignore_a_substitute():
    skus = ["HP2/01/155gsm/1990mm/1200-1450", "HP2/01/155gsm/1990mm/1200-1450/Subst.", "K/01/110gsm/2000mm/1200-1450"]
    with pytest.raises(ValueError):
        match_planner_sku(skus, "HP2/01/1/155gsm/1990mm")
    assert match_planner_sku(skus, "HP2/01/1/155gsm/1990mm", ignore_substitutes=True) == "HP2/01/155gsm/1990mm/1200-1450"
    only_subst = ["X/01/100gsm/2000mm/1/Subst.", "X/01/100gsm/2000mm/2/Subst."]
    with pytest.raises(ValueError):
        match_planner_sku(only_subst, "X/01/1/100gsm/2000mm", ignore_substitutes=True)  # still ambiguous


def test_run_series_returns_three_tables_tagged_with_the_series():
    days = pd.bdate_range("2025-01-06", "2026-03-31", freq="C", weekmask="Mon Tue Wed Thu Fri Sat")
    rng = np.random.default_rng(0)
    vals = np.where(rng.random(len(days)) < 0.4, rng.gamma(3.0, 1.0, len(days)), 0.0)
    df = pd.DataFrame({"planta": "P", "sku": "A", "fecha": days, "consumo": vals})
    df = df[df["consumo"] > 0]
    filler = pd.DataFrame({"planta": "P", "sku": "Z", "fecha": pd.to_datetime(["2025-01-06", "2026-03-31"]), "consumo": 1.0})
    df = pd.concat([df, filler], ignore_index=True)
    months = pd.PeriodIndex(["2026-01", "2026-02", "2026-03"], freq="M")
    out = cbt.run_series(df, "P", "A", pd.Series([50.0, 50.0, 50.0], index=months), "2026-01-01", draws=100, seed=1)
    assert set(out) == {"daily", "weeks", "months"}
    assert all((t["series"] == "P|A").all() for t in out.values())
    assert list(out["months"]["month"]) == ["2026-01", "2026-02", "2026-03"]
    assert out["daily"]["target"].min() >= pd.Timestamp("2026-01-01")
