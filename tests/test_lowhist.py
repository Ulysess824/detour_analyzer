"""Strategies for SKUs with little history: attributes parse, and nothing after the origin is used."""

import dataclasses

import numpy as np
import pytest

from src.utils.feature_utils import make_frame
from src.utils.lowhist_utils import (
    SKU_FEATURES, add_sku_features, attribute_table, cohort_forecast, cohort_ratio, family_forecast, fit_predict_l1,
    history_months, parse_sku, weighted_median,
)  # fmt: skip


@pytest.mark.parametrize(
    "sku, expected",
    [
        ("K/01/160gsm/2100mm/1200-1450", ("K", 1, 160, 2100, 0)),
        ("HP2/01/160gsm/1790mm/1200-1450/Subst.", ("HP2", 1, 160, 1790, 1)),
        ("KCH/337/135gsm/2800mm/1200-1250", ("KCH", 337, 135, 2800, 0)),
    ],
)
def test_parse_sku(sku, expected):
    p = parse_sku(sku)
    assert (p["type"], p["sub"], p["gsm"], p["width"], p["subst"]) == expected


def test_weighted_median_minimises_the_weighted_absolute_error():
    values, weights = np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 5.0])
    assert weighted_median(values, weights) == 2.0
    assert weighted_median(np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0])) == 0.0


def test_sku_features_follow_each_row(toy_panel):
    attrs = attribute_table(toy_panel)
    frame = add_sku_features(make_frame(toy_panel, 1), attrs)
    assert set(SKU_FEATURES) <= set(frame.columns)
    row = frame.iloc[10]
    assert row["sku_gsm"] == attrs.loc[row["series"], "gsm"] == 100 and row["sku_width"] == 2100


def _hide_after(S, origin):
    r"""Copy of the panel with every month after `origin` replaced by a different value."""
    monthly = S.monthly.copy()
    monthly[:, origin + 1 :] = np.where(np.isnan(monthly[:, origin + 1 :]), np.nan, monthly[:, origin + 1 :] * 5 + 7)
    return dataclasses.replace(S, monthly=monthly)


def test_cohort_ratio_and_family_forecast_ignore_the_future(toy_panel):
    S = toy_panel
    origin = 20
    S2 = _hide_after(S, origin)
    # cohort ratios use only outcomes observed up to the origin
    for age in (1, 2, 3):
        a, b = cohort_ratio(S, origin, age), cohort_ratio(S2, origin, age)
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b)
    attrs = attribute_table(S)
    rows = make_frame(S, 1)
    rows = rows[rows["o"] == origin]
    rows2 = make_frame(S2, 1)
    rows2 = rows2[rows2["o"] == origin]
    assert np.allclose(family_forecast(S, rows, attrs, 1.0), family_forecast(S2, rows2, attrs, 1.0), equal_nan=True)
    assert np.allclose(cohort_forecast(S, rows), cohort_forecast(S2, rows2), equal_nan=True)


def test_rules_only_forecast_skus_with_little_history(toy_panel):
    S = toy_panel
    rows = make_frame(S, 1)
    rows = rows[rows["o"] == S.monthly.shape[1] - 2]
    age = history_months(S, rows)
    for p in (cohort_forecast(S, rows), family_forecast(S, rows, attribute_table(S), 1.0)):
        assert np.isnan(p[age > 3]).all()


def test_median_objective_model_is_never_negative(toy_panel):
    frame = make_frame(toy_panel, 1)
    origin = toy_panel.monthly.shape[1] - 2
    p = fit_predict_l1(frame[frame["t"] <= origin], frame[frame["o"] == origin])
    assert (p >= 0).all() and np.isfinite(p).all()
