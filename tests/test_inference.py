"""The next-month forecast uses only observed months and reproduces the rolling-origin features."""

import numpy as np
import pandas as pd
import pytest

from src.utils.feature_utils import SERIES_FEATURES, make_frame
from src.utils.inference_utils import check_last_month_complete, extend_panel, forecast_next_month
from src.utils.panel_utils import build_panel


def _cut(df, n_months):
    r"""The rows of the first `n_months` months (the toy data starts on 2023-01-02)."""
    return df[df["fecha"] < pd.Timestamp("2023-01-01") + pd.DateOffset(months=n_months)]


def test_extend_panel_adds_empty_months_and_keeps_the_past(toy_panel):
    S = toy_panel
    n = S.monthly.shape[1]
    E = extend_panel(S, 2)
    assert E.monthly.shape[1] == n + 2 and len(E.labels) == n + 2
    assert np.array_equal(E.monthly[:, :n], S.monthly, equal_nan=True)
    assert np.isnan(E.monthly[:, n:]).all() and np.isnan(E.total_monthly[n:]).all()
    assert E.labels[:n] == S.labels
    assert (E.business_days[n:] > 20).all() and (E.business_days[n:] < E.calendar_days[n:]).all()
    assert (E.month_of_year[n:] == ((S.month_of_year[-1] + np.arange(1, 3) - 1) % 12) + 1).all()


def test_features_of_the_future_rows_equal_the_rolling_features(toy_df):
    r"""A month forecast from truncated data has the same features as the row the rolling frame builds for it."""
    n = 29
    full = build_panel(_cut(toy_df, n))
    truncated = build_panel(_cut(toy_df, n - 1))
    rolling = make_frame(full, 1)
    rolling = rolling[rolling["o"] == n - 2]
    future = make_frame(extend_panel(truncated, 1), 1)
    future = future[future["o"] == n - 2]
    key = lambda S, f: [S.keys[s] for s in f["series"]]  # noqa: E731
    a = rolling.assign(key=key(full, rolling)).set_index("key").sort_index()
    b = future.assign(key=key(truncated, future)).set_index("key").sort_index()
    assert list(a.index) == list(b.index)
    assert np.allclose(a[SERIES_FEATURES].to_numpy(float), b[SERIES_FEATURES].to_numpy(float), equal_nan=True)
    assert b["y"].isna().all()  # the target is unknown


def test_forecast_next_month_rows_labels_and_ensemble(toy_panel):
    S = toy_panel
    out = forecast_next_month(S, {})
    alive = int((S.first_month <= S.monthly.shape[1] - 1).sum())
    assert len(out) == alive
    assert out["mes"].iloc[0] == str(pd.Period(S.labels[-1], freq="M") + 1)
    assert (out[["lgbm", "xgb", "rf", "ml_mean"]] >= 0).all().all()
    assert np.allclose(out["ml_mean"], out[["lgbm", "xgb", "rf"]].mean(axis=1))


def test_forecast_ignores_the_unknown_target(toy_panel):
    r"""Changing nothing but the (empty) future month cannot change the forecast: it is deterministic."""
    a = forecast_next_month(toy_panel, {})
    b = forecast_next_month(toy_panel, {})
    assert np.allclose(a["ml_mean"], b["ml_mean"])


def test_incomplete_last_month_is_rejected(toy_df):
    complete = _cut(toy_df, 12)
    check_last_month_complete(complete)  # ends the last working day of the month: fine
    partial = complete[complete["fecha"] < pd.Timestamp("2023-12-15")]
    with pytest.raises(SystemExit):
        check_last_month_complete(partial)
