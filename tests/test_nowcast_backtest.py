"""Nowcast backtest tables: only days before the cutoff are read, the full month is first days + forecast, and the aggregations add up."""

import dataclasses

import numpy as np
import pandas as pd

from src.experiments.nowcast_backtest_utils import (
    closer_counts,
    decision_criteria,
    global_scores,
    h1_reference,
    light_month_table,
    month_table,
    planta_scores,
    series_summary,
)

DAY = 15


def fake_cache(S, months):
    r"""A one-month-ahead cache with a constant forecast for every series alive in the previous month."""
    rows = []
    for t in months:
        alive = np.where(~np.isnan(S.monthly[:, t - 1]))[0]
        rows.append(pd.DataFrame({"series": alive, "t": t, "p_lgbm": 1.0, "p_xgb": 2.0, "p_rf": 3.0}))
    return pd.concat(rows, ignore_index=True)


def tables(S, months, h1=None):
    return pd.concat([month_table(S, m, DAY, draws=20, h1=h1) for m in months], ignore_index=True)


def test_h1_reference_is_the_mean_of_the_three_trees(toy_panel):
    h1 = h1_reference(fake_cache(toy_panel, [toy_panel.monthly.shape[1] - 1]))
    assert np.allclose(h1["ml_mean_h1"], 2.0)


def test_full_month_is_first_days_plus_forecast_and_real_month_is_the_panel(toy_panel):
    S = toy_panel
    month = S.monthly.shape[1] - 2
    long = month_table(S, month, DAY, draws=20)
    assert np.allclose(long["mes_media"], long["real_primeros"] + long["resto_media"])
    assert np.allclose(long["real_mes"], S.monthly[long["series"].to_numpy(), month])
    assert np.allclose(long["mes_mes_anterior"], S.monthly[long["series"].to_numpy(), month - 1])


def test_the_light_table_equals_the_full_one_for_media_and_mediana(toy_panel):
    S = toy_panel
    month = S.monthly.shape[1] - 2
    full, light = month_table(S, month, DAY, draws=20), light_month_table(S, month, DAY)
    for col in ("resto_media", "resto_mediana", "mes_media", "real_resto"):
        assert np.allclose(full[col], light[col])


def test_days_after_the_cutoff_cannot_change_the_forecast(toy_panel):
    S = toy_panel
    month = S.monthly.shape[1] - 2
    cutoff = int(S.month_first_date[month]) + DAY
    daily = S.daily.copy()
    daily[:, cutoff:] *= 3
    S2 = dataclasses.replace(S, daily=daily)
    a, b = light_month_table(S, month, DAY), light_month_table(S2, month, DAY)
    assert np.allclose(a["resto_media"], b["resto_media"]) and np.allclose(a["resto_mediana"], b["resto_mediana"])
    assert not np.allclose(a["real_resto"], b["real_resto"])


def test_levels_add_up(toy_panel):
    S = toy_panel
    months = [S.monthly.shape[1] - 3, S.monthly.shape[1] - 2]
    long = tables(S, months, h1_reference(fake_cache(S, months)))
    g = global_scores(long, len(S.plantas), "mes")
    one = g[(g["mes"] == S.labels[months[0]]) & (g["modelo"] == "media")].iloc[0]
    part = long[long["mes"] == S.labels[months[0]]]
    total_real, total_f = part["real_mes"].sum(), part["mes_media"].sum()
    assert np.isclose(one["acierto_total"], 1 - abs(total_f - total_real) / total_real)
    assert np.isclose(one["sesgo_total"], total_f / total_real - 1)
    pooled = g[(g["mes"] == "todos") & (g["modelo"] == "media")].iloc[0]
    assert np.isclose(pooled["real_TO"], long["real_mes"].sum()) and np.isclose(pooled["pronostico_TO"], long["mes_media"].sum())
    p = planta_scores(long, "mes")
    mine = p[(p["mes"] == "todos") & (p["modelo"] == "media")]
    assert np.isclose(mine["real_TO"].sum(), long["real_mes"].sum())  # planta rows add up to the total
    s = series_summary(long, "mes")
    assert np.isclose(s["real_mes_TO"].sum(), long["real_mes"].sum()) and len(s) == long[["planta", "sku"]].drop_duplicates().shape[0]


def test_closer_counts_leave_ties_to_nobody():
    long = pd.DataFrame({"real_mes": [10.0, 10.0, 10.0], "mes_media": [10.0, 5.0, 9.0], "mes_ml_mean_h1": [8.0, 5.0, 12.0]})
    assert closer_counts(long, "media", "ml_mean_h1") == {"media": 2, "ml_mean_h1": 0, "empates": 1, "total": 3}


def test_decision_criteria_returns_the_four_rows(toy_panel):
    S = toy_panel
    months = list(range(S.monthly.shape[1] - 7, S.monthly.shape[1] - 1))
    long = tables(S, months, h1_reference(fake_cache(S, months)))
    n = len(S.plantas)
    out = decision_criteria(global_scores(long, n, "resto"), global_scores(long, n, "mes"), planta_scores(long, "mes"))
    assert len(out) == 4 and set(out.columns) == {"criterio", "resultado", "detalle"}
