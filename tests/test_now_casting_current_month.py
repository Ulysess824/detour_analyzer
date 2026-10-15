"""Nowcasting month benchmark: no model reads past the cutoff, the rules are what the docstring says, and the shapes agree."""

import dataclasses

import numpy as np

from src.utils.now_casting_current_month import MODELS, forecast_media, forecast_mediana, level_scores, nowcast_month, planta_table, seen_days

DAY = 15


def _month(S):
    return S.monthly.shape[1] - 1


def test_days_after_the_cutoff_cannot_change_any_forecast(toy_panel):
    S, m = toy_panel, _month(toy_panel)
    cutoff = int(S.month_first_date[m]) + DAY
    daily = S.daily.copy()
    daily[:, cutoff:] *= 3
    a, b = nowcast_month(S, m, DAY, draws=30, seed=1), nowcast_month(dataclasses.replace(S, daily=daily), m, DAY, draws=30, seed=1)
    for model in MODELS:
        assert np.array_equal(a.forecasts[model], b.forecasts[model]), model
    assert not np.allclose(a.real_rest, b.real_rest)  # the real rest did change: only scoring sees it


def test_the_month_only_simulator_ignores_the_history_before_the_month(toy_panel):
    S, m = toy_panel, _month(toy_panel)
    start = int(S.month_first_date[m])
    daily = S.daily.copy()
    daily[:, :start] *= 5  # every earlier day
    a, b = nowcast_month(S, m, DAY, draws=30, seed=2), nowcast_month(dataclasses.replace(S, daily=daily), m, DAY, draws=30, seed=2)
    assert np.array_equal(a.forecasts["montecarlo_solo_mes"], b.forecasts["montecarlo_solo_mes"])
    assert not np.array_equal(a.forecasts["montecarlo_historia_mes"], b.forecasts["montecarlo_historia_mes"])


def test_media_and_mediana_follow_their_definition():
    days = np.array([[0.0, 0.0, 6.0, 0.0], [2.0, 4.0, 6.0, 8.0]])
    assert np.allclose(forecast_media(days, 3), [4.5, 15.0])
    assert np.allclose(forecast_mediana(days, 3), [0.0, 15.0])


def test_seen_days_skip_sundays_and_count_the_days_left(toy_panel):
    S, m = toy_panel, _month(toy_panel)
    series = np.arange(S.daily.shape[0])
    days, left = seen_days(S, m, DAY, series)
    weekday = np.asarray(S.dates[int(S.month_first_date[m]) : int(S.month_first_date[m]) + DAY].dayofweek)
    assert days.shape[1] == int((weekday != 6).sum())
    assert days.shape[1] + left == int(S.business_days[m])


def test_scores_are_consistent(toy_panel):
    S, m = toy_panel, _month(toy_panel)
    nc = nowcast_month(S, m, DAY, draws=30, seed=0)
    scores = level_scores(nc, len(S.plantas))
    assert list(scores.index) == MODELS and (scores["pronostico_resto_TO"] >= 0).all()
    table = planta_table(nc, S.plantas)
    assert np.isclose(table["real_resto_TO"].sum(), nc.real_rest.sum())
    assert (np.asarray(nc.planta) == S.planta_code[nc.series]).all()
