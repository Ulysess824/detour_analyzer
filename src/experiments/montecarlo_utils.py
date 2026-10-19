"""
Monte Carlo completion of the rest of a month from its first days (experiment, not part of the pipeline).

At a cutoff (e.g. the 15th) the days after it are unknown. Each series is simulated day by day with only what is known
at the cutoff:

    planta works     Bernoulli(w[planta, weekday]) shared by all the SKUs of the planta on that day (holidays and
                     Sundays come out of the data: w is the share of such weekdays on which the planta consumed anything)
    SKU consumes     Bernoulli(q), q = share of the working days with consumption, from the last `lookback` days and the
                     days already seen of the month (shrunk towards the long-run share)
    size             a daily value drawn at random from the positive daily values of the series in the same window

The sum of the simulated days is one draw of the rest of the month; `draws` of them give its distribution.
"""

from __future__ import annotations

import numpy as np

from src.engine.panel_utils import Panel

LOOKBACK_DAYS = 365
SHRINK = 10.0  # weight, in working days, of the long-run share of consumption days against the days seen this month


def planta_daily_totals(S: Panel, lo: int, hi: int) -> np.ndarray:
    r"""(n_plantas, hi - lo) total consumption of every planta on the days lo..hi-1 of the calendar."""
    totals = np.zeros((len(S.plantas), hi - lo))
    np.add.at(totals, S.planta_code, S.daily[:, lo:hi].astype(float))
    return totals


def working_day_profile(S: Panel, cutoff: int, lookback: int = LOOKBACK_DAYS) -> np.ndarray:
    r"""(n_plantas, 7) share of each weekday (Monday = 0) on which the planta consumed anything, using days before `cutoff`."""
    lo = max(cutoff - lookback, 0)
    works = planta_daily_totals(S, lo, cutoff) > 0
    weekday = S.dates[lo:cutoff].dayofweek.to_numpy()
    profile = np.zeros((len(S.plantas), 7))
    for d in range(7):
        mine = weekday == d
        profile[:, d] = works[:, mine].mean(axis=1) if mine.any() else 0.0
    return profile


def consumption_share(S: Panel, cutoff: int, month_start: int, lookback: int = LOOKBACK_DAYS, shrink: float = SHRINK) -> np.ndarray:
    r"""
    (n_series,) probability that a series consumes on a day its planta works.

    Long-run share over the last `lookback` days before `cutoff`, updated with the days of the month already seen
    (`month_start` .. `cutoff`) as `shrink` pseudo working days of prior.
    """
    lo = max(cutoff - lookback, 0)
    works = planta_daily_totals(S, lo, cutoff) > 0  # (n_plantas, days)
    works_series = works[S.planta_code]  # (n_series, days)
    consumes = S.daily[:, lo:cutoff] > 0
    long_run = (consumes & works_series).sum(axis=1) / np.maximum(works_series.sum(axis=1), 1)
    seen = slice(month_start - lo, cutoff - lo)
    n_seen = works_series[:, seen].sum(axis=1)
    n_pos = (consumes[:, seen] & works_series[:, seen]).sum(axis=1)
    return np.clip((n_pos + shrink * long_run) / np.maximum(n_seen + shrink, 1e-9), 0.0, 1.0)


def positive_sizes(S: Panel, cutoff: int, lookback: int = LOOKBACK_DAYS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Positive daily values of every series before `cutoff` as (flat values, start of each series, count of each series)."""
    lo = max(cutoff - lookback, 0)
    block = S.daily[:, lo:cutoff]
    count = (block > 0).sum(axis=1)
    start = np.cumsum(count) - count
    return block[block > 0].astype(float), start, count  # row-major order keeps the series together


def simulate_rest_of_month(
    S: Panel,
    month: int,
    cutoff_day: int,
    draws: int = 200,
    seed: int = 0,
    series: np.ndarray | None = None,
    lookback: int = LOOKBACK_DAYS,
    shrink: float = SHRINK,
) -> np.ndarray:
    r"""
    (draws, n_series) simulated consumption of the days after the first `cutoff_day` days of `month`.

    Only the days before the cutoff are read (`S.daily[:, :cutoff]`). With `lookback = cutoff_day` and `shrink = 0` the
    simulator learns from the month in progress alone.
    """
    month_start = int(S.month_first_date[month])
    cutoff = month_start + cutoff_day
    month_end = month_start + int(S.calendar_days[month])
    rng = np.random.default_rng(seed)

    profile = working_day_profile(S, cutoff, lookback)
    share = consumption_share(S, cutoff, month_start, lookback, shrink)
    sizes, start, count = positive_sizes(S, cutoff, lookback)
    n_series = S.daily.shape[0]
    out = np.zeros((draws, n_series))
    can_draw = count > 0
    for date_pos in range(cutoff, month_end):
        weekday = S.dates[date_pos].dayofweek
        works = rng.random((draws, len(S.plantas))) < profile[:, weekday][None, :]  # one draw per planta and day
        consumes = (rng.random((draws, n_series)) < share[None, :]) & works[:, S.planta_code] & can_draw[None, :]
        pick = start[None, :] + np.floor(rng.random((draws, n_series)) * np.maximum(count, 1)[None, :]).astype(int)
        pick = np.minimum(pick, max(len(sizes) - 1, 0))
        out += np.where(consumes, sizes[pick] if len(sizes) else 0.0, 0.0)
    return out if series is None else out[:, series]
