"""
Weekly series of one planta and SKU, built from the daily master (data/consumos_long.csv).

A week runs Monday to Sunday and is complete only if every one of its days lies inside the date range of the master (not of the
SKU): the first and last weeks of the data are dropped. A week without a row of consumption is 0, but only from the week of the
first consumption of the SKU onwards (before it the series does not exist). Sunday consumption counts, as in the rest of the project.
"""

from __future__ import annotations

import re

import pandas as pd


def match_planner_sku(skus, planner_name: str, ignore_substitutes: bool = False) -> str:
    r"""
    The consumption SKU of a planner name (grade/sub-grade/gsm/width): same grade, gsm and width, ignoring the core width and
    any extra part. Exactly one SKU must match, otherwise it stops and lists the candidates. With `ignore_substitutes`, SKUs ending
    in "/Subst." are dropped when another SKU matches too (the substitute of a material is not the material).
    """
    grade = planner_name.split("/")[0].lower()
    gsm = re.search(r"(\d+)gsm", planner_name.lower())
    width = re.search(r"(\d+)mm", planner_name.lower())
    if not (gsm and width):
        raise ValueError(f"{planner_name!r} has no 'NNNgsm' and 'NNNNmm' parts")
    found = [s for s in skus if s.split("/")[0].lower() == grade
             and re.search(rf"/{gsm.group(1)}gsm(/|$)", s.lower()) and re.search(rf"/{width.group(1)}mm(/|$)", s.lower())]
    if ignore_substitutes and len(found) > 1:
        found = [f for f in found if not f.lower().endswith("/subst.")] or found
    if len(found) != 1:
        raise ValueError(f"{planner_name!r} matches {len(found)} SKUs: {found}")
    return found[0]


def complete_weeks(first_day: pd.Timestamp, last_day: pd.Timestamp) -> pd.DatetimeIndex:
    r"""Mondays of the weeks (Monday to Sunday) that lie entirely inside [first_day, last_day]."""
    first_day, last_day = first_day.normalize(), last_day.normalize()
    first_monday = first_day + pd.Timedelta(days=(7 - first_day.dayofweek) % 7)
    last_monday = last_day - pd.Timedelta(days=(last_day.dayofweek + 1) % 7 + 6)  # the week ends on the last Sunday <= last_day
    return pd.date_range(first_monday, last_monday, freq="7D")


def weekly_series(df: pd.DataFrame, planta: str, sku: str) -> pd.Series:
    r"""
    Weekly consumption (TO) of one series, indexed by the Monday of each complete week, from the week of its first consumption.
    `df` is the whole master: its date range decides which weeks are complete.
    """
    mondays = complete_weeks(df["fecha"].min(), df["fecha"].max())
    rows = df[(df["planta"] == planta) & (df["sku"] == sku)]
    if rows.empty:
        raise ValueError(f"no rows for planta {planta!r}, SKU {sku!r}")
    daily = rows.groupby("fecha")["consumo"].sum()
    week_of = daily.index - pd.to_timedelta(daily.index.dayofweek, unit="D")
    weekly = daily.groupby(week_of).sum().reindex(mondays, fill_value=0.0)
    first = week_of.min()
    return weekly[weekly.index >= first].rename(f"{planta}|{sku}")
