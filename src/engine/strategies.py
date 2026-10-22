"""
Production forecasts of the strategies: one month ahead, direct, direct + first days of the month in progress, the month completed
with the mean, and iterated.

`S` is the panel of the data as loaded, so when the last month is in progress it holds that month's partial total. `origin` is the
position of the last complete month; everything after it is hidden (`mask_panel`) except the first days of the month in progress, which
the partial and nowcast strategies read from `daily` up to the cutoff day. Every forecast has the columns of `forecast_next_month`
(`planta, sku, mes, lgbm, xgb, rf, ml_mean, naive`) plus `mes_origen` and `estrategia`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.feature_utils import SERIES_FEATURES, frame_at_origin, make_frame
from src.engine.inference_utils import extend_panel, forecast_next_month
from src.engine.member_utils import ML_MODELS, ml_forecasts
from src.engine.month_in_progress import completed_month, complete_calendar, mask_panel, with_pseudo_month
from src.engine.panel_utils import Panel
from src.engine.partial_utils import PARTIAL_FEATURES, add_partial_features


def _known_panel(S: Panel, origin: int, horizon: int) -> Panel:
    r"""Panel with the months after `origin` hidden and enough empty months to hold the target `origin + horizon`."""
    return extend_panel(mask_panel(S, origin), max(0, origin + horizon - (S.monthly.shape[1] - 1)))


def _table(S: Panel, extended: Panel, series: np.ndarray, models: dict, naive: np.ndarray, origin: int, target: int, name: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "planta": [S.keys[s][0] for s in series],
            "sku": [S.keys[s][1] for s in series],
            "mes": extended.labels[target],
            **{kind: models[kind] for kind in ML_MODELS},
            "ml_mean": models["ml_mean"],
            "naive": naive,
            "mes_origen": S.labels[origin],
            "estrategia": name,
        }
    )
    return out


def direct_forecast(S: Panel, tuned: dict, origin: int, horizon: int, cutoff_day: int | None = None, name: str = "directa") -> pd.DataFrame:
    r"""
    One model trained with the target `horizon` months ahead, for the month `origin + horizon`.

    With `cutoff_day` the four month-to-date features of the month after the origin (its first `cutoff_day` days) are added; the training rows
    get the same features at the same day for every month of the history.
    """
    extended = _known_panel(S, origin, horizon)
    frame = make_frame(extended, horizon)
    features = None
    if cutoff_day is not None:
        frame = add_partial_features(frame[frame["o"] <= origin].reset_index(drop=True), S, cutoff_day)  # month-to-date reads the real days of S
        features = SERIES_FEATURES + PARTIAL_FEATURES
    train = frame[frame["t"] <= origin]
    rows = frame[frame["o"] == origin].reset_index(drop=True)
    models = ml_forecasts(train, rows, tuned, features)
    return _table(S, extended, rows["series"].to_numpy(), models, rows["p_naive"].to_numpy(), origin, origin + horizon, name)


def nowcast_input_forecast(S: Panel, frame1: pd.DataFrame, origin: int, tuned: dict, cutoff_day: int = 15) -> pd.DataFrame:
    r"""
    Forecast of month `origin + 2` from the last complete month `origin` and the month `origin + 1` completed with the mean of its first days.

    `frame1` is `make_frame(S, 1)`. Returns one row per series alive at `origin` with the forecast of each tree model and their mean
    (`ml_mean`). The model is trained on targets known at `origin` (real complete months only), so the completed month is the only estimated input.
    """
    train = frame1[frame1["t"] <= origin]
    rows = frame1[frame1["o"] == origin]
    series = rows["series"].to_numpy()
    values, active = completed_month(S, origin + 1, cutoff_day, series)
    pseudo = with_pseudo_month(mask_panel(S, origin), origin + 1, series, values, active)
    rows2 = frame_at_origin(pseudo, origin + 1, 1)
    rows2 = rows2[rows2["series"].isin(series)].reset_index(drop=True)
    out = ml_forecasts(train, rows2, tuned)
    return pd.DataFrame({"series": rows2["series"].to_numpy(), **out})


def nowcast_media_forecast(S: Panel, tuned: dict, origin: int, cutoff_day: int) -> pd.DataFrame:
    r"""`nowcast_input_forecast` on a panel prepared for production, in the table format of the other strategies."""
    extended = _known_panel(S, origin, 2)
    out = nowcast_input_forecast(extended, make_frame(extended, 1), origin, tuned, cutoff_day)
    series = out["series"].to_numpy()
    naive = extended.monthly[series, origin]  # last complete month, like every `naive`
    return _table(S, extended, series, out, naive, origin, origin + 2, "nowcast_media")


def iterated_models(S: Panel, frame1: pd.DataFrame, origin: int, tuned: dict) -> pd.DataFrame:
    r"""
    Two applications of the one-month-ahead model, trained on the targets known at `origin`.

    Month `origin + 1` is filled with the first forecasts (active days from the recent occupancy) and the features of origin `origin + 1`
    are rebuilt from the panel with the later months hidden. Returns the forecast of month `origin + 2` of each model, by series.
    """
    train = frame1[frame1["t"] <= origin]
    rows = frame1[frame1["o"] == origin].reset_index(drop=True)
    first = ml_forecasts(train, rows, tuned)["ml_mean"]
    days = S.business_days[origin + 1]
    active = np.minimum(rows["occ6"].fillna(0.0).to_numpy() * days, days)
    pseudo = with_pseudo_month(mask_panel(S, origin), origin + 1, rows["series"].to_numpy(), first, active)
    rows2 = frame_at_origin(pseudo, origin + 1, 1)
    rows2 = rows2[rows2["series"].isin(rows["series"])].reset_index(drop=True)
    return pd.DataFrame({"series": rows2["series"].to_numpy(), **ml_forecasts(train, rows2, tuned)})


def iterated_forecast(S: Panel, frame1: pd.DataFrame, origin: int, tuned: dict) -> pd.Series:
    r"""`ml_mean` of `iterated_models`, indexed by series."""
    out = iterated_models(S, frame1, origin, tuned)
    return pd.Series(out["ml_mean"].to_numpy(), index=out["series"].to_numpy())


def iterated_table(S: Panel, tuned: dict, origin: int) -> pd.DataFrame:
    r"""`iterated_models` in the table format of the other strategies."""
    extended = _known_panel(S, origin, 2)
    out = iterated_models(extended, make_frame(extended, 1), origin, tuned)
    series = out["series"].to_numpy()
    return _table(S, extended, series, out, extended.monthly[series, origin], origin, origin + 2, "iterada")


def forecast_with_strategy(S: Panel, tuned: dict, strategy: str, origin: int, cutoff_day: int | None = None) -> pd.DataFrame:
    r"""
    Forecast with one strategy from the last complete month `origin` (a position in the panel `S`).

    `h1` forecasts the month after `origin` (the month in progress when there is one); the others forecast `origin + 2`. `parcial` and
    `nowcast_media` need the month in progress (`origin` is the month before the last one) and a `cutoff_day` not after its last record.
    """
    n = S.monthly.shape[1]
    if origin < n - 1:
        S = complete_calendar(S)  # the month in progress must know how many days it has
    if strategy in ("parcial", "nowcast_media"):
        if origin != n - 2 or cutoff_day is None:
            raise ValueError(f"{strategy} needs the month in progress after the last complete month and a cutoff day")
    if strategy == "h1":
        if origin == n - 1:
            return forecast_next_month(S, tuned).assign(mes_origen=S.labels[origin], estrategia="h1")
        return direct_forecast(S, tuned, origin, 1, name="h1")
    if strategy == "directa":
        return direct_forecast(S, tuned, origin, 2)
    if strategy == "parcial":
        return direct_forecast(S, tuned, origin, 2, cutoff_day, "parcial")
    if strategy == "nowcast_media":
        return nowcast_media_forecast(S, tuned, origin, cutoff_day)
    if strategy == "iterada":
        return iterated_table(S, tuned, origin)
    raise ValueError(f"unknown strategy {strategy!r}")
