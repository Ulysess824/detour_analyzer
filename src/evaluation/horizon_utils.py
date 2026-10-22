"""
Forecast the month after next, as a planner's mid-month deadline requires.

At mid-month `m` the last complete month is `o = m - 1`, and the month to forecast is `t = o + 2`.
Strategies compared (the model of the report, `ml_mean`, is the mean of LightGBM, XGBoost and Random Forest):

    direct      one model trained with the target two months ahead
    iterated    a one-month-ahead model applied twice: its forecast of month `o + 1` is used as if it were observed
    partial     the direct model plus the consumption of the first days of month `o + 1` (mid-month cutoff)

Every forecast of month `t` uses only what is known at the end of month `o` (and, for `partial`, the first days of `o + 1`).
"""

from __future__ import annotations

import pandas as pd

from src.engine.feature_utils import SERIES_FEATURES
from src.engine.member_utils import ML_MODELS, ml_forecasts
from src.engine.month_in_progress import mask_panel, with_pseudo_month  # noqa: F401  (re-exported for the scripts and tests)
from src.engine.panel_utils import Panel
from src.engine.partial_utils import PARTIAL_FEATURES
from src.engine.strategies import iterated_forecast

HORIZON = 2
BASELINES = ["naive"]


def strategy_forecasts(
    S: Panel,
    frame1: pd.DataFrame,
    frame2: pd.DataFrame,
    frame2_partial: pd.DataFrame,
    tuned: dict,
    test_months: list[int],
    log=print,
) -> pd.DataFrame:
    r"""
    One row per (series, test month t) with the actual `y` and a column p_<strategy>.

    frame1 / frame2 are `make_frame` at horizon 1 / 2; frame2_partial is frame2 with the month-to-date features.
    The series are those alive at the last complete month `t - 2`, the same for every strategy.
    """
    parts = []
    for t in test_months:
        origin = t - HORIZON
        rows = frame2[frame2["o"] == origin].reset_index(drop=True)
        block = rows[["series", "o", "t", "planta_code", "y"]].copy()

        direct = ml_forecasts(frame2[frame2["t"] <= origin], rows, tuned)
        for kind in ML_MODELS:
            block[f"p_direct_{kind}"] = direct[kind]
        block["p_direct"] = direct["ml_mean"]

        rows_p = frame2_partial[frame2_partial["o"] == origin].reset_index(drop=True)
        partial = ml_forecasts(frame2_partial[frame2_partial["t"] <= origin], rows_p, tuned, SERIES_FEATURES + PARTIAL_FEATURES)
        block["p_partial"] = partial["ml_mean"]

        block["p_iterated"] = iterated_forecast(S, frame1, origin, tuned).reindex(block["series"]).to_numpy()

        for name in BASELINES:
            block[f"p_{name}"] = rows[f"p_{name}"].to_numpy()
        parts.append(block)
        log(f"    target {S.labels[t]} done")
    return pd.concat(parts, ignore_index=True)
