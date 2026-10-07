"""
Forecasts for SKUs with little history (1 to 3 months of consumption at the forecast origin).

Strategies, all one month ahead and using only what is known at the origin:

    e1  SKU attributes (type, sub-type, gsm, width, substitute flag) added to the features of the trees
    e2  family: the SKU's own mean shrunk towards the median of its mature siblings (same planta, type and gsm)
    e3  cohort rule: the SKU's own mean times the weighted median ratio next month / own mean of earlier SKUs of the same age
    e5  median objective: LightGBM trained with the absolute-error loss (WAPE is minimised by a median, not a mean)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.utils.feature_utils import SERIES_FEATURES
from src.utils.panel_utils import Panel, window_mean

SKU_FEATURES = ["sku_type", "sku_sub", "sku_gsm", "sku_width", "sku_subst"]
LOW_HISTORY = 3  # months of history that count as little


def parse_sku(sku: str) -> dict:
    r"""Attributes in the SKU description, e.g. `K/01/160gsm/2100mm/1200-1450` or `HP2/01/160gsm/1790mm/1200-1450/Subst.`."""
    parts = sku.split("/")
    gsm, width = re.search(r"(\d+)gsm", sku), re.search(r"(\d+)mm", sku)
    return {
        "type": parts[0],
        "sub": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1,
        "gsm": int(gsm.group(1)) if gsm else -1,
        "width": int(width.group(1)) if width else -1,
        "subst": int("subst" in sku.lower()),
    }


def attribute_table(S: Panel) -> pd.DataFrame:
    r"""One row per series index with the parsed attributes, the planta and an integer code of the SKU type."""
    table = pd.DataFrame([parse_sku(sku) for _, sku in S.keys])
    table["planta"] = [planta for planta, _ in S.keys]
    table["type_code"] = pd.Categorical(table["type"]).codes
    return table


def add_sku_features(frame: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    r"""Add the SKU_FEATURES columns of each row's series."""
    rows = attrs.loc[frame["series"].to_numpy()]
    out = frame.copy()
    out["sku_type"], out["sku_sub"] = rows["type_code"].to_numpy(), rows["sub"].to_numpy()
    out["sku_gsm"], out["sku_width"], out["sku_subst"] = rows["gsm"].to_numpy(), rows["width"].to_numpy(), rows["subst"].to_numpy()
    return out


def history_months(S: Panel, frame: pd.DataFrame) -> np.ndarray:
    r"""Months of history of each row's series at its origin (1 = the origin is its first month)."""
    return frame["o"].to_numpy() - S.first_month[frame["series"].to_numpy()] + 1


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def cohort_ratio(S: Panel, origin: int, age: int) -> float:
    r"""
    Weighted median of (consumption next month) / (own mean over the first `age` months).

    Uses the SKUs that had `age` months of history at an earlier origin `o2 <= origin - 1`, so their next month is
    already observed at `origin`. SKUs present since the first month of the data are left out (their history is cut).
    Weights are the own means: with them the median minimises the sum of absolute errors. NaN if no cohort exists.
    """
    ratios, weights = [], []
    for o2 in range(age, origin):
        born = np.flatnonzero(S.first_month == o2 - age + 1)
        born = born[S.first_month[born] >= 1]
        if len(born) == 0:
            continue
        base = S.monthly[born, o2 - age + 1 : o2 + 1].mean(axis=1)
        nxt = S.monthly[born, o2 + 1]
        keep = base > 0
        ratios.append(nxt[keep] / base[keep])
        weights.append(base[keep])
    if not ratios:
        return float("nan")
    return weighted_median(np.concatenate(ratios), np.concatenate(weights))


def cohort_forecast(S: Panel, rows: pd.DataFrame) -> np.ndarray:
    r"""e3 for the rows of one origin; NaN for rows with more than LOW_HISTORY months of history."""
    origin = int(rows["o"].iloc[0])
    age = history_months(S, rows)
    out = np.full(len(rows), np.nan)
    for a in range(1, LOW_HISTORY + 1):
        mask = age == a
        if mask.any():
            out[mask] = rows["mean3"].to_numpy()[mask] * cohort_ratio(S, origin, a)
    return np.clip(out, 0.0, None)


def family_forecast(S: Panel, rows: pd.DataFrame, attrs: pd.DataFrame, k0: float) -> np.ndarray:
    r"""
    e2 for the rows of one origin; NaN for rows with more than LOW_HISTORY months of history.

    Reference = median of `mean6` of the mature siblings (6+ months of history) with the same planta, type and gsm; if there
    are none, the same type and gsm in any planta. Forecast = w * own mean + (1 - w) * reference, with w = age / (age + k0).
    """
    origin = int(rows["o"].iloc[0])
    age = history_months(S, rows)
    mature = np.flatnonzero(S.first_month <= origin - 5)
    ref = attrs.loc[mature, ["planta", "type", "gsm"]].copy()
    ref["mean6"] = window_mean(S.monthly, origin - 5, origin)[mature]
    ref = ref[ref["mean6"] > 0]
    by_planta = ref.groupby(["planta", "type", "gsm"])["mean6"].median()
    by_type = ref.groupby(["type", "gsm"])["mean6"].median()
    out = np.full(len(rows), np.nan)
    series, own = rows["series"].to_numpy(), rows["mean3"].to_numpy()
    for i in np.flatnonzero(age <= LOW_HISTORY):
        a = attrs.loc[series[i]]
        reference = by_planta.get((a["planta"], a["type"], a["gsm"]), by_type.get((a["type"], a["gsm"]), np.nan))
        weight = age[i] / (age[i] + k0)
        out[i] = own[i] if np.isnan(reference) else weight * own[i] + (1 - weight) * reference
    return np.clip(out, 0.0, None)


def fit_predict_l1(train: pd.DataFrame, test: pd.DataFrame, features: list[str] | None = None, seed: int = 0) -> np.ndarray:
    r"""e5: LightGBM with the absolute-error loss; early stopping on the last two target months, then refit on all of train."""
    import lightgbm as lgb

    cols = features or SERIES_FEATURES
    last_two = (train["t"] >= train["t"].max() - 1).to_numpy()
    params = dict(
        objective="l1", learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, verbose=-1, random_state=seed,
    )  # fmt: skip
    probe = lgb.LGBMRegressor(n_estimators=1_500, **params)
    probe.fit(
        train.loc[~last_two, cols], train.loc[~last_two, "y"], eval_set=[(train.loc[last_two, cols], train.loc[last_two, "y"])],
        eval_metric="mae", categorical_feature=["planta_code"], callbacks=[lgb.early_stopping(50, verbose=False)],
    )  # fmt: skip
    final = lgb.LGBMRegressor(n_estimators=max(probe.best_iteration_, 20), **params)
    final.fit(train[cols], train["y"], categorical_feature=["planta_code"])
    return np.clip(final.predict(test[cols]), 0.0, None).astype(float)
