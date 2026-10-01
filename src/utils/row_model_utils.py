"""
Models on the raw observation rows (one row per planta, sku and day with data).

Each observed row is treated as one time period in sequence; gaps from days without a row
are not reconstructed on the calendar. This is the simple first comparison; the monthly
forecasting code uses the zero-filled calendar instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.tsa.arima.model import ARIMA

from src.utils.panel_utils import GROUP_COLS

LGBM_FEATURES = [
    "lag_1", "roll_mean_3", "periods_since_positive",
    "sku_mean", "sku_std", "sku_max", "sku_occurrence_rate", "sku_median_positive", "sku_cv",
    "day_of_month", "days_to_end_of_month", "day_of_week", "month", "quarter",
    "planta_code",
]  # fmt: skip

ARIMA_ORDERS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0)]


def load_rows(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fecha"])
    return df.sort_values(GROUP_COLS + ["fecha"]).reset_index(drop=True)


# ---------------------------------------------------------------- features


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    r"""Add lag_1 and a 3-period rolling mean of prior observed values, per series."""
    df = df.copy()
    by_series = df.groupby(GROUP_COLS)["consumo"]
    df["lag_1"] = by_series.shift(1)
    df["roll_mean_3"] = by_series.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return df


def _periods_since_positive(values: np.ndarray) -> np.ndarray:
    r"""For each position, observations since the last positive value (only looks at earlier rows)."""
    out = np.full(len(values), np.nan)
    since = np.nan
    for i, value in enumerate(values):
        out[i] = since
        if value > 0:
            since = 1.0
        elif not np.isnan(since):
            since += 1.0
    return out


def add_recency_feature(df: pd.DataFrame) -> pd.DataFrame:
    r"""Add periods_since_positive; row t only reflects rows before t, so there is no leakage."""
    df = df.copy()
    df["periods_since_positive"] = df.groupby(GROUP_COLS)["consumo"].transform(
        lambda s: pd.Series(_periods_since_positive(s.to_numpy()), index=s.index)
    )
    return df


def add_series_stats(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    r"""
    Attach per-series level features computed on train rows only (no leakage).

    They give the model the series identity: typical level, how often it consumes, how erratic
    it is. Series absent from train fall back to the train-wide averages.
    """
    by_series = df[train_mask].groupby(GROUP_COLS)["consumo"]
    stats = pd.DataFrame(
        {
            "sku_mean": by_series.mean(),
            "sku_std": by_series.std(),
            "sku_max": by_series.max(),
            "sku_occurrence_rate": by_series.apply(lambda s: (s > 0).mean()),
            "sku_median_positive": by_series.apply(lambda s: s[s > 0].median()),
        }
    )
    stats["sku_cv"] = stats["sku_std"] / stats["sku_mean"].replace(0.0, np.nan)
    out = df.merge(stats, on=GROUP_COLS, how="left")
    return out.fillna({col: stats[col].mean() for col in stats.columns})


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    r"""Add calendar covariates from the date (known in advance, so no lookahead)."""
    df = df.copy()
    df["day_of_month"] = df["fecha"].dt.day
    df["days_to_end_of_month"] = (df["fecha"] + pd.offsets.MonthEnd(0) - df["fecha"]).dt.days
    df["day_of_week"] = df["fecha"].dt.dayofweek
    df["month"] = df["fecha"].dt.month
    df["quarter"] = df["fecha"].dt.quarter
    return df


def build_row_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    r"""All features of the row-level models; series statistics use only rows up to the cutoff."""
    df = add_lag_features(df)
    df = add_calendar_features(df)
    df = add_recency_feature(df)
    return add_series_stats(df, df["fecha"] <= cutoff)


# ---------------------------------------------------------------- models


def naive_lag1_forecast(df: pd.DataFrame) -> pd.Series:
    r"""forecast(t) = last observed value of the series."""
    return df["lag_1"]


def croston_series(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    r"""Croston recursion for one chronological series; forecast[t] only uses data up to t-1."""
    forecast = np.full(len(values), np.nan)
    positive = np.flatnonzero(values > 0)
    if positive.size == 0:
        return forecast  # nothing to learn from before the first demand

    first = positive[0]
    size = values[first]  # smoothed demand size
    interval = float(first + 1)  # smoothed inter-demand interval
    since_last = 0  # periods since the last non-zero demand

    for t in range(first + 1, len(values)):
        forecast[t] = size / interval if interval > 0 else 0.0
        since_last += 1
        if values[t] > 0:
            size = alpha * values[t] + (1 - alpha) * size
            interval = alpha * since_last + (1 - alpha) * interval
            since_last = 0
    return forecast


def croston_forecast(df: pd.DataFrame, alpha: float = 0.1) -> pd.Series:
    r"""Croston applied independently to each (planta, sku) series."""
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(GROUP_COLS).groups.items():
        out.loc[idx] = croston_series(df.loc[idx, "consumo"].to_numpy(), alpha=alpha)
    return out


def hurdle_forecast(df: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    r"""
    Pooled two-part model fit once on all training rows.

    Features: lag_1, roll_mean_3, day_of_month, days_to_end_of_month plus dummies for planta,
    day_of_week, month and quarter.
    Part A: logistic regression of P(consumo > 0). Part B: linear regression of log1p(consumo)
    on the positive rows. Forecast = P(consumo > 0) * E[consumo | consumo > 0].
    """
    numeric = ["lag_1", "roll_mean_3", "day_of_month", "days_to_end_of_month"]
    features = pd.concat(
        [
            df[numeric],
            pd.get_dummies(df["planta"], prefix="planta"),
            pd.get_dummies(df["day_of_week"], prefix="dow"),
            pd.get_dummies(df["month"], prefix="month"),
            pd.get_dummies(df["quarter"], prefix="quarter"),
        ],
        axis=1,
    )
    valid = df[numeric].notna().all(axis=1)

    train_idx = df.index[train_mask & valid]
    x_train = features.loc[train_idx].to_numpy()
    y_train = df.loc[train_idx, "consumo"].to_numpy()
    is_positive = y_train > 0

    occurrence = LogisticRegression(max_iter=1_000).fit(x_train, is_positive)
    magnitude = LinearRegression().fit(x_train[is_positive], np.log1p(y_train[is_positive]))

    score_idx = df.index[valid]
    x_all = features.loc[score_idx].to_numpy()
    expected_size = np.clip(np.expm1(magnitude.predict(x_all)), 0.0, None)

    out = pd.Series(np.nan, index=df.index)
    out.loc[score_idx] = occurrence.predict_proba(x_all)[:, 1] * expected_size
    return out


def _fit_and_validation_rows(df: pd.DataFrame, train_idx: pd.Index, val_days: int):
    r"""
    Chronological validation slice for early stopping: the last `val_days` of train.

    The slice shrinks to a fifth of the train span when train is short. Returns
    (fit_idx, val_idx, use_early_stopping); with no usable slice, everything is fit.
    """
    dates = df.loc[train_idx, "fecha"]
    span_days = (dates.max() - dates.min()).days
    split_date = dates.max() - pd.Timedelta(days=min(val_days, max(1, span_days // 5)))
    fit_idx = train_idx[dates <= split_date]
    val_idx = train_idx[dates > split_date]
    use_early_stopping = len(fit_idx) > 0 and len(val_idx) > 0
    return (fit_idx if use_early_stopping else train_idx), val_idx, use_early_stopping


def lgbm_tweedie_forecast(df: pd.DataFrame, train_mask: pd.Series, val_days: int = 21) -> tuple[pd.Series, pd.DataFrame]:
    r"""
    One global LightGBM with a Tweedie objective across every series.

    Tweedie fits non-negative data with a point mass at zero, which is what daily consumption
    looks like. One model for all series; series identity enters through the train-only level
    features. Returns (predictions, feature importance by gain).
    """
    import lightgbm as lgb

    df = df.copy()
    df["planta_code"] = df["planta"].astype("category").cat.codes
    valid = df[LGBM_FEATURES].notna().all(axis=1)
    fit_idx, val_idx, early = _fit_and_validation_rows(df, df.index[train_mask & valid], val_days)

    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.2, n_estimators=2_000 if early else 300, learning_rate=0.05,
        num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    )  # fmt: skip
    fit_args: dict = {"categorical_feature": ["planta_code"]}
    if early:
        fit_args.update(
            eval_X=df.loc[val_idx, LGBM_FEATURES],
            eval_y=df.loc[val_idx, "consumo"],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    model.fit(df.loc[fit_idx, LGBM_FEATURES], df.loc[fit_idx, "consumo"], **fit_args)

    out = pd.Series(np.nan, index=df.index)
    score_idx = df.index[valid]
    out.loc[score_idx] = model.predict(df.loc[score_idx, LGBM_FEATURES])

    importance = pd.DataFrame({"feature": LGBM_FEATURES, "gain": model.booster_.feature_importance("gain")})
    importance = importance.sort_values("gain", ascending=False)
    importance["gain_pct"] = (importance["gain"] / importance["gain"].sum() * 100).round(1)
    return out, importance


def xgboost_forecast(df: pd.DataFrame, train_mask: pd.Series, val_days: int = 21) -> pd.Series:
    r"""Global XGBoost with the same features, Tweedie objective and validation rules as the LightGBM."""
    import xgboost as xgb

    df = df.copy()
    df["planta_code"] = df["planta"].astype("category").cat.codes
    valid = df[LGBM_FEATURES].notna().all(axis=1)
    fit_idx, val_idx, early = _fit_and_validation_rows(df, df.index[train_mask & valid], val_days)

    model = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.2, n_estimators=2_000 if early else 300, learning_rate=0.05,
        max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=100 if early else None, verbosity=0,
    )  # fmt: skip
    fit_args: dict = {"verbose": False}
    if early:
        fit_args["eval_set"] = [(df.loc[val_idx, LGBM_FEATURES], df.loc[val_idx, "consumo"])]
    model.fit(df.loc[fit_idx, LGBM_FEATURES], df.loc[fit_idx, "consumo"], **fit_args)

    out = pd.Series(np.nan, index=df.index)
    score_idx = df.index[valid]
    out.loc[score_idx] = np.clip(model.predict(df.loc[score_idx, LGBM_FEATURES]), 0.0, None)
    return out


def _fit_best_arima(y_train: np.ndarray):
    r"""Fit every order of the small grid and keep the one with the lowest AIC (None if all fail)."""
    best_aic, best_fit, best_order = np.inf, None, None
    for order in ARIMA_ORDERS:
        try:
            fit = ARIMA(y_train, order=order).fit()
        except Exception:
            continue
        if np.isfinite(fit.aic) and fit.aic < best_aic:
            best_aic, best_fit, best_order = fit.aic, fit, order
    return best_fit, best_order


def arima_forecast(df: pd.DataFrame, train_mask: pd.Series, min_train_obs: int = 30) -> tuple[pd.Series, pd.Series]:
    r"""
    Per-series ARIMA with AIC order selection, rolling one-step-ahead on the test rows.

    Parameters are estimated on train only; the test observations are then appended without
    refitting, so each prediction sees the same information as the other h=1 models. Series
    with fewer than `min_train_obs` training points are skipped (left NaN).
    Returns (predictions, selected order per series).
    """
    out = pd.Series(np.nan, index=df.index)
    chosen: dict[tuple, str] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for keys, g in df.groupby(GROUP_COLS):
            g = g.sort_values("fecha")
            is_train = train_mask.loc[g.index]
            y_train = g.loc[is_train, "consumo"].to_numpy()
            test_idx = g.index[~is_train]
            if len(y_train) < min_train_obs or len(test_idx) == 0:
                continue

            fit, order = _fit_best_arima(y_train)
            if fit is None:
                continue
            chosen[keys] = str(order)

            y_test = df.loc[test_idx, "consumo"].to_numpy()
            try:
                preds = fit.append(y_test, refit=False).predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
            except Exception:
                preds = np.full(len(y_test), float(np.mean(y_train)))
            out.loc[test_idx] = np.clip(np.asarray(preds), 0.0, None)

    return out, pd.Series(chosen, name="order")
