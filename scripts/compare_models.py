"""
Compare 3 simple forecasting models on the long-format consumption table
(planta, sku, fecha, consumo), evaluated with a chronological train/test split
(never random split, per project convention).

Models:
    1. Naive lag-1 (persistence baseline).
    2. Croston's method (classic intermittent-demand model, fit per series).
    3. Hurdle / two-part model (pooled logistic occurrence + log-linear
       magnitude regression, fit once across the whole panel, with planta
       dummies and calendar covariates: day of month, days to end of month,
       day of week, month and quarter).

Models 1 and 2 are univariate by construction and take no covariates; only the
hurdle model consumes the calendar features.

Each observed row is treated as one time period in sequence (gaps from
missing/no-report days are not reconstructed on the calendar) -- a
simplification appropriate for a first, simple comparison.

Usage:
    python scripts/compare_models.py data/consumos_long.csv --cutoff 2026-04-30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

GROUP_COLS = ["planta", "sku"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fecha"])
    return df.sort_values(GROUP_COLS + ["fecha"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    r"""Add lag_1 and a 3-period rolling mean of prior observed values, per series."""
    df = df.copy()
    g = df.groupby(GROUP_COLS)["consumo"]
    df["lag_1"] = g.shift(1)
    df["roll_mean_3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    r"""Add calendar covariates derived from fecha (no lookahead: the date is known upfront)."""
    df = df.copy()
    df["day_of_month"] = df["fecha"].dt.day
    df["days_to_end_of_month"] = (
        df["fecha"] + pd.offsets.MonthEnd(0) - df["fecha"]
    ).dt.days
    df["day_of_week"] = df["fecha"].dt.dayofweek
    df["month"] = df["fecha"].dt.month
    df["quarter"] = df["fecha"].dt.quarter
    return df


def naive_lag1_forecast(df: pd.DataFrame) -> pd.Series:
    r"""Model 1: forecast(t) = last observed value of the series (lag_1)."""
    return df["lag_1"]


def croston_series(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    r"""
    Standard Croston recursion for a single chronological series.

    *   forecast[t] is produced using only information available up to t-1.
    *   Before the first non-zero observation there is no basis for a
        forecast, so those entries stay NaN.
    *
    """
    n = len(values)
    forecast = np.full(n, np.nan)

    nonzero_idx = np.flatnonzero(values > 0)
    if nonzero_idx.size == 0:
        return forecast

    first = nonzero_idx[0]
    a = values[first]          # smoothed demand size
    p = float(first + 1)       # smoothed inter-demand interval
    q = 0                      # periods since the last non-zero demand

    for t in range(first + 1, n):
        forecast[t] = a / p if p > 0 else 0.0
        q += 1
        if values[t] > 0:
            a = alpha * values[t] + (1 - alpha) * a
            p = alpha * q + (1 - alpha) * p
            q = 0

    return forecast


def croston_forecast(df: pd.DataFrame, alpha: float = 0.1) -> pd.Series:
    r"""Model 2: apply the Croston recursion independently to each (planta, sku) series."""
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(GROUP_COLS).groups.items():
        values = df.loc[idx, "consumo"].to_numpy()
        out.loc[idx] = croston_series(values, alpha=alpha)
    return out


def hurdle_forecast(df: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    r"""
    Model 3: pooled two-part model, fit once on all training rows with a lag_1.

    *   Features: [lag_1, roll_mean_3, day_of_month, days_to_end_of_month] plus
        dummies for planta, day_of_week, month and quarter.
    *   Part A: logistic regression P(consumo > 0) on those features.
    *   Part B: linear regression of log1p(consumo) given consumo > 0, same features.
    *   Final forecast = P(consumo > 0) * E[consumo | consumo > 0].
    *
    """
    numeric_cols = ["lag_1", "roll_mean_3", "day_of_month", "days_to_end_of_month"]
    features = pd.concat(
        [
            df[numeric_cols],
            pd.get_dummies(df["planta"], prefix="planta"),
            pd.get_dummies(df["day_of_week"], prefix="dow"),
            pd.get_dummies(df["month"], prefix="month"),
            pd.get_dummies(df["quarter"], prefix="quarter"),
        ],
        axis=1,
    )
    feature_cols = list(features.columns)
    valid = df[numeric_cols].notna().all(axis=1)

    train_idx = df.index[train_mask & valid]
    x_train = features.loc[train_idx, feature_cols].to_numpy()
    y_train = df.loc[train_idx, "consumo"].to_numpy()
    is_positive = y_train > 0

    occurrence_model = LogisticRegression(max_iter=1_000)
    occurrence_model.fit(x_train, is_positive)

    magnitude_model = LinearRegression()
    magnitude_model.fit(x_train[is_positive], np.log1p(y_train[is_positive]))

    out = pd.Series(np.nan, index=df.index)
    score_idx = df.index[valid]
    x_all = features.loc[score_idx, feature_cols].to_numpy()

    p_positive = occurrence_model.predict_proba(x_all)[:, 1]
    magnitude = np.expm1(magnitude_model.predict(x_all))
    out.loc[score_idx] = p_positive * np.clip(magnitude, a_min=0.0, a_max=None)
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    wape = np.sum(np.abs(err)) / np.sum(np.abs(y_true))
    bias = np.mean(err)
    return {"n": len(y_true), "mae": mae, "rmse": rmse, "wape": wape, "bias": bias}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--cutoff", type=str, default="2026-04-30")
    args = parser.parse_args()

    df = load(args.input)
    df = add_lag_features(df)
    df = add_calendar_features(df)
    cutoff = pd.Timestamp(args.cutoff)
    train_mask = df["fecha"] <= cutoff
    test_mask = df["fecha"] > cutoff

    df["pred_naive"] = naive_lag1_forecast(df)
    df["pred_croston"] = croston_forecast(df)
    df["pred_hurdle"] = hurdle_forecast(df, train_mask)

    print(f"train rows: {train_mask.sum():_} | test rows: {test_mask.sum():_} | cutoff: {cutoff.date()}")
    print()

    results = []
    for model_name, pred_col in [
        ("naive_lag1", "pred_naive"),
        ("croston", "pred_croston"),
        ("hurdle", "pred_hurdle"),
    ]:
        test_df = df.loc[test_mask, ["consumo", pred_col]].dropna()
        metrics = compute_metrics(test_df["consumo"].to_numpy(), test_df[pred_col].to_numpy())
        metrics["model"] = model_name
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("model")[["n", "mae", "rmse", "wape", "bias"]]
    baseline_mae = results_df.loc["naive_lag1", "mae"]
    results_df["mae_lift_pct"] = ((baseline_mae - results_df["mae"]) / baseline_mae * 100).round(1)
    print(results_df.round(4))


if __name__ == "__main__":
    main()
