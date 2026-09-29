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

    4. Global LightGBM with a Tweedie objective (cross-learning across every
       series; series identity enters via train-only level features).
    5. Global XGBoost with the same Tweedie objective and feature set as (4).
    6. Per-series ARIMA with AIC order selection, rolling one-step-ahead.

Models 1 and 2 are univariate by construction and take no covariates; the hurdle
and LightGBM models consume the calendar and series-level features.

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


def add_recency_feature(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Add periods_since_positive: observations elapsed since the last positive consumption.

    *   Causal by construction: the value on row t reflects only rows strictly before t,
        so it can be computed over train and test alike without leakage.
    *
    """

    def _since(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan)
        since = np.nan
        for i, v in enumerate(values):
            out[i] = since
            if v > 0:
                since = 1.0
            elif not np.isnan(since):
                since += 1.0
        return out

    df = df.copy()
    df["periods_since_positive"] = df.groupby(GROUP_COLS)["consumo"].transform(
        lambda s: pd.Series(_since(s.to_numpy()), index=s.index)
    )
    return df


def add_series_stats(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    r"""
    Attach per-series level features, computed on TRAIN ROWS ONLY (no leakage).

    *   These give the model the series identity that the hurdle model lacked: its
        typical level, how often it consumes at all, and how erratic it is.
    *   Series absent from train fall back to the train-wide averages.
    *
    """
    train = df[train_mask]
    g = train.groupby(GROUP_COLS)["consumo"]
    stats = pd.DataFrame(
        {
            "sku_mean": g.mean(),
            "sku_std": g.std(),
            "sku_max": g.max(),
            "sku_occurrence_rate": g.apply(lambda s: (s > 0).mean()),
            "sku_median_positive": g.apply(lambda s: s[s > 0].median()),
        }
    )
    stats["sku_cv"] = stats["sku_std"] / stats["sku_mean"].replace(0.0, np.nan)

    out = df.merge(stats, on=GROUP_COLS, how="left")
    return out.fillna({col: stats[col].mean() for col in stats.columns})


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


def xgboost_forecast(df: pd.DataFrame, train_mask: pd.Series, val_days: int = 21) -> pd.Series:
    r"""
    Model 6: global XGBoost, same features and Tweedie objective as the LightGBM model.

    *   Same family as LightGBM (gradient-boosted trees), included so the comparison
        does not rest on a single boosting implementation.
    *   Validation slice and early stopping follow the same chronological rules.
    *
    """
    import xgboost as xgb

    df = df.copy()
    df["planta_code"] = df["planta"].astype("category").cat.codes

    valid = df[LGBM_FEATURES].notna().all(axis=1)
    train_idx = df.index[train_mask & valid]

    train_dates = df.loc[train_idx, "fecha"]
    span_days = (train_dates.max() - train_dates.min()).days
    effective_val_days = min(val_days, max(1, span_days // 5))
    split_date = train_dates.max() - pd.Timedelta(days=effective_val_days)
    fit_idx = train_idx[train_dates <= split_date]
    val_idx = train_idx[train_dates > split_date]

    use_early_stopping = len(fit_idx) > 0 and len(val_idx) > 0
    if not use_early_stopping:
        fit_idx = train_idx

    model = xgb.XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.2,
        n_estimators=2_000 if use_early_stopping else 300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=100 if use_early_stopping else None,
        verbosity=0,
    )
    fit_kwargs: dict = {"verbose": False}
    if use_early_stopping:
        fit_kwargs["eval_set"] = [(df.loc[val_idx, LGBM_FEATURES], df.loc[val_idx, "consumo"])]
    model.fit(df.loc[fit_idx, LGBM_FEATURES], df.loc[fit_idx, "consumo"], **fit_kwargs)

    out = pd.Series(np.nan, index=df.index)
    score_idx = df.index[valid]
    out.loc[score_idx] = np.clip(model.predict(df.loc[score_idx, LGBM_FEATURES]), 0.0, None)
    return out


ARIMA_ORDERS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0)]


def arima_forecast(
    df: pd.DataFrame, train_mask: pd.Series, min_train_obs: int = 30
) -> tuple[pd.Series, pd.Series]:
    r"""
    Model 5: per-series ARIMA with AIC order selection over a small grid.

    *   Rolling one-step-ahead on test: parameters are estimated on train only, then
        the test observations are appended without refitting, so each prediction sees
        the same information the other h=1 models do.
    *   Series with fewer than min_train_obs training points are skipped (left NaN).
    *   Returns (predictions, selected order per series).
    *
    """
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

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

            best_aic, best_res, best_order = np.inf, None, None
            for order in ARIMA_ORDERS:
                try:
                    res = ARIMA(y_train, order=order).fit()
                except Exception:
                    continue
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best_aic, best_res, best_order = res.aic, res, order
            if best_res is None:
                continue

            chosen[keys] = str(best_order)
            y_test = df.loc[test_idx, "consumo"].to_numpy()
            try:
                extended = best_res.append(y_test, refit=False)
                preds = extended.predict(
                    start=len(y_train), end=len(y_train) + len(y_test) - 1
                )
            except Exception:
                preds = np.full(len(y_test), float(np.mean(y_train)))
            out.loc[test_idx] = np.clip(np.asarray(preds), 0.0, None)

    return out, pd.Series(chosen, name="order")


LGBM_FEATURES = [
    "lag_1",
    "roll_mean_3",
    "periods_since_positive",
    "sku_mean",
    "sku_std",
    "sku_max",
    "sku_occurrence_rate",
    "sku_median_positive",
    "sku_cv",
    "day_of_month",
    "days_to_end_of_month",
    "day_of_week",
    "month",
    "quarter",
    "planta_code",
]


def lgbm_tweedie_forecast(
    df: pd.DataFrame, train_mask: pd.Series, val_days: int = 21
) -> tuple[pd.Series, pd.DataFrame]:
    r"""
    Model 4: one global LightGBM with a Tweedie objective across every series.

    *   Tweedie fits continuous non-negative data with a point mass at zero, which is
        what daily consumption looks like here (16 pct zeros on average).
    *   Cross-learning: a single model for all series; series identity enters through
        the train-only level features rather than 1_424 dummies.
    *   Early stopping uses the tail of train as a chronological validation slice,
        never a random split. The slice is val_days, shrunk to a fifth of the train
        span when train is short, and skipped entirely when even that leaves no
        rows to fit on.
    *
    """
    import lightgbm as lgb

    df = df.copy()
    df["planta_code"] = df["planta"].astype("category").cat.codes

    valid = df[LGBM_FEATURES].notna().all(axis=1)
    train_idx = df.index[train_mask & valid]

    train_dates = df.loc[train_idx, "fecha"]
    span_days = (train_dates.max() - train_dates.min()).days
    effective_val_days = min(val_days, max(1, span_days // 5))
    split_date = train_dates.max() - pd.Timedelta(days=effective_val_days)
    fit_idx = train_idx[train_dates <= split_date]
    val_idx = train_idx[train_dates > split_date]

    use_early_stopping = len(fit_idx) > 0 and len(val_idx) > 0
    if not use_early_stopping:
        fit_idx = train_idx

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.2,
        n_estimators=2_000 if use_early_stopping else 300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
    )
    fit_kwargs: dict = {"categorical_feature": ["planta_code"]}
    if use_early_stopping:
        fit_kwargs.update(
            eval_X=df.loc[val_idx, LGBM_FEATURES],
            eval_y=df.loc[val_idx, "consumo"],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    model.fit(df.loc[fit_idx, LGBM_FEATURES], df.loc[fit_idx, "consumo"], **fit_kwargs)

    out = pd.Series(np.nan, index=df.index)
    score_idx = df.index[valid]
    out.loc[score_idx] = model.predict(df.loc[score_idx, LGBM_FEATURES])

    importance = pd.DataFrame(
        {"feature": LGBM_FEATURES, "gain": model.booster_.feature_importance("gain")}
    ).sort_values("gain", ascending=False)
    importance["gain_pct"] = (importance["gain"] / importance["gain"].sum() * 100).round(1)
    return out, importance


def _ranked(results: pd.DataFrame) -> pd.DataFrame:
    r"""Add a 1-based rank column (best MAE first) and sort the table by it."""
    out = results.copy()
    out.insert(0, "rank", out["mae"].rank(method="min").astype(int))
    return out.sort_values("rank").round(4)


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
    parser.add_argument(
        "--neural",
        action="store_true",
        help="also train the DNN and LSTM models (slow; requires torch)",
    )
    args = parser.parse_args()

    df = load(args.input)
    df = add_lag_features(df)
    df = add_calendar_features(df)
    df = add_recency_feature(df)
    cutoff = pd.Timestamp(args.cutoff)
    df = add_series_stats(df, df["fecha"] <= cutoff)
    train_mask = df["fecha"] <= cutoff
    test_mask = df["fecha"] > cutoff

    df["pred_naive"] = naive_lag1_forecast(df)
    df["pred_croston"] = croston_forecast(df)
    df["pred_hurdle"] = hurdle_forecast(df, train_mask)
    df["pred_lgbm"], importance = lgbm_tweedie_forecast(df, train_mask)
    df["pred_xgb"] = xgboost_forecast(df, train_mask)
    df["pred_arima"], arima_orders = arima_forecast(df, train_mask)

    neural_runs: dict[str, pd.DataFrame] = {}
    if args.neural:
        from neural_models import build_windows, neural_forecast

        df = build_windows(df)
        for kind, col in [("dnn", "pred_dnn"), ("lstm", "pred_lstm")]:
            df[col], neural_runs[kind] = neural_forecast(df, train_mask, kind=kind)

    print(f"train rows: {train_mask.sum():_} | test rows: {test_mask.sum():_} | cutoff: {cutoff.date()}")
    print()

    results = []
    for model_name, pred_col in [
        ("naive_lag1", "pred_naive"),
        ("croston", "pred_croston"),
        ("hurdle", "pred_hurdle"),
        ("lgbm_tweedie", "pred_lgbm"),
        ("xgboost_tweedie", "pred_xgb"),
        ("arima", "pred_arima"),
        *([("dnn", "pred_dnn"), ("lstm", "pred_lstm")] if args.neural else []),
    ]:
        test_df = df.loc[test_mask, ["consumo", pred_col]].dropna()
        metrics = compute_metrics(test_df["consumo"].to_numpy(), test_df[pred_col].to_numpy())
        metrics["model"] = model_name
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("model")[["n", "mae", "rmse", "wape", "bias"]]
    baseline_mae = results_df.loc["naive_lag1", "mae"]
    results_df["mae_lift_pct"] = ((baseline_mae - results_df["mae"]) / baseline_mae * 100).round(1)
    print(_ranked(results_df))

    # ARIMA skips short series, so repeat the comparison on the rows it did cover.
    pred_cols = [
        "pred_naive",
        "pred_croston",
        "pred_hurdle",
        "pred_lgbm",
        "pred_xgb",
        "pred_arima",
        *(["pred_dnn", "pred_lstm"] if args.neural else []),
    ]
    common = df.loc[test_mask, ["consumo", *pred_cols]].dropna()
    print()
    print(f"Same-rows comparison (n={len(common):_}, where every model has a prediction):")
    common_rows = []
    for model_name, pred_col in zip(
        [
            "naive_lag1",
            "croston",
            "hurdle",
            "lgbm_tweedie",
            "xgboost_tweedie",
            "arima",
            *(["dnn", "lstm"] if args.neural else []),
        ],
        pred_cols,
    ):
        m = compute_metrics(common["consumo"].to_numpy(), common[pred_col].to_numpy())
        m["model"] = model_name
        common_rows.append(m)
    common_df = pd.DataFrame(common_rows).set_index("model")[["mae", "rmse", "wape", "bias"]]
    print(_ranked(common_df))

    for kind, runs in neural_runs.items():
        print()
        print(f"{kind.upper()} per-seed runs (the paper's unstable-estimation check):")
        print(runs.to_string(index=False))

    print()
    print("ARIMA order chosen by AIC:")
    print(arima_orders.value_counts().to_string())
    print()
    print("LightGBM feature importance (gain):")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
