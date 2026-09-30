"""
Daily-then-aggregate forecaster: predict every calendar day of the target month, then sum.

The daily target is zero-filled: a day without a row counts as 0 consumption, so the model
predicts the expected consumption of every day, including the many days with none. That is
what makes the sum an honest monthly forecast; summing predictions only over days that
actually had a row would smuggle in the (unknown) number of active days.

*   One global LightGBM with a Tweedie objective (log link), so its output is a conditional
    mean and sums coherently over days. It is trained on daily rows, sampled to keep the
    fit fast, with series-level features known at the forecast origin plus day features
    (weekday, day of month, position in the month).
*   Rows are the monthly frame of forecast_monthly.py expanded to the days of the target
    month, so the daily and monthly models see exactly the same series-level information.
*   Early stopping uses the last two target months of the training window (chronological).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY_FEATURES = ["dow", "dom", "day_idx", "days_left"]


def _expand(S, rows: pd.DataFrame, features: list[str], frac: float | None, seed: int):
    r"""Repeat each monthly row once per calendar day of its target month."""
    t = rows["t"].to_numpy()
    counts = S.cd[t]
    rep = np.repeat(np.arange(len(rows)), counts)
    starts = np.cumsum(counts) - counts
    day_idx = np.arange(rep.size) - np.repeat(starts, counts)
    if frac is not None:
        keep = np.random.default_rng(seed).random(rep.size) < frac
        rep, day_idx = rep[keep], day_idx[keep]
    j = S.month_first_date[t[rep]] + day_idx

    base = rows[features].to_numpy(dtype=np.float32)[rep]
    day = np.column_stack(
        [
            S.dates.dayofweek.to_numpy()[j],
            S.dates.day.to_numpy()[j],
            day_idx,
            counts[rep] - 1 - day_idx,
        ]
    ).astype(np.float32)
    X = pd.DataFrame(np.hstack([base, day]), columns=features + DAY_FEATURES)
    X["planta_code"] = X["planta_code"].astype(int)
    y = S.D[rows["series"].to_numpy()[rep], j]
    return X, y, rep


def daily_lgbm_forecast(
    S, frame: pd.DataFrame, origin: int, features: list[str], frac: float = 0.35, seed: int = 0
) -> pd.Series:
    r"""Fit on daily rows whose month is known at the origin; return the monthly sum per row."""
    import lightgbm as lgb

    train_rows = frame[frame["t"] <= origin]
    test_rows = frame[frame["o"] == origin]

    x_train, y_train, rep_train = _expand(S, train_rows, features, frac, seed)
    val = train_rows["t"].to_numpy()[rep_train] >= origin - 1

    params = dict(
        objective="tweedie",
        tweedie_variance_power=1.3,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        max_bin=63,
        verbose=-1,
    )
    probe = lgb.LGBMRegressor(n_estimators=400, **params)
    probe.fit(
        x_train[~val],
        y_train[~val],
        eval_X=x_train[val],
        eval_y=y_train[val],
        categorical_feature=["planta_code"],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    final = lgb.LGBMRegressor(n_estimators=max(probe.best_iteration_, 20), **params)
    final.fit(x_train, y_train, categorical_feature=["planta_code"])

    x_test, _, rep_test = _expand(S, test_rows, features, None, seed)
    daily = np.clip(final.predict(x_test), 0.0, None)
    monthly = np.bincount(rep_test, weights=daily, minlength=len(test_rows))
    return pd.Series(monthly, index=test_rows.index)
