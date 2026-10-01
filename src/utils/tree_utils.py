"""Tree ensembles (LightGBM, XGBoost, Random Forest) on the monthly frame, plus the daily LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.daily_utils import expand_to_days
from src.utils.feature_utils import SERIES_FEATURES

MAX_TREES = 1_500
PATIENCE = 50
RF_TREES = 200

# Hyperparameters used before any tuning (the baseline of the Optuna comparison).
TREE_BASE = {
    "lgbm": dict(
        tweedie_variance_power=1.2, learning_rate=0.05, num_leaves=15, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8,
    ),
    "xgb": dict(
        tweedie_variance_power=1.2, learning_rate=0.05, max_depth=6, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8,
    ),
    "rf": dict(criterion="squared_error", max_features=1.0, min_samples_leaf=1, max_samples=None),
}  # fmt: skip


def fit_predict_trees(
    kind: str, params: dict, train: pd.DataFrame, test: pd.DataFrame, seed: int | None = None
) -> np.ndarray:
    r"""
    Fit on `train` and predict `test`; the monthly total of each row.

    LightGBM and XGBoost: early stopping on the last two target months of train picks the
    number of trees, then the model is refit on all of train with that number.
    Random Forest: no early stopping; missing values are filled with -1 (trees need a sentinel).
    """
    y = train["y"]
    last_two_months = (train["t"] >= train["t"].max() - 1).to_numpy()
    seed_args = {} if seed is None else {"random_state": seed}

    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=RF_TREES, n_jobs=-1, random_state=seed, **params)
        model.fit(train[SERIES_FEATURES].fillna(-1.0), y)
        return np.clip(model.predict(test[SERIES_FEATURES].fillna(-1.0)), 0.0, None).astype(float)

    fit_part, val_part = train[~last_two_months], train[last_two_months]

    if kind == "lgbm":
        import lightgbm as lgb

        p = dict(objective="tweedie", subsample_freq=1, verbose=-1, **seed_args, **params)
        probe = lgb.LGBMRegressor(n_estimators=MAX_TREES, **p)
        probe.fit(
            fit_part[SERIES_FEATURES],
            fit_part["y"],
            eval_X=val_part[SERIES_FEATURES],
            eval_y=val_part["y"],
            eval_metric="mae",
            categorical_feature=["planta_code"],
            callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
        )
        final = lgb.LGBMRegressor(n_estimators=max(probe.best_iteration_, 20), **p)
        final.fit(train[SERIES_FEATURES], y, categorical_feature=["planta_code"])
        return np.clip(final.predict(test[SERIES_FEATURES]), 0.0, None).astype(float)

    import xgboost as xgb

    p = dict(objective="reg:tweedie", tree_method="hist", verbosity=0, **seed_args, **params)
    probe = xgb.XGBRegressor(n_estimators=MAX_TREES, early_stopping_rounds=PATIENCE, eval_metric="mae", **p)
    probe.fit(
        fit_part[SERIES_FEATURES], fit_part["y"], eval_set=[(val_part[SERIES_FEATURES], val_part["y"])], verbose=False
    )
    final = xgb.XGBRegressor(n_estimators=max(int(probe.best_iteration) + 1, 20), **p)
    final.fit(train[SERIES_FEATURES], y, verbose=False)
    return np.clip(final.predict(test[SERIES_FEATURES]), 0.0, None).astype(float)


def lgbm_forecast(frame: pd.DataFrame, origin: int) -> pd.Series:
    r"""LightGBM with the baseline parameters: fit on rows known at `origin`, predict the rows at it."""
    train = frame[frame["t"] <= origin]
    test = frame[frame["o"] == origin]
    pred = fit_predict_trees("lgbm", TREE_BASE["lgbm"], train, test)
    return pd.Series(pred, index=test.index)


def daily_lgbm_forecast(S, frame: pd.DataFrame, origin: int, frac: float = 0.35, seed: int = 0) -> pd.Series:
    r"""
    Daily-then-aggregate LightGBM: predict every calendar day of the target month, then sum.

    The daily target is zero-filled (a day without a row is 0), so the sum is an honest monthly
    forecast. Tweedie loss keeps the daily predictions mean-coherent. Early stopping uses the
    daily rows of the last two target months of the training window.
    """
    import lightgbm as lgb

    train_rows = frame[frame["t"] <= origin]
    test_rows = frame[frame["o"] == origin]

    x_train, y_train, row_pos = expand_to_days(S, train_rows, SERIES_FEATURES, frac, seed)
    is_val = train_rows["t"].to_numpy()[row_pos] >= origin - 1

    params = dict(
        objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.1, num_leaves=31, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=63, verbose=-1,
    )  # fmt: skip
    probe = lgb.LGBMRegressor(n_estimators=400, **params)
    probe.fit(
        x_train[~is_val],
        y_train[~is_val],
        eval_X=x_train[is_val],
        eval_y=y_train[is_val],
        categorical_feature=["planta_code"],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    final = lgb.LGBMRegressor(n_estimators=max(probe.best_iteration_, 20), **params)
    final.fit(x_train, y_train, categorical_feature=["planta_code"])

    x_test, _, test_pos = expand_to_days(S, test_rows, SERIES_FEATURES, None, seed)
    daily = np.clip(final.predict(x_test), 0.0, None)
    monthly = np.bincount(test_pos, weights=daily, minlength=len(test_rows))
    return pd.Series(monthly, index=test_rows.index)
