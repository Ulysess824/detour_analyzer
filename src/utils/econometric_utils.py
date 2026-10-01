"""Classical univariate time-series models, fit per series on its monthly totals."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from src.utils.panel_utils import Panel

MIN_OBS = 12  # series with fewer months use the mean of what they have
ARIMA_ORDERS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]
ECONOMETRIC_MODELS = ["ses", "damped_holt", "arima"]


def _ses(y: np.ndarray, horizon: int) -> float:
    r"""Simple exponential smoothing with the smoothing level estimated by maximum likelihood."""
    fit = SimpleExpSmoothing(y, initialization_method="estimated").fit()
    return float(fit.forecast(horizon)[-1])


def _damped_holt(y: np.ndarray, horizon: int) -> float:
    r"""Holt's linear trend with a damped trend."""
    fit = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
    return float(fit.forecast(horizon)[-1])


def _arima(y: np.ndarray, horizon: int) -> float:
    r"""ARIMA with the order (from a small grid with d=0) chosen by AIC."""
    best_aic, best_forecast = np.inf, None
    for order in ARIMA_ORDERS:
        try:
            fit = ARIMA(y, order=order).fit()
        except Exception:
            continue
        if np.isfinite(fit.aic) and fit.aic < best_aic:
            best_aic, best_forecast = fit.aic, float(fit.forecast(horizon)[-1])
    if best_forecast is None:
        raise ValueError("no ARIMA order converged")
    return best_forecast


_FITTERS = {"ses": _ses, "damped_holt": _damped_holt, "arima": _arima}


def _forecast_one_series(y: np.ndarray, horizon: int) -> dict[str, float]:
    r"""Forecast one series with every model; any failure falls back to the mean of the last 6 months."""
    fallback = float(np.mean(y[-6:]))
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fitter in _FITTERS.items():
            if len(y) < MIN_OBS:
                out[name] = fallback
                continue
            try:
                out[name] = max(fitter(y, horizon), 0.0)
            except Exception:
                out[name] = fallback
    return out


def econometric_forecasts(S: Panel, rows: pd.DataFrame, origin: int, horizon: int, n_jobs: int = 4) -> pd.DataFrame:
    r"""
    SES, damped Holt and ARIMA forecasts for the rows of one origin.

    Each model sees only the months of the series up to the origin. Returns a frame aligned
    with `rows` and one column per model.
    """
    histories = [S.monthly[s, S.first_month[s] : origin + 1] for s in rows["series"].to_numpy()]
    results = Parallel(n_jobs=n_jobs)(delayed(_forecast_one_series)(y, horizon) for y in histories)
    return pd.DataFrame(results, index=rows.index)[ECONOMETRIC_MODELS]
