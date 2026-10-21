"""
Croston as two forecasts: the probability that a working day has consumption and the size when it does (Shenstone and Hyndman 2005 fit
sizes and intervals separately; the flat forecast is their ratio, Z / P = (1 / P) x Z).

Every origin reads only the history up to it. Probabilities are scored with the Brier score and log loss against the historical
frequency of consumption days; sizes are scored on the days that had consumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics_utils import score
from src.experiments import croston_utils as cu
from src.experiments.weekday_profile import SHRINK, weekday_profile

PROB_MODELS = ("croston", "freq", "croston_dow", "freq_dow")
SIZE_MODELS = ("z", "mean", "median", "last3")
CLIP = 1e-3  # probabilities are clipped to [CLIP, 1 - CLIP] for the log loss
CALIBRATION_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0)


def rolling_probabilities(series: pd.Series, test_start: str, weekdays: int = 5, shrink: float = SHRINK) -> pd.DataFrame:
    r"""
    One row per target day >= test_start with the realised value `y` and indicator `has`, the probabilities `p_<model>` and the size
    forecasts `s_<model>`. `croston` is 1 / P (the Croston state of the interval), `freq` the share of consumption days in the history, and
    the `_dow` versions multiply it by the weekday profile of the consumption indicator. Targets whose history has fewer than
    cu.MIN_DEMANDS consumption days are skipped.
    """
    values, index = series.to_numpy(dtype=float), series.index
    rows = []
    for pos in np.flatnonzero(index >= pd.Timestamp(test_start)):
        history = values[:pos]
        fit = cu.fit(history)
        if fit is None:
            continue
        sizes = history[history > 0]
        has = (history > 0).astype(float)
        v = weekday_profile(has, index.dayofweek[:pos], weekdays, shrink)[index[pos].dayofweek]
        pi, freq = min(1.0, 1.0 / fit.p), float(has.mean())
        rows.append({"target": index[pos], "origin": index[pos - 1], "y": values[pos], "has": int(values[pos] > 0),
                     "p_croston": pi, "p_freq": freq, "p_croston_dow": min(1.0, pi * v), "p_freq_dow": min(1.0, freq * v),
                     "s_z": fit.z, "s_mean": float(sizes.mean()), "s_median": float(np.median(sizes)), "s_last3": float(sizes[-3:].mean())})
    return pd.DataFrame(rows)


def probability_scores(table: pd.DataFrame, models: tuple[str, ...] = PROB_MODELS, reference: str = "freq") -> pd.DataFrame:
    r"""Brier score, log loss, Brier skill against `reference`, mean forecast and observed frequency, per probability model."""
    y = table["has"].to_numpy(dtype=float)
    brier_ref = float(np.mean((table[f"p_{reference}"].to_numpy() - y) ** 2))
    out = []
    for m in models:
        p = table[f"p_{m}"].to_numpy()
        pc = np.clip(p, CLIP, 1.0 - CLIP)
        brier = float(np.mean((p - y) ** 2))
        out.append({"model": m, "n": len(y), "brier": brier, "brier_skill": 1.0 - brier / brier_ref,
                    "logloss": float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))),
                    "mean_forecast": float(p.mean()), "observed": float(y.mean())})
    return pd.DataFrame(out).set_index("model")


def calibration(table: pd.DataFrame, model: str, edges: tuple[float, ...] = CALIBRATION_EDGES) -> pd.DataFrame:
    r"""Reliability table of one probability model: per bin of the forecast, how many days, the mean forecast and the observed frequency."""
    p = table[f"p_{model}"].to_numpy()
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.any():
            out.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()), "mean_forecast": float(p[mask].mean()),
                        "observed": float(table["has"].to_numpy()[mask].mean())})
    return pd.DataFrame(out)


def size_scores(table: pd.DataFrame, models: tuple[str, ...] = SIZE_MODELS) -> pd.DataFrame:
    r"""WAPE, MAE and bias (positive = too low) of each size forecast over the days that had consumption."""
    pos = table[table["has"] == 1]
    out = []
    for m in models:
        s = score(pos["y"].to_numpy(), pos[f"s_{m}"].to_numpy())
        out.append({"model": m, "n": len(pos), "mae": float(np.mean(np.abs(pos["y"] - pos[f"s_{m}"]))), **s})
    return pd.DataFrame(out).set_index("model").round(4)
