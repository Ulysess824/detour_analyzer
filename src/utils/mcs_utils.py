"""
Model Confidence Set (Hansen, Lunde and Nason, 2011, Econometrica 79(2)).

The MCS keeps the models that are not statistically worse than the best one, at a confidence
level 1 - alpha, for a user-chosen loss.

Procedure (range statistic T_R with the elimination rule of the paper):
    1. Take the loss differences d_ij,t = loss_i,t - loss_j,t over the models still in the set.
    2. Test the null "all models in the set have equal expected loss" with
       T_R = max_ij |t_ij|, t_ij = mean(d_ij) / se(d_ij). The distribution of T_R under the null
       and the standard errors come from a moving-block bootstrap over time.
    3. If the null is rejected at level alpha, remove the model with the largest standardised
       loss relative to some other model (the worst one) and repeat; otherwise stop.
    4. The MCS p-value of a model is the largest bootstrap p-value reached up to the step in
       which it is removed (so p-values never decrease along the elimination order). A model is in
       the MCS at level alpha when its p-value is at least alpha.

Time units here are the target months. Each month has many SKU observations, so the losses of a
month are summed and the bootstrap resamples whole months in blocks, which keeps the
cross-sectional dependence between SKUs of the same month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _block_bootstrap_indices(n_periods: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    r"""Circular moving-block bootstrap: (n_boot, n_periods) period indices."""
    n_blocks = int(np.ceil(n_periods / block))
    starts = rng.integers(0, n_periods, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_periods
    return idx.reshape(n_boot, -1)[:, :n_periods]


def model_confidence_set(
    loss: pd.DataFrame, period: np.ndarray, block: int = 3, n_boot: int = 5_000, seed: int = 0
) -> pd.DataFrame:
    r"""
    Run the MCS.

    loss    one row per observation, one column per model (the loss of that model)
    period  the time period (target month) of each row
    Returns a table sorted by elimination order with the mean loss, the elimination step and the
    MCS p-value of each model; the best model has p-value 1.
    """
    models = list(loss.columns)
    periods = np.unique(period)
    block = min(block, max(1, len(periods) // 3))  # a block as long as the sample would make every draw identical
    count = np.array([(period == p).sum() for p in periods], dtype=float)
    # Sum of the losses of every period: (n_periods, n_models).
    period_sum = np.vstack([loss.to_numpy()[period == p].sum(axis=0) for p in periods])

    rng = np.random.default_rng(seed)
    idx = _block_bootstrap_indices(len(periods), block, n_boot, rng)
    boot_mean = period_sum[idx].sum(axis=1) / count[idx].sum(axis=1)[:, None]  # (n_boot, n_models)
    sample_mean = period_sum.sum(axis=0) / count.sum()

    alive = list(range(len(models)))
    p_value = {}
    order = []
    running_max = 0.0
    while len(alive) > 1:
        sub = np.array(alive)
        d = sample_mean[sub][:, None] - sample_mean[sub][None, :]
        d_boot = boot_mean[:, sub][:, :, None] - boot_mean[:, sub][:, None, :]
        se = np.maximum(d_boot.std(axis=0), 1e-12)
        np.fill_diagonal(se, 1.0)
        t_ij = d / se
        statistic = np.abs(t_ij).max()
        null_stat = (np.abs(d_boot - d[None, :, :]) / se[None, :, :]).max(axis=(1, 2))
        p = float((null_stat >= statistic).mean())
        running_max = max(running_max, p)

        worst = sub[np.argmax(t_ij.max(axis=1))]  # largest standardised excess loss over some other model
        p_value[worst] = running_max
        order.append(worst)
        alive.remove(worst)
    p_value[alive[0]] = 1.0
    order.append(alive[0])

    table = pd.DataFrame(
        {
            "model": [models[i] for i in order],
            "elimination_step": range(1, len(order) + 1),
            "mean_loss": [sample_mean[i] for i in order],
            "mcs_p_value": [p_value[i] for i in order],
        }
    ).set_index("model")
    return table
