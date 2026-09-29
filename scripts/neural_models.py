"""
Neural network forecasters following the strategy in Ouwehand, Krieg and Willems
(2024), "Deep learning for time series forecasting and nowcasting" (CBS Discussion
Paper, August 2024).

What the paper prescribes, and how it is applied here:

    - Architectures: a feed-forward DNN over lagged inputs and an LSTM over the
      look-back window (paper sections 2.2, 4.2).
    - Look-back window as the input representation; the paper tests 2, 8 and 16 lags.
    - One or two hidden layers with few units; shape found by trial and error.
    - Early stopping on a validation loss, patience of 5 epochs, learning rate 0.0005.
    - Input normalisation, which the paper calls essential for the optimiser.
    - Unstable estimation: random initialisation lands on different local optima, so
      the paper trains every model several times and reports the best by RMSE. It
      also warns that one frequent local optimum is a constant equal to the training
      mean, so the seed spread is reported here rather than hidden.
    - The paper notes series are usually too short for deep learning and suggests
      pretraining on similar series. With 1_424 short series that becomes a single
      global model trained across all of them (cross-learning), which is also the
      "large number of time series" case the paper flags as promising.

Requires: torch (pip install torch)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

GROUP_COLS = ["planta", "sku"]

LOOK_BACK = 8
LEARNING_RATE = 0.0005
PATIENCE = 5
MAX_EPOCHS = 200
BATCH_SIZE = 256

STATIC_FEATURES = [
    "sku_mean",
    "sku_std",
    "sku_max",
    "sku_occurrence_rate",
    "sku_median_positive",
    "sku_cv",
    "day_of_month",
    "days_to_end_of_month",
    "day_of_week",
]


class DNN(nn.Module):
    r"""Feed-forward network over the flattened feature vector (paper section 2.2.1)."""

    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_seq, x_static], dim=1)).squeeze(-1)


class LSTMNet(nn.Module):
    r"""LSTM over the look-back window, with static series features concatenated after."""

    def __init__(self, n_static: int, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_seq.unsqueeze(-1))
        return self.head(torch.cat([h_n[-1], x_static], dim=1)).squeeze(-1)


def build_windows(df: pd.DataFrame, look_back: int = LOOK_BACK) -> pd.DataFrame:
    r"""Add lag_1..lag_{look_back} columns per series, the paper's look-back inputs."""
    df = df.copy()
    g = df.groupby(GROUP_COLS)["consumo"]
    for k in range(1, look_back + 1):
        df[f"lb_{k}"] = g.shift(k)
    return df


def _tensors(
    df: pd.DataFrame, idx: pd.Index, seq_cols: list[str], stats: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = (df.loc[idx, seq_cols].to_numpy(dtype=np.float32) - stats["seq_mu"]) / stats["seq_sd"]
    static = (
        df.loc[idx, STATIC_FEATURES].to_numpy(dtype=np.float32) - stats["st_mu"]
    ) / stats["st_sd"]
    y = (df.loc[idx, "consumo"].to_numpy(dtype=np.float32) - stats["y_mu"]) / stats["y_sd"]
    return torch.from_numpy(seq), torch.from_numpy(static), torch.from_numpy(y)


def _train_once(
    model: nn.Module,
    train_batch: tuple,
    val_batch: tuple,
    seed: int,
) -> nn.Module:
    r"""Train with Adam, early stopping on validation loss (patience 5), per the paper."""
    torch.manual_seed(seed)
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.LSTM)):
            for name, param in layer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param) if param.dim() > 1 else None

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    x_seq, x_st, y = train_batch
    vx_seq, vx_st, vy = val_batch

    n = len(y)
    best_loss, best_state, waited = np.inf, None, 0
    generator = torch.Generator().manual_seed(seed)

    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            batch = perm[start : start + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(x_seq[batch], x_st[batch]), y[batch])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(vx_seq, vx_st), vy))
        if val_loss < best_loss - 1e-6:
            best_loss, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def neural_forecast(
    df: pd.DataFrame,
    train_mask: pd.Series,
    kind: str = "dnn",
    seeds: tuple[int, ...] = (0, 1, 2),
    val_days: int = 21,
) -> tuple[pd.Series, pd.DataFrame]:
    r"""
    Train a global DNN or LSTM across every series and return (predictions, per-seed runs).

    *   Predictions come from the seed with the best validation loss, which is the
        paper's rule of training several times and keeping the best run.
    *   The returned frame reports every seed so the instability the paper warns about
        stays visible instead of being averaged away.
    *
    """
    seq_cols = [f"lb_{k}" for k in range(1, LOOK_BACK + 1)]
    needed = seq_cols + STATIC_FEATURES
    valid = df[needed].notna().all(axis=1)

    train_idx = df.index[train_mask & valid]
    score_idx = df.index[valid]

    train_dates = df.loc[train_idx, "fecha"]
    span = (train_dates.max() - train_dates.min()).days
    split_date = train_dates.max() - pd.Timedelta(days=min(val_days, max(1, span // 5)))
    fit_idx = train_idx[train_dates <= split_date]
    val_idx = train_idx[train_dates > split_date]
    if len(fit_idx) == 0 or len(val_idx) == 0:
        fit_idx = val_idx = train_idx

    fit_seq = df.loc[fit_idx, seq_cols].to_numpy(dtype=np.float32)
    fit_static = df.loc[fit_idx, STATIC_FEATURES].to_numpy(dtype=np.float32)
    fit_y = df.loc[fit_idx, "consumo"].to_numpy(dtype=np.float32)
    stats = {
        "seq_mu": fit_seq.mean(),
        "seq_sd": fit_seq.std() + 1e-8,
        "st_mu": fit_static.mean(axis=0),
        "st_sd": fit_static.std(axis=0) + 1e-8,
        "y_mu": fit_y.mean(),
        "y_sd": fit_y.std() + 1e-8,
    }

    train_batch = _tensors(df, fit_idx, seq_cols, stats)
    val_batch = _tensors(df, val_idx, seq_cols, stats)
    score_batch = _tensors(df, score_idx, seq_cols, stats)

    runs, best = [], (np.inf, None)
    for seed in seeds:
        model = DNN(LOOK_BACK + len(STATIC_FEATURES)) if kind == "dnn" else LSTMNet(len(STATIC_FEATURES))
        model = _train_once(model, train_batch, val_batch, seed)

        model.eval()
        with torch.no_grad():
            val_loss = float(nn.MSELoss()(model(val_batch[0], val_batch[1]), val_batch[2]))
            preds = model(score_batch[0], score_batch[1]).numpy()
        preds = np.clip(preds * stats["y_sd"] + stats["y_mu"], 0.0, None)

        # The paper warns a common local optimum is a constant at the training mean.
        runs.append(
            {
                "seed": seed,
                "val_loss": round(val_loss, 5),
                "pred_std": round(float(np.std(preds)), 4),
                "pred_mean": round(float(np.mean(preds)), 4),
            }
        )
        if val_loss < best[0]:
            best = (val_loss, preds)

    out = pd.Series(np.nan, index=df.index)
    out.loc[score_idx] = best[1]
    return out, pd.DataFrame(runs)
