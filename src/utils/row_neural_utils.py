"""
Row-level DNN and LSTM forecasters, following Ouwehand, Krieg and Willems (2024),
"Deep learning for time series forecasting and nowcasting" (CBS Discussion Paper).

Applied as in the paper:
    - A feed-forward DNN over lagged inputs and an LSTM over the look-back window.
    - One or two small hidden layers, Adam with learning rate 0.0005, early stopping with
      patience 5 on a validation loss, and normalised inputs.
    - Random initialisation lands on different local optima, so every model is trained with
      several seeds and the spread is reported. One frequent bad optimum is a constant equal
      to the training mean.
    - One global model trained across all series (cross-learning), since the series are short.

Requires: torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.utils.panel_utils import GROUP_COLS

LOOK_BACK = 8
LEARNING_RATE = 0.0005
PATIENCE = 5
MAX_EPOCHS = 200
BATCH_SIZE = 256

STATIC_FEATURES = [
    "sku_mean", "sku_std", "sku_max", "sku_occurrence_rate", "sku_median_positive", "sku_cv",
    "day_of_month", "days_to_end_of_month", "day_of_week",
]  # fmt: skip


class DNN(nn.Module):
    r"""Feed-forward network over the look-back values and the static features (32 -> 16 -> 1)."""

    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_seq, x_static], dim=1)).squeeze(-1)


class LSTMNet(nn.Module):
    r"""LSTM over the look-back window, with the static features concatenated before the head."""

    def __init__(self, n_static: int, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_seq.unsqueeze(-1))
        return self.head(torch.cat([h_n[-1], x_static], dim=1)).squeeze(-1)


def build_windows(df: pd.DataFrame, look_back: int = LOOK_BACK) -> pd.DataFrame:
    r"""Add the columns lb_1..lb_{look_back}: the previous observations of each series."""
    df = df.copy()
    by_series = df.groupby(GROUP_COLS)["consumo"]
    for k in range(1, look_back + 1):
        df[f"lb_{k}"] = by_series.shift(k)
    return df


def _tensors(df: pd.DataFrame, idx: pd.Index, seq_cols: list[str], stats: dict):
    r"""Normalised (sequence, static, target) tensors of the rows in `idx`."""
    seq = (df.loc[idx, seq_cols].to_numpy(dtype=np.float32) - stats["seq_mu"]) / stats["seq_sd"]
    static = (df.loc[idx, STATIC_FEATURES].to_numpy(dtype=np.float32) - stats["st_mu"]) / stats["st_sd"]
    y = (df.loc[idx, "consumo"].to_numpy(dtype=np.float32) - stats["y_mu"]) / stats["y_sd"]
    return torch.from_numpy(seq), torch.from_numpy(static), torch.from_numpy(y)


def _init_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.LSTM)):
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param)


def _train_once(model: nn.Module, train_batch: tuple, val_batch: tuple, seed: int) -> nn.Module:
    r"""Train with Adam and MSE; early stopping on the validation loss, best weights restored."""
    torch.manual_seed(seed)
    _init_weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    x_seq, x_static, y = train_batch
    val_seq, val_static, val_y = val_batch

    best_loss, best_state, waited = np.inf, None, 0
    generator = torch.Generator().manual_seed(seed)

    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(x_seq[batch], x_static[batch]), y[batch]).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_seq, val_static), val_y))
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


def _normalisation_stats(df: pd.DataFrame, idx: pd.Index, seq_cols: list[str]) -> dict:
    seq = df.loc[idx, seq_cols].to_numpy(dtype=np.float32)
    static = df.loc[idx, STATIC_FEATURES].to_numpy(dtype=np.float32)
    y = df.loc[idx, "consumo"].to_numpy(dtype=np.float32)
    return {
        "seq_mu": seq.mean(), "seq_sd": seq.std() + 1e-8,
        "st_mu": static.mean(axis=0), "st_sd": static.std(axis=0) + 1e-8,
        "y_mu": y.mean(), "y_sd": y.std() + 1e-8,
    }  # fmt: skip


def neural_forecast(
    df: pd.DataFrame, train_mask: pd.Series, kind: str = "dnn", seeds: tuple[int, ...] = (0, 1, 2), val_days: int = 21
) -> tuple[pd.Series, pd.DataFrame]:
    r"""
    Train a global DNN or LSTM across every series; return (predictions, per-seed runs).

    Predictions come from the seed with the best validation loss (the paper's rule of training
    several times and keeping the best). The per-seed table keeps the instability visible.
    """
    seq_cols = [f"lb_{k}" for k in range(1, LOOK_BACK + 1)]
    valid = df[seq_cols + STATIC_FEATURES].notna().all(axis=1)
    train_idx = df.index[train_mask & valid]
    score_idx = df.index[valid]

    # Chronological validation slice: the last `val_days` of train (a fifth of the span if short).
    dates = df.loc[train_idx, "fecha"]
    span = (dates.max() - dates.min()).days
    split_date = dates.max() - pd.Timedelta(days=min(val_days, max(1, span // 5)))
    fit_idx = train_idx[dates <= split_date]
    val_idx = train_idx[dates > split_date]
    if len(fit_idx) == 0 or len(val_idx) == 0:
        fit_idx = val_idx = train_idx

    stats = _normalisation_stats(df, fit_idx, seq_cols)
    train_batch = _tensors(df, fit_idx, seq_cols, stats)
    val_batch = _tensors(df, val_idx, seq_cols, stats)
    score_batch = _tensors(df, score_idx, seq_cols, stats)

    runs, best = [], (np.inf, None)
    for seed in seeds:
        torch.manual_seed(seed)  # the initial biases are drawn here, so seed before building the model
        model = DNN(LOOK_BACK + len(STATIC_FEATURES)) if kind == "dnn" else LSTMNet(len(STATIC_FEATURES))
        model = _train_once(model, train_batch, val_batch, seed)

        model.eval()
        with torch.no_grad():
            val_loss = float(nn.MSELoss()(model(val_batch[0], val_batch[1]), val_batch[2]))
            preds = model(score_batch[0], score_batch[1]).numpy()
        preds = np.clip(preds * stats["y_sd"] + stats["y_mu"], 0.0, None)

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
