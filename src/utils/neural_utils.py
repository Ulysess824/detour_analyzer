"""
Daily-then-aggregate neural forecasters (DNN and LSTM) on the monthly frame.

Every monthly row (series, origin, target month) is expanded to the calendar days of its
target month with a zero-filled daily target. The network predicts the expected consumption
of each day and the daily predictions are summed per row to get the monthly forecast.

    Loss        Tweedie deviance with a log link (the network outputs log(mu)), so predictions
                are never negative and sums over days are mean-coherent.
    Optimiser   Adam (AdamW when weight decay > 0), gradient clipping at 1.0.
    Inputs      series features known at the origin (signed log1p, standardised with train
                statistics, missing values set to 0 with a missing indicator), embeddings for
                planta, month of year and weekday, and the position of the day in the month.
                The LSTM also reads the last monthly totals of the series.
    Stopping    early stopping (patience 5) on the daily rows of the last two target months of
                the training window; the best epoch's weights are restored. No refit afterwards.

Requires: torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.utils.daily_utils import expand_to_days
from src.utils.feature_utils import SERIES_FEATURES

CATEGORICAL = ["planta_code", "month_t"]
NUMERIC = [f for f in SERIES_FEATURES if f not in CATEGORICAL]
DAY_NUMERIC = ["dom", "day_idx", "days_left"]
N_PLANTA_EMBEDDINGS = 64
MAX_EPOCHS = 40
PATIENCE = 5
TRAIN_FRAC = 0.2  # share of the daily rows used to train
CLIP_NORM = 1.0
MIN_SCALE = 0.1  # floor of the standard deviation used to standardise
MAX_Z = 5.0  # standardised values are clipped to +-MAX_Z

# Configuration used before tuning: the paper's architecture with the Tweedie loss.
NEURAL_BASE = {
    "dnn": dict(n_layers=2, units=32, dropout=0.0, weight_decay=0.0, lr=5e-4, batch_size=512, tweedie_p=1.2),
    "lstm": dict(
        hidden=32, units=16, dropout=0.0, weight_decay=0.0, lr=5e-4, batch_size=512, tweedie_p=1.2, look_back=8
    ),
}  # fmt: skip


def tweedie_loss(out: torch.Tensor, y: torch.Tensor, p: float) -> torch.Tensor:
    r"""Mean Tweedie deviance (up to a constant) of mu = exp(out) against y."""
    out = out.clamp(-12.0, 6.0)
    return (-y * torch.exp((1 - p) * out) / (1 - p) + torch.exp((2 - p) * out) / (2 - p)).mean()


class Net(nn.Module):
    r"""DNN (funnel of layers) or LSTM (reads the monthly history), both ending in log(mu)."""

    def __init__(self, kind: str, n_dense: int, params: dict) -> None:
        super().__init__()
        self.kind = kind
        self.emb_planta = nn.Embedding(N_PLANTA_EMBEDDINGS, 4)
        self.emb_month = nn.Embedding(12, 3)
        self.emb_weekday = nn.Embedding(7, 2)
        width_in = n_dense + 4 + 3 + 2
        if kind == "lstm":
            self.lstm = nn.LSTM(input_size=2, hidden_size=params["hidden"], batch_first=True)
            width_in += params["hidden"]
            widths = [params["units"]]
        else:
            widths = [max(4, params["units"] // 2**k) for k in range(params["n_layers"])]  # 64, 32, 16...
        layers = []
        for width in widths:
            layers += [nn.Linear(width_in, width), nn.ReLU(), nn.Dropout(params["dropout"])]
            width_in = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(width_in, 1)
        nn.init.zeros_(self.out.bias)

    def forward(self, dense, categories, weekday, history=None):
        parts = [dense, self.emb_planta(categories[:, 0]), self.emb_month(categories[:, 1]), self.emb_weekday(weekday)]
        if self.kind == "lstm":
            _, (hidden, _) = self.lstm(history)
            parts.append(hidden[-1])
        return self.out(self.body(torch.cat(parts, dim=1))).squeeze(-1)


def _signed_log(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


def _history(S, rows: pd.DataFrame, look_back: int) -> np.ndarray:
    r"""Last `look_back` monthly totals up to the origin: (n, look_back, 2) = [log1p value, is_valid]."""
    months = rows["o"].to_numpy()[:, None] + np.arange(1 - look_back, 1)[None, :]
    series = rows["series"].to_numpy()[:, None]
    values = S.monthly[series, np.clip(months, 0, None)]
    valid = (months >= 0) & ~np.isnan(values)
    return np.stack([np.log1p(np.where(valid, values, 0.0)), valid.astype(float)], axis=-1).astype(np.float32)


def _daily_block(S, rows: pd.DataFrame, frac: float | None, seed: int, look_back: int | None):
    r"""Daily design arrays of a block of monthly rows, the daily target, and each day's row position."""
    X, y, row_pos = expand_to_days(S, rows, SERIES_FEATURES, frac, seed)
    block = {
        "numeric": X[NUMERIC].to_numpy(np.float32),
        "day": X[DAY_NUMERIC].to_numpy(np.float32),
        "categories": np.column_stack([X["planta_code"].to_numpy(), X["month_t"].to_numpy().astype(int) % 12]).astype(
            np.int64
        ),
        "weekday": X["dow"].to_numpy().astype(np.int64),
        "row_pos": row_pos,
    }
    if look_back:
        block["history"] = _history(S, rows, look_back)[row_pos]
    return block, np.asarray(y, dtype=np.float32)


def _fit_scaler(train: dict, look_back: int | None) -> dict:
    r"""Standardisation statistics from the training block only."""
    numeric = _signed_log(train["numeric"])
    scaler = {
        "mean": np.nanmean(numeric, axis=0),
        "std": np.maximum(np.nanstd(numeric, axis=0), MIN_SCALE),
        "missing_cols": np.where(np.isnan(numeric).any(axis=0))[0],
        "day_mean": train["day"].mean(axis=0),
        "day_std": np.maximum(train["day"].std(axis=0), MIN_SCALE),
    }
    if look_back:
        valid_values = train["history"][..., 0][train["history"][..., 1] > 0]
        scaler["hist_mean"] = float(valid_values.mean())
        scaler["hist_std"] = float(max(valid_values.std(), MIN_SCALE))
    return scaler


def _to_tensors(block: dict, scaler: dict, look_back: int | None) -> list[torch.Tensor]:
    r"""Standardise a block and convert it to the tensors the network takes."""
    numeric = _signed_log(block["numeric"])
    missing = np.isnan(numeric[:, scaler["missing_cols"]]).astype(np.float32)
    z = np.nan_to_num((numeric - scaler["mean"]) / scaler["std"], nan=0.0)
    z = np.clip(z, -MAX_Z, MAX_Z)
    day = (block["day"] - scaler["day_mean"]) / scaler["day_std"]
    dense = np.hstack([z, missing, day]).astype(np.float32)
    tensors = [torch.from_numpy(dense), torch.from_numpy(block["categories"]), torch.from_numpy(block["weekday"])]
    if look_back:
        history = block["history"].copy()
        z_hist = np.clip((history[..., 0] - scaler["hist_mean"]) / scaler["hist_std"], -MAX_Z, MAX_Z)
        history[..., 0] = np.where(history[..., 1] > 0, z_hist, 0.0)
        tensors.append(torch.from_numpy(history))
    return tensors


def _train(net: Net, train_tensors, y_train, val_tensors, y_val, params: dict, seed: int) -> None:
    r"""Adam training with early stopping on the validation Tweedie loss; restores the best weights."""
    p, weight_decay = params["tweedie_p"], params["weight_decay"]
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(net.parameters(), lr=params["lr"], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=params["lr"])

    batch_size = int(params["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, waited = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        order = torch.randperm(len(y_train), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = tweedie_loss(net(*[t[batch] for t in train_tensors]), y_train[batch], p)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP_NORM)
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = float(tweedie_loss(net(*val_tensors), y_val, p))
        if val_loss < best_loss - 1e-7:
            best_loss, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)


def fit_predict_neural(S, kind: str, params: dict, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    r"""Train on the daily rows of `train`; return the summed daily forecast of each row of `test`."""
    look_back = params["look_back"] if kind == "lstm" else None
    torch.manual_seed(seed)

    # The last two target months of train are the validation set for early stopping.
    first_val_month = train["t"].max() - 1
    fit_rows, val_rows = train[train["t"] < first_val_month], train[train["t"] >= first_val_month]
    train_block, y_train = _daily_block(S, fit_rows, TRAIN_FRAC, seed, look_back)
    val_block, y_val = _daily_block(S, val_rows, TRAIN_FRAC, seed + 1, look_back)
    test_block, _ = _daily_block(S, test, None, seed, look_back)

    scaler = _fit_scaler(train_block, look_back)
    train_tensors = _to_tensors(train_block, scaler, look_back)
    val_tensors = _to_tensors(val_block, scaler, look_back)
    test_tensors = _to_tensors(test_block, scaler, look_back)

    # The target is divided by its training mean so the network works around 1.
    scale = float(y_train.mean()) + 1e-6
    y_train, y_val = torch.from_numpy(y_train / scale), torch.from_numpy(y_val / scale)

    net = Net(kind, train_tensors[0].shape[1], params)
    _train(net, train_tensors, y_train, val_tensors, y_val, params, seed)

    net.eval()
    with torch.no_grad():
        daily = torch.exp(net(*test_tensors).clamp(-12.0, 6.0)).numpy() * scale
    monthly = np.bincount(test_block["row_pos"], weights=daily, minlength=len(test))
    return monthly.astype(float)
