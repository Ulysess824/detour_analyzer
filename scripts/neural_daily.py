"""
Daily-then-aggregate neural forecasters (DNN and LSTM) on the monthly frame of forecast_monthly.py.

Every monthly row (series, origin, target month) is expanded to the calendar days of its target
month with a zero-filled daily target, exactly like daily_model.py. The network predicts the
expected consumption of each day and the predictions are summed per row, which gives the monthly
forecast.

*   Loss: Tweedie deviance with a log link (the network outputs log(mu)), so predictions are
    never negative and sums over days are mean-coherent.
*   Optimiser: Adam (AdamW when weight decay > 0) with gradient clipping at 1.0.
*   Inputs: the series features known at the origin (signed log1p, standardised with train
    statistics (sd floored at 0.1, clipped at +-5), missing values at 0 with a missing indicator), embeddings for planta, month of
    year and weekday, and the day position in the month. The LSTM also reads the last w monthly
    totals of the series (log1p, standardised, with a validity mask).
*   Early stopping uses the daily rows of the last two target months of the training window
    (chronological), patience 5; the best epoch's weights are restored. There is no refit on the
    full window afterwards.

Requires: torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

import daily_model as dm
import forecast_monthly as fm

FEATURES = fm.SERIES_FEATURES
CAT = ["planta_code", "month_t"]
NUM = [f for f in FEATURES if f not in CAT]
DAY_NUM = ["dom", "day_idx", "days_left"]
MAX_LOOK_BACK = 12
MAX_EPOCHS = 40
PATIENCE = 5
TRAIN_FRAC = 0.2
CLIP = 1.0
torch.set_num_threads(4)

BASE = {
    "dnn": dict(n_layers=2, units=32, dropout=0.0, weight_decay=0.0, lr=5e-4, batch_size=512, tweedie_p=1.2),
    "lstm": dict(hidden=32, units=16, dropout=0.0, weight_decay=0.0, lr=5e-4, batch_size=512, tweedie_p=1.2, look_back=8),
}


def tweedie_loss(out: torch.Tensor, y: torch.Tensor, p: float) -> torch.Tensor:
    r"""Mean Tweedie deviance (up to a constant) of mu = exp(out) against y."""
    out = out.clamp(-12.0, 6.0)
    return (-y * torch.exp((1 - p) * out) / (1 - p) + torch.exp((2 - p) * out) / (2 - p)).mean()


class Net(nn.Module):
    def __init__(self, kind: str, n_dense: int, params: dict) -> None:
        super().__init__()
        self.kind = kind
        self.emb_planta = nn.Embedding(len(self.PLANTAS), 4)
        self.emb_month = nn.Embedding(12, 3)
        self.emb_dow = nn.Embedding(7, 2)
        d_in = n_dense + 4 + 3 + 2
        drop = params["dropout"]
        if kind == "lstm":
            self.lstm = nn.LSTM(input_size=2, hidden_size=params["hidden"], batch_first=True)
            d_in += params["hidden"]
            widths = [params["units"]]
        else:
            widths = [max(4, params["units"] // 2**k) for k in range(params["n_layers"])]
        layers, d = [], d_in
        for w in widths:
            layers += [nn.Linear(d, w), nn.ReLU(), nn.Dropout(drop)]
            d = w
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.bias)

    PLANTAS = range(64)

    def forward(self, dense, cat, dow, seq=None):
        z = [dense, self.emb_planta(cat[:, 0]), self.emb_month(cat[:, 1]), self.emb_dow(dow)]
        if self.kind == "lstm":
            _, (h, _) = self.lstm(seq)
            z.append(h[-1])
        return self.out(self.body(torch.cat(z, dim=1))).squeeze(-1)


def _slog(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


def _seq(S, rows: pd.DataFrame, look_back: int) -> np.ndarray:
    r"""Last look_back monthly totals up to the origin: (n, look_back, 2) = [log1p value, valid]."""
    o = rows["o"].to_numpy()[:, None] + np.arange(1 - look_back, 1)[None, :]
    s = rows["series"].to_numpy()[:, None]
    ok = o >= 0
    v = S.Y[s, np.clip(o, 0, None)]
    ok &= ~np.isnan(v)
    return np.stack([np.log1p(np.where(ok, v, 0.0)), ok.astype(float)], axis=-1).astype(np.float32)


def _prep(S, rows: pd.DataFrame, frac, seed: int, look_back: int | None):
    r"""Daily design pieces for a block of monthly rows; returns dict of arrays plus y and row index."""
    X, y, rep = dm._expand(S, rows, FEATURES, frac, seed)
    out = {
        "num": X[NUM].to_numpy(np.float32),
        "day": X[DAY_NUM].to_numpy(np.float32),
        "cat": np.column_stack([X["planta_code"].to_numpy(), X["month_t"].to_numpy().astype(int) % 12]).astype(np.int64),
        "dow": X["dow"].to_numpy().astype(np.int64),
        "rep": rep,
    }
    if look_back:
        out["seq"] = _seq(S, rows, look_back)[rep]
    return out, np.asarray(y, dtype=np.float32)


def _scaler(train: dict, look_back: int | None) -> dict:
    num = _slog(train["num"])
    mu, sd = np.nanmean(num, axis=0), np.maximum(np.nanstd(num, axis=0), 0.1)
    miss_cols = np.where(np.isnan(num).any(axis=0))[0]
    dmu, dsd = train["day"].mean(axis=0), np.maximum(train["day"].std(axis=0), 0.1)
    st = {"mu": mu, "sd": sd, "miss": miss_cols, "dmu": dmu, "dsd": dsd}
    if look_back:
        v = train["seq"][..., 0][train["seq"][..., 1] > 0]
        st["smu"], st["ssd"] = float(v.mean()), float(max(v.std(), 0.1))
    return st


def _tensors(block: dict, st: dict, look_back: int | None) -> list[torch.Tensor]:
    num = _slog(block["num"])
    miss = np.isnan(num[:, st["miss"]]).astype(np.float32)
    num = np.clip(np.nan_to_num((num - st["mu"]) / st["sd"], nan=0.0), -5.0, 5.0)
    day = (block["day"] - st["dmu"]) / st["dsd"]
    dense = np.hstack([num, miss, day]).astype(np.float32)
    ts = [torch.from_numpy(dense), torch.from_numpy(block["cat"]), torch.from_numpy(block["dow"])]
    if look_back:
        s = block["seq"].copy()
        s[..., 0] = np.where(s[..., 1] > 0, np.clip((s[..., 0] - st["smu"]) / st["ssd"], -5.0, 5.0), 0.0)
        ts.append(torch.from_numpy(s))
    return ts


def fit_predict(S, kind: str, params: dict, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    r"""Train on daily rows of `train`, return the summed daily forecast for each row of `test`."""
    lb = params["look_back"] if kind == "lstm" else None
    torch.manual_seed(seed)
    tail_t = train["t"].max() - 1
    fit_rows, val_rows = train[train["t"] < tail_t], train[train["t"] >= tail_t]

    tr, y_tr = _prep(S, fit_rows, TRAIN_FRAC, seed, lb)
    va, y_va = _prep(S, val_rows, TRAIN_FRAC, seed + 1, lb)
    te, _ = _prep(S, test, None, seed, lb)
    st = _scaler(tr, lb)
    scale = float(y_tr.mean()) + 1e-6
    T_tr, T_va, T_te = _tensors(tr, st, lb), _tensors(va, st, lb), _tensors(te, st, lb)
    y_tr, y_va = torch.from_numpy(y_tr / scale), torch.from_numpy(y_va / scale)

    n_dense = T_tr[0].shape[1]
    net = Net(kind, n_dense, params)
    p = params["tweedie_p"]
    wd = params["weight_decay"]
    opt = (torch.optim.AdamW if wd > 0 else torch.optim.Adam)(net.parameters(), lr=params["lr"], **({"weight_decay": wd} if wd > 0 else {}))

    n, bs = len(y_tr), int(params["batch_size"])
    gen = torch.Generator().manual_seed(seed)
    best, best_state, waited = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for a in range(0, n, bs):
            b = perm[a : a + bs]
            opt.zero_grad()
            loss = tweedie_loss(net(*[t[b] for t in T_tr]), y_tr[b], p)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(tweedie_loss(net(*T_va), y_va, p))
        if v < best - 1e-7:
            best, waited = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        daily = torch.exp(net(*T_te).clamp(-12.0, 6.0)).numpy() * scale
    monthly = np.bincount(te["rep"], weights=daily, minlength=len(test))
    return monthly.astype(float)
