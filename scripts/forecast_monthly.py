"""
Monthly forecasting of consumption per (planta, sku), validated with a rolling origin.

Daily consumption is close to noise around each series' level, so this aggregates to
monthly totals (empty months count as 0) and forecasts those directly, at three levels:
SKU (planta + sku), planta, and grand total.

Validation (never a random split):
    - Rolling origin: for every test month t, models are fit using only months up to
      t - h and asked for month t. The last --n-test months are the test months.
    - Horizons h are months ahead; h=1 is next month, h=3 is a quarter ahead.
    - The SKU universe at each origin is the series that already exist at that origin.
      A SKU that first appears later cannot be forecast; its volume is reported as a
      cold-start share and stays inside the planta and total actuals.

Models:
    naive, mean3, mean6, mean12        trailing means of monthly totals (zeros included)
    seasonal_naive                     same month last year (mean6 if unavailable)
    mean6_x_global / mean6_x_planta    mean6 times last year's seasonal ratio, taken from
                                       the grand total or the planta total
    lgbm_tweedie                       one global LightGBM over every series, Tweedie loss
    lgbm_reconciled / mean6_reconciled the SKU forecasts of each planta rescaled to add up
                                       to the direct planta forecast (mean6 times last
                                       year's ratio), so SKU and planta figures agree
    At planta and total level each simple model is also applied directly to the
    aggregated series ("direct"), in addition to summing the SKU forecasts ("bottom-up").

Metrics: WAPE = sum|error| / sum(actual); accuracy = 1 - WAPE; bias_pct = sum(actual -
forecast) / sum(actual), so a positive bias means the model under-forecasts.

Input is the long csv from transform_consumos.py. Combine the exports first, e.g.:
    python scripts/transform_consumos.py --drop-zeros \
        data/consumos_2024_2025.xlsx data/consumos_2026.xlsx -o data/consumos_long.csv
    python scripts/forecast_monthly.py data/consumos_long.csv
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLS = ["planta", "sku"]
MIN_ORIGIN = 3
BASELINES = ["naive", "mean3", "mean6", "mean12", "seasonal_naive", "mean6_x_global", "mean6_x_planta"]
DIRECT = ["naive", "mean3", "mean6", "mean12", "seasonal_naive", "mean6_x_seasonal"]
LGBM_FEATURES = [
    "lag1",
    "lag2",
    "lag3",
    "mean3",
    "mean6",
    "mean12",
    "std6",
    "zero_share12",
    "months_active",
    "ly",
    "planta_code",
    "month_t",
    "ratio_planta",
    "ratio_global",
    "lyrel_planta",
    "lyrel_global",
]


def _nmean(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo = max(lo, 0)
    if hi < lo:
        return np.full(a.shape[:-1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[..., lo : hi + 1], axis=-1)


def _nstd(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo = max(lo, 0)
    if hi < lo:
        return np.full(a.shape[:-1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a[..., lo : hi + 1], axis=-1)


@dataclass
class Panel:
    Y: np.ndarray
    first: np.ndarray
    planta_code: np.ndarray
    plantas: list[str]
    P: np.ndarray
    T: np.ndarray
    month_of_year: np.ndarray
    labels: list[str]


def build_panel(df: pd.DataFrame) -> Panel:
    r"""Monthly totals per series; NaN before a series first appears, 0 for empty months after."""
    start = df["fecha"].min().to_period("M")
    m = (df["fecha"].dt.year - start.year) * 12 + (df["fecha"].dt.month - start.month)
    df = df.assign(m=m.to_numpy())
    n_months = int(df["m"].max()) + 1

    tab = df.groupby(GROUP_COLS + ["m"])["consumo"].sum().unstack("m").reindex(columns=range(n_months))
    Y = tab.to_numpy(dtype=float)
    first = np.argmax(~np.isnan(Y), axis=1)
    cols = np.arange(n_months)[None, :]
    Y = np.where(cols >= first[:, None], np.nan_to_num(Y, nan=0.0), np.nan)

    plantas = sorted(tab.index.get_level_values("planta").unique())
    planta_code = pd.Categorical(tab.index.get_level_values("planta"), categories=plantas).codes
    Yz = np.nan_to_num(Y, nan=0.0)
    P = np.zeros((len(plantas), n_months))
    np.add.at(P, planta_code, Yz)

    return Panel(
        Y=Y,
        first=first,
        planta_code=np.asarray(planta_code),
        plantas=plantas,
        P=P,
        T=Yz.sum(axis=0),
        month_of_year=((start.month - 1 + np.arange(n_months)) % 12) + 1,
        labels=[str(p) for p in pd.period_range(start, periods=n_months, freq="M")],
    )


def _direct_preds(A: np.ndarray, o: int, h: int) -> dict[str, np.ndarray]:
    r"""Apply the simple models to aggregated series A of shape (k, n_months)."""
    t = o + h
    mean6 = _nmean(A, o - 5, o)
    ly = A[:, t - 12] if t - 12 >= 0 else np.full(A.shape[0], np.nan)
    base = _nmean(A, o - 17, o - 12) if o - 17 >= 0 else np.full(A.shape[0], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ly / base
    analog = np.where(np.isfinite(ratio), mean6 * ratio, mean6)
    return {
        "naive": A[:, o],
        "mean3": _nmean(A, o - 2, o),
        "mean6": mean6,
        "mean12": _nmean(A, o - 11, o),
        "seasonal_naive": np.where(np.isnan(ly), mean6, ly),
        "mean6_x_seasonal": analog,
    }


def make_frame(S: Panel, h: int) -> pd.DataFrame:
    r"""
    One row per (origin, series alive at that origin), with features known at the origin.

    *   Every window ends at or before the origin o, so nothing from the target month
        t = o + h is used; the seasonal windows around t - 12 stay <= o for h <= 9.
    *   Baseline forecasts are stored as p_* columns alongside the LightGBM features.
    *
    """
    n_months = S.Y.shape[1]
    frames = []
    for o in range(MIN_ORIGIN, n_months - h):
        t = o + h
        idx = np.flatnonzero(S.first <= o)
        Ya = S.Y[idx]
        pc = S.planta_code[idx]
        nan_col = np.full(len(idx), np.nan)

        def lag(k: int) -> np.ndarray:
            return Ya[:, o - k] if o - k >= 0 else nan_col

        mean3, mean6, mean12 = _nmean(Ya, o - 2, o), _nmean(Ya, o - 5, o), _nmean(Ya, o - 11, o)
        zeros = np.where(np.isnan(Ya), np.nan, (Ya == 0).astype(float))
        ly = Ya[:, t - 12] if t - 12 >= 0 else nan_col

        with np.errstate(divide="ignore", invalid="ignore"):
            base_p = _nmean(S.P, o - 17, o - 12) if o - 17 >= 0 else np.full(len(S.plantas), np.nan)
            ratio_p = (S.P[:, t - 12] / base_p) if t - 12 >= 0 else np.full(len(S.plantas), np.nan)
            base_g = _nmean(S.T[None, :], o - 17, o - 12) if o - 17 >= 0 else np.array([np.nan])
            ratio_g = (S.T[t - 12] / base_g)[0] if t - 12 >= 0 else np.nan
            if t - 15 >= 0:
                lyrel_p = S.P[:, t - 12] / _nmean(S.P, t - 15, t - 9)
                lyrel_g = S.T[t - 12] / _nmean(S.T[None, :], t - 15, t - 9)[0]
            else:
                lyrel_p, lyrel_g = np.full(len(S.plantas), np.nan), np.nan

        rp = ratio_p[pc]
        frame = pd.DataFrame(
            {
                "series": idx,
                "o": o,
                "t": t,
                "planta_code": pc,
                "lag1": lag(0),
                "lag2": lag(1),
                "lag3": lag(2),
                "mean3": mean3,
                "mean6": mean6,
                "mean12": mean12,
                "std6": _nstd(Ya, o - 5, o),
                "zero_share12": _nmean(zeros, o - 11, o),
                "months_active": o - S.first[idx] + 1,
                "ly": ly,
                "month_t": S.month_of_year[t],
                "ratio_planta": rp,
                "ratio_global": ratio_g,
                "lyrel_planta": lyrel_p[pc],
                "lyrel_global": lyrel_g,
                "y": Ya[:, t],
            }
        )
        frame["p_naive"] = frame["lag1"]
        frame["p_mean3"], frame["p_mean6"], frame["p_mean12"] = mean3, mean6, mean12
        frame["p_seasonal_naive"] = np.where(np.isnan(ly), mean6, ly)
        frame["p_mean6_x_global"] = np.where(np.isfinite(ratio_g), mean6 * ratio_g, mean6)
        frame["p_mean6_x_planta"] = np.where(np.isfinite(rp), mean6 * rp, mean6)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def lgbm_forecast(frame: pd.DataFrame, origin: int) -> pd.Series:
    r"""Fit on rows whose target is known at the origin, predict the rows at the origin."""
    import lightgbm as lgb

    train = frame[frame["t"] <= origin]
    test = frame[frame["o"] == origin]
    val = train["t"] >= origin - 1
    params = dict(
        objective="tweedie",
        tweedie_variance_power=1.2,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
    )
    probe = lgb.LGBMRegressor(n_estimators=1_000, **params)
    probe.fit(
        train.loc[~val, LGBM_FEATURES],
        train.loc[~val, "y"],
        eval_X=train.loc[val, LGBM_FEATURES],
        eval_y=train.loc[val, "y"],
        eval_metric="mae",
        categorical_feature=["planta_code"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    final = lgb.LGBMRegressor(n_estimators=max(probe.best_iteration_, 20), **params)
    final.fit(train[LGBM_FEATURES], train["y"], categorical_feature=["planta_code"])
    return pd.Series(np.clip(final.predict(test[LGBM_FEATURES]), 0.0, None), index=test.index)


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = y - p
    wape = float(np.abs(e).sum() / y.sum())
    return {
        "n": len(y),
        "mae": float(np.abs(e).mean()),
        "wape": wape,
        "accuracy": 1.0 - wape,
        "bias_pct": float(e.sum() / y.sum() * 100.0),
    }


def _ranked(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows).set_index("model")
    out.insert(0, "rank", out["wape"].rank(method="min").astype(int))
    return out.sort_values("rank").round(4)


SKU_MODELS = BASELINES + ["lgbm_tweedie", "lgbm_reconciled", "mean6_reconciled"]


def _reconcile(res: pd.DataFrame, S: Panel, h: int, test_months: list[int], col: str) -> pd.Series:
    r"""
    Scale each planta's SKU forecasts so they add up to the direct planta forecast.

    *   The direct planta model (mean6 times last year's seasonal ratio) beat summing the
        SKU forecasts at planta level, so it is used as the planta control total and
        allocated to SKUs in proportion to their own forecasts.
    *
    """
    out = res[col].copy()
    for t in test_months:
        control = _direct_preds(S.P, t - h, h)["mean6_x_seasonal"]
        mask = (res["t"] == t).to_numpy()
        sums = res.loc[mask].groupby("planta_code")[col].sum()
        factor = pd.Series(control[sums.index.to_numpy()], index=sums.index) / sums.replace(0.0, np.nan)
        scaled = res.loc[mask, col].to_numpy() * res.loc[mask, "planta_code"].map(factor).fillna(1.0).to_numpy()
        out.loc[mask] = scaled
    return out


def evaluate(S: Panel, h: int, n_test: int) -> dict[str, pd.DataFrame]:
    n_months = S.Y.shape[1]
    test_months = list(range(n_months - n_test, n_months))
    frame = make_frame(S, h)

    parts = []
    for t in test_months:
        origin = t - h
        block = frame[frame["o"] == origin].copy()
        block["p_lgbm_tweedie"] = lgbm_forecast(frame, origin)
        parts.append(block)
    res = pd.concat(parts, ignore_index=True)
    res["p_lgbm_reconciled"] = _reconcile(res, S, h, test_months, "p_lgbm_tweedie")
    res["p_mean6_reconciled"] = _reconcile(res, S, h, test_months, "p_mean6")

    sku = _ranked([{"model": n, **score(res["y"].to_numpy(), res[f"p_{n}"].to_numpy())} for n in SKU_MODELS])

    # planta and total: the simple models applied directly to the aggregated series, and the
    # SKU forecasts summed bottom-up. Actuals are the full totals, cold-start SKUs included.
    level_rows: dict[str, list[dict]] = {"planta": [], "total": []}
    by_planta_parts = []
    for level in level_rows:
        actual: list[np.ndarray] = []
        direct: dict[str, list[np.ndarray]] = {n: [] for n in DIRECT}
        bottom: dict[str, list[np.ndarray]] = {n: [] for n in SKU_MODELS}
        for t in test_months:
            o = t - h
            blk = res[res["t"] == t]
            if level == "planta":
                actual.append(S.P[:, t])
                d = _direct_preds(S.P, o, h)
                for n in SKU_MODELS:
                    sums = blk.groupby("planta_code")[f"p_{n}"].sum()
                    bottom[n].append(sums.reindex(range(len(S.plantas))).fillna(0.0).to_numpy())
            else:
                actual.append(np.array([S.T[t]]))
                d = _direct_preds(S.T[None, :], o, h)
                for n in SKU_MODELS:
                    bottom[n].append(np.array([blk[f"p_{n}"].sum()]))
            for n in DIRECT:
                direct[n].append(d[n])
        y = np.concatenate(actual)
        for n in DIRECT:
            level_rows[level].append({"model": f"{n} (direct)", **score(y, np.concatenate(direct[n]))})
        for n in SKU_MODELS:
            level_rows[level].append({"model": f"{n} (bottom-up)", **score(y, np.concatenate(bottom[n]))})

        if level == "planta":
            planta_id = np.tile(np.arange(len(S.plantas)), len(test_months))
            for code, name in enumerate(S.plantas):
                sel = planta_id == code
                by_planta_parts.append(
                    {
                        "planta": name,
                        "volume_pct": round(y[sel].sum() / y.sum() * 100, 1),
                        "acc_direct_seasonal": round(score(y[sel], np.concatenate(direct["mean6_x_seasonal"])[sel])["accuracy"], 4),
                        "acc_direct_mean6": round(score(y[sel], np.concatenate(direct["mean6"])[sel])["accuracy"], 4),
                        "acc_lgbm_bottom_up": round(score(y[sel], np.concatenate(bottom["lgbm_tweedie"])[sel])["accuracy"], 4),
                    }
                )

    monthly = []
    for t in test_months:
        o = t - h
        blk = res[res["t"] == t]
        d_p = _direct_preds(S.P, o, h)["mean6_x_seasonal"]
        d_t = _direct_preds(S.T[None, :], o, h)["mean6_x_seasonal"][0]
        monthly.append(
            {
                "month": S.labels[t],
                "sku_acc_lgbm": round(score(blk["y"].to_numpy(), blk["p_lgbm_tweedie"].to_numpy())["accuracy"], 4),
                "sku_acc_naive": round(score(blk["y"].to_numpy(), blk["p_naive"].to_numpy())["accuracy"], 4),
                "planta_acc_direct": round(score(S.P[:, t], d_p)["accuracy"], 4),
                "total_error_pct": round((S.T[t] - d_t) / S.T[t] * 100, 2),
                "cold_start_pct": round((1 - blk["y"].sum() / S.T[t]) * 100, 2),
            }
        )

    res["tercile"] = pd.cut(
        res.groupby("t")["mean12"].rank(pct=True), [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"]
    )
    return {
        "sku": sku,
        "planta": _ranked(level_rows["planta"]),
        "total": _ranked(level_rows["total"]),
        "by_planta": pd.DataFrame(by_planta_parts).set_index("planta").sort_values("volume_pct", ascending=False),
        "monthly": pd.DataFrame(monthly).set_index("month"),
        "res": res,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--n-test", type=int, default=12)
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    df = pd.read_csv(args.input, parse_dates=["fecha"])
    S = build_panel(df)
    n_series, n_months = S.Y.shape
    print(f"months: {S.labels[0]} .. {S.labels[-1]} ({n_months}) | series: {n_series:_} | plantas: {len(S.plantas)}")
    print(f"test months: {S.labels[n_months - args.n_test]} .. {S.labels[-1]} ({args.n_test})")

    for h in args.horizons:
        out = evaluate(S, h, args.n_test)
        print(f"\n===== horizon h={h} month(s) ahead =====")
        print(f"\nSKU level (n = SKU-months, {out['sku']['n'].iloc[0]:_} each):")
        print(out["sku"])
        print("\nPlanta level (180 planta-months):")
        print(out["planta"])
        print("\nTotal level (12 months):")
        print(out["total"])
        print("\nAccuracy by planta:")
        print(out["by_planta"])
        print("\nMonth by month:")
        print(out["monthly"])

        res = out["res"]
        by_tercile = []
        for terc, g in res.groupby("tercile", observed=True):
            for name in ("mean6", "lgbm_tweedie", "lgbm_reconciled"):
                sc = score(g["y"].to_numpy(), g[f"p_{name}"].to_numpy())
                by_tercile.append(
                    {
                        "tercile": terc,
                        "model": name,
                        "volume_pct": round(g["y"].sum() / res["y"].sum() * 100, 1),
                        "wape": round(sc["wape"], 4),
                        "accuracy": round(sc["accuracy"], 4),
                        "bias_pct": round(sc["bias_pct"], 2),
                    }
                )
        print("\nSKU level by size (mean12 tercile within each month):")
        print(pd.DataFrame(by_tercile).set_index(["tercile", "model"]))


if __name__ == "__main__":
    main()
