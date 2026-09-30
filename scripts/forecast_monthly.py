"""
Monthly forecasting of consumption per (planta, sku), validated with a rolling origin.

Two ways of getting a monthly number are compared under identical conditions:
    - monthly-direct: forecast the monthly total itself (simple rules and a global LightGBM).
    - daily-then-aggregate (--daily): forecast every calendar day of the target month with a
      zero-filled daily model, then sum the days (see daily_model.py).
Both use the same series-level features, so the only difference is the target granularity.

Validation (never a random split):
    - Rolling origin: for every test month t, models are fit using only months up to
      t - h and asked for month t. The last --n-test months are the test months.
    - Horizons h are months ahead; h=1 is next month, h=3 is a quarter ahead.
    - The SKU universe at each origin is the series that already exist at that origin.
      A SKU that first appears later cannot be forecast; its volume is reported as a
      cold-start share and stays inside the planta and total actuals.
    - The calendar (days per month, weekdays) is known in advance, so using the number of
      non-Sunday days of the target month is not leakage.

Models:
    naive, mean3, mean6, mean12        trailing means of monthly totals (zeros included)
    perday6                            trailing daily rate over 6 months times the number of
                                       non-Sunday days of the target month
    seasonal_naive                     same month last year (mean6 if unavailable)
    mean6_x_global / mean6_x_planta    mean6 times last year's seasonal ratio, taken from
                                       the grand total or the planta total
    lgbm_tweedie                       one global LightGBM over every series, Tweedie loss
    daily_lgbm                         daily model summed over the month (--daily)
    *_reconciled                       the SKU forecasts of each planta rescaled to add up
                                       to the direct planta forecast (mean6 times last
                                       year's ratio), so SKU and planta figures agree
    At planta and total level each simple model is also applied directly to the
    aggregated series ("direct"), in addition to summing the SKU forecasts ("bottom-up").

Metrics: WAPE = sum|error| / sum(actual); accuracy = 1 - WAPE; bias_pct = sum(actual -
forecast) / sum(actual), so a positive bias means the model under-forecasts.

Input is the long csv from transform_consumos.py. Combine the exports first, e.g.:
    python scripts/transform_consumos.py --drop-zeros \
        data/consumos_2024_2025.xlsx data/consumos_2026.xlsx -o data/consumos_long.csv
    python scripts/forecast_monthly.py data/consumos_long.csv --daily
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
BASELINES = ["naive", "mean3", "mean6", "mean12", "perday6", "seasonal_naive", "mean6_x_global", "mean6_x_planta"]
DIRECT = ["naive", "mean3", "mean6", "mean12", "perday6", "seasonal_naive", "mean6_x_seasonal"]
SERIES_FEATURES = [
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
    "n_days_t",
    "rate3",
    "rate6",
    "rate12",
    "occ3",
    "occ6",
    "occ12",
    "size6",
    "ly_rate",
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


def _rate(a: np.ndarray, bd: np.ndarray, lo: int, hi: int) -> np.ndarray:
    r"""Sum over the window divided by the non-Sunday days of the months each series was alive."""
    lo = max(lo, 0)
    if hi < lo:
        return np.full(a.shape[:-1], np.nan)
    w = a[..., lo : hi + 1]
    num = np.nansum(w, axis=-1)
    den = ((~np.isnan(w)) * bd[lo : hi + 1]).sum(axis=-1)
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


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
    bd: np.ndarray
    cd: np.ndarray
    NZ: np.ndarray
    D: np.ndarray
    dates: pd.DatetimeIndex
    month_first_date: np.ndarray


def build_panel(df: pd.DataFrame) -> Panel:
    r"""
    Monthly totals per series; NaN before a series first appears, 0 for empty months after.

    *   Also builds the zero-filled daily matrix D over every calendar day, the count of
        active days per month, and the days per month (bd excludes Sundays, cd is all).
    *
    """
    start = df["fecha"].min().to_period("M")
    m = ((df["fecha"].dt.year - start.year) * 12 + (df["fecha"].dt.month - start.month)).to_numpy()
    n_months = int(m.max()) + 1

    tab = df.assign(m=m).groupby(GROUP_COLS + ["m"])["consumo"].sum().unstack("m").reindex(columns=range(n_months))
    Y = tab.to_numpy(dtype=float)
    first = np.argmax(~np.isnan(Y), axis=1)
    cols = np.arange(n_months)[None, :]
    Y = np.where(cols >= first[:, None], np.nan_to_num(Y, nan=0.0), np.nan)
    Yz = np.nan_to_num(Y, nan=0.0)
    n_series = Y.shape[0]

    plantas = sorted(tab.index.get_level_values("planta").unique())
    planta_code = np.asarray(pd.Categorical(tab.index.get_level_values("planta"), categories=plantas).codes)
    P = np.zeros((len(plantas), n_months))
    np.add.at(P, planta_code, Yz)

    dates = pd.date_range(start.to_timestamp(), df["fecha"].max())
    date_month = ((dates.year - start.year) * 12 + (dates.month - start.month)).to_numpy()
    cd = np.bincount(date_month, minlength=n_months)
    bd = np.bincount(date_month[dates.dayofweek != 6], minlength=n_months)
    month_first_date = np.searchsorted(date_month, np.arange(n_months))

    sid = tab.index.get_indexer(pd.MultiIndex.from_frame(df[GROUP_COLS]))
    did = dates.get_indexer(df["fecha"])
    value = df["consumo"].to_numpy()
    D = np.zeros((n_series, len(dates)), dtype=np.float32)
    np.add.at(D, (sid, did), value.astype(np.float32))
    active = np.zeros((n_series, n_months))
    np.add.at(active, (sid, m), (value > 0).astype(float))
    NZ = np.where(cols >= first[:, None], active, np.nan)

    if not np.allclose(np.add.reduceat(D, month_first_date, axis=1), Yz, rtol=1e-4, atol=0.05):
        raise ValueError("daily matrix does not add up to the monthly totals")

    return Panel(
        Y=Y,
        first=first,
        planta_code=planta_code,
        plantas=plantas,
        P=P,
        T=Yz.sum(axis=0),
        month_of_year=((start.month - 1 + np.arange(n_months)) % 12) + 1,
        labels=[str(p) for p in pd.period_range(start, periods=n_months, freq="M")],
        bd=bd,
        cd=cd,
        NZ=NZ,
        D=D,
        dates=dates,
        month_first_date=month_first_date,
    )


def _direct_preds(A: np.ndarray, o: int, h: int, bd: np.ndarray) -> dict[str, np.ndarray]:
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
        "perday6": _rate(A, bd, o - 5, o) * bd[t],
        "seasonal_naive": np.where(np.isnan(ly), mean6, ly),
        "mean6_x_seasonal": analog,
    }


def make_frame(S: Panel, h: int) -> pd.DataFrame:
    r"""
    One row per (origin, series alive at that origin), with features known at the origin.

    *   Every window ends at or before the origin o, so nothing from the target month
        t = o + h is used; the seasonal windows around t - 12 stay <= o for h <= 9.
    *   Baseline forecasts are stored as p_* columns alongside the model features.
    *
    """
    n_months = S.Y.shape[1]
    frames = []
    for o in range(MIN_ORIGIN, n_months - h):
        t = o + h
        idx = np.flatnonzero(S.first <= o)
        Ya = S.Y[idx]
        NZa = S.NZ[idx]
        pc = S.planta_code[idx]
        nan_col = np.full(len(idx), np.nan)

        def lag(k: int) -> np.ndarray:
            return Ya[:, o - k] if o - k >= 0 else nan_col

        mean3, mean6, mean12 = _nmean(Ya, o - 2, o), _nmean(Ya, o - 5, o), _nmean(Ya, o - 11, o)
        rate3, rate6, rate12 = _rate(Ya, S.bd, o - 2, o), _rate(Ya, S.bd, o - 5, o), _rate(Ya, S.bd, o - 11, o)
        occ6 = _rate(NZa, S.bd, o - 5, o)
        zeros = np.where(np.isnan(Ya), np.nan, (Ya == 0).astype(float))
        ly = Ya[:, t - 12] if t - 12 >= 0 else nan_col
        ly_rate = ly / S.bd[t - 12] if t - 12 >= 0 else nan_col

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
            size6 = np.where(occ6 > 0, rate6 / occ6, np.nan)

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
                "n_days_t": S.bd[t],
                "rate3": rate3,
                "rate6": rate6,
                "rate12": rate12,
                "occ3": _rate(NZa, S.bd, o - 2, o),
                "occ6": occ6,
                "occ12": _rate(NZa, S.bd, o - 11, o),
                "size6": size6,
                "ly_rate": ly_rate,
                "y": Ya[:, t],
            }
        )
        frame["p_naive"] = frame["lag1"]
        frame["p_mean3"], frame["p_mean6"], frame["p_mean12"] = mean3, mean6, mean12
        frame["p_perday6"] = rate6 * S.bd[t]
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
        train.loc[~val, SERIES_FEATURES],
        train.loc[~val, "y"],
        eval_X=train.loc[val, SERIES_FEATURES],
        eval_y=train.loc[val, "y"],
        eval_metric="mae",
        categorical_feature=["planta_code"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    final = lgb.LGBMRegressor(n_estimators=max(probe.best_iteration_, 20), **params)
    final.fit(train[SERIES_FEATURES], train["y"], categorical_feature=["planta_code"])
    return pd.Series(np.clip(final.predict(test[SERIES_FEATURES]), 0.0, None), index=test.index)


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


def _reconcile(res: pd.DataFrame, S: Panel, h: int, test_months: list[int], col: str) -> pd.Series:
    r"""
    Scale each planta's SKU forecasts so they sum to the direct planta forecast.

    *   The direct planta model (mean6 times last year's seasonal ratio) is used as the
        planta control total and allocated to SKUs in proportion to their own forecasts.
    *
    """
    out = res[col].copy()
    for t in test_months:
        control = _direct_preds(S.P, t - h, h, S.bd)["mean6_x_seasonal"]
        mask = (res["t"] == t).to_numpy()
        sums = res.loc[mask].groupby("planta_code")[col].sum()
        factor = pd.Series(control[sums.index.to_numpy()], index=sums.index) / sums.replace(0.0, np.nan)
        scaled = res.loc[mask, col].to_numpy() * res.loc[mask, "planta_code"].map(factor).fillna(1.0).to_numpy()
        out.loc[mask] = scaled
    return out


def evaluate(S: Panel, h: int, n_test: int, daily: bool = False) -> dict[str, pd.DataFrame]:
    n_months = S.Y.shape[1]
    test_months = list(range(n_months - n_test, n_months))
    frame = make_frame(S, h)

    parts = []
    for t in test_months:
        origin = t - h
        block = frame[frame["o"] == origin].copy()
        block["p_lgbm_tweedie"] = lgbm_forecast(frame, origin)
        if daily:
            from daily_model import daily_lgbm_forecast

            block["p_daily_lgbm"] = daily_lgbm_forecast(S, frame, origin, SERIES_FEATURES)
        parts.append(block)
    res = pd.concat(parts, ignore_index=True)

    learned = ["lgbm_tweedie"] + (["daily_lgbm"] if daily else [])
    reconciled = []
    for name in ["mean6"] + learned:
        res[f"p_{name}_reconciled"] = _reconcile(res, S, h, test_months, f"p_{name}")
        reconciled.append(f"{name}_reconciled")
    sku_models = BASELINES + learned + reconciled

    sku = _ranked([{"model": n, **score(res["y"].to_numpy(), res[f"p_{n}"].to_numpy())} for n in sku_models])

    # planta and total: the simple models applied directly to the aggregated series, and the
    # SKU forecasts summed bottom-up. Actuals are the full totals, cold-start SKUs included.
    level_rows: dict[str, list[dict]] = {"planta": [], "total": []}
    by_planta_rows = []
    for level in level_rows:
        actual: list[np.ndarray] = []
        direct: dict[str, list[np.ndarray]] = {n: [] for n in DIRECT}
        bottom: dict[str, list[np.ndarray]] = {n: [] for n in sku_models}
        for t in test_months:
            o = t - h
            blk = res[res["t"] == t]
            if level == "planta":
                actual.append(S.P[:, t])
                d = _direct_preds(S.P, o, h, S.bd)
                for n in sku_models:
                    sums = blk.groupby("planta_code")[f"p_{n}"].sum()
                    bottom[n].append(sums.reindex(range(len(S.plantas))).fillna(0.0).to_numpy())
            else:
                actual.append(np.array([S.T[t]]))
                d = _direct_preds(S.T[None, :], o, h, S.bd)
                for n in sku_models:
                    bottom[n].append(np.array([blk[f"p_{n}"].sum()]))
            for n in DIRECT:
                direct[n].append(d[n])
        y = np.concatenate(actual)
        for n in DIRECT:
            level_rows[level].append({"model": f"{n} (direct)", **score(y, np.concatenate(direct[n]))})
        for n in sku_models:
            level_rows[level].append({"model": f"{n} (bottom-up)", **score(y, np.concatenate(bottom[n]))})

        if level == "planta":
            planta_id = np.tile(np.arange(len(S.plantas)), len(test_months))
            best = "daily_lgbm" if daily else "lgbm_tweedie"
            for code, name in enumerate(S.plantas):
                sel = planta_id == code
                by_planta_rows.append(
                    {
                        "planta": name,
                        "volume_pct": round(y[sel].sum() / y.sum() * 100, 1),
                        "acc_direct_seasonal": round(score(y[sel], np.concatenate(direct["mean6_x_seasonal"])[sel])["accuracy"], 4),
                        "acc_direct_perday6": round(score(y[sel], np.concatenate(direct["perday6"])[sel])["accuracy"], 4),
                        f"acc_{best}_bottom_up": round(score(y[sel], np.concatenate(bottom[best])[sel])["accuracy"], 4),
                    }
                )

    monthly = []
    for t in test_months:
        o = t - h
        blk = res[res["t"] == t]
        d_p = _direct_preds(S.P, o, h, S.bd)["mean6_x_seasonal"]
        d_t = _direct_preds(S.T[None, :], o, h, S.bd)["mean6_x_seasonal"][0]
        row = {
            "month": S.labels[t],
            "sku_acc_lgbm": round(score(blk["y"].to_numpy(), blk["p_lgbm_tweedie"].to_numpy())["accuracy"], 4),
            "sku_acc_naive": round(score(blk["y"].to_numpy(), blk["p_naive"].to_numpy())["accuracy"], 4),
        }
        if daily:
            row["sku_acc_daily"] = round(score(blk["y"].to_numpy(), blk["p_daily_lgbm"].to_numpy())["accuracy"], 4)
        row.update(
            {
                "planta_acc_direct": round(score(S.P[:, t], d_p)["accuracy"], 4),
                "total_error_pct": round((S.T[t] - d_t) / S.T[t] * 100, 2),
                "cold_start_pct": round((1 - blk["y"].sum() / S.T[t]) * 100, 2),
            }
        )
        monthly.append(row)

    planta_tbl, total_tbl = _ranked(level_rows["planta"]), _ranked(level_rows["total"])
    summary_models = ["naive", "perday6", "lgbm_tweedie", "lgbm_tweedie_reconciled"]
    if daily:
        summary_models += ["daily_lgbm", "daily_lgbm_reconciled"]
    summary = pd.DataFrame(
        {
            "sku_acc": [sku.loc[n, "accuracy"] for n in summary_models],
            "planta_acc": [planta_tbl.loc[f"{n} (bottom-up)", "accuracy"] for n in summary_models],
            "total_acc": [total_tbl.loc[f"{n} (bottom-up)", "accuracy"] for n in summary_models],
        },
        index=summary_models,
    )

    res["tercile"] = pd.cut(
        res.groupby("t")["mean12"].rank(pct=True), [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"]
    )
    return {
        "sku": sku,
        "planta": planta_tbl,
        "total": total_tbl,
        "summary": summary,
        "by_planta": pd.DataFrame(by_planta_rows).set_index("planta").sort_values("volume_pct", ascending=False),
        "monthly": pd.DataFrame(monthly).set_index("month"),
        "res": res,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--daily", action="store_true", help="also run the daily-then-aggregate model (slower)")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    df = pd.read_csv(args.input, parse_dates=["fecha"])
    S = build_panel(df)
    n_series, n_months = S.Y.shape
    print(f"months: {S.labels[0]} .. {S.labels[-1]} ({n_months}) | series: {n_series:_} | plantas: {len(S.plantas)}")
    print(f"test months: {S.labels[n_months - args.n_test]} .. {S.labels[-1]} ({args.n_test})")

    for h in args.horizons:
        out = evaluate(S, h, args.n_test, daily=args.daily)
        print(f"\n===== horizon h={h} month(s) ahead =====")
        print("\nSummary (accuracy = 1 - WAPE; planta and total are bottom-up sums of the SKU forecasts):")
        print(out["summary"].round(4))
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
        models = ["mean6", "lgbm_tweedie", "lgbm_tweedie_reconciled"] + (
            ["daily_lgbm", "daily_lgbm_reconciled"] if args.daily else []
        )
        for terc, g in res.groupby("tercile", observed=True):
            for name in models:
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
