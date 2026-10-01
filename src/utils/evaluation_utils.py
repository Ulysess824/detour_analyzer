"""Rolling-origin evaluation of the monthly forecasts at SKU, planta and total level."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.feature_utils import BASELINES, DIRECT, SERIES_FEATURES, direct_forecasts, make_frame
from src.utils.metrics_utils import rank_by_wape, score
from src.utils.panel_utils import Panel
from src.utils.tree_utils import daily_lgbm_forecast, lgbm_forecast


def test_months_of(S: Panel, n_test: int) -> list[int]:
    r"""Indices of the last `n_test` months."""
    n_months = S.monthly.shape[1]
    return list(range(n_months - n_test, n_months))


def reconcile(res: pd.DataFrame, S: Panel, horizon: int, test_months: list[int], col: str) -> pd.Series:
    r"""
    Scale each planta's SKU forecasts so they add up to the direct planta forecast.

    The direct planta model (mean6 times last year's seasonal ratio) is the planta control
    total; it is allocated to the SKUs in proportion to their own forecasts.
    """
    out = res[col].copy()
    for t in test_months:
        control = direct_forecasts(S.planta_monthly, t - horizon, horizon, S.business_days)["mean6_x_seasonal"]
        in_month = (res["t"] == t).to_numpy()
        planta_sums = res.loc[in_month].groupby("planta_code")[col].sum()
        factor = pd.Series(control[planta_sums.index.to_numpy()], index=planta_sums.index)
        factor = factor / planta_sums.replace(0.0, np.nan)
        row_factor = res.loc[in_month, "planta_code"].map(factor).fillna(1.0).to_numpy()
        out.loc[in_month] = res.loc[in_month, col].to_numpy() * row_factor
    return out


def level_actual_and_bottom_up(S: Panel, res: pd.DataFrame, t: int, level: str, models: list[str]):
    r"""Actual values of month `t` at planta or total level, and the bottom-up sums of each model."""
    block = res[res["t"] == t]
    if level == "planta":
        actual = S.planta_monthly[:, t]
        sums = {n: block.groupby("planta_code")[f"p_{n}"].sum() for n in models}
        return actual, {n: sums[n].reindex(range(len(S.plantas))).fillna(0.0).to_numpy() for n in models}
    return np.array([S.total_monthly[t]]), {n: np.array([block[f"p_{n}"].sum()]) for n in models}


def _rolling_forecasts(S: Panel, frame: pd.DataFrame, horizon: int, test_months: list[int], daily: bool) -> pd.DataFrame:
    r"""Forecast every test month using only what was known `horizon` months before."""
    parts = []
    for t in test_months:
        origin = t - horizon
        block = frame[frame["o"] == origin].copy()
        block["p_lgbm_tweedie"] = lgbm_forecast(frame, origin)
        if daily:
            block["p_daily_lgbm"] = daily_lgbm_forecast(S, frame, origin)
        parts.append(block)
    return pd.concat(parts, ignore_index=True)


def _level_tables(S: Panel, res: pd.DataFrame, horizon: int, test_months: list[int], sku_models: list[str], level: str):
    r"""Score the simple models applied directly to the aggregate, and the SKU forecasts summed bottom-up."""
    actual = []
    direct = {n: [] for n in DIRECT}
    bottom = {n: [] for n in sku_models}
    for t in test_months:
        origin = t - horizon
        month_actual, month_bottom = level_actual_and_bottom_up(S, res, t, level, sku_models)
        series = S.planta_monthly if level == "planta" else S.total_monthly[None, :]
        month_direct = direct_forecasts(series, origin, horizon, S.business_days)
        actual.append(month_actual)
        for n in DIRECT:
            direct[n].append(month_direct[n])
        for n in sku_models:
            bottom[n].append(month_bottom[n])

    y = np.concatenate(actual)
    direct = {n: np.concatenate(v) for n, v in direct.items()}
    bottom = {n: np.concatenate(v) for n, v in bottom.items()}
    rows = [{"model": f"{n} (direct)", **score(y, direct[n])} for n in DIRECT]
    rows += [{"model": f"{n} (bottom-up)", **score(y, bottom[n])} for n in sku_models]
    return y, direct, bottom, rank_by_wape(rows)


def _by_planta_table(S: Panel, y, direct, bottom, n_months: int, best: str) -> pd.DataFrame:
    r"""Accuracy of three planta-level approaches, for each planta."""
    planta_id = np.tile(np.arange(len(S.plantas)), n_months)
    rows = []
    for code, name in enumerate(S.plantas):
        sel = planta_id == code
        rows.append(
            {
                "planta": name,
                "volume_pct": round(y[sel].sum() / y.sum() * 100, 1),
                "acc_direct_seasonal": round(score(y[sel], direct["mean6_x_seasonal"][sel])["accuracy"], 4),
                "acc_direct_perday6": round(score(y[sel], direct["perday6"][sel])["accuracy"], 4),
                f"acc_{best}_bottom_up": round(score(y[sel], bottom[best][sel])["accuracy"], 4),
            }
        )
    return pd.DataFrame(rows).set_index("planta").sort_values("volume_pct", ascending=False)


def _monthly_table(S: Panel, res: pd.DataFrame, horizon: int, test_months: list[int], daily: bool) -> pd.DataFrame:
    rows = []
    for t in test_months:
        block = res[res["t"] == t]
        y = block["y"].to_numpy()
        control_planta = direct_forecasts(S.planta_monthly, t - horizon, horizon, S.business_days)["mean6_x_seasonal"]
        control_total = direct_forecasts(S.total_monthly[None, :], t - horizon, horizon, S.business_days)["mean6_x_seasonal"][0]
        row = {
            "month": S.labels[t],
            "sku_acc_lgbm": round(score(y, block["p_lgbm_tweedie"].to_numpy())["accuracy"], 4),
            "sku_acc_naive": round(score(y, block["p_naive"].to_numpy())["accuracy"], 4),
        }
        if daily:
            row["sku_acc_daily"] = round(score(y, block["p_daily_lgbm"].to_numpy())["accuracy"], 4)
        row["planta_acc_direct"] = round(score(S.planta_monthly[:, t], control_planta)["accuracy"], 4)
        row["total_error_pct"] = round((S.total_monthly[t] - control_total) / S.total_monthly[t] * 100, 2)
        row["cold_start_pct"] = round((1 - y.sum() / S.total_monthly[t]) * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def evaluate(S: Panel, horizon: int, n_test: int, daily: bool = False) -> dict[str, pd.DataFrame]:
    r"""
    Rolling-origin evaluation: for each test month t the models see only months up to t - horizon.

    The SKU universe at each origin is the series that already exist at that origin; the
    volume of SKUs that appear later (cold start) stays inside the planta and total actuals.
    """
    test_months = test_months_of(S, n_test)
    frame = make_frame(S, horizon)
    res = _rolling_forecasts(S, frame, horizon, test_months, daily)

    learned = ["lgbm_tweedie"] + (["daily_lgbm"] if daily else [])
    for name in ["mean6"] + learned:
        res[f"p_{name}_reconciled"] = reconcile(res, S, horizon, test_months, f"p_{name}")
    sku_models = BASELINES + learned + [f"{n}_reconciled" for n in ["mean6"] + learned]

    sku = rank_by_wape([{"model": n, **score(res["y"].to_numpy(), res[f"p_{n}"].to_numpy())} for n in sku_models])

    best = "daily_lgbm" if daily else "lgbm_tweedie"
    y_p, direct_p, bottom_p, planta_tbl = _level_tables(S, res, horizon, test_months, sku_models, "planta")
    _, _, _, total_tbl = _level_tables(S, res, horizon, test_months, sku_models, "total")

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

    # Size terciles of the series within each month (by their trailing 12-month mean).
    res["tercile"] = pd.cut(res.groupby("t")["mean12"].rank(pct=True), [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"])
    return {
        "sku": sku,
        "planta": planta_tbl,
        "total": total_tbl,
        "summary": summary,
        "by_planta": _by_planta_table(S, y_p, direct_p, bottom_p, len(test_months), best),
        "monthly": _monthly_table(S, res, horizon, test_months, daily),
        "res": res,
    }


def tercile_table(res: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    r"""Accuracy of each model within the low, mid and high volume terciles of SKUs."""
    rows = []
    for tercile, group in res.groupby("tercile", observed=True):
        for name in models:
            sc = score(group["y"].to_numpy(), group[f"p_{name}"].to_numpy())
            rows.append(
                {
                    "tercile": tercile,
                    "model": name,
                    "volume_pct": round(group["y"].sum() / res["y"].sum() * 100, 1),
                    "wape": round(sc["wape"], 4),
                    "accuracy": round(sc["accuracy"], 4),
                    "bias_pct": round(sc["bias_pct"], 2),
                }
            )
    return pd.DataFrame(rows).set_index(["tercile", "model"])


def score_models_by_level(S: Panel, res: pd.DataFrame, models: list[str], test_months: list[int]) -> dict[str, pd.DataFrame]:
    r"""Ranked accuracy tables of `models` at SKU level and, summing the SKU forecasts, at planta and total level."""
    tables = {"sku": rank_by_wape([{"model": n, **score(res["y"].to_numpy(), res[f"p_{n}"].to_numpy())} for n in models])}
    for level in ("planta", "total"):
        actual, bottom = [], {n: [] for n in models}
        for t in test_months:
            month_actual, month_bottom = level_actual_and_bottom_up(S, res, t, level, models)
            actual.append(month_actual)
            for n in models:
                bottom[n].append(month_bottom[n])
        y = np.concatenate(actual)
        tables[level] = rank_by_wape([{"model": n, **score(y, np.concatenate(bottom[n]))} for n in models])
    return tables
