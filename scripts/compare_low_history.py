"""
SKUs with little history (1 to 3 months): compare strategies against the current `ml_mean`, one month ahead.

Strategies (src/utils/lowhist_utils.py): e1 SKU attributes in the trees, e2 family shrinkage, e3 cohort rule, e5 median
objective (LightGBM with absolute-error loss). The policy `pol_*` uses a strategy only for SKUs with up to 3 months of history
and `ml_mean` for the rest. The parameter of e2 is chosen before the test months. Metrics: WAPE, accuracy, bias, MAE.

Usage:
    python scripts/compare_low_history.py data/consumos_long.csv
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.cache_utils import check_alignment, check_meta, expected_meta, panel_fingerprint, write_meta
from src.utils.evaluation_utils import test_months_of
from src.utils.feature_utils import SERIES_FEATURES, make_frame
from src.utils.lowhist_utils import (
    LOW_HISTORY, SKU_FEATURES, add_sku_features, attribute_table, cohort_forecast, family_forecast, fit_predict_l1, history_months,
)  # fmt: skip
from src.utils.member_utils import ML_MODELS, ml_forecast
from src.utils.metrics_utils import score
from src.utils.panel_utils import build_panel, load_consumption
from src.utils.tuning_utils import check_tuning_window

K0_GRID = [0.5, 1, 3, 6, 12]
GROUPS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-11", 6, 11), ("12-23", 12, 23), ("24+", 24, 999)]
STRATEGIES = ["e1", "e2", "e3", "e5"]


def pick_k0(S, frame, attrs, last_origin: int) -> tuple[float, dict]:
    r"""k0 of e2 with the lowest WAPE on SKUs with little history, using only origins whose target is before the test."""
    errors = {k0: [0.0, 0.0] for k0 in K0_GRID}
    for origin in range(8, last_origin + 1):
        rows = frame[frame["o"] == origin]
        low = rows[history_months(S, rows) <= LOW_HISTORY]
        if low.empty:
            continue
        for k0 in K0_GRID:
            p = family_forecast(S, low, attrs, k0)
            errors[k0][0] += float(np.abs(low["y"].to_numpy() - p).sum())
            errors[k0][1] += float(low["y"].sum())
    wape = {k0: e[0] / e[1] for k0, e in errors.items()}
    return min(wape, key=wape.get), wape


def strategy_forecasts(S, frame, attrs, tuned, test_months, k0) -> pd.DataFrame:
    r"""One row per (series, test month) with p_e1 .. p_e5; each month uses only the targets known at its origin."""
    parts = []
    for t in test_months:
        origin = t - 1
        train, rows = frame[frame["t"] <= origin], frame[frame["o"] == origin].reset_index(drop=True)
        block = rows[["series", "o", "t"]].copy()
        block["p_e1"] = np.mean([ml_forecast(k, train, rows, tuned, SERIES_FEATURES + SKU_FEATURES) for k in ML_MODELS], axis=0)
        block["p_e2"] = family_forecast(S, rows, attrs, k0)
        block["p_e3"] = cohort_forecast(S, rows)
        block["p_e5"] = fit_predict_l1(train, rows)
        parts.append(block)
        print(f"    target {S.labels[t]} done", flush=True)
    return pd.concat(parts, ignore_index=True)


def table(res: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = {}
    for name in models:
        ok = res[f"p_{name}"].notna()
        if ok.any():
            rows[name] = score(res.loc[ok, "y"].to_numpy(), res.loc[ok, f"p_{name}"].to_numpy())
    return pd.DataFrame(rows).T[["n", "mae", "wape", "accuracy", "bias_pct"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--tuning", type=Path, default=Path("results/tuning_trees.json"))
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--base", type=Path, default=Path("results/member_forecasts.csv"), help="one-month forecasts of run_ensembles.py")
    parser.add_argument("--predictions", type=Path, default=Path("results/low_history_preds.csv"))
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)

    S = build_panel(load_consumption(args.input))
    test_months = test_months_of(S, args.n_test)
    tuned = json.loads(args.tuning.read_text()) if args.tuning.exists() else {}
    check_tuning_window(tuned, S, args.n_test)
    attrs = attribute_table(S)
    frame = add_sku_features(make_frame(S, 1), attrs)

    k0, wape_by_k0 = pick_k0(S, frame, attrs, last_origin=test_months[0] - 2)
    print("e2 k0 chosen before the test (WAPE of SKUs with <=3 months, origins up to", S.labels[test_months[0] - 2], "):", {k: round(v, 3) for k, v in wape_by_k0.items()}, "->", k0)

    meta = {**expected_meta(S, 1, args.n_test, tuned), "k0": k0}
    if args.predictions.exists() and not args.refit:
        check_meta(args.predictions, meta)
        new = pd.read_csv(args.predictions)
        print(f"loaded {args.predictions}")
    else:
        new = strategy_forecasts(S, frame, attrs, tuned, test_months, k0)
        new.to_csv(args.predictions, index=False)
        write_meta(args.predictions, meta)

    check_meta(args.base, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1}, keys=("data", "n_series", "horizon"))
    base = pd.read_csv(args.base)
    check_alignment(base, S)
    base["p_base"] = base[[f"p_{k}" for k in ML_MODELS]].mean(axis=1)
    res = base[["series", "o", "t", "y", "p_base", "p_naive"]].merge(new[["series", "t", "p_e1", "p_e2", "p_e3", "p_e5"]], on=["series", "t"], how="left")
    res["age"] = history_months(S, res)
    low = res["age"] <= LOW_HISTORY
    for name in STRATEGIES:
        res[f"p_pol_{name}"] = np.where(low & res[f"p_{name}"].notna(), res[f"p_{name}"], res["p_base"])

    models = ["base", "naive", *STRATEGIES]
    all_scores = []
    print("\n=== By months of history at the origin (one month ahead, test months) ===")
    for label, lo, hi in GROUPS:
        part = res[(res["age"] >= lo) & (res["age"] <= hi)]
        tab = table(part, models)
        print(f"\n[{label} months] rows {len(part)}, share of the real consumption {100 * part['y'].sum() / res['y'].sum():.1f}%")
        print(tab.round(3))
        all_scores.append(tab.reset_index().rename(columns={"index": "model"}).assign(group=label))

    print("\n=== Policy: strategy for SKUs with up to 3 months of history, ml_mean for the rest ===")
    policies = ["base"] + [f"pol_{n}" for n in STRATEGIES]
    for label, part in (("all rows", res), ("<=3 months", res[low]), (">3 months", res[~low])):
        print(f"\n[{label}] rows {len(part)}")
        print(table(part, policies).round(3))

    print("\n=== Months (of 12) in which the strategy beats ml_mean, SKUs with 1 / up to 3 months of history ===")
    for label, part in (("1 month", res[res["age"] == 1]), ("<=3 months", res[low])):
        wins = {}
        for name in STRATEGIES:
            monthly = part.groupby("t").apply(
                lambda g, n=name: (np.abs(g["y"] - g[f"p_{n}"]).sum() < np.abs(g["y"] - g["p_base"]).sum()) if g[f"p_{n}"].notna().all() else np.nan
            )
            wins[name] = f"{int(monthly.sum())} of {int(monthly.notna().sum())}"
        print(f"  [{label}]", wins)

    print("\n=== Secondary: every origin since the start (no training needed: e2, e3, naive), SKUs with 1 to 3 months of history ===")
    sec = frame[(frame["t"] <= S.monthly.shape[1] - 1)].copy()
    sec["age"] = history_months(S, sec)
    sec = sec[(sec["age"] <= LOW_HISTORY) & (sec["o"] >= 8)]
    parts = []
    for origin, rows in sec.groupby("o"):
        rows = rows.copy()
        rows["p_e2"] = family_forecast(S, rows, attrs, k0)
        rows["p_e3"] = cohort_forecast(S, rows)
        parts.append(rows)
    sec = pd.concat(parts)
    for a in (1, 2, 3):
        part = sec[sec["age"] == a]
        print(f"\n[{a} month(s)] rows {len(part)} (origins {S.labels[int(part['o'].min())]} .. {S.labels[int(part['o'].max())]})")
        print(table(part, ["naive", "e2", "e3"]).round(3))

    out = args.predictions.with_name("low_history_scores.csv")
    pd.concat(all_scores).to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
