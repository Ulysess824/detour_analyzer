"""
Ensembles of machine-learning and classical econometric forecasts, and a Model Confidence Set.

Members (forecast of the monthly total of each SKU, rolling origin, horizon --horizon):
    machine learning   LightGBM, XGBoost, Random Forest (Optuna-tuned parameters from the tuning json)
    classical          naive, mean3/6/12, per-day rate, seasonal naive, mean6 x planta seasonal
                       ratio, simple exponential smoothing, damped Holt, ARIMA (AIC)

Ensembles, for each of the three groups (ml, econ, ml_econ = both together) and four methods
(mean, median, trimmed mean, weighted by inverse past MAE): see src/utils/ensemble_utils.py.

The Model Confidence Set of Hansen, Lunde and Nason (2011) is then run on the absolute-error loss
at SKU-month, planta-month and total-month level to see which models can be discarded.

The member forecasts are cached in --predictions; pass --refit to compute them again.

Usage:
    python scripts/run_ensembles.py data/consumos_long.csv --tuning results/tuning_trees.json
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.ensemble_utils import METHODS, add_ensembles
from src.utils.evaluation_utils import score_models_by_level, test_months_of
from src.utils.feature_utils import make_frame
from src.utils.mcs_utils import model_confidence_set
from src.utils.member_utils import ENSEMBLE_GROUPS, ML_MODELS, collect_member_forecasts
from src.utils.panel_utils import build_panel, load_consumption
from src.utils.tuning_utils import to_jsonable


def mcs_losses(S, res: pd.DataFrame, models: list[str], test_months: list[int], level: str):
    r"""Absolute-error loss of every model, and the target month of each loss observation."""
    if level == "sku":
        loss = pd.DataFrame({m: (res["y"] - res[f"p_{m}"]).abs() for m in models}).reset_index(drop=True)
        return loss, res["t"].to_numpy()
    rows, periods = [], []
    for t in test_months:
        block = res[res["t"] == t]
        if level == "planta":
            actual = S.planta_monthly[:, t]
            sums = {m: block.groupby("planta_code")[f"p_{m}"].sum().reindex(range(len(S.plantas))).fillna(0.0).to_numpy() for m in models}
        else:
            actual = np.array([S.total_monthly[t]])
            sums = {m: np.array([block[f"p_{m}"].sum()]) for m in models}
        for i in range(len(actual)):
            rows.append({m: abs(actual[i] - sums[m][i]) for m in models})
            periods.append(t)
    return pd.DataFrame(rows), np.array(periods)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--tuning", type=Path, default=Path("results/tuning_trees.json"))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--predictions", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--refit", action="store_true", help="recompute the member forecasts even if cached")
    parser.add_argument("--out", type=Path, default=Path("results/ensembles.json"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.25])
    parser.add_argument("--boot", type=int, default=5_000)
    parser.add_argument("--block", type=int, default=3, help="block length of the bootstrap, in months")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)

    S = build_panel(load_consumption(args.input))
    test_months = test_months_of(S, args.n_test)
    print(f"test: {S.labels[test_months[0]]} .. {S.labels[-1]} | horizon h={args.horizon}")

    if args.predictions.exists() and not args.refit:
        res = pd.read_csv(args.predictions)
        print(f"loaded member forecasts from {args.predictions}")
    else:
        tuned = json.loads(args.tuning.read_text()) if args.tuning.exists() else {}
        res = collect_member_forecasts(S, make_frame(S, args.horizon), test_months, args.horizon, tuned, args.jobs)
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.predictions, index=False)

    res = add_ensembles(res, ENSEMBLE_GROUPS, args.horizon)
    members = [m for group in ("ml", "econ") for m in ENSEMBLE_GROUPS[group]]
    ensembles = [f"{g}_{m}" for g in ENSEMBLE_GROUPS for m in METHODS]
    models = members + ensembles

    tables = score_models_by_level(S, res, models, test_months)
    for level, table in tables.items():
        print(f"\n=== accuracy at {level} level (rank 1 = lowest WAPE) ===")
        print(table[["rank", "n", "mae", "wape", "accuracy", "bias_pct"]])

    results = {"accuracy": {k: v.reset_index().to_dict("records") for k, v in tables.items()}, "mcs": {}}
    for level in ("sku", "planta", "total"):
        loss, period = mcs_losses(S, res, models, test_months, level)
        mcs = model_confidence_set(loss, period, block=args.block, n_boot=args.boot)
        mcs = mcs.sort_values(["mcs_p_value", "mean_loss"], ascending=[False, True])
        mcs.insert(0, "rank", range(1, len(mcs) + 1))
        for alpha in args.alphas:
            mcs[f"in_mcs_{int(alpha * 100)}"] = mcs["mcs_p_value"] >= alpha
        print(f"\n=== Model Confidence Set, {level} level (absolute-error loss, block {args.block}, {args.boot:_} bootstrap draws) ===")
        print(mcs.round(4))
        for alpha in args.alphas:
            keep = mcs.index[mcs["mcs_p_value"] >= alpha].tolist()
            out = [m for m in models if m not in keep]
            print(f"  alpha={alpha}: {len(keep)} models stay, {len(out)} can be discarded: {out}")
        results["mcs"][level] = mcs.reset_index().to_dict("records")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(results), indent=1))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
