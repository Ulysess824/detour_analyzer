"""
Compare the forecast of the planners with the models, SKU by SKU, for the plantas and months of the planner files.

The planner files (data/planner_forecast_<month>.csv, written by transform_planner.py) have one row per planner SKU
(planta, mes, sku_planner, strategy, forecast_to). The planner SKU has no core width, so the series of the dataset that share
planta, type, gsm and width are added up. The model forecasts are the ones saved by run_ensembles.py (h=1, so made one month
before). Every model and the planner are scored on exactly the same SKUs and month.

A planner SKU-month is left out of the comparison when its strategy is not selected, the SKU is not in the data, its sub-grade
differs from the data, its series have less than --min-history months before the forecast origin, or the models have no
forecast for it. The left-out rows are saved: results/skus_poca_historia.csv (planta, sku, first date with consumption) and
results/filas_excluidas_planificador.csv (the other reasons).

Usage:
    python scripts/compare_planner.py data/consumos_long.csv --months 2026-09
    python scripts/compare_planner.py data/consumos_long.csv --months 2026-09 --plantas SCAN --strategies VMI "NO VMI" \\
        --min-history 0 --prefix planner_comparison     # the SCAN files the dashboard reads
"""

import argparse

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.cache_utils import check_alignment, check_meta, panel_fingerprint
from src.utils.ensemble_utils import add_ensembles
from src.utils.member_utils import ENSEMBLE_GROUPS
from src.utils.metrics_utils import score
from src.utils.panel_utils import build_panel, load_consumption
from src.utils.planner_utils import LOW_HISTORY, NO_SERIES, accuracy_table, compare_month_audit, first_consumption_dates, planner_key, series_table

HIGHLIGHT = ["ml_mean", "lgbm", "xgb", "rf", "naive", "mean6"]


def by_planta(table: pd.DataFrame) -> pd.DataFrame:
    r"""Rows, real and planner volume, accuracy of the planner, of ml_mean and of naive, and who was closer, per planta."""
    rows = {}
    for planta, g in table.groupby("planta"):
        real = g["real"].to_numpy()
        closer = int(((g["real"] - g["ml_mean"]).abs() < (g["real"] - g["planner"]).abs()).sum())
        rows[planta] = {
            "rows": len(g),
            "real_TO": round(real.sum()),
            "planner_TO": round(g["planner"].sum()),
            **{f"acc_{c}": round(100 * score(real, g[c].to_numpy())["accuracy"], 1) for c in ("planner", "ml_mean", "naive")},
            "ml_mean_closer": closer,
        }
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--planner-glob", default="data/planner_forecast_*.csv")
    parser.add_argument("--months", nargs="+", default=None, help="months to compare (YYYY-MM); default: the last month of the planner files")
    parser.add_argument("--plantas", nargs="+", default=None, help="plantas to compare; default: all the plantas of the planner files")
    parser.add_argument("--strategies", nargs="+", default=["VMI"], help='planner strategies to keep, or ALL (default: VMI)')
    parser.add_argument("--min-history", type=int, default=3, help="months of history before the origin a SKU needs (0: no filter)")
    parser.add_argument("--predictions", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="planner_plantas", help="the comparison of each month is saved as <out-dir>/<prefix>_<month>.csv")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    strategies = None if args.strategies == ["ALL"] else args.strategies

    df = load_consumption(args.input)
    S = build_panel(df)
    files = sorted(Path().glob(args.planner_glob))
    if not files:
        raise SystemExit(f"No planner files match {args.planner_glob}.")
    planner = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    months = args.months or [sorted(planner["mes"].unique())[-1]]
    if args.plantas:
        planner = planner[planner["planta"].isin(args.plantas)]

    check_meta(args.predictions, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1}, keys=("data", "n_series", "horizon"))
    res = pd.read_csv(args.predictions)
    check_alignment(res, S)
    res = add_ensembles(res, ENSEMBLE_GROUPS, horizon=1)
    all_models = [c[2:] for c in res.columns if c.startswith("p_")]
    pairs = series_table(S, df)

    tables, left_out = [], []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for month in months:
        if month not in S.labels:
            raise SystemExit(f"Month {month} is not in the data ({S.labels[0]} .. {S.labels[-1]}).")
        if S.labels.index(month) not in set(res["t"]):
            raise SystemExit(f"The saved forecasts do not include {month}; run run_ensembles.py with a test window that does.")
        wanted = planner[planner["mes"] == month]
        if wanted.empty:
            raise SystemExit(f"No planner rows for {month} in {args.planner_glob}.")
        table, excluded = compare_month_audit(wanted, res, pairs, S.labels.index(month), all_models, args.min_history, strategies)
        table.insert(0, "mes", month)
        out = args.out_dir / f"{args.prefix}_{month}.csv"
        table.to_csv(out, index=False)
        print(f"{month}: {len(table)} SKUs compared, {len(excluded)} left out | real {table['real'].sum():,.1f}, planner {table['planner'].sum():,.1f} TO | saved {out}")
        tables.append(table)
        left_out.append(excluded)

    T = pd.concat(tables, ignore_index=True)
    X = pd.concat(left_out, ignore_index=True)
    kept = len(T)
    print(f"\nstrategies {strategies or 'ALL'}, min history {args.min_history} months | compared {kept} SKU-months, left out {len(X)}")
    if len(X):
        print("left out by reason:", X["reason"].value_counts().to_dict())

    print("\nAccuracy on the compared SKUs (rank 1 = lowest WAPE):")
    print(accuracy_table(T, ["planner", *all_models]).head(8).round(4))
    print("\nBy planta (accuracy in %):")
    print(by_planta(T))
    if T["mes"].nunique() > 1:
        print("\nBy month (accuracy in %):")
        print(pd.DataFrame({m: {c: 100 * score(g["real"].to_numpy(), g[c].to_numpy())["accuracy"] for c in ["planner", *HIGHLIGHT]} for m, g in T.groupby("mes")}).round(1))
    wins = int(((T["real"] - T["ml_mean"]).abs() < (T["real"] - T["planner"]).abs()).sum())
    print(f"\nml_mean closer to the real than the planner in {wins} of {kept} SKU-months")

    first = first_consumption_dates(df)
    short = X[X["reason"].isin([LOW_HISTORY, NO_SERIES])].drop_duplicates(["planta", "sku"]).copy()
    short["primera_fecha_consumo"] = [first.get((p, planner_key(s)), pd.NaT) if r == LOW_HISTORY else pd.NaT for p, s, r in zip(short["planta"], short["sku"], short["reason"])]
    short = short.sort_values(["planta", "sku"])[["planta", "sku", "primera_fecha_consumo"]]
    short["primera_fecha_consumo"] = pd.to_datetime(short["primera_fecha_consumo"]).dt.strftime("%Y-%m-%d").fillna("")
    short.to_csv(args.out_dir / "skus_poca_historia.csv", index=False)
    X[X["reason"] != LOW_HISTORY].to_csv(args.out_dir / "filas_excluidas_planificador.csv", index=False)
    print(f"\nsaved {args.out_dir / 'skus_poca_historia.csv'} ({len(short)} SKUs) and {args.out_dir / 'filas_excluidas_planificador.csv'} ({int((X['reason'] != LOW_HISTORY).sum())} rows)")


if __name__ == "__main__":
    main()
