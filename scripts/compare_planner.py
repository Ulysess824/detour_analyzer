"""
Compare the forecast of the planners with the models, SKU by SKU, for the plantas and months of the planner files.

The planner files (data/planner_forecast_<month>.csv, written by transform_planner.py) have one row per planner SKU
(planta, mes, sku_planner, strategy, forecast_to). The planner SKU has no core width, so the series of the dataset that share
planta, type, gsm and width are added up. The model forecasts are the ones saved by run_ensembles.py (h=1, so made one month
before). Every model and the planner are scored on exactly the same SKUs and month.

With --internal (data/internal_forecast_plantas.csv, from transform_internal.py) the forecast of the company's internal model is added
and every metric uses only the SKU-months the three sources share (planner, internal model and ml_mean): the SKUs the internal model
has no forecast for are kept in the saved table with internal_status "faltante" but stay out of the metrics.

A planner SKU-month is left out of the comparison when its strategy is not selected, the SKU is not in the data, its sub-grade
differs from the data, its series have less than --min-history months before the forecast origin, or the models have no
forecast for it. The left-out rows are saved: results/skus_poca_historia.csv (planta, sku, first date with consumption) and
results/filas_excluidas_planificador.csv (the other reasons).

Usage:
    python scripts/compare_planner.py data/consumos_long.csv --months 2026-09
    python scripts/compare_planner.py data/consumos_long.csv --months 2026-09 --internal data/internal_forecast_plantas.csv
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
from src.utils.planner_utils import LOW_HISTORY, NO_SERIES, accuracy_table, add_internal, compare_month_audit, first_consumption_dates, planner_key, series_table

HIGHLIGHT = ["ml_mean", "lgbm", "xgb", "rf", "naive", "mean6"]


def closest_counts(g: pd.DataFrame, sources: list[str]) -> dict:
    r"""How many SKU-months each source has the smallest absolute error in; a tie for the smallest counts for nobody."""
    err = g[sources].sub(g["real"], axis=0).abs()
    best = err.min(axis=1)
    unique = (err.eq(best, axis=0)).sum(axis=1) == 1
    return err[unique].idxmin(axis=1).value_counts().to_dict()


def by_planta(table: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    r"""Rows, real and planner volume, accuracy of each source and of naive, and how many SKUs each source was the closest to, per planta."""
    rows = {}
    for planta, g in table.groupby("planta"):
        real = g["real"].to_numpy()
        closest = closest_counts(g, sources)
        rows[planta] = {
            "rows": len(g),
            "real_TO": round(real.sum()),
            "planner_TO": round(g["planner"].sum()),
            **{f"acc_{c}": round(100 * score(real, g[c].to_numpy())["accuracy"], 1) for c in (*sources, "naive")},
            **{f"closest_{c}": int(closest.get(c, 0)) for c in sources},
        }
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--planner-glob", default="data/planner_forecast_*.csv")
    parser.add_argument("--months", nargs="+", default=None, help="months to compare (YYYY-MM); default: the last month of the planner files")
    parser.add_argument("--plantas", nargs="+", default=None, help="plantas to compare; default: all the plantas of the planner files")
    parser.add_argument("--strategies", nargs="+", default=["VMI"], help='planner strategies to keep, or ALL (default: VMI)')
    parser.add_argument("--internal", type=Path, default=None, help="internal model forecasts (long csv): adds the column and restricts the metrics to the SKUs of the three sources")
    parser.add_argument("--min-history", type=int, default=3, help="months of history before the origin a SKU needs (0: no filter)")
    parser.add_argument("--predictions", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="planner_plantas", help="the comparison of each month is saved as <out-dir>/<prefix>_<month>.csv")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 30)
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

    internal = pd.read_csv(args.internal) if args.internal else None
    sources = ["planner", "ml_mean"] + (["internal"] if internal is not None else [])
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
        if internal is not None:
            table, missing = add_internal(table, internal, month)
            excluded = pd.concat([excluded, missing], ignore_index=True)
        out = args.out_dir / f"{args.prefix}_{month}.csv"
        table.to_csv(out, index=False)
        in_metrics = table if internal is None else table[table["internal"].notna()]
        print(f"{month}: {len(table)} SKUs compared, {len(in_metrics)} in the metrics, {len(excluded)} left out | real {in_metrics['real'].sum():,.1f}, planner {in_metrics['planner'].sum():,.1f} TO | saved {out}")
        tables.append(table)
        left_out.append(excluded)

    T_all = pd.concat(tables, ignore_index=True)
    T = T_all if internal is None else T_all[T_all["internal"].notna()].reset_index(drop=True)
    X = pd.concat(left_out, ignore_index=True)
    kept = len(T)
    print(f"\nstrategies {strategies or 'ALL'}, min history {args.min_history} months | compared {len(T_all)} SKU-months, left out {len(X)}")
    if internal is not None:
        print(f"metrics on the {kept} SKU-months of the three sources; {len(T_all) - kept} more have the internal model as \"faltante\" (in the table, outside the metrics)")
    if len(X):
        print("left out by reason:", X["reason"].value_counts().to_dict())

    print("\nAccuracy on the compared SKUs (rank 1 = lowest WAPE):")
    print(accuracy_table(T, [*sources, "naive"]).round(4))
    print("\nBy planta (accuracy in %):")
    print(by_planta(T, sources))
    if T["mes"].nunique() > 1:
        print("\nBy month (accuracy in %):")
        print(pd.DataFrame({m: {c: 100 * score(g["real"].to_numpy(), g[c].to_numpy())["accuracy"] for c in [*sources[:1], *sources[2:], *HIGHLIGHT]} for m, g in T.groupby("mes")}).round(1))
    closest = closest_counts(T, sources)
    print(f"\nclosest to the real, in SKU-months of {kept} (ties count for nobody): {closest}")

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
