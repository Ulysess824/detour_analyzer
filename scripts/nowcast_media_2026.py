"""
Backtest of the mean nowcast over the months of 2026, by planta and SKU.

At the cutoff day (15) the rest of each month is forecast with media, mediana and two Monte Carlo simulators; the full month is
the real first days plus that rest. The full month is compared with the one-month-ahead ml_mean (results/member_forecasts.csv,
made before the month starts) and with the previous complete month. Tables: global, by planta and by planta and SKU (csv).
Other cutoff days are scored for media and mediana only. Logic: src/experiments/nowcast_backtest_utils.py.

Usage:
    python scripts/nowcast_media_2026.py data/consumos_long.csv
    python scripts/nowcast_media_2026.py data/consumos_long.csv --months 2026-07 2026-08 --draws 100 --out-dir results
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.panel_utils import build_panel, load_consumption
from src.evaluation.cache_utils import check_alignment, check_meta, panel_fingerprint
from src.experiments.nowcast_backtest_utils import (
    NOWCAST_MODELS,
    REFERENCES,
    closer_counts,
    decision_criteria,
    global_scores,
    h1_reference,
    light_month_table,
    month_table,
    planta_scores,
    series_summary,
)

LEVELS = ["acierto_SKU", "acierto_planta", "acierto_total"]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.to_string())


def ranking_table(scores: pd.DataFrame) -> pd.DataFrame:
    r"""Position of each model at every level (1 = best) and the mean position, sorted by it."""
    ranks = scores[LEVELS].rank(ascending=False, method="min").astype(int)
    ranks["posicion_media"] = ranks.mean(axis=1)
    return ranks.sort_values("posicion_media")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--months", nargs="+", default=None, help="months (YYYY-MM); default every month of 2026 in the data")
    parser.add_argument("--cutoff-day", type=int, default=15)
    parser.add_argument("--cutoffs", type=int, nargs="*", default=[10, 20], help="other cutoff days scored for media and mediana only")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", type=Path, default=Path("results/member_forecasts.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    S = build_panel(load_consumption(args.input))
    cache = pd.read_csv(args.reference)
    check_meta(args.reference, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1}, keys=("data", "n_series", "horizon"))
    check_alignment(cache, S)
    h1 = h1_reference(cache)

    months = args.months or [m for m in S.labels if m.startswith("2026")]
    idx = [S.labels.index(m) for m in months]
    print(f"months {months[0]}..{months[-1]} ({len(idx)}), cutoff day {args.cutoff_day}, {args.draws} draws")
    parts = []
    for m in idx:
        part = month_table(S, m, args.cutoff_day, args.draws, args.seed, h1)
        cover = part["real_mes"].sum() / S.total_monthly[m]
        print(f"  {S.labels[m]}: {len(part):,} series, real {part['real_mes'].sum():,.0f} TO ({cover:.1%} of the month's volume), ml_mean reference for {part['mes_ml_mean_h1'].notna().mean():.1%}", flush=True)
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    n = len(S.plantas)

    full, rest = global_scores(long, n, "mes"), global_scores(long, n, "resto")
    pooled = full[full["mes"] == "todos"].set_index("modelo")
    show("MES COMPLETO, acumulado de los meses: acierto, sesgo del total y pronostico (TO)", pooled[LEVELS + ["sesgo_total", "pronostico_TO", "real_TO"]])
    show("RANKING del mes completo (1 = mejor)", ranking_table(pooled))
    for level in LEVELS:
        show(f"MES COMPLETO por mes: {level}", full[full["mes"] != "todos"].pivot(index="mes", columns="modelo", values=level))
    show("MES COMPLETO por mes: sesgo del total (pronostico / real - 1)", full[full["mes"] != "todos"].pivot(index="mes", columns="modelo", values="sesgo_total"))
    show("RESTO DEL MES, acumulado: acierto, sesgo del total", rest[rest["mes"] == "todos"].set_index("modelo")[LEVELS + ["sesgo_total", "pronostico_TO", "real_TO"]])

    pf, pr = planta_scores(long, "mes"), planta_scores(long, "resto")
    show("POR PLANTA, mes completo, acumulado: acierto del total de la planta", pf[pf["mes"] == "todos"].pivot(index="planta", columns="modelo", values="acierto_planta"))
    show("POR PLANTA, mes completo, acumulado: acierto de sus SKU juntos", pf[pf["mes"] == "todos"].pivot(index="planta", columns="modelo", values="acierto_SKU"))

    print("\nSKU-MESES MAS CERCANOS AL REAL, mes completo (empate no cuenta)")
    for ref in REFERENCES:
        print(f"  media vs {ref}: {closer_counts(long, 'media', ref)}")

    show("CRITERIOS DE DECISION", decision_criteria(rest, full, pf).set_index("criterio"))

    print("\nOTROS DIAS DE CORTE (media y mediana, acumulado de los meses)")
    rows = []
    for cut in sorted({*args.cutoffs, args.cutoff_day}):
        sub = long if cut == args.cutoff_day else pd.concat([light_month_table(S, m, cut, h1) for m in idx], ignore_index=True)
        for target in ("resto", "mes"):
            sc = global_scores(sub, n, target)
            sc = sc[(sc["mes"] == "todos") & sc["modelo"].isin(["media", "mediana"])]
            rows.extend(sc.assign(corte=cut).to_dict("records"))
    show("", pd.DataFrame(rows).set_index(["objetivo", "corte", "modelo"])[LEVELS + ["sesgo_total"]])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    year = months[0][:4]
    outs = {
        f"nowcast_{year}_global.csv": pd.concat([full, rest], ignore_index=True),
        f"nowcast_{year}_planta.csv": pd.concat([pf, pr], ignore_index=True),
        f"nowcast_{year}_planta_sku.csv": long.drop(columns=["series", "planta_code"]),
        f"nowcast_{year}_planta_sku_resumen.csv": series_summary(long, "mes"),
    }
    for name, table in outs.items():
        table.to_csv(args.out_dir / name, index=False, float_format="%.4f")
        print(f"saved {args.out_dir / name}  ({len(table):,} rows)")


if __name__ == "__main__":
    main()
