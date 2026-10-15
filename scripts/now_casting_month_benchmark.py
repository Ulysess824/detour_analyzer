"""
Nowcasting month benchmark: media, mediana and two Monte Carlo simulators on the rest of the month in progress.

Models and information rules: src/utils/now_casting_current_month.py. Scores: accuracy (1 - WAPE) at SKU level (all series
pooled), planta level (all plantas pooled) and total level, and the ranking of the models by each.

Usage:
    python scripts/now_casting_month_benchmark.py data/consumos_long.csv --month 2026-09
    python scripts/now_casting_month_benchmark.py data/consumos_long.csv --month 2026-09 --last-months 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.now_casting_current_month import MODELS, accuracy, by_planta, level_scores, nowcast_month, planta_table, ranking
from src.utils.panel_utils import build_panel, load_consumption

LEVELS = ["acierto_SKU", "acierto_planta", "acierto_total"]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.to_string())


def ranking_table(scores: pd.DataFrame) -> pd.DataFrame:
    r"""Position of each model at every level and the mean position (lower is better), sorted by it."""
    ranks = pd.DataFrame({level: ranking(scores[level]) for level in LEVELS})
    ranks["posicion_media"] = ranks.mean(axis=1)
    return ranks.sort_values("posicion_media")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--month", default="2026-09")
    parser.add_argument("--cutoff-day", type=int, default=15)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--last-months", type=int, default=0, help="also score this many months up to --month and pool them")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    S = build_panel(load_consumption(args.input))
    n_plantas = len(S.plantas)
    month = S.labels.index(args.month)
    nc = nowcast_month(S, month, args.cutoff_day, args.draws, args.seed)
    print(f"month {args.month}, cutoff day {args.cutoff_day}, {args.draws} draws, {len(nc.series):,} series | real rest {nc.real_rest.sum():,.0f} TO")

    scores = level_scores(nc, n_plantas)
    show(f"GLOBAL {args.month}: accuracy of the rest of the month and total forecast (TO)", scores)
    show("RANKING by accuracy (1 = best)", ranking_table(scores))
    pt = planta_table(nc, S.plantas)
    show("BY PLANTA: accuracy of the planta total (planta:) and of its SKUs pooled (SKU:)", pt)

    if args.last_months:
        months = range(month - args.last_months + 1, month + 1)
        results = [nowcast_month(S, m, args.cutoff_day, args.draws, args.seed) for m in months]
        per_month = [level_scores(r, n_plantas) for r in results]
        mean = sum(per_month) / len(per_month)
        show(f"GLOBAL mean of {args.last_months} months (monthly scores averaged)", mean)
        show("RANKING by accuracy, 12-month mean (1 = best)", ranking_table(mean))
        counts = pd.DataFrame({lv: pd.Series([s[lv].idxmax() for s in per_month]).value_counts() for lv in LEVELS}).reindex(MODELS).fillna(0).astype(int)
        show("MONTHS WON (best accuracy in the month)", counts)
        pooled = {}
        for model in MODELS:
            real = np.concatenate([r.real_rest for r in results])
            fore = np.concatenate([r.forecasts[model] for r in results])
            real_p = np.concatenate([by_planta(r.real_rest, r.planta, n_plantas) for r in results])
            fore_p = np.concatenate([by_planta(r.forecasts[model], r.planta, n_plantas) for r in results])
            pooled[model] = {"acierto_SKU": accuracy(real, fore), "acierto_planta": accuracy(real_p, fore_p),
                             "sesgo_total_%": 100 * (fore.sum() / real.sum() - 1)}
        show("POOLED over all months (one WAPE over every SKU-month / planta-month)", pd.DataFrame(pooled).T)
        pl = {}
        for code, name in enumerate(S.plantas):
            row = {}
            for model in MODELS:
                real = np.array([by_planta(r.real_rest, r.planta, n_plantas)[code] for r in results])
                fore = np.array([by_planta(r.forecasts[model], r.planta, n_plantas)[code] for r in results])
                if real.sum() > 0:
                    row[model] = accuracy(real, fore)
            if row:
                pl[name] = row
        show("BY PLANTA over the months: accuracy of the planta total, pooled", pd.DataFrame(pl).T)


if __name__ == "__main__":
    main()
