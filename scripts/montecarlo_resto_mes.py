"""
Predict the rest of a month (the days after a cutoff) by Monte Carlo and score it against what really happened.

Experiment of docs/plan_montecarlo_mes_parcial.md (phase 2: the simulator on its own). For every series alive in the
previous month, with only the days up to the cutoff known:

    montecarlo   mean of the simulated rest of the month (src/experiments/montecarlo_utils.py)
    prorata      consumption so far / non-Sunday days elapsed * non-Sunday days left
    hist_rate    mean daily rate of the previous 6 months * non-Sunday days left   (the `perday6` idea)

Scores: accuracy (1 - WAPE) of the rest of the month and of the whole month (first days real + rest) at SKU, planta and
total level, the bias (forecast / real - 1), and how often the real value falls inside the 10-90% band of the draws.

Usage:
    python scripts/montecarlo_resto_mes.py data/consumos_long.csv --month 2026-09
    python scripts/montecarlo_resto_mes.py data/consumos_long.csv --month 2026-09 --last-months 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.evaluation.metrics_utils import score
from src.experiments.montecarlo_utils import simulate_rest_of_month
from src.engine.panel_utils import build_panel, load_consumption
from src.engine.partial_utils import month_to_date


def evaluate_month(S, month: int, cutoff_day: int, draws: int, seed: int) -> dict:
    r"""Forecasts and real values of the rest of `month`, summed at SKU, planta and total level."""
    alive = ~np.isnan(S.monthly[:, month - 1])  # series that existed in the previous month
    start = int(S.month_first_date[month])
    cutoff, end = start + cutoff_day, start + int(S.calendar_days[month])
    mtd, _, elapsed = month_to_date(S, cutoff_day)
    left_days = S.business_days[month] - elapsed[month]  # non-Sunday days after the cutoff

    sim = simulate_rest_of_month(S, month, cutoff_day, draws, seed)[:, alive]
    real_rest = S.daily[alive, cutoff:end].sum(axis=1, dtype=float)
    first = mtd[alive, month]
    prior = np.nansum(S.monthly[alive, month - 6 : month], axis=1) / S.business_days[month - 6 : month].sum()
    forecasts = {
        "montecarlo": sim.mean(axis=0),
        "prorata": first / max(elapsed[month], 1) * left_days,
        "hist_rate": prior * left_days,
    }
    planta = S.planta_code[alive]
    return {"real_rest": real_rest, "first": first, "forecasts": forecasts, "sim": sim, "planta": planta, "n_plantas": len(S.plantas)}


def by_level(values: np.ndarray, planta: np.ndarray, n_plantas: int) -> dict[str, np.ndarray]:
    r"""Sum an array (..., n_series) at planta and total level; the series level is the array itself."""
    per_planta = np.stack([values[..., planta == p].sum(axis=-1) for p in range(n_plantas)], axis=-1)
    return {"SKU": values, "planta": per_planta, "total": values.sum(axis=-1, keepdims=True)}


def summarize(r: dict) -> pd.DataFrame:
    r"""Accuracy and bias of every forecast of the rest of the month and of the whole month, plus the band coverage."""
    rows = []
    real = by_level(r["real_rest"], r["planta"], r["n_plantas"])
    first = by_level(r["first"], r["planta"], r["n_plantas"])
    band = by_level(r["sim"], r["planta"], r["n_plantas"])
    for name, forecast in r["forecasts"].items():
        pred = by_level(forecast, r["planta"], r["n_plantas"])
        for level in ("SKU", "planta", "total"):
            y, p = real[level].ravel(), pred[level].ravel()
            row = {"model": name, "level": level, "acc_rest": score(y, p)["accuracy"], "bias_rest": p.sum() / y.sum() - 1}
            full_y, full_p = (first[level] + real[level]).ravel(), (first[level] + pred[level]).ravel()
            row["acc_month"] = score(full_y, full_p)["accuracy"]
            if name == "montecarlo":
                lo, hi = np.percentile(band[level], [10, 90], axis=0)
                row["coverage_10_90"] = float(((y >= lo.ravel()) & (y <= hi.ravel())).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--month", default="2026-09", help="month to complete (YYYY-MM)")
    parser.add_argument("--cutoff-day", type=int, default=15, help="calendar days of the month already known")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--last-months", type=int, default=0, help="also score this many months up to --month, one summary line each")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    S = build_panel(load_consumption(args.input))
    if args.month not in S.labels:
        raise SystemExit(f"{args.month} is not in the data ({S.labels[0]} .. {S.labels[-1]}).")
    month = S.labels.index(args.month)
    r = evaluate_month(S, month, args.cutoff_day, args.draws, args.seed)
    n = len(r["real_rest"])
    print(f"month {args.month}, cutoff day {args.cutoff_day}, {args.draws} draws, {n:,} series alive in the previous month")
    print(f"real first {args.cutoff_day} days {r['first'].sum():,.0f} TO | real rest {r['real_rest'].sum():,.0f} TO | "
          f"simulated rest (mean) {r['sim'].sum(axis=1).mean():,.0f} TO (10-90%: {np.percentile(r['sim'].sum(axis=1), 10):,.0f} - {np.percentile(r['sim'].sum(axis=1), 90):,.0f})")
    print("\nrest of the month and whole month (acc = 1 - WAPE; bias = forecast / real - 1):")
    print(summarize(r).to_string(index=False))

    if args.last_months:
        lines = []
        for m in range(month - args.last_months + 1, month + 1):
            s = summarize(evaluate_month(S, m, args.cutoff_day, args.draws, args.seed))
            sku = s[s.level == "SKU"].set_index("model")
            tot = s[(s.level == "total")].set_index("model")
            lines.append({
                "month": S.labels[m],
                "acc_rest_SKU_mc": sku.loc["montecarlo", "acc_rest"], "prorata": sku.loc["prorata", "acc_rest"], "hist_rate": sku.loc["hist_rate", "acc_rest"],
                "bias_total_mc": tot.loc["montecarlo", "bias_rest"], "cover_SKU": sku.loc["montecarlo", "coverage_10_90"], "cover_total": tot.loc["montecarlo", "coverage_10_90"],
            })
        table = pd.DataFrame(lines).set_index("month")
        print(f"\nlast {args.last_months} months (SKU accuracy of the rest of the month):")
        print(table.to_string())
        print("\nmean:", table.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
