"""
Load the saved predictions into the SQLite database (table consumos_predichos), overwriting each block.

Beta rule: every load replaces the rows of the same estrategia and mes_origen and stamps creado_en. Sources (each is skipped when the
file is missing): results/member_forecasts.csv -> h1; results/horizon_strategies.csv -> directa, parcial, iterada;
results/nowcast_input_ml_mean_2026-06_09.csv -> nowcast_media; results/forecast_<month>.csv -> h1 (real forecast of the next month).
The series index stored in the caches is checked against the panel before it is translated to (planta, sku).

Usage:
    python scripts/load_predictions.py data/consumos_long.csv --db data/detour.db
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.engine.db_utils import connect, create_schema, predictions_from_cache, predictions_from_forecast, replace_predictions, upsert_run
from src.engine.panel_utils import build_panel, load_consumption
from src.evaluation.cache_utils import check_alignment

DIRECT = {"p_direct_lgbm": ("directa", "lgbm"), "p_direct_xgb": ("directa", "xgb"), "p_direct_rf": ("directa", "rf"),
          "p_direct": ("directa", "ml_mean"), "p_naive": ("directa", "naive"), "p_partial": ("parcial", "ml_mean"),
          "p_iterated": ("iterada", "ml_mean")}
NOWCAST = {"p_nowcast": ("nowcast_media", "ml_mean"), "p_nowcast_lgbm": ("nowcast_media", "lgbm"),
           "p_nowcast_xgb": ("nowcast_media", "xgb"), "p_nowcast_rf": ("nowcast_media", "rf")}
MEMBERS = {"p_lgbm": ("h1", "lgbm"), "p_xgb": ("h1", "xgb"), "p_rf": ("h1", "rf"), "p_naive": ("h1", "naive"), "ml_mean": ("h1", "ml_mean")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="master csv (to translate the series index of the caches)")
    parser.add_argument("--db", type=Path, default=Path("data/detour.db"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    S = build_panel(load_consumption(args.input))
    con = connect(args.db)
    create_schema(con)
    batches = []

    path = args.results / "member_forecasts.csv"
    if path.exists():
        frame = pd.read_csv(path)
        check_alignment(frame, S)
        frame["ml_mean"] = frame[["p_lgbm", "p_xgb", "p_rf"]].mean(axis=1)
        batches.append((path.name, predictions_from_cache(frame, S, MEMBERS)))
    for name, columns in (("horizon_strategies.csv", DIRECT), ("nowcast_input_ml_mean_2026-06_09.csv", NOWCAST)):
        path = args.results / name
        if path.exists():
            frame = pd.read_csv(path)
            check_alignment(frame, S)
            batches.append((path.name, predictions_from_cache(frame, S, columns)))
    for path in sorted(args.results.glob("forecast_*.csv")):
        manifest_file = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else None
        batches.append((path.name, predictions_from_forecast(pd.read_csv(path)), manifest))

    for name, preds, *manifest in [(b[0], b[1], b[2] if len(b) > 2 else None) for b in batches]:
        run_id = manifest[0]["run"]["run_id"] if manifest and manifest[0] else None
        if run_id:
            upsert_run(con, manifest[0])
        n = replace_predictions(con, preds, run_id=run_id)
        print(f"{name}: {n:_} predictions ({', '.join(sorted(preds['estrategia'].unique()))}){f', run {run_id}' if run_id else ''}")
    total = con.execute("SELECT estrategia, COUNT(*) FROM consumos_predichos GROUP BY estrategia ORDER BY estrategia").fetchall()
    print("in the database:", dict(total))
    con.close()


if __name__ == "__main__":
    main()
