"""Tracking of a closed month: scores at the three levels, alerts at their thresholds, storage in the csv and in SQLite, and the schema migration."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.db_utils import (
    connect, create_schema, load_tracking_csv, read_predictions, read_tracking_csv, replace_predictions, replace_tracking, upsert_run, write_consumption,
)
from src.engine.fingerprints import panel_fingerprint
from src.engine.panel_utils import build_panel
from src.evaluation.cache_utils import write_meta
from src.evaluation.tracking_run import is_backtest, strategy_of, track_horizon_backtests, track_saved_forecast
from src.evaluation.tracking_utils import (
    THRESHOLDS, THRESHOLDS_TWO_MONTHS, check_forecast_matches, check_month_closed, coverage_ok, find_alerts, forecast_coverage, month_scores, thresholds_for, tracking_rows,
    write_tracking_csv,
)

REPO = Path(__file__).resolve().parents[1]


def consumption():
    r"""January and February by hand; B/s4 only exists in February (no forecast for it)."""
    rows = [
        ("A", "s1", "2024-01-10", 10.0), ("A", "s2", "2024-01-10", 20.0), ("B", "s3", "2024-01-10", 30.0),
        ("A", "s1", "2024-02-10", 12.0), ("A", "s2", "2024-02-10", 18.0), ("B", "s3", "2024-02-10", 30.0), ("B", "s4", "2024-02-10", 5.0),
    ]
    return pd.DataFrame(rows, columns=["planta", "sku", "fecha", "consumo"]).assign(fecha=lambda d: pd.to_datetime(d["fecha"]))


def forecast():
    r"""The February forecast: ml_mean wrong by 2, 2 and 6; naive is January."""
    return pd.DataFrame(
        {"planta": ["A", "A", "B"], "sku": ["s1", "s2", "s3"], "mes": "2024-02", "lgbm": [11.0, 19.0, 33.0], "xgb": [9.0, 21.0, 39.0], "rf": [10.0, 20.0, 36.0],
         "ml_mean": [10.0, 20.0, 36.0], "naive": [10.0, 20.0, 30.0]}
    )


@pytest.fixture()
def scores():
    return month_scores(build_panel(consumption()), forecast(), "2024-02")


def pick(table, nivel, modelo="ml_mean", planta=""):
    row = table[(table["nivel"] == nivel) & (table["modelo"] == modelo) & (table["planta"] == planta)]
    assert len(row) == 1, (nivel, modelo, planta, len(row))
    return row.iloc[0]


def test_scores_at_the_three_levels_match_a_hand_calculation(scores):
    sku = pick(scores, "sku")  # actual 12, 18, 30 against 10, 20, 36: absolute error 10 of 60
    assert (sku["n"], sku["real_TO"], sku["pronostico_TO"]) == (3, 60.0, 66.0)
    assert sku["acierto"] == pytest.approx(1 - 10 / 60) and sku["wape"] == pytest.approx(10 / 60) and sku["bias_pct"] == pytest.approx(-6 / 60 * 100)
    planta = pick(scores, "planta")  # A: 30 against 30; B: 35 (with the new SKU) against 36
    assert (planta["real_TO"], planta["pronostico_TO"]) == (65.0, 66.0) and planta["acierto"] == pytest.approx(1 - 1 / 65)
    total = pick(scores, "total")
    assert (total["real_TO"], total["pronostico_TO"], total["n"]) == (65.0, 66.0, 1) and total["acierto"] == pytest.approx(1 - 1 / 65)
    assert pick(scores, "planta", planta="A")["acierto"] == pytest.approx(1.0) and pick(scores, "planta", planta="B")["acierto"] == pytest.approx(1 - 1 / 35)


def test_every_model_is_scored_and_the_naive_is_last_month(scores):
    assert set(scores["modelo"]) >= {"ml_mean", "lgbm", "xgb", "rf", "naive"}
    assert pick(scores, "sku", "naive")["acierto"] == pytest.approx(1 - 4 / 60)
    assert pick(scores, "sku", "lgbm")["pronostico_TO"] == 63.0


def test_volume_of_series_without_a_forecast_is_reported(scores):
    extra = scores[scores["nivel"] == "sin_pronostico"]
    assert len(extra) == 1 and extra.iloc[0]["real_TO"] == 5.0 and extra.iloc[0]["n"] == 1


def test_alerts_fire_strictly_below_their_threshold(scores):
    assert find_alerts(scores).query("tipo == 'alerta' and codigo != 'peor_que_naive'").empty  # 0.833 / 0.985 / 0.985 against 0.64 / 0.85 / 0.90
    exact = float(pick(scores, "sku")["acierto"])
    assert "sku_bajo" not in set(find_alerts(scores, {**THRESHOLDS, "sku": exact})["codigo"])  # equal to the threshold: no alert
    assert "sku_bajo" in set(find_alerts(scores, {**THRESHOLDS, "sku": exact + 1e-6})["codigo"])
    assert "planta_bajo" in set(find_alerts(scores, {**THRESHOLDS, "planta": 0.99})["codigo"])
    assert "total_bajo" in set(find_alerts(scores, {**THRESHOLDS, "total": 0.99})["codigo"])


def test_ml_mean_worse_than_the_naive_is_an_alert(scores):
    alerts = find_alerts(scores)  # ml_mean 0.833 against naive 0.933 at SKU level
    row = alerts[alerts["codigo"] == "peor_que_naive"]
    assert len(row) == 1 and row.iloc[0]["tipo"] == "alerta"


def test_plants_to_review_are_informational(scores):
    alerts = find_alerts(scores, {**THRESHOLDS, "planta_revisar": 0.99})
    review = alerts[alerts["codigo"] == "planta_revisar"]
    assert list(review["planta"]) == ["B"] and (review["tipo"] == "info").all()


def test_a_month_must_be_in_the_data_and_closed():
    days = pd.date_range("2024-05-01", "2024-06-30")
    df = pd.DataFrame({"planta": "A", "sku": "s", "fecha": days[days.dayofweek != 6], "consumo": 1.0})
    check_month_closed(df, "2024-06")
    with pytest.raises(SystemExit, match="incomplete"):
        check_month_closed(df[df["fecha"] <= "2024-06-20"], "2024-06")
    with pytest.raises(SystemExit, match="not in the data"):
        check_month_closed(df, "2024-07")


def test_rows_for_storage_carry_the_run_and_the_alerts(scores):
    rows = tracking_rows("run-1", scores, find_alerts(scores), created_at="2026-11-01 00:00:00")
    assert set(rows["run_id"]) == {"run-1"} and (rows["creado_en"] == "2026-11-01 00:00:00").all()
    flagged = rows[rows["alerta"] != ""]
    assert list(flagged["alerta"]) == ["peor_que_naive"] and flagged.iloc[0]["nivel"] == "sku" and flagged.iloc[0]["modelo"] == "ml_mean"


def test_the_csv_and_the_database_replace_the_same_run_and_keep_others(scores, tmp_path):
    csv, con = tmp_path / "seguimiento.csv", connect()
    create_schema(con)
    one = tracking_rows("run-1", scores, find_alerts(scores), created_at="t1")
    for _ in range(2):  # twice: no duplicates
        write_tracking_csv(csv, one)
        replace_tracking(con, one)
    assert len(pd.read_csv(csv)) == len(one) == con.execute("SELECT COUNT(*) FROM seguimiento").fetchone()[0]
    two = tracking_rows("run-2", scores, find_alerts(scores), created_at="t2")
    write_tracking_csv(csv, two)
    replace_tracking(con, two)
    assert len(pd.read_csv(csv)) == 2 * len(one) == con.execute("SELECT COUNT(*) FROM seguimiento").fetchone()[0]
    fresh = connect()
    create_schema(fresh)
    load_tracking_csv(fresh, csv)  # rebuild: the history comes back from the csv
    assert fresh.execute("SELECT COUNT(*) FROM seguimiento").fetchone()[0] == 2 * len(one)
    assert fresh.execute("SELECT acierto FROM seguimiento WHERE run_id='run-1' AND nivel='total' AND modelo='ml_mean'").fetchone()[0] == pytest.approx(1 - 1 / 65)


def test_runs_are_stored_from_the_manifest():
    con = connect()
    create_schema(con)
    manifest = {"run": {"run_id": "r1", "created_at": "2026-10-19T17:10:18+00:00", "forecast_month": "2026-10", "as_of": None},
                "code": {"commit": "abc123"}, "data": {"panel_fingerprint": "f1"}, "params": {"params_hash": "p1"}, "output": {"csv_sha256": "s1", "total_ml_mean": 53219.6}}
    upsert_run(con, manifest)
    upsert_run(con, manifest)  # same run twice
    assert con.execute("SELECT run_id, mes_pronosticado, commit_git, huella_datos, params_hash, csv_sha256, total_ml_mean FROM corridas").fetchall() == [
        ("r1", "2026-10", "abc123", "f1", "p1", "s1", 53219.6)]


def test_predictions_can_carry_the_run_that_made_them(toy_df):
    con = connect()
    create_schema(con)
    write_consumption(con, toy_df)
    planta, sku = toy_df.iloc[0][["planta", "sku"]]
    preds = pd.DataFrame([{"planta": planta, "sku": sku, "mes_objetivo": "2025-07", "mes_origen": "2025-06", "estrategia": "h1", "modelo": "ml_mean", "valor": 1.0}])
    replace_predictions(con, preds, run_id="r1")
    assert read_predictions(con)["run_id"].tolist() == ["r1"]
    replace_predictions(con, preds)  # beta rule: overwritten; no run given
    assert read_predictions(con)["run_id"].isna().all()


V1_PREDICTIONS = """
CREATE TABLE plantas (planta_id INTEGER PRIMARY KEY, codigo TEXT NOT NULL UNIQUE);
CREATE TABLE skus (sku_id INTEGER PRIMARY KEY, descripcion TEXT NOT NULL UNIQUE, grade TEXT, subgrade TEXT, gsm INTEGER, ancho_mm INTEGER,
    ancho_core TEXT, sufijo TEXT, clave_planificador TEXT NOT NULL);
CREATE TABLE consumos_predichos (
    planta_id INTEGER NOT NULL, sku_id INTEGER NOT NULL, mes_objetivo TEXT NOT NULL, mes_origen TEXT NOT NULL, estrategia TEXT NOT NULL,
    modelo TEXT NOT NULL, valor REAL NOT NULL, creado_en TEXT NOT NULL,
    PRIMARY KEY (planta_id, sku_id, mes_objetivo, mes_origen, estrategia, modelo)) WITHOUT ROWID;
INSERT INTO plantas VALUES (1, 'A');
INSERT INTO skus VALUES (1, 's', NULL, NULL, NULL, NULL, NULL, NULL, 's');
INSERT INTO consumos_predichos VALUES (1, 1, '2025-07', '2025-06', 'h1', 'ml_mean', 4.5, '2026-01-01 00:00:00');
PRAGMA user_version = 1;
"""


def test_a_version_1_database_is_migrated_without_losing_predictions(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(V1_PREDICTIONS)
    old.commit()
    old.close()
    con = connect(path)
    for _ in range(2):  # a second call changes nothing
        create_schema(con)
    columns = [r[1] for r in con.execute("PRAGMA table_info(consumos_predichos)")]
    assert "run_id" in columns and con.execute("PRAGMA user_version").fetchone()[0] == 3
    assert con.execute("SELECT valor, run_id FROM consumos_predichos").fetchall() == [(4.5, None)]
    assert {"corridas", "seguimiento"} <= {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_the_scripts_track_a_closed_month_end_to_end(toy_df, tmp_path):
    data, out, seg, db = tmp_path / "consumos.csv", tmp_path / "forecast_2025-06.csv", tmp_path / "seguimiento.csv", tmp_path / "t.db"
    toy_df.to_csv(data, index=False)
    run = lambda *args: subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=REPO)  # noqa: E731
    made = run("scripts/predict_next_month.py", str(data), "--as-of", "2025-05", "--out", str(out), "--tuning", str(tmp_path / "none.json"))
    assert made.returncode == 0, made.stderr[-400:]
    for _ in range(2):  # run twice: the second replaces the first
        tracked = run("scripts/track_forecast.py", str(out), "--data", str(data), "--db", str(db), "--csv", str(seg))
        assert tracked.returncode in (0, 2), tracked.stderr[-400:]  # 2 = an alert fired (the toy data is noisy)
    table = pd.read_csv(seg)
    assert {"total", "planta", "sku"} <= set(table["nivel"]) and table["run_id"].nunique() == 1 and set(table["mes"]) == {"2025-06"}
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM seguimiento").fetchone()[0] == len(table)
    assert con.execute("SELECT COUNT(*) FROM corridas").fetchone()[0] == 1


# ---- a forecast that does not belong to the data, and infinite values --------------------------------------------------------


def test_a_month_with_no_real_consumption_is_scored_empty_not_infinite():
    other = forecast().assign(planta="Z", sku=["x1", "x2", "x3"])  # series that are not in the data: their real is 0
    result = month_scores(build_panel(consumption()), other, "2024-02")
    sku = result[(result["nivel"] == "sku") & (result["modelo"] == "ml_mean")].iloc[0]
    assert sku["real_TO"] == 0.0 and sku["pronostico_TO"] > 0
    assert np.isnan(sku["acierto"]) and np.isnan(sku["wape"]) and np.isnan(sku["bias_pct"])
    assert not np.isinf(result.select_dtypes("number").to_numpy(dtype=float)).any()


def test_infinite_scores_are_refused_for_storage(scores):
    bad = scores.copy()
    bad.loc[0, "wape"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        tracking_rows("run", bad, find_alerts(scores))


def test_a_forecast_needs_at_least_half_of_its_series_in_the_data():
    S = build_panel(consumption())
    full = forecast()
    assert forecast_coverage(S, full) == (3, 3) and coverage_ok(3, 3)
    mixed = full.assign(sku=["s1", "x2", "x3"])
    assert forecast_coverage(S, mixed) == (1, 3) and not coverage_ok(1, 3) and coverage_ok(2, 3) and not coverage_ok(0, 0)
    check_forecast_matches(S, full, "2024-02")  # does not stop
    with pytest.raises(SystemExit, match="does not belong to these data"):
        check_forecast_matches(S, mixed, "2024-02")


def test_no_forecast_made_by_a_test_is_committed_in_results():
    r"""A forecast made inside a test (its manifest points to a pytest temp folder) must not end up in results/: the app would offer to measure it."""
    for manifest in (REPO / "results").glob("forecast_*.manifest.json"):
        assert "pytest" not in json.loads(manifest.read_text(encoding="utf-8"))["data"]["path"], f"{manifest.name} comes from a test run"


def test_the_history_file_reads_infinite_values_as_empty(tmp_path):
    path = tmp_path / "seguimiento.csv"
    path.write_text(
        "run_id,mes,nivel,planta,modelo,acierto,wape,bias_pct,n,real_TO,pronostico_TO,alerta,creado_en\n"
        "r,2025-07,sku,,ml_mean,-inf,inf,-inf,12,0.0,1730.0,,2026-10-20 17:53:32\n",
        encoding="utf-8",
    )
    assert read_tracking_csv(path)[["acierto", "wape", "bias_pct"]].isna().all().all()


# ---- the strategy of every tracked run ------------------------------------------------------------------------------------------


V2_TRACKING = """
CREATE TABLE corridas (run_id TEXT PRIMARY KEY, creado_en TEXT NOT NULL, mes_pronosticado TEXT, as_of TEXT, commit_git TEXT, huella_datos TEXT, params_hash TEXT,
    csv_sha256 TEXT, total_ml_mean REAL);
CREATE TABLE seguimiento (run_id TEXT NOT NULL, mes TEXT NOT NULL, nivel TEXT NOT NULL, planta TEXT NOT NULL DEFAULT '', modelo TEXT NOT NULL, acierto REAL, wape REAL,
    bias_pct REAL, n INTEGER, real_TO REAL, pronostico_TO REAL, alerta TEXT NOT NULL DEFAULT '', creado_en TEXT NOT NULL, PRIMARY KEY (run_id, mes, nivel, planta, modelo)) WITHOUT ROWID;
INSERT INTO corridas VALUES ('r1', '2026-01-01 00:00:00', '2025-07', NULL, 'abc', 'fp', 'ph', 'sha', 1.5);
INSERT INTO seguimiento VALUES ('r1', '2025-07', 'total', '', 'ml_mean', 0.9, 0.1, 1.0, 1, 10.0, 9.0, '', '2026-01-01 00:00:00');
PRAGMA user_version = 2;
"""


def test_a_version_2_database_gets_the_strategy_columns_and_keeps_its_rows(tmp_path):
    path = tmp_path / "v2.db"
    old = sqlite3.connect(path)
    old.executescript(V2_TRACKING)
    old.commit()
    old.close()
    con = connect(path)
    for _ in range(2):
        create_schema(con)
    assert con.execute("PRAGMA user_version").fetchone()[0] == 3
    assert con.execute("SELECT run_id, estrategia, corte FROM seguimiento").fetchall() == [("r1", "h1", None)]  # the old runs were the one-month forecast
    assert con.execute("SELECT run_id, total_ml_mean, estrategia, corte FROM corridas").fetchall() == [("r1", 1.5, None, None)]
    upsert_run(con, {"run": {"run_id": "r2", "created_at": "2026-02-01 00:00:00", "forecast_month": "2025-08"}, "strategy": {"used": "parcial", "cutoff_day": 15}})
    assert con.execute("SELECT estrategia, corte FROM corridas WHERE run_id = 'r2'").fetchone() == ("parcial", 15)
    upsert_run(con, {"run": {"run_id": "r3", "created_at": "2026-02-01 00:00:00"}})  # a manifest from before the strategies
    assert con.execute("SELECT estrategia FROM corridas WHERE run_id = 'r3'").fetchone() == ("h1",)


def test_rows_without_the_strategy_columns_are_stored_as_the_one_month_forecast(tmp_path):
    scores = month_scores(build_panel(consumption()), forecast(), "2024-02")
    rows = tracking_rows("r", scores, find_alerts(scores)).drop(columns=["estrategia", "corte"])  # what a caller from before the strategies passes
    con = connect()
    create_schema(con)
    assert replace_tracking(con, rows) == len(rows)
    assert {r[0] for r in con.execute("SELECT estrategia FROM seguimiento")} == {"h1"}


def test_tracking_rows_carry_the_strategy_and_the_cutoff():
    scores = month_scores(build_panel(consumption()), forecast(), "2024-02")
    rows = tracking_rows("r", scores, find_alerts(scores), strategy="parcial", cutoff=15)
    assert set(rows["estrategia"]) == {"parcial"} and set(rows["corte"]) == {15}
    assert list(rows.columns)[-2:] == ["estrategia", "corte"]
    assert set(tracking_rows("r", scores, find_alerts(scores))["estrategia"]) == {"h1"}


def test_an_old_history_file_is_read_as_the_one_month_forecast(tmp_path):
    path = tmp_path / "old.csv"
    pd.DataFrame({"run_id": ["r"], "mes": ["2025-07"], "nivel": ["total"], "planta": [""], "modelo": ["ml_mean"], "acierto": [0.9], "wape": [0.1], "bias_pct": [1.0], "n": [1],
                  "real_TO": [10.0], "pronostico_TO": [9.0], "alerta": [""], "creado_en": ["2026-01-01 00:00:00"]}).to_csv(path, index=False)
    table = read_tracking_csv(path)
    assert set(table["estrategia"]) == {"h1"} and table["corte"].isna().all()
    scores = month_scores(build_panel(consumption()), forecast(), "2024-02")
    write_tracking_csv(path, tracking_rows("r2", scores, find_alerts(scores), strategy="directa"))  # adding rows of another strategy keeps the old ones
    both = read_tracking_csv(path)
    assert set(both.loc[both["run_id"] == "r", "estrategia"]) == {"h1"} and set(both.loc[both["run_id"] == "r2", "estrategia"]) == {"directa"}


def test_the_strategy_of_a_saved_forecast_comes_from_the_manifest_then_the_name_then_the_default():
    assert strategy_of("results/forecast_2026-11_parcial.csv", {"strategy": {"used": "nowcast_media", "cutoff_day": 12}}) == ("nowcast_media", 12)
    assert strategy_of("results/forecast_2026-11_parcial.csv", {"run": {}}) == ("parcial", None)
    assert strategy_of("results/forecast_2026-11_nowcast_media.csv", None) == ("nowcast_media", None)
    assert strategy_of("results/forecast_2026-11.csv", None) == ("h1", None)
    assert strategy_of("other.csv", {}) == ("h1", None)


def test_the_two_month_strategies_have_their_own_alert_thresholds():
    assert thresholds_for("h1") == THRESHOLDS and all(thresholds_for(s) == THRESHOLDS_TWO_MONTHS for s in ("directa", "parcial", "nowcast_media", "iterada"))
    assert all(THRESHOLDS_TWO_MONTHS[k] < THRESHOLDS[k] for k in ("sku", "planta", "total"))
    scores = pd.DataFrame([
        {"mes": "2025-07", "nivel": "sku", "planta": "", "modelo": "ml_mean", "acierto": 0.62}, {"mes": "2025-07", "nivel": "sku", "planta": "", "modelo": "naive", "acierto": 0.55},
        {"mes": "2025-07", "nivel": "planta", "planta": "", "modelo": "ml_mean", "acierto": 0.9}, {"mes": "2025-07", "nivel": "total", "planta": "", "modelo": "ml_mean", "acierto": 0.95},
    ])
    assert list(find_alerts(scores, thresholds_for("h1"))["codigo"]) == ["sku_bajo"]  # 62% is under the one-month threshold of 64%
    assert find_alerts(scores, thresholds_for("parcial")).empty  # and above the two-month one of 60%


def test_a_saved_forecast_is_tracked_with_its_strategy_and_its_thresholds(tmp_path):
    df = pd.concat([consumption(), pd.DataFrame({"planta": ["A"], "sku": ["s1"], "fecha": [pd.Timestamp("2024-02-29")], "consumo": [0.0]})], ignore_index=True)  # February ends
    S = build_panel(df)
    csv = tmp_path / "forecast_2024-02_parcial.csv"
    forecast().to_csv(csv, index=False)
    run = track_saved_forecast(csv, df, S)
    assert (run.strategy, run.cutoff) == ("parcial", None) and set(run.rows["estrategia"]) == {"parcial"}
    (tmp_path / "forecast_2024-02_parcial.manifest.json").write_text(json.dumps({"run": {"run_id": "r9", "forecast_month": "2024-02"}, "strategy": {"used": "parcial", "cutoff_day": 14}}))
    run = track_saved_forecast(csv, df, S)
    assert (run.run_id, run.strategy, run.cutoff) == ("r9", "parcial", 14) and set(run.rows["corte"]) == {14} and run.manifest["strategy"]["cutoff_day"] == 14


def test_the_backtest_runs_are_told_apart_from_the_real_ones():
    assert is_backtest("backtest-h1") and is_backtest("backtest-parcial") and not is_backtest("20261022T101010Z-abc123") and not is_backtest("sin-manifiesto-ab12")


def _horizon_cache(S, tmp_path, cutoff=15):
    r"""A two-month cache for the toy panel: three months, every series alive at the month two before."""
    n = S.monthly.shape[1]
    rows = []
    for t in (n - 3, n - 2, n - 1):
        for series in range(len(S.keys)):
            y = float(np.nan_to_num(S.monthly[series, t]))
            rows.append({"series": series, "o": t - 2, "t": t, "planta_code": int(S.planta_code[series]), "y": S.monthly[series, t], "p_direct_lgbm": y * 1.1 + 1, "p_direct_xgb": y * 0.9, "p_direct_rf": y,
                         "p_direct": y, "p_partial": y * 1.05 + 0.5, "p_iterated": y * 0.8, "p_naive": y * 1.3})
    path = tmp_path / "horizon_strategies.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    write_meta(path, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 2, "cutoff_day": cutoff})
    return path


def test_the_two_month_backtests_are_scored_for_each_strategy(toy_panel, tmp_path):
    S = toy_panel
    path = _horizon_cache(S, tmp_path)
    runs = track_horizon_backtests(S, path, repo_dir=tmp_path)
    assert {r.run_id for r in runs} == {"backtest-directa", "backtest-parcial", "backtest-iterada"} and len(runs) == 9  # three months each
    by = {(r.run_id, r.month): r for r in runs}
    month = S.labels[S.monthly.shape[1] - 1]
    direct = by[("backtest-directa", month)]
    assert direct.strategy == "directa" and direct.cutoff is None and set(direct.rows["modelo"]) == {"ml_mean", "naive", "lgbm", "xgb", "rf", "-"}
    partial = by[("backtest-parcial", month)]
    assert partial.cutoff == 15 and set(partial.rows["corte"]) == {15} and set(partial.rows["modelo"]) == {"ml_mean", "naive", "-"}
    assert by[("backtest-iterada", month)].manifest["strategy"] == {"used": "iterada", "cutoff_day": None}
    sku = lambda run: float(run.scores[(run.scores["nivel"] == "sku") & (run.scores["modelo"] == "ml_mean")]["acierto"].iloc[0])  # noqa: E731
    assert sku(direct) == pytest.approx(1.0) and sku(by[("backtest-iterada", month)]) < sku(direct)  # the cache holds y for direct and 80% of it for iterated
    write_meta(path, {"data": "other", "n_series": len(S.keys), "horizon": 2, "cutoff_day": 15})
    with pytest.raises(SystemExit):
        track_horizon_backtests(S, path)
    write_meta(path, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1})
    with pytest.raises(SystemExit):  # the one-month cache is not a two-month one
        track_horizon_backtests(S, path)


def test_the_alert_thresholds_follow_the_strategy_of_the_saved_forecast(tmp_path):
    df = pd.concat([consumption(), pd.DataFrame({"planta": ["A"], "sku": ["s1"], "fecha": [pd.Timestamp("2024-02-29")], "consumo": [0.0]})], ignore_index=True)
    S = build_panel(df)
    weak = forecast().assign(**{m: [0.62 * 12, 0.62 * 18, 0.62 * 30] for m in ("lgbm", "xgb", "rf", "ml_mean")})  # 62% accuracy at SKU level
    codes = {}
    for name in ("forecast_2024-02.csv", "forecast_2024-02_parcial.csv"):
        weak.to_csv(tmp_path / name, index=False)
        codes[name] = set(track_saved_forecast(tmp_path / name, df, S).alerts["codigo"])
    assert "sku_bajo" in codes["forecast_2024-02.csv"] and "sku_bajo" not in codes["forecast_2024-02_parcial.csv"]  # 62% is under 64% and over 60%
