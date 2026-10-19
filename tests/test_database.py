"""SQLite store: constraints, round trip with the csv, and the beta rule (predictions are overwritten by block)."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.engine.db_utils import (
    connect,
    create_schema,
    parse_sku,
    predictions_from_cache,
    predictions_from_forecast,
    read_consumption,
    read_predictions,
    replace_predictions,
    write_consumption,
)
from src.engine.panel_utils import build_panel, load_consumption
from src.evaluation.cache_utils import panel_fingerprint


@pytest.fixture()
def con(toy_df):
    con = connect()
    create_schema(con)
    write_consumption(con, toy_df)
    return con


def preds(con, value=1.0, estrategia="h1", origin="2024-06", target="2024-07", skus=2):
    rows = con.execute("SELECT p.codigo, s.descripcion FROM consumos c JOIN plantas p USING (planta_id) JOIN skus s USING (sku_id) GROUP BY 1, 2").fetchall()[:skus]
    return pd.DataFrame(
        [{"planta": p, "sku": s, "mes_objetivo": target, "mes_origen": origin, "estrategia": estrategia, "modelo": "ml_mean", "valor": value} for p, s in rows]
    )


def test_round_trip_keeps_the_panel(toy_df, con):
    back = read_consumption(con)
    assert len(back) == len(toy_df)
    assert panel_fingerprint(build_panel(back)) == panel_fingerprint(build_panel(toy_df))


def test_load_consumption_reads_a_db_file(toy_df, tmp_path):
    path = tmp_path / "t.db"
    con = connect(path)
    create_schema(con)
    write_consumption(con, toy_df)
    con.close()
    assert panel_fingerprint(build_panel(load_consumption(path))) == panel_fingerprint(build_panel(toy_df))


def test_constraints_reject_bad_rows(toy_df, con):
    with pytest.raises(ValueError):
        write_consumption(con, pd.concat([toy_df, toy_df.head(1)]))
    with pytest.raises(ValueError):
        write_consumption(con, toy_df.assign(consumo=-1.0))
    with pytest.raises(sqlite3.IntegrityError):  # unknown planta and sku
        con.execute("INSERT INTO consumos VALUES (999, 999, '2024-01-02', 1.0)")
    with pytest.raises(sqlite3.IntegrityError):  # negative consumption
        con.execute("INSERT INTO consumos VALUES (1, 1, '2024-01-02', -1.0)")


def test_rewriting_the_consumption_keeps_ids_and_does_not_duplicate(toy_df, con):
    before = con.execute("SELECT planta_id, codigo FROM plantas ORDER BY 1").fetchall()
    n = write_consumption(con, toy_df)
    assert n["consumos"] == len(toy_df)
    assert con.execute("SELECT planta_id, codigo FROM plantas ORDER BY 1").fetchall() == before


def test_parse_sku_parts():
    full = parse_sku("HP2/01/115gsm/1790mm/1200-1400/Subst.")
    assert (full["grade"], full["subgrade"], full["gsm"], full["ancho_mm"], full["ancho_core"], full["sufijo"]) == (
        "HP2", "01", 115, 1790, "1200-1400", "Subst.")
    assert full["clave_planificador"] == "hp2/01/115gsm/1790mm"
    assert parse_sku("KBA/157/157GSM/2200mm/1200-1450")["gsm"] == 157
    odd = parse_sku("SF1/200GSM/2100MM")  # three parts: kept, with empty parts
    assert odd["grade"] is None and odd["clave_planificador"] == "sf1/200gsm/2100mm"


def test_predictions_are_overwritten_by_block_and_stamped(con):
    first = replace_predictions(con, preds(con, 1.0), created_at="2026-01-01 00:00:00")
    other = preds(con, 5.0, estrategia="directa")
    replace_predictions(con, other, created_at="2026-01-01 00:00:00")
    replace_predictions(con, preds(con, 2.0, skus=1), created_at="2026-02-01 00:00:00")  # h1 block: one SKU less, new values
    h1 = read_predictions(con, estrategia="h1")
    assert first == 2 and len(h1) == 1 and h1["valor"].iloc[0] == 2.0 and h1["creado_en"].iloc[0] == "2026-02-01 00:00:00"
    assert (read_predictions(con, estrategia="directa")["valor"] == 5.0).all()  # other blocks untouched
    assert len(read_predictions(con)) == 1 + 2


def test_failed_load_keeps_the_previous_version(con):
    replace_predictions(con, preds(con, 1.0))
    bad = preds(con, 3.0).assign(valor=[3.0, np.nan])
    with pytest.raises(ValueError):
        replace_predictions(con, bad)
    with pytest.raises(ValueError):
        replace_predictions(con, preds(con).assign(sku="no existe"))
    assert (read_predictions(con)["valor"] == 1.0).all() and len(read_predictions(con)) == 2


def test_a_failure_in_the_middle_of_the_insert_rolls_the_delete_back(con):
    replace_predictions(con, preds(con, 1.0))
    broken = preds(con, 9.0)
    broken.loc[1, "mes_origen"] = "2024-08"  # origin after the target: the database rejects it after the old block was deleted
    with pytest.raises(sqlite3.IntegrityError):
        replace_predictions(con, broken)
    assert (read_predictions(con)["valor"] == 1.0).all() and len(read_predictions(con)) == 2


def test_prediction_rules(con):
    with pytest.raises(sqlite3.IntegrityError):  # origin must precede the target
        replace_predictions(con, preds(con, origin="2024-07", target="2024-07"))
    with pytest.raises(ValueError):  # repeated key
        p = preds(con)
        replace_predictions(con, pd.concat([p, p]))


def test_cache_translation_uses_the_series_index(toy_panel):
    S = toy_panel
    n = S.monthly.shape[1]
    frame = pd.DataFrame({"series": [0, 3], "o": [n - 3, n - 3], "t": [n - 2, n - 2], "p_a": [1.5, 2.5]})
    out = predictions_from_cache(frame, S, {"p_a": ("h1", "ml_mean")})
    assert list(out["planta"]) == [S.keys[0][0], S.keys[3][0]] and list(out["sku"]) == [S.keys[0][1], S.keys[3][1]]
    assert set(out["mes_objetivo"]) == {S.labels[n - 2]} and set(out["mes_origen"]) == {S.labels[n - 3]}


def test_forecast_file_origin_is_the_previous_month():
    frame = pd.DataFrame({"planta": ["A"], "sku": ["x"], "mes": ["2026-01"], "lgbm": [1.0], "ml_mean": [2.0]})
    out = predictions_from_forecast(frame)
    assert set(out["mes_origen"]) == {"2025-12"} and set(out["modelo"]) == {"lgbm", "ml_mean"}
