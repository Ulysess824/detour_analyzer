"""Ajustes: the saved strategy and cutoff, the plan they give for the data loaded, and the forecast that follows them."""

import json

import pandas as pd
import pytest

from src.app import settings_service
from src.app.paths import AppPaths
from tests.conftest import make_consumption
from tests.test_app_api import write_data
from tests.test_app_forecast import make_client, wait


def _in_progress(root, last="2025-05-20"):
    df = make_consumption()
    write_data(root, df[df["fecha"] <= pd.Timestamp(last)])
    return make_client(root)


def test_the_screen_loads_without_data_and_without_a_saved_file(tmp_path):
    body = make_client(tmp_path).get("/api/ajustes").json()
    assert body["ajustes"] == {"estrategia": "automatica", "corte_manual": None} and body["estado"] is None and body["plan"] is None
    assert [o["id"] for o in body["opciones"]][0] == "automatica" and body["corte_minimo"] == 10 and len(body["curva_parcial"]) == 10


def test_settings_are_saved_and_come_back(tmp_path):
    client = _in_progress(tmp_path)
    saved = client.put("/api/ajustes", json={"estrategia": "nowcast_media", "corte_manual": 12})
    assert saved.status_code == 200 and saved.json()["ajustes"] == {"estrategia": "nowcast_media", "corte_manual": 12}
    assert json.loads((tmp_path / "config" / "ajustes_app.json").read_text()) == {"estrategia": "nowcast_media", "corte_manual": 12}
    assert make_client(tmp_path).get("/api/ajustes").json()["ajustes"]["estrategia"] == "nowcast_media"  # a new app reads the file


@pytest.mark.parametrize("body", [{"estrategia": "otra"}, {"estrategia": "h1", "corte_manual": 0}, {"estrategia": "h1", "corte_manual": 32}, {"estrategia": "h1", "corte_manual": True}])
def test_invalid_settings_are_rejected_and_nothing_is_saved(tmp_path, body):
    client = _in_progress(tmp_path)
    response = client.put("/api/ajustes", json=body)
    assert response.status_code == 422
    assert not (tmp_path / "config" / "ajustes_app.json").exists()


def test_a_damaged_settings_file_falls_back_to_the_defaults(tmp_path):
    for text in ("{ not json", '{"estrategia": "otra"}', "[]", '{"estrategia": "h1", "corte_manual": "x"}'):
        (tmp_path / "config").mkdir(exist_ok=True)
        (tmp_path / "config" / "ajustes_app.json").write_text(text)
        assert settings_service.load(AppPaths(tmp_path)) == {"estrategia": "automatica", "corte_manual": None}


def test_the_plan_follows_the_state_of_the_data(tmp_path):
    body = _in_progress(tmp_path, "2025-05-20").get("/api/ajustes").json()
    assert body["estado"]["estado"] == "in_progress" and body["estado"]["dia_de_corte"] == 20 and body["estado"]["ultimo_mes_completo"] == "2025-04"
    assert (body["plan"]["estrategia"], body["plan"]["corte"], body["plan"]["mes_origen"], body["plan"]["mes_pronosticado"]) == ("parcial", 20, "2025-04", "2025-06")
    few = _in_progress(tmp_path, "2025-05-07").get("/api/ajustes").json()
    assert few["plan"]["estrategia"] == "directa" and few["plan"]["avisos"]
    done = _in_progress(tmp_path, "2025-06-30").get("/api/ajustes").json()
    assert done["estado"]["estado"] == "complete" and done["plan"]["estrategia"] == "h1" and done["plan"]["mes_pronosticado"] == "2025-07"


def test_a_cutoff_after_the_last_record_is_shown_as_an_error_and_blocks_the_forecast(tmp_path):
    client = _in_progress(tmp_path, "2025-05-12")
    client.put("/api/ajustes", json={"estrategia": "parcial", "corte_manual": 25})
    body = client.get("/api/ajustes").json()
    assert body["plan"] is None and "12" in body["error"]
    response = client.post("/api/pronostico/generar", json={})
    assert response.status_code == 422 and "12" in response.json()["detalle"]


def test_the_forecast_follows_the_saved_strategy_and_cutoff(tmp_path):
    client = _in_progress(tmp_path, "2025-05-20")
    client.put("/api/ajustes", json={"estrategia": "nowcast_media", "corte_manual": 12})
    start = client.post("/api/pronostico/generar", json={}).json()
    assert (start["mes"], start["estrategia"], start["corte"]) == ("2025-06", "nowcast_media", 12)
    state = wait(client, start["trabajo"])
    assert state["estado"] == "listo" and state["resultado"]["estrategia"] == "nowcast_media" and state["resultado"]["corte"] == 12
    manifest = json.loads((tmp_path / "results" / "forecast_2025-06_nowcast_media.manifest.json").read_text())
    assert manifest["strategy"]["used"] == "nowcast_media" and manifest["strategy"]["cutoff_day"] == 12 and manifest["strategy"]["estimated_months"] == ["2025-05"]


def test_forecasts_are_listed_with_the_one_month_forecast_first_in_a_month(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for name in ("forecast_2025-06_parcial.csv", "forecast_2025-06.csv", "forecast_2025-05.csv", "forecast_2025-06_directa.csv", "forecast_bad.csv"):
        pd.DataFrame({"planta": ["A"], "sku": ["S"], "mes": ["x"], "ml_mean": [1.0]}).to_csv(results / name, index=False)
    listed = [(month, strategy) for month, strategy, _ in AppPaths(tmp_path).forecast_files()]
    assert listed == [("2025-06", "h1"), ("2025-06", "parcial"), ("2025-06", "directa"), ("2025-05", "h1")]
    assert AppPaths(tmp_path).forecast_csv("2025-06").name == "forecast_2025-06.csv" and AppPaths(tmp_path).forecast_csv("2025-06", "parcial").name == "forecast_2025-06_parcial.csv"


def test_the_excel_of_a_forecast_with_a_strategy_names_it(tmp_path):
    client = _in_progress(tmp_path, "2025-05-20")
    wait(client, client.post("/api/pronostico/generar", json={}).json()["trabajo"])
    response = client.get("/api/descargas/pronostico")
    assert response.status_code == 200 and "pronostico_2025-06_parcial.xlsx" in response.headers["content-disposition"]
    assert client.get("/api/descargas/pronostico", params={"mes": "2025-06", "estrategia": "parcial"}).status_code == 200
    assert client.get("/api/descargas/pronostico", params={"mes": "2025-06", "estrategia": "directa"}).status_code == 404


def test_validate_rejects_booleans_and_non_integers_directly():
    for bad in (True, 1.5, "12"):
        with pytest.raises(ValueError, match="entero"):
            settings_service.validate("h1", bad)
    assert settings_service.validate("h1", 12) == {"estrategia": "h1", "corte_manual": 12}


def test_only_the_one_month_forecast_of_a_closed_month_is_offered_to_measure(tmp_path):
    df = make_consumption()
    write_data(tmp_path, df[df["fecha"] <= pd.Timestamp("2025-05-31")])
    client = make_client(tmp_path)
    wait(client, client.post("/api/pronostico/generar", json={}).json()["trabajo"])  # h1 for 2025-06
    results = tmp_path / "results"
    (results / "forecast_2025-06_directa.csv").write_text((results / "forecast_2025-06.csv").read_text())
    write_data(tmp_path, df)  # June is now in the data: the forecast of 2025-06 can be measured
    rows = {(r["mes"], r["estrategia"]): r["por_medir"] for r in make_client(tmp_path).get("/api/historial").json()["pronosticos"]}
    assert rows == {("2025-06", "h1"): True, ("2025-06", "directa"): False}
