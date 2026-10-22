"""Seguimiento by strategy: the forecasts of one month made with different strategies are measured one by one and compared."""

import io
import shutil

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src.app.paths import AppPaths
from tests.conftest import make_consumption
from tests.test_app_api import make_tracking, write_data
from tests.test_app_forecast import make_client, wait
from src.evaluation.tracking_utils import write_tracking_csv


def _generate(client):
    start = client.post("/api/pronostico/generar", json={"sobrescribir": True})
    assert start.status_code == 200, start.text
    assert wait(client, start.json()["trabajo"])["estado"] == "listo"


@pytest.fixture(scope="module")
def two_strategies(tmp_path_factory):
    r"""A folder with two forecasts of 2025-06: the mid-month one (data to 2025-05-20, `parcial`) and the one-month one (data to 2025-05-31), June closed."""
    root = tmp_path_factory.mktemp("strategies")
    df = make_consumption()
    write_data(root, df[df["fecha"] <= pd.Timestamp("2025-05-20")])
    _generate(make_client(root))
    write_data(root, df[df["fecha"] <= pd.Timestamp("2025-05-31")])
    _generate(make_client(root))
    write_data(root, df)  # June is in the data now
    return root


@pytest.fixture()
def workdir(two_strategies, tmp_path):
    shutil.copytree(two_strategies, tmp_path / "w")
    return tmp_path / "w"


def test_the_two_forecasts_of_the_month_exist_and_are_both_pending(workdir):
    assert [(m, s) for m, s, _ in AppPaths(workdir).forecast_files()] == [("2025-06", "h1"), ("2025-06", "parcial")]
    client = make_client(workdir)
    assert client.get("/api/seguimiento").json()["por_medir"] == [{"mes": "2025-06", "estrategia": "h1"}, {"mes": "2025-06", "estrategia": "parcial"}]
    status = client.get("/api/estado").json()
    assert status["seguimiento_pendiente"] == {"mes": "2025-06", "estrategia": "h1"}  # the oldest first, the one-month forecast before the mid-month one


def test_each_strategy_is_measured_on_its_own_and_both_stay_in_the_history(workdir):
    client = make_client(workdir)
    first = client.post("/api/seguimiento/medir", json={"mes": "2025-06"}).json()
    assert first["estrategia"] == "h1" and first["detalle"]["estrategia"] == "h1" and first["detalle"]["corte"] is None
    assert client.get("/api/seguimiento").json()["por_medir"] == [{"mes": "2025-06", "estrategia": "parcial"}]
    second = client.post("/api/seguimiento/medir", json={"mes": "2025-06", "estrategia": "parcial"}).json()
    assert second["estrategia"] == "parcial" and second["detalle"]["corte"] == 20
    assert first["corrida"] != second["corrida"]
    history = client.get("/api/seguimiento").json()
    assert history["por_medir"] == [] and [(r["mes"], r["estrategia"], r["corte"]) for r in history["produccion"]] == [("2025-06", "h1", None), ("2025-06", "parcial", 20)]
    stored = pd.read_csv(workdir / "results" / "seguimiento.csv")
    assert set(stored["estrategia"]) == {"h1", "parcial"} and stored.groupby("run_id")["estrategia"].nunique().max() == 1
    assert client.get("/api/estado").json()["seguimiento_pendiente"] is None


def test_the_detail_picks_the_strategy_and_lists_how_the_others_did(workdir):
    client = make_client(workdir)
    for strategy in ("h1", "parcial"):
        client.post("/api/seguimiento/medir", json={"mes": "2025-06", "estrategia": strategy})
    default = client.get("/api/seguimiento/detalle", params={"mes": "2025-06"}).json()
    assert default["estrategia"] == "h1" and default["estrategias"] == ["h1", "parcial"]
    assert [o["estrategia"] for o in default["otras_estrategias"]] == ["parcial"] and set(default["otras_estrategias"][0]) == {"estrategia", "sku", "planta", "total"}
    mid = client.get("/api/seguimiento/detalle", params={"mes": "2025-06", "estrategia": "parcial"}).json()
    assert mid["estrategia"] == "parcial" and mid["corte"] == 20 and [o["estrategia"] for o in mid["otras_estrategias"]] == ["h1"]
    assert mid["modelos"][0]["sku"] != default["modelos"][0]["sku"]  # two different forecasts of the same month
    assert client.get("/api/seguimiento/detalle", params={"mes": "2025-06", "estrategia": "iterada"}).status_code == 404


def test_the_biggest_misses_come_from_the_file_of_the_strategy(workdir):
    client = make_client(workdir)
    client.post("/api/seguimiento/medir", json={"mes": "2025-06", "estrategia": "parcial"})
    errors = client.get("/api/seguimiento/detalle", params={"mes": "2025-06", "estrategia": "parcial"}).json()["errores"]
    forecast = pd.read_csv(workdir / "results" / "forecast_2025-06_parcial.csv")
    real = make_consumption()
    real = real[real["fecha"].dt.strftime("%Y-%m") == "2025-06"].groupby(["planta", "sku"])["consumo"].sum()
    forecast["error"] = forecast["ml_mean"] - [float(real.get((p, s), 0.0)) for p, s in zip(forecast["planta"], forecast["sku"])]
    worst = forecast.reindex(forecast["error"].abs().sort_values(ascending=False).index).head(len(errors))
    assert errors and [(e["planta"], e["sku"]) for e in errors] == list(zip(worst["planta"], worst["sku"]))
    assert [e["error_TO"] for e in errors] == pytest.approx(list(worst["error"]), abs=0.1)
    one_month = pd.read_csv(workdir / "results" / "forecast_2025-06.csv")
    assert not np.allclose(one_month["ml_mean"], forecast["ml_mean"])  # the two files really differ, so the list above is the mid-month one


def test_measuring_a_strategy_without_a_saved_forecast_is_refused(workdir):
    refused = make_client(workdir).post("/api/seguimiento/medir", json={"mes": "2025-06", "estrategia": "directa"})
    assert refused.status_code == 422 and "directa" in refused.json()["detalle"]


def test_a_regenerated_forecast_measured_again_keeps_only_the_newest_run(workdir):
    client = make_client(workdir)
    old = client.post("/api/seguimiento/medir", json={"mes": "2025-06"}).json()["corrida"]
    stored = pd.read_csv(workdir / "results" / "seguimiento.csv")
    stored.loc[stored["run_id"] == old, "creado_en"] = "2020-01-01 00:00:00"  # an older measurement of the same forecast
    stored.to_csv(workdir / "results" / "seguimiento.csv", index=False)
    new = client.post("/api/seguimiento/medir", json={"mes": "2025-06"}).json()["corrida"]
    assert new == old  # the same forecast file has the same run id: its rows are replaced
    other = stored.assign(run_id="otra-corrida", creado_en="2030-01-01 00:00:00", pronostico_TO=1.0, acierto=0.1)
    pd.concat([pd.read_csv(workdir / "results" / "seguimiento.csv"), other[other["estrategia"] == "h1"]]).to_csv(workdir / "results" / "seguimiento.csv", index=False)
    history = client.get("/api/seguimiento").json()
    assert [(r["mes"], r["estrategia"]) for r in history["produccion"]] == [("2025-06", "h1")] and history["produccion"][0]["sku"] == 10.0  # the newest run wins


def test_the_comparison_averages_each_strategy_in_production_and_in_the_historical_test(tmp_path):
    paths = AppPaths(tmp_path)
    paths.results.mkdir()
    for month, h1, parcial in (("2025-04", 0.70, 0.66), ("2025-05", 0.72, 0.68), ("2025-06", 0.74, 0.70)):
        write_tracking_csv(paths.tracking_csv, make_tracking(month, f"r-h1-{month}", {"sku": h1, "planta": 0.9, "total": 0.95}))
        write_tracking_csv(paths.tracking_csv, make_tracking(month, f"r-p-{month}", {"sku": parcial, "planta": 0.88, "total": 0.94}, strategy="parcial", cutoff=15))
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-04", "backtest-parcial", {"sku": 0.60, "planta": 0.8, "total": 0.9}, strategy="parcial", cutoff=15))
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-05", "backtest-parcial", {"sku": 0.70, "planta": 0.8, "total": 0.9}, strategy="parcial", cutoff=15))
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-04", "backtest-directa", {"sku": 0.50}, strategy="directa"))
    comparison = {c["estrategia"]: c for c in make_client(tmp_path).get("/api/seguimiento").json()["comparacion"]}
    assert list(comparison) == ["h1", "directa", "parcial"]  # the order of the strategies
    assert comparison["h1"]["produccion"] == {"meses": 3, "sku": 72.0, "planta": 90.0, "total": 95.0} and comparison["h1"]["prueba_historica"] is None
    assert comparison["parcial"]["produccion"]["sku"] == 68.0 and comparison["parcial"]["produccion"]["meses"] == 3
    assert comparison["parcial"]["prueba_historica"] == {"meses": 2, "sku": 65.0, "planta": 80.0, "total": 90.0}
    assert comparison["directa"]["produccion"] is None and comparison["directa"]["prueba_historica"]["meses"] == 1


def test_the_detail_of_a_historical_test_is_by_strategy_too(tmp_path):
    paths = AppPaths(tmp_path)
    paths.results.mkdir()
    for strategy in ("h1", "directa", "parcial"):
        write_tracking_csv(paths.tracking_csv, make_tracking("2025-05", f"backtest-{strategy}", {"sku": 0.7, "planta": 0.9, "total": 0.95}, strategy=strategy))
    client = make_client(tmp_path)
    detail = client.get("/api/seguimiento/detalle", params={"mes": "2025-05", "origen": "backtest", "estrategia": "directa"}).json()
    assert detail["estrategia"] == "directa" and detail["estrategias"] == ["h1", "directa", "parcial"] and detail["errores"] == []
    assert client.get("/api/seguimiento/detalle", params={"mes": "2025-05", "origen": "backtest"}).json()["estrategia"] == "h1"


def test_the_tracking_excel_names_the_strategy_and_has_the_comparison(workdir):
    client = make_client(workdir)
    client.post("/api/seguimiento/medir", json={"mes": "2025-06", "estrategia": "parcial"})
    response = client.get("/api/descargas/seguimiento", params={"mes": "2025-06", "estrategia": "parcial"})
    assert response.status_code == 200 and "seguimiento_2025-06_parcial.xlsx" in response.headers["content-disposition"]
    book = openpyxl.load_workbook(io.BytesIO(response.content))
    assert "Comparación" in book.sheetnames
    header = [c.value for c in next(book["Historial"].iter_rows(min_row=1, max_row=1))]
    assert {"Estrategia", "Corte"} <= set(header)


def test_the_status_card_names_the_strategy_of_the_last_measured_month(workdir):
    client = make_client(workdir)
    client.post("/api/seguimiento/medir", json={"mes": "2025-06", "estrategia": "parcial"})
    last = client.get("/api/estado").json()["seguimiento"]
    assert (last["mes"], last["estrategia"], last["origen"]) == ("2025-06", "parcial", "produccion")


@pytest.mark.parametrize("older, newer", [("a-old", "z-new"), ("z-old", "a-new")])
def test_when_a_month_and_strategy_were_measured_twice_the_newest_run_is_used_everywhere(tmp_path, older, newer):
    paths = AppPaths(tmp_path)
    paths.results.mkdir()
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-06", older, {"sku": 0.50, "planta": 0.8, "total": 0.9}, created="2026-10-01 10:00:00"))
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-06", newer, {"sku": 0.90, "planta": 0.95, "total": 0.99}, created="2026-10-20 10:00:00"))
    client = make_client(tmp_path)
    history = client.get("/api/seguimiento").json()
    assert [(r["mes"], r["sku"]) for r in history["produccion"]] == [("2025-06", 90.0)]
    assert client.get("/api/seguimiento/detalle", params={"mes": "2025-06"}).json()["modelos"][0]["sku"] == 90.0
    assert client.get("/api/estado").json()["seguimiento"]["acierto_sku"] == 90.0  # the status card reads the same history


def test_a_tracked_strategy_without_its_forecast_file_still_shows_its_detail_without_misses(workdir):
    paths = AppPaths(workdir)
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-06", "run-d", {"sku": 0.66, "planta": 0.9, "total": 0.94}, strategy="directa"))  # only the one-month file exists
    detail = make_client(workdir).get("/api/seguimiento/detalle", params={"mes": "2025-06", "estrategia": "directa"})
    assert detail.status_code == 200 and detail.json()["errores"] == [] and detail.json()["estrategia"] == "directa"


@pytest.mark.parametrize("older, newer", [("a-old", "z-new"), ("z-old", "a-new")])
def test_the_history_of_forecasts_shows_the_newest_measurement_of_each_strategy(workdir, older, newer):
    paths = AppPaths(workdir)
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-06", older, {"sku": 0.50, "planta": 0.8, "total": 0.9}, created="2026-10-01 10:00:00"))
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-06", newer, {"sku": 0.90, "planta": 0.95, "total": 0.99}, created="2026-10-20 10:00:00"))
    rows = {(r["mes"], r["estrategia"]): r for r in make_client(workdir).get("/api/historial").json()["pronosticos"]}
    assert rows[("2025-06", "h1")]["medido"] == {"sku": 90.0, "planta": 95.0, "total": 99.0} and rows[("2025-06", "h1")]["por_medir"] is False
    assert rows[("2025-06", "parcial")]["medido"] is None
