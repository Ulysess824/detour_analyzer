"""Predicción and Seguimiento in the app: running a forecast as a background task, its detail, measuring a closed month and the history."""

import shutil
import threading
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app.api import create_app
from src.app.jobs import Busy, JobRunner
from src.app.paths import AppPaths
from src.evaluation.tracking_utils import write_tracking_csv
from tests.conftest import make_consumption
from tests.test_app_api import make_tracking, write_data

TODAY = date(2026, 10, 20)


def wait(client, job_id, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        state = client.get(f"/api/trabajos/{job_id}").json()
        if state["estado"] != "corriendo":
            return state
        time.sleep(0.2)
    raise AssertionError("the task did not finish")


def make_client(root):
    return TestClient(create_app(AppPaths(root), today=lambda: TODAY))


@pytest.fixture(scope="module")
def forecasted(tmp_path_factory):
    r"""A folder with the toy data (30 months, to 2025-06) and the forecast of 2025-07 made through the API."""
    root = tmp_path_factory.mktemp("app")
    write_data(root, make_consumption())
    client = make_client(root)
    start = client.post("/api/pronostico/generar", json={})
    assert start.status_code == 200 and start.json()["mes"] == "2025-07"
    state = wait(client, start.json()["trabajo"])
    assert state["estado"] == "listo", state
    return root, state


@pytest.fixture()
def workdir(forecasted, tmp_path):
    shutil.copytree(forecasted[0], tmp_path / "w")
    return tmp_path / "w"


# ---- running a forecast ------------------------------------------------------------------------------------------------------


def test_the_forecast_runs_in_the_background_and_leaves_the_csv_and_its_manifest(forecasted):
    root, state = forecasted
    assert state["progreso"] == 1.0 and state["resultado"]["mes"] == "2025-07"
    assert (root / "results" / "forecast_2025-07.csv").exists() and (root / "results" / "forecast_2025-07.manifest.json").exists()
    table = pd.read_csv(root / "results" / "forecast_2025-07.csv")
    assert state["resultado"]["series"] == len(table) and state["resultado"]["total_TO"] == pytest.approx(table["ml_mean"].sum(), abs=0.1)


def test_asking_again_needs_permission_to_overwrite(workdir):
    client = make_client(workdir)
    again = client.post("/api/pronostico/generar", json={})
    assert again.status_code == 409 and again.json()["existe"] is True and again.json()["mes"] == "2025-07"


def test_a_partial_last_month_fails_the_task_with_a_clear_message(tmp_path):
    df = make_consumption()
    write_data(tmp_path, df[df["fecha"] <= pd.Timestamp("2025-05-12")])
    client = make_client(tmp_path)
    start = client.post("/api/pronostico/generar", json={})
    assert start.json()["mes"] == "2025-05"
    state = wait(client, start.json()["trabajo"])
    assert state["estado"] == "error" and "incompleto" in state["error"]
    assert not (tmp_path / "results").exists() or not list((tmp_path / "results").glob("forecast_*.csv"))


def test_job_runner_runs_one_task_at_a_time_and_reports_progress_and_errors():
    runner, gate = JobRunner(), threading.Event()

    def slow(progress):
        progress("a mitad", 0.5)
        gate.wait(5)
        return {"ok": True}

    first = runner.start("x", slow)
    time.sleep(0.2)
    with pytest.raises(Busy):
        runner.start("y", lambda progress: {})
    running = runner.get(first)
    assert running["estado"] == "corriendo" and running["mensaje"] == "a mitad" and running["progreso"] == 0.5
    gate.set()
    for _ in range(50):
        if runner.get(first)["estado"] != "corriendo":
            break
        time.sleep(0.05)
    assert runner.get(first)["estado"] == "listo" and runner.get(first)["resultado"] == {"ok": True}

    def stops(progress):
        raise SystemExit("Los datos tienen errores.")

    second = runner.start("z", stops)
    for _ in range(50):
        if runner.get(second)["estado"] != "corriendo":
            break
        time.sleep(0.05)
    assert runner.get(second)["estado"] == "error" and runner.get(second)["error"] == "Los datos tienen errores."
    assert runner.get("nope") is None


def test_an_unknown_task_is_a_404(workdir):
    assert make_client(workdir).get("/api/trabajos/zzz").status_code == 404


# ---- the forecast on screen --------------------------------------------------------------------------------------------------


def test_forecast_detail_for_everything_has_totals_the_spread_and_a_table_of_plantas(workdir):
    client = make_client(workdir)
    table = pd.read_csv(workdir / "results" / "forecast_2025-07.csv")
    body = client.get("/api/pronostico").json()
    assert body["mes"] == "2025-07" and body["disponibles"] == ["2025-07"]
    assert body["pronostico"]["ml_mean_TO"] == pytest.approx(table["ml_mean"].sum(), abs=0.1)
    assert body["pronostico"]["minimo_TO"] <= body["pronostico"]["ml_mean_TO"] <= body["pronostico"]["maximo_TO"]
    assert body["historia"]["etiquetas"][-1] == "2025-06" and body["ultimo_real"]["mes"] == "2025-06"
    assert body["ultimo_real"]["cambio_pct"] == pytest.approx((body["pronostico"]["ml_mean_TO"] / body["ultimo_real"]["TO"] - 1) * 100, abs=0.1)
    assert body["tabla"]["tipo"] == "planta" and [r["planta"] for r in body["tabla"]["filas"]] and body["tabla"]["total_filas"] == 3
    assert sum(r["pronostico_TO"] for r in body["tabla"]["filas"]) == pytest.approx(body["pronostico"]["ml_mean_TO"], abs=0.2)


def test_forecast_detail_of_a_planta_and_of_one_series(workdir):
    client = make_client(workdir)
    table = pd.read_csv(workdir / "results" / "forecast_2025-07.csv")
    mine = table[table["planta"] == "A"]
    body = client.get("/api/pronostico", params={"planta": "a", "limite": 2}).json()
    assert body["pronostico"]["ml_mean_TO"] == pytest.approx(mine["ml_mean"].sum(), abs=0.1) and body["pronostico"]["series"] == len(mine)
    assert body["tabla"]["tipo"] == "sku" and len(body["tabla"]["filas"]) == 2 and body["tabla"]["total_filas"] == len(mine)
    values = [r["pronostico_TO"] for r in body["tabla"]["filas"]]
    assert values == sorted(values, reverse=True)
    one = mine.iloc[0]
    body = client.get("/api/pronostico", params={"planta": "A", "sku": one["sku"]}).json()
    assert body["pronostico"]["ml_mean_TO"] == pytest.approx(one["ml_mean"], abs=0.1) and body["pronostico"]["series"] == 1


def test_forecast_errors_are_404s_with_a_message(workdir, tmp_path):
    client = make_client(workdir)
    assert client.get("/api/pronostico", params={"mes": "2020-01"}).status_code == 404
    assert client.get("/api/pronostico", params={"planta": "ZZ"}).status_code == 404
    write_data(tmp_path, make_consumption())
    empty = make_client(tmp_path).get("/api/pronostico")
    assert empty.status_code == 404 and "Genera" in empty.json()["detalle"]
    assert make_client(tmp_path).get("/api/pronosticos").json() == {"pronosticos": []}


# ---- tracking ----------------------------------------------------------------------------------------------------------------


def close_the_month(root):
    r"""Replace the data by one with a month more, so the forecast of 2025-07 can be measured."""
    write_data(root, make_consumption(n_months=31))


def test_without_tracking_the_history_is_empty_and_the_detail_says_so(workdir):
    client = make_client(workdir)
    body = client.get("/api/seguimiento").json()
    assert body["backtest"] == [] and body["produccion"] == [] and body["por_medir"] == [] and body["umbrales"]["sku"] == 64.0
    assert client.get("/api/seguimiento/detalle").status_code == 404


def test_a_month_that_is_still_open_cannot_be_measured(workdir):
    client = make_client(workdir)
    refused = client.post("/api/seguimiento/medir", json={"mes": "2025-07"})
    assert refused.status_code == 422 and "completo" in refused.json()["detalle"]
    assert client.post("/api/seguimiento/medir", json={"mes": "2024-01"}).status_code == 422  # no forecast saved for it


def test_measuring_a_closed_month_stores_the_scores_and_lists_the_biggest_misses(workdir):
    close_the_month(workdir)
    client = make_client(workdir)
    assert client.get("/api/seguimiento").json()["por_medir"] == ["2025-07"]
    done = client.post("/api/seguimiento/medir", json={"mes": "2025-07"})
    assert done.status_code == 200
    body = done.json()["detalle"]
    assert body["mes"] == "2025-07" and body["origen"] == "produccion" and [m["modelo"] for m in body["modelos"]][0] == "ml_mean"
    assert {"sku", "planta", "total"} <= set(body["modelos"][0])
    accuracy = [p["acierto"] for p in body["plantas"]]
    assert accuracy == sorted(accuracy) and len(accuracy) == 3

    forecast = pd.read_csv(workdir / "results" / "forecast_2025-07.csv")
    real = make_consumption(n_months=31)
    real = real[real["fecha"].dt.strftime("%Y-%m") == "2025-07"].groupby(["planta", "sku"])["consumo"].sum()
    forecast["real"] = [float(real.get((p, s), 0.0)) for p, s in zip(forecast["planta"], forecast["sku"])]
    worst = (forecast["ml_mean"] - forecast["real"]).abs().max()
    errors = body["errores"]
    assert errors and abs(errors[0]["error_TO"]) == pytest.approx(worst, abs=0.1)
    miss = forecast.assign(error=forecast["ml_mean"] - forecast["real"]).iloc[(forecast["ml_mean"] - forecast["real"]).abs().argmax()]
    assert errors[0]["sku"] == miss["sku"] and errors[0]["error_TO"] == pytest.approx(miss["error"], abs=0.1)  # error = forecast - real
    assert [abs(e["error_TO"]) for e in errors] == sorted([abs(e["error_TO"]) for e in errors], reverse=True)
    assert body["total"]["real_TO"] == pytest.approx(forecast["real"].sum(), abs=0.5)

    history = client.get("/api/seguimiento").json()
    assert [r["mes"] for r in history["produccion"]] == ["2025-07"] and history["por_medir"] == []
    stored = pd.read_csv(workdir / "results" / "seguimiento.csv")
    again = client.post("/api/seguimiento/medir", json={"mes": "2025-07"})  # measuring twice replaces the rows
    assert again.status_code == 200 and len(pd.read_csv(workdir / "results" / "seguimiento.csv")) == len(stored)


def test_the_detail_picks_production_over_the_backtest_and_can_show_either(workdir):
    close_the_month(workdir)
    client = make_client(workdir)
    paths = AppPaths(workdir)
    paths.results.mkdir(exist_ok=True)
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-07", "backtest-h1", {"sku": 0.7, "planta": 0.9, "total": 0.97}, alert="sku_bajo"))
    write_tracking_csv(paths.tracking_csv, make_tracking("2025-06", "backtest-h1", {"sku": 0.7, "planta": 0.9, "total": 0.97}))
    only_backtest = client.get("/api/seguimiento/detalle", params={"mes": "2025-07"}).json()
    assert only_backtest["disponibles"] == ["2025-07", "2025-06"]  # newest first
    assert only_backtest["origen"] == "backtest" and only_backtest["errores"] == []
    assert [a["codigo"] for a in only_backtest["alertas"]] == ["sku_bajo"]
    client.post("/api/seguimiento/medir", json={"mes": "2025-07"})
    assert client.get("/api/seguimiento/detalle").json()["origen"] == "produccion"
    assert client.get("/api/seguimiento/detalle", params={"mes": "2025-07", "origen": "backtest"}).json()["origen"] == "backtest"
    assert client.get("/api/seguimiento/detalle", params={"mes": "2030-01"}).status_code == 404
    history = client.get("/api/seguimiento").json()
    assert [r["mes"] for r in history["backtest"]] == ["2025-06", "2025-07"] and len(history["produccion"]) == 1


def test_a_partial_month_is_not_closed_and_is_not_offered_for_measuring(tmp_path):
    from src.app.consumption_service import ConsumptionStore

    df = make_consumption()
    write_data(tmp_path, df[df["fecha"] <= pd.Timestamp("2025-05-12")])
    store = ConsumptionStore(AppPaths(tmp_path))
    assert store.is_closed("2025-04") and not store.is_closed("2025-05") and not store.is_closed("2030-01")
    paths = AppPaths(tmp_path)
    paths.results.mkdir()
    pd.DataFrame({"planta": ["A"], "sku": ["x"], "mes": ["2025-05"], "ml_mean": [1.0]}).to_csv(paths.forecast_csv("2025-05"), index=False)
    assert make_client(tmp_path).get("/api/seguimiento").json()["por_medir"] == []
