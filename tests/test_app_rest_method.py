"""Ajustes: how the rest of the month in progress is estimated (mean or Monte Carlo) and what the card of Predicción shows."""

import numpy as np
import pandas as pd
import pytest

from src.app import forecast_service, settings_service
from src.app.consumption_service import ConsumptionStore
from src.app.paths import AppPaths
from src.engine.month_in_progress import complete_calendar
from src.experiments.montecarlo_utils import simulate_rest_of_month
from tests.conftest import make_consumption
from tests.test_app_api import write_data
from tests.test_app_forecast import make_client, wait


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    r"""A folder with data to 2025-05-20 and its `parcial` forecast of 2025-06 (cutoff 20)."""
    root = tmp_path_factory.mktemp("rest")
    write_data(root, make_consumption()[lambda d: d["fecha"] <= pd.Timestamp("2025-05-20")])
    client = make_client(root)
    wait(client, client.post("/api/pronostico/generar", json={}).json()["trabajo"])
    return root


@pytest.fixture()
def client(generated):
    return make_client(generated)


def test_the_method_defaults_to_the_mean_and_the_options_are_listed(tmp_path):
    body = make_client(tmp_path).get("/api/ajustes").json()
    assert body["ajustes"]["metodo_resto"] == "media"
    assert [m["id"] for m in body["metodos_resto"]] == ["media", "montecarlo"] and body["metodos_resto"][0]["recomendada"] and not body["metodos_resto"][1]["recomendada"]
    assert set(body["metodos_resto"][1]["acierto"]) == {"sku", "planta", "total"}


def test_the_method_is_saved_kept_and_validated(tmp_path):
    client = make_client(tmp_path)
    assert client.put("/api/ajustes", json={"estrategia": "parcial", "metodo_resto": "montecarlo"}).json()["ajustes"]["metodo_resto"] == "montecarlo"
    kept = client.put("/api/ajustes", json={"estrategia": "directa"}).json()["ajustes"]  # a save that does not name the method keeps it
    assert (kept["estrategia"], kept["metodo_resto"]) == ("directa", "montecarlo")
    refused = client.put("/api/ajustes", json={"estrategia": "directa", "metodo_resto": "adivinar"})
    assert refused.status_code == 422 and "adivinar" in refused.json()["detalle"]
    assert make_client(tmp_path).get("/api/ajustes").json()["ajustes"]["metodo_resto"] == "montecarlo"  # the refused one saved nothing


def test_a_damaged_method_in_the_file_falls_back_to_the_default(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "ajustes_app.json").write_text('{"estrategia": "h1", "corte_manual": null, "metodo_resto": "otro"}')
    assert settings_service.load(AppPaths(tmp_path)) == {"estrategia": "automatica", "corte_manual": None, "metodo_resto": "media"}


def _block(client, **params):
    return client.get("/api/pronostico", params=params).json()["mes_en_curso"]


def test_the_card_uses_the_mean_by_default_and_shows_the_monte_carlo_total_next_to_it(client):
    block = _block(client)
    assert block["metodo"] == "media" and block["metodo_fijo"] is False and block["otro_metodo"]["metodo"] == "montecarlo"
    assert block["total_estimado_TO"] == pytest.approx(block["real_hasta_corte_TO"] + block["resto_estimado_TO"], abs=0.2)
    other = block["otro_metodo"]
    assert other["total_estimado_TO"] == pytest.approx(block["real_hasta_corte_TO"] + other["resto_estimado_TO"], abs=0.2)
    assert other["resto_estimado_TO"] != block["resto_estimado_TO"]


def test_the_monte_carlo_estimate_equals_the_simulator_on_the_series_chosen(generated):
    client = make_client(generated)
    client.put("/api/ajustes", json={"estrategia": "automatica", "metodo_resto": "montecarlo"})
    store = ConsumptionStore(AppPaths(generated))
    S = complete_calendar(store.panel)
    month = S.labels.index("2025-05")
    for params, selection in (({}, np.arange(len(S.keys))), ({"planta": "A"}, np.array([i for i, (p, _) in enumerate(S.keys) if p == "A"]))):
        block = _block(client, **params)
        expected = simulate_rest_of_month(S, month, 20, 200, 0)[:, selection].mean(axis=0).sum()
        assert block["metodo"] == "montecarlo" and block["resto_estimado_TO"] == pytest.approx(expected, abs=0.1)
        assert block["otro_metodo"]["metodo"] == "media"
    everything, planta = _block(client), _block(client, planta="A")
    assert 0 < planta["resto_estimado_TO"] < everything["resto_estimado_TO"]


def test_the_chosen_method_changes_the_estimate_but_not_the_real_part(generated):
    client = make_client(generated)
    client.put("/api/ajustes", json={"estrategia": "automatica", "metodo_resto": "media"})
    mean = _block(client)
    client.put("/api/ajustes", json={"estrategia": "automatica", "metodo_resto": "montecarlo"})
    simulated = _block(client)
    assert mean["real_hasta_corte_TO"] == simulated["real_hasta_corte_TO"] and mean["resto_estimado_TO"] != simulated["resto_estimado_TO"]
    assert simulated["otro_metodo"]["resto_estimado_TO"] == pytest.approx(mean["resto_estimado_TO"], abs=0.1)


def test_the_nowcast_strategy_always_shows_the_mean_because_its_forecast_used_it(tmp_path):
    write_data(tmp_path, make_consumption()[lambda d: d["fecha"] <= pd.Timestamp("2025-05-20")])
    client = make_client(tmp_path)
    client.put("/api/ajustes", json={"estrategia": "nowcast_media", "metodo_resto": "montecarlo"})
    wait(client, client.post("/api/pronostico/generar", json={}).json()["trabajo"])
    block = _block(client)
    assert block["metodo"] == "media" and block["metodo_fijo"] is True and block["otro_metodo"]["metodo"] == "montecarlo"


def test_a_selection_with_no_series_has_zero_estimates(generated):
    store = ConsumptionStore(AppPaths(generated))
    block = forecast_service.month_in_progress(store, "2025-06", "parcial", 20, np.array([], dtype=int), "montecarlo")
    assert block["real_hasta_corte_TO"] == 0 and block["resto_estimado_TO"] == 0 and block["otro_metodo"]["resto_estimado_TO"] == 0
    assert forecast_service.month_in_progress(store, "2025-06", "directa", 20, np.array([0]), "montecarlo") is None  # strategies that do not use the month in progress
    assert forecast_service.month_in_progress(store, "2025-06", "parcial", None, np.array([0]), "media") is None
