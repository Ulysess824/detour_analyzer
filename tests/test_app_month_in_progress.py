"""Predicción: the card of the month in progress (real first days + the rest estimated with the mean) and settings files from before it was simplified."""

import json

import numpy as np
import pandas as pd
import pytest

from src.app import forecast_service, settings_service
from src.app.consumption_service import ConsumptionStore
from src.app.paths import AppPaths
from tests.conftest import make_consumption
from tests.test_app_api import write_data
from tests.test_app_forecast import make_client, wait


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    r"""A folder with data to 2025-05-20 and its `nowcast_media` forecast of 2025-06 (cutoff 20)."""
    root = tmp_path_factory.mktemp("month_in_progress")
    write_data(root, make_consumption()[lambda d: d["fecha"] <= pd.Timestamp("2025-05-20")])
    client = make_client(root)
    wait(client, client.post("/api/pronostico/generar", json={}).json()["trabajo"])
    return root


def test_the_card_estimates_the_rest_of_the_month_with_the_mean(generated):
    block = make_client(generated).get("/api/pronostico").json()["mes_en_curso"]
    assert block["total_estimado_TO"] == pytest.approx(block["real_hasta_corte_TO"] + block["resto_estimado_TO"], abs=0.2)
    assert set(block) == {"mes", "corte", "dias_vistos", "dias_restantes", "real_hasta_corte_TO", "resto_estimado_TO", "total_estimado_TO", "real_completo_TO"}


def test_a_selection_with_no_series_has_zero_estimates(generated):
    store = ConsumptionStore(AppPaths(generated))
    block = forecast_service.month_in_progress(store, "2025-06", "nowcast_media", 20, np.array([], dtype=int))
    assert block["real_hasta_corte_TO"] == 0 and block["resto_estimado_TO"] == 0
    assert forecast_service.month_in_progress(store, "2025-06", "iterada", 20, np.array([0])) is None  # strategies that do not use the month in progress
    assert forecast_service.month_in_progress(store, "2025-06", "nowcast_media", None, np.array([0])) is None


def test_a_settings_file_with_a_removed_strategy_or_method_falls_back_to_what_is_valid(tmp_path):
    (tmp_path / "config").mkdir()
    file = tmp_path / "config" / "ajustes_app.json"
    file.write_text('{"estrategia": "parcial", "corte_manual": 15, "metodo_resto": "montecarlo"}')
    assert settings_service.load(AppPaths(tmp_path)) == {"estrategia": "automatica", "corte_manual": None}  # `parcial` no longer exists
    file.write_text('{"estrategia": "h1", "corte_manual": null, "metodo_resto": "montecarlo"}')
    assert settings_service.load(AppPaths(tmp_path)) == {"estrategia": "h1", "corte_manual": None}  # the old method is ignored
    assert json.loads(file.read_text())["metodo_resto"] == "montecarlo"  # reading never rewrites the file
