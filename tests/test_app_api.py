"""Local API of the app on a synthetic consumption table: filters, series, summary, suggestions, status and errors."""

import json
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app import consumption_service
from src.app.api import create_app
from src.app.paths import AppPaths
from src.app.status_service import add_months, last_forecast, last_tracking, pending_tracking, retune_status
from src.engine.db_utils import TRACKING_COLUMNS
from src.evaluation.tracking_utils import write_tracking_csv
from tests.conftest import make_consumption

TODAY = date(2026, 10, 20)


def write_data(root, df):
    (root / "data").mkdir(exist_ok=True)
    df.to_csv(root / "data" / "consumos_long.csv", index=False)


@pytest.fixture()
def root(tmp_path, toy_df):
    write_data(tmp_path, toy_df)
    return tmp_path


@pytest.fixture()
def client(root):
    return TestClient(create_app(AppPaths(root), today=lambda: TODAY))


def test_health_and_options(client, toy_df):
    assert client.get("/api/salud").json() == {"ok": True}
    assert client.get("/api/plantas").json()["plantas"] == ["A", "B", "C"]
    assert len(client.get("/api/skus").json()["skus"]) == 4
    assert client.get("/api/skus", params={"planta": "a"}).json()["skus"] == sorted(toy_df[toy_df["planta"] == "A"]["sku"].unique())


def test_unknown_planta_is_a_404_with_the_closest_names(client):
    r = client.get("/api/consumo", params={"planta": "Z"})
    assert r.status_code == 404
    assert "Z" in r.json()["detalle"] and r.json()["sugerencias"] == []
    r = client.get("/api/consumo", params={"planta": "A", "sku": "SKU9/01/100gsm/2100mm/1200-1450"})
    assert r.status_code == 404 and r.json()["campo"] == "el SKU"


def test_suggestions_for_planta_and_for_the_skus_of_one_planta(client):
    assert client.get("/api/sugerencias", params={"campo": "planta", "q": "b"}).json()["sugerencias"] == ["B"]
    got = client.get("/api/sugerencias", params={"campo": "sku", "q": "sku2", "planta": "A"}).json()["sugerencias"]
    assert got == ["SKU2/01/100gsm/2100mm/1200-1450"]
    assert client.get("/api/sugerencias", params={"campo": "otro"}).status_code == 422


def test_monthly_total_and_breakdown_by_planta_add_up(client, toy_df):
    body = client.get("/api/consumo").json()
    assert body["nivel"] == "mensual" and body["filtro"] == {"planta": None, "sku": None}
    assert sum(body["valores"]) == pytest.approx(toy_df["consumo"].sum(), rel=1e-6)
    assert body["desglose"]["por"] == "planta"
    names = [s["nombre"] for s in body["desglose"]["series"]]
    assert sorted(names) == ["A", "B", "C"]  # biggest first
    stacked = [sum(s["valores"][i] or 0 for s in body["desglose"]["series"]) for i in range(len(body["valores"]))]
    assert stacked == pytest.approx(body["valores"], abs=0.01)


def test_breakdown_keeps_the_biggest_series_and_groups_the_rest_as_otros(client, toy_df, monkeypatch):
    mine = toy_df[toy_df["planta"] == "A"]
    body = client.get("/api/consumo", params={"planta": "A"}).json()
    assert body["resumen"]["n_series"] == 4 and body["resumen"]["total_TO"] == pytest.approx(mine["consumo"].sum(), abs=0.1)
    assert [s["nombre"] for s in body["desglose"]["series"]].count("Otros") == 0
    monkeypatch.setattr(consumption_service, "TOP_SERIES", 2)
    body = client.get("/api/consumo", params={"planta": "A"}).json()
    series = body["desglose"]["series"]
    assert [s["nombre"] for s in series][-1] == "Otros" and len(series) == 3
    assert sum(sum(s["valores"]) for s in series) == pytest.approx(sum(body["valores"]), abs=0.01)
    totals = [sum(s["valores"]) for s in series[:-1]]
    assert totals == sorted(totals, reverse=True)
    top_plantas = client.get("/api/consumo").json()["desglose"]["series"]
    assert [s["nombre"] for s in top_plantas][-1] == "Otros" and len(top_plantas) == 3


def test_one_series_is_empty_before_its_first_month_and_has_a_summary(client, toy_df):
    sku = "SKU0/01/100gsm/2100mm/1200-1450"  # starts late
    body = client.get("/api/consumo", params={"planta": "a", "sku": sku}).json()
    assert body["valores"][0] is None and body["desglose"] is None
    first = body["resumen"]["primer_mes"]
    assert first > body["etiquetas"][0]
    mine = toy_df[(toy_df["planta"] == "A") & (toy_df["sku"] == sku)]
    assert body["resumen"]["total_TO"] == pytest.approx(mine["consumo"].sum(), abs=0.1)
    last = body["resumen"]["ultimo_mes_completo"]
    by_month = mine.groupby(mine["fecha"].dt.strftime("%Y-%m"))["consumo"].sum()
    previous = by_month.iloc[-2]
    assert body["resumen"]["ultimo_mes_TO"] == pytest.approx(by_month[last], abs=0.1)
    assert body["resumen"]["cambio_vs_mes_anterior_pct"] == pytest.approx((by_month[last] / previous - 1) * 100, abs=0.1)


def test_sku_without_planta_adds_the_same_name_across_plantas(client, toy_df):
    sku = "SKU1/01/100gsm/2100mm/1200-1450"
    body = client.get("/api/consumo", params={"sku": sku}).json()
    assert body["resumen"]["n_series"] == 3
    assert body["resumen"]["total_TO"] == pytest.approx(toy_df[toy_df["sku"] == sku]["consumo"].sum(), abs=0.1)


def test_partial_last_month_is_flagged_and_left_out_of_the_summary(root, toy_df):
    cut = toy_df[toy_df["fecha"] <= pd.Timestamp("2025-05-12")]
    write_data(root, cut)
    client = TestClient(create_app(AppPaths(root), today=lambda: TODAY))
    summary = client.get("/api/consumo").json()["resumen"]
    assert summary["mes_parcial"] == "2025-05" and summary["ultimo_mes_completo"] == "2025-04"
    assert summary["ultimo_dato"] == "2025-05-12"
    data = client.get("/api/estado").json()["datos"]
    assert data["ultimo_mes_completo"] == "2025-04" and data["mes_parcial"] == "2025-05"


def test_daily_series_defaults_to_the_last_90_days_and_takes_a_range(client, toy_df):
    body = client.get("/api/consumo", params={"nivel": "diario"}).json()
    assert len(body["etiquetas"]) == 90 and body["etiquetas"][-1] == "2025-06-30"
    r = client.get("/api/consumo", params={"nivel": "diario", "planta": "B", "desde": "2025-06-02", "hasta": "2025-06-08"}).json()
    assert len(r["etiquetas"]) == 7
    day = toy_df[(toy_df["planta"] == "B") & (toy_df["fecha"] >= "2025-06-02") & (toy_df["fecha"] <= "2025-06-08")]
    assert r["resumen"]["total_TO"] == pytest.approx(day["consumo"].sum(), abs=0.1)
    assert r["resumen"]["dias_con_consumo"] == day["fecha"].nunique()
    bad = client.get("/api/consumo", params={"nivel": "diario", "desde": "2025-06-09", "hasta": "2025-06-02"})
    assert bad.status_code == 422


def test_without_data_the_api_says_so_and_status_still_answers(tmp_path):
    client = TestClient(create_app(AppPaths(tmp_path), today=lambda: TODAY))
    r = client.get("/api/consumo")
    assert r.status_code == 503 and "Datos" in r.json()["detalle"]
    status = client.get("/api/estado").json()
    assert status["datos"] is None and status["pronostico"] is None and status["ajuste"]["estado"] == "sin_registro"


def test_the_panel_is_rebuilt_when_the_file_changes(root):
    client = TestClient(create_app(AppPaths(root), today=lambda: TODAY))
    assert len(client.get("/api/skus").json()["skus"]) == 4
    write_data(root, make_consumption(extra_sku=True))
    assert len(client.get("/api/skus").json()["skus"]) == 5


# ---- status ------------------------------------------------------------------------------------------------------------------


def write_adjustments(paths, last, months=6):
    paths.results.mkdir(exist_ok=True)
    paths.adjustments_json.write_text(json.dumps({"ultimo_ajuste": last, "intervalo_meses": months}), encoding="utf-8")


def test_add_months_clips_to_the_end_of_a_shorter_month():
    assert add_months(date(2026, 8, 31), 6) == date(2027, 2, 28)
    assert add_months(date(2026, 10, 5), 6) == date(2027, 4, 5)


@pytest.mark.parametrize(
    "last, state, left",
    [("2026-10-05", "al_dia", 167), ("2026-04-25", "por_vencer", 5), ("2026-04-19", "vencido", -1)],
)
def test_retune_state_depends_on_the_days_left(tmp_path, last, state, left):
    paths = AppPaths(tmp_path)
    write_adjustments(paths, last)
    got = retune_status(paths, TODAY)
    assert (got["estado"], got["dias_restantes"]) == (state, left)


def make_tracking(month, run_id, ml_levels, alert=""):
    rows = [
        {"run_id": run_id, "mes": month, "nivel": level, "planta": "", "modelo": "ml_mean", "acierto": value, "wape": 1 - value, "bias_pct": 0.0, "n": 10,
         "real_TO": 1.0, "pronostico_TO": 1.0, "alerta": alert if level == "sku" else "", "creado_en": "2026-10-20 10:00:00"}
        for level, value in ml_levels.items()
    ]
    return pd.DataFrame(rows)[TRACKING_COLUMNS]


def test_last_forecast_reads_the_manifest_when_there_is_one(tmp_path):
    paths = AppPaths(tmp_path)
    paths.results.mkdir()
    assert last_forecast(paths) is None
    pd.DataFrame({"planta": ["A"], "sku": ["x"], "mes": ["2026-10"], "ml_mean": [2.5]}).to_csv(paths.results / "forecast_2026-10.csv", index=False)
    assert last_forecast(paths)["total_ml_mean_TO"] == 2.5 and last_forecast(paths)["creado_en"] is None
    manifest = {"run": {"created_at": "2026-10-19T17:10:18+00:00"}, "output": {"total_ml_mean": 9.0, "rows": 3}}
    (paths.results / "forecast_2026-10.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    got = last_forecast(paths)
    assert (got["total_ml_mean_TO"], got["series"], got["creado_en"][:10]) == (9.0, 3, "2026-10-19")


def test_tracking_prefers_production_runs_over_the_backtest_and_collects_alerts(tmp_path):
    paths = AppPaths(tmp_path)
    paths.results.mkdir()
    assert last_tracking(paths) is None
    write_tracking_csv(paths.tracking_csv, make_tracking("2026-09", "backtest-h1", {"sku": 0.66, "planta": 0.88, "total": 0.99}))
    got = last_tracking(paths)
    assert (got["origen"], got["mes"], got["acierto_sku"], got["acierto_total"]) == ("backtest", "2026-09", 66.0, 99.0)
    write_tracking_csv(paths.tracking_csv, make_tracking("2026-10", "run-1", {"sku": 0.6, "planta": 0.9, "total": 0.95}, alert="sku_bajo;peor_que_naive"))
    got = last_tracking(paths)
    assert (got["origen"], got["mes"], got["alertas"]) == ("produccion", "2026-10", ["peor_que_naive", "sku_bajo"])


def test_a_forecast_is_pending_only_when_its_month_closed_and_is_not_tracked(tmp_path):
    paths = AppPaths(tmp_path)
    paths.results.mkdir()
    forecast = {"mes": "2026-10"}
    assert pending_tracking(paths, "2026-09", forecast) is None  # the month has not closed
    assert pending_tracking(paths, "2026-10", None) is None and pending_tracking(paths, None, forecast) is None
    assert pending_tracking(paths, "2026-10", forecast) == "2026-10"
    write_tracking_csv(paths.tracking_csv, make_tracking("2026-10", "backtest-h1", {"sku": 0.6}))
    assert pending_tracking(paths, "2026-10", forecast) == "2026-10"  # a backtest row does not count
    write_tracking_csv(paths.tracking_csv, make_tracking("2026-10", "run-1", {"sku": 0.6}))
    assert pending_tracking(paths, "2026-10", forecast) is None


def test_status_names_the_next_forecast_month(root):
    client = TestClient(create_app(AppPaths(root), today=lambda: TODAY))
    status = client.get("/api/estado").json()
    assert status["proximo_pronostico"] == "2025-07" and status["datos"]["filas"] > 0


# ---- screens -----------------------------------------------------------------------------------------------------------------


def test_the_screens_are_served_and_every_file_they_load_exists(client):
    import re

    from src.app.paths import STATIC

    page = client.get("/")
    assert page.status_code == 200 and "detour" in page.text
    for ref in re.findall(r'(?:src|href)="([^"]+)"', page.text):
        assert (STATIC / ref).is_file(), ref
        assert client.get("/" + ref).status_code == 200
    assert client.get("/api/salud").json() == {"ok": True}  # the API routes win over the static mount


def test_the_javascript_has_no_syntax_errors():
    import shutil
    import subprocess

    from src.app.paths import STATIC

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    done = subprocess.run([node, "--check", str(STATIC / "app.js")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
