"""The Ajuste section: password (hash only), sessions, background retune, adopting or discarding the candidate and the schedule."""

import json
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app import security, tuning_service
from src.app.api import create_app
from src.app.consumption_service import ConsumptionStore
from src.app.paths import AppPaths
from src.app.security import Locked, NoPassword, Sessions, Unauthorized, WrongPassword, hash_password, save_record, verify_password
from tests.conftest import make_consumption
from tests.test_app_api import write_data
from tests.test_app_forecast import wait

TODAY = date(2026, 10, 20)
PASSWORD = "clave-de-prueba"


# ---- password and sessions ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["scrypt", "pbkdf2"])
def test_only_a_hash_is_stored_and_it_verifies_only_the_right_password(algorithm, monkeypatch, tmp_path):
    monkeypatch.setattr(security, "PBKDF2_ROUNDS", 1000)
    record = hash_password(PASSWORD, algorithm=algorithm)
    assert record["algoritmo"] == algorithm and verify_password(PASSWORD, record)
    assert not verify_password("otra", record) and not verify_password(PASSWORD.upper(), record)
    path = tmp_path / "config" / "seguridad.json"
    save_record(path, record)
    assert PASSWORD not in path.read_text() and security.load_record(path) == record
    assert hash_password(PASSWORD, algorithm=algorithm)["sal"] != record["sal"]  # a new salt each time


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def make_sessions(clock=None):
    record = hash_password(PASSWORD, algorithm="pbkdf2")
    return Sessions(lambda: record, clock or Clock())


def test_a_session_lasts_while_it_is_used_and_expires_when_idle(monkeypatch):
    monkeypatch.setattr(security, "PBKDF2_ROUNDS", 1000)
    clock = Clock()
    sessions = make_sessions(clock)
    token = sessions.login(PASSWORD)
    sessions.check(token)
    clock.now += security.IDLE_SECONDS - 1
    sessions.check(token)  # using it extends it
    clock.now += security.IDLE_SECONDS - 1
    sessions.check(token)
    clock.now += security.IDLE_SECONDS + 1
    with pytest.raises(Unauthorized):
        sessions.check(token)
    for bad in (None, "", "inventado"):
        with pytest.raises(Unauthorized):
            sessions.check(bad)
    token = sessions.login(PASSWORD)
    sessions.logout(token)
    with pytest.raises(Unauthorized):
        sessions.check(token)


def test_five_wrong_passwords_lock_the_screen_for_a_minute(monkeypatch):
    monkeypatch.setattr(security, "PBKDF2_ROUNDS", 1000)
    clock = Clock()
    sessions = make_sessions(clock)
    for remaining in (4, 3, 2, 1):
        with pytest.raises(WrongPassword) as error:
            sessions.login("mal")
        assert error.value.remaining == remaining
    with pytest.raises(Locked):
        sessions.login("mal")
    with pytest.raises(Locked) as locked:
        sessions.login(PASSWORD)  # the right one does not help while it is locked
    assert 1 <= locked.value.seconds <= security.LOCK_SECONDS
    clock.now += security.LOCK_SECONDS + 1
    assert sessions.login(PASSWORD)
    with pytest.raises(WrongPassword) as error:  # the count started again
        sessions.login("mal")
    assert error.value.remaining == 4


def test_a_success_resets_the_failed_attempts(monkeypatch):
    monkeypatch.setattr(security, "PBKDF2_ROUNDS", 1000)
    sessions = make_sessions()
    for _ in range(3):
        with pytest.raises(WrongPassword):
            sessions.login("mal")
    sessions.login(PASSWORD)
    with pytest.raises(WrongPassword) as error:
        sessions.login("mal")
    assert error.value.remaining == 4


def test_without_a_configured_password_nothing_unlocks():
    sessions = Sessions(lambda: None)
    assert not sessions.configured()
    with pytest.raises(NoPassword):
        sessions.login("x")


# ---- the screen's API --------------------------------------------------------------------------------------------------------


@pytest.fixture()
def root(tmp_path, monkeypatch):
    write_data(tmp_path, make_consumption())
    save_record(AppPaths(tmp_path).security_json, hash_password(PASSWORD))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ajustes.json").write_text(json.dumps({"ultimo_ajuste": "2026-10-05", "intervalo_meses": 6, "historial": []}), encoding="utf-8")
    monkeypatch.setattr(tuning_service, "TEST_MONTHS", 3)
    monkeypatch.setitem(tuning_service.PRESETS["rapido"], "trials", {"lgbm": 2})
    return tmp_path


@pytest.fixture()
def client(root):
    return TestClient(create_app(AppPaths(root), today=lambda: TODAY))


def unlock(client):
    reply = client.post("/api/ajuste/desbloquear", json={"clave": PASSWORD})
    assert reply.status_code == 200
    return {"X-Ajuste-Token": reply.json()["token"]}


def retune_and_wait(client, headers, kind="rapido"):
    start = client.post("/api/ajuste/reafinar", json={"tipo": kind}, headers=headers)
    assert start.status_code == 200, start.text
    return wait(client, start.json()["trabajo"])


def test_the_state_is_public_and_the_actions_need_the_session(client):
    body = client.get("/api/ajuste").json()
    assert body["estado"]["estado"] == "al_dia" and body["estado"]["proximo"] == "2027-04-05" and body["historial"] == [] and body["candidato"] is None
    assert [t["id"] for t in body["tipos"]] == ["completo", "rapido"] and body["clave_configurada"] and not body["mes_parcial"]
    for path in ("reafinar", "adoptar", "descartar"):
        for headers in ({}, {"X-Ajuste-Token": "inventado"}):
            reply = client.post(f"/api/ajuste/{path}", json={}, headers=headers)
            assert reply.status_code == 401 and reply.json()["bloqueado"] is True


def test_the_wrong_password_says_how_many_attempts_are_left_and_five_lock_it(client):
    first = client.post("/api/ajuste/desbloquear", json={"clave": "mal"})
    assert first.status_code == 401 and first.json()["intentos"] == 4
    for _ in range(3):
        client.post("/api/ajuste/desbloquear", json={"clave": "mal"})
    locked = client.post("/api/ajuste/desbloquear", json={"clave": "mal"})
    assert locked.status_code == 429 and locked.json()["segundos"] == security.LOCK_SECONDS
    assert client.post("/api/ajuste/desbloquear", json={"clave": PASSWORD}).status_code == 429


def test_locking_ends_the_session(client):
    headers = unlock(client)
    assert client.post("/api/ajuste/bloquear", headers=headers).status_code == 200
    assert client.post("/api/ajuste/descartar", headers=headers).status_code == 401


def test_without_a_password_file_the_section_cannot_be_unlocked(tmp_path):
    write_data(tmp_path, make_consumption())
    client = TestClient(create_app(AppPaths(tmp_path), today=lambda: TODAY))
    assert client.get("/api/ajuste").json()["clave_configurada"] is False
    assert client.post("/api/ajuste/desbloquear", json={"clave": "x"}).status_code == 503


def test_a_retune_leaves_a_candidate_and_changes_nothing_until_it_is_adopted(client, root):
    headers = unlock(client)
    state = retune_and_wait(client, headers)
    assert state["estado"] == "listo", state
    paths = AppPaths(root)
    assert paths.tuning_candidate.exists() and not paths.production_tuning.exists()
    body = client.get("/api/ajuste").json()
    candidate = body["candidato"]
    assert [m["modelo"] for m in candidate["modelos"]] == ["lgbm"] and candidate["tipo"] == "rapido"
    assert {"sku_base", "sku_ajustado", "mejora_sku_pts", "planta_base", "total_ajustado"} <= set(candidate["modelos"][0])
    assert candidate["mejora_sku_pts"] == candidate["modelos"][0]["mejora_sku_pts"]
    assert body["estado"]["ultimo"] == "2026-10-05"  # the schedule did not move
    assert client.get("/api/estado").json()["ajuste"]["candidato"] is True


def test_adopting_makes_it_the_production_file_updates_the_schedule_and_the_forecast_uses_it(client, root):
    headers = unlock(client)
    retune_and_wait(client, headers)
    adopted = client.post("/api/ajuste/adoptar", headers=headers)
    assert adopted.status_code == 200
    paths = AppPaths(root)
    assert paths.production_tuning.exists() and not paths.tuning_candidate.exists()
    state = adopted.json()["estado"]
    assert state["ultimo"] == "2026-10-20" and state["proximo"] == "2027-04-20" and state["estado"] == "al_dia"
    entry = adopted.json()["historial"][0]
    assert entry["fecha"] == "2026-10-20" and entry["origen"] == "app" and entry["tipo"] == "rapido"
    assert json.loads(paths.adjustments_json.read_text())["ultimo_ajuste"] == "2026-10-20"

    start = client.post("/api/pronostico/generar", json={})
    assert wait(client, start.json()["trabajo"])["estado"] == "listo"
    manifest = json.loads((root / "results" / "forecast_2025-07.manifest.json").read_text())
    assert manifest["params"]["tuning_file"].endswith("tuning_produccion.json")

    retune_and_wait(client, headers)  # a second adoption keeps the first one in the archive
    client.post("/api/ajuste/adoptar", headers=headers)
    assert [f.name for f in paths.tuning_archive.glob("*.json")] == ["tuning_produccion_2026-10-20.json"]
    assert len(client.get("/api/ajuste").json()["historial"]) == 2


def test_discarding_removes_the_candidate_and_keeps_the_schedule(client, root):
    headers = unlock(client)
    assert client.post("/api/ajuste/descartar", headers=headers).status_code == 404  # nothing to discard
    assert client.post("/api/ajuste/adoptar", headers=headers).status_code == 404
    retune_and_wait(client, headers)
    assert client.post("/api/ajuste/descartar", headers=headers).status_code == 200
    paths = AppPaths(root)
    assert not paths.tuning_candidate.exists() and not paths.production_tuning.exists()
    assert client.get("/api/ajuste").json()["estado"]["ultimo"] == "2026-10-05"


def test_a_retune_is_refused_for_an_unknown_type_and_for_a_partial_month(client, root):
    headers = unlock(client)
    state = retune_and_wait(client, headers, "inventado")
    assert state["estado"] == "error" and "inventado" in state["error"]
    df = make_consumption()
    write_data(root, df[df["fecha"] <= pd.Timestamp("2025-05-12")])
    assert client.get("/api/ajuste").json()["mes_parcial"] is True
    state = retune_and_wait(client, headers)
    assert state["estado"] == "error" and "incompleto" in state["error"]
    assert not AppPaths(root).tuning_candidate.exists()


def test_the_retune_reports_each_trial_to_the_progress_bar(root):
    store = ConsumptionStore(AppPaths(root))
    steps = []
    tuning_service.retune(AppPaths(root), store, "rapido", lambda message, fraction: steps.append((message, fraction)))
    messages = [m for m, _ in steps]
    assert "Ajustando LightGBM: prueba 1 de 2" in messages and "Ajustando LightGBM: prueba 2 de 2" in messages
    fractions = [f for _, f in steps]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
