"""Starting the app (server thread, log file, window or browser) and the maintenance tools of the packaged build."""

import base64
import json
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from src.app import launcher, tools
from src.app.paths import AppPaths, default_paths
from src.app.security import hash_password, save_record, verify_password
from tests.conftest import make_consumption
from tests.test_app_api import write_data


@pytest.fixture()
def home(tmp_path):
    write_data(tmp_path, make_consumption())
    save_record(AppPaths(tmp_path).security_json, hash_password("prueba"))
    return AppPaths(tmp_path)


# ---- where the files are -----------------------------------------------------------------------------------------------------


def test_the_folder_is_the_repository_unless_the_environment_or_the_package_say_otherwise(monkeypatch, tmp_path):
    monkeypatch.delenv("DETOUR_HOME", raising=False)
    assert default_paths().root == Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "detour.exe"))
    assert default_paths().root == tmp_path.resolve()  # the folder of the .exe
    monkeypatch.setenv("DETOUR_HOME", str(tmp_path / "otra"))
    assert default_paths().root == (tmp_path / "otra").resolve()


# ---- server, log and window --------------------------------------------------------------------------------------------------


def test_the_server_answers_on_this_machine_only_and_stops(home):
    server = launcher.Server(home)
    assert server.url == f"http://127.0.0.1:{server.port}" and 1024 < server.port < 65536
    server.start()
    try:
        with urllib.request.urlopen(server.url + "/api/salud", timeout=5) as reply:
            assert json.loads(reply.read()) == {"ok": True}
    finally:
        server.stop()
    assert not server._thread.is_alive()


def test_a_server_that_cannot_start_says_so(home):
    first = launcher.Server(home)
    first.start()
    try:
        second = launcher.Server(home, port=first.port)  # the port is taken
        with pytest.raises(RuntimeError, match="detuvo|respondió"):
            second.start(timeout=5)
    finally:
        first.stop()


def test_the_log_goes_to_a_file_and_replaces_the_missing_console_of_a_windowed_program(home, monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    path = launcher.setup_logging(home)
    assert path == home.logs / "detour.log" and sys.stdout is not None and sys.stderr is not None
    print("texto de prueba")
    import logging

    logging.getLogger("prueba").info("registro de prueba")
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = path.read_text(encoding="utf-8")
    assert "texto de prueba" in text and "registro de prueba" in text
    sys.stdout.close()


def test_the_window_saves_files_through_a_native_dialog(tmp_path):
    chosen = {"value": str(tmp_path / "pronostico.xlsx")}

    class Window:
        def create_file_dialog(self, kind, save_filename=""):
            assert kind == "SAVE" and save_filename == "pronostico.xlsx"
            return chosen["value"]

    api = launcher.WindowApi(lambda: Window(), "SAVE")
    done = api.save_file("pronostico.xlsx", base64.b64encode(b"contenido").decode())
    assert done == {"guardado": True, "ruta": chosen["value"]} and Path(chosen["value"]).read_bytes() == b"contenido"
    chosen["value"] = (str(tmp_path / "otro.png"),)  # some systems answer with a tuple
    assert api.save_file("pronostico.xlsx", base64.b64encode(b"x").decode())["ruta"].endswith("otro.png")
    chosen["value"] = None  # the user cancelled
    assert api.save_file("pronostico.xlsx", "eA==") == {"guardado": False}


def fake_webview(fail_on=None):
    module = types.ModuleType("webview")
    module.settings = {"ALLOW_DOWNLOADS": False}
    module.FileDialog = types.SimpleNamespace(SAVE="SAVE")
    module.calls = []

    def create_window(*args, **kwargs):
        if fail_on == "create":
            raise RuntimeError("no window system")
        module.calls.append((args, kwargs))
        return object()

    def start(*args, **kwargs):
        if fail_on == "start":
            raise RuntimeError("no GUI")

    module.create_window, module.start = create_window, start
    return module


def test_the_window_opens_with_downloads_allowed_and_falls_back_when_it_cannot(monkeypatch):
    module = fake_webview()
    monkeypatch.setitem(sys.modules, "webview", module)
    assert launcher.run_window("http://127.0.0.1:1") is True
    (args, kwargs), = module.calls
    assert args == (launcher.TITLE, "http://127.0.0.1:1") and "Luis Moreno" in launcher.TITLE and kwargs["width"] >= 1000 and kwargs["js_api"] is not None and module.settings["ALLOW_DOWNLOADS"] is True
    for failure in ("create", "start"):
        monkeypatch.setitem(sys.modules, "webview", fake_webview(failure))
        assert launcher.run_window("http://127.0.0.1:1") is False
    monkeypatch.setitem(sys.modules, "webview", None)  # not installed
    assert launcher.run_window("http://127.0.0.1:1") is False


class FakeServer:
    def __init__(self, paths, port=None):
        self.url, self.stopped = "http://127.0.0.1:9", False

    def start(self):
        pass

    def stop(self):
        self.stopped = True


def test_main_opens_the_window_or_the_browser_and_stops_the_server(home, monkeypatch, capsys):
    servers = []
    monkeypatch.setattr(launcher, "Server", lambda paths, port=None: servers.append(FakeServer(paths, port)) or servers[-1])
    monkeypatch.setattr(launcher, "run_window", lambda url: True)
    assert launcher.main(["--home", str(home.root)]) == 0 and servers[-1].stopped

    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt))  # "Ctrl+C" at once
    assert launcher.main(["--browser", "--home", str(home.root)]) == 0 and opened == ["http://127.0.0.1:9"] and servers[-1].stopped
    assert launcher.main(["--browser", "--no-open", "--home", str(home.root)]) == 0 and opened == ["http://127.0.0.1:9"]
    assert "http://127.0.0.1:9" in capsys.readouterr().out


def test_main_reports_a_start_failure_and_the_version(home, monkeypatch, capsys):
    def broken(paths, port=None):
        raise RuntimeError("puerto ocupado")

    boxes = []
    monkeypatch.setattr(launcher, "_message_box", lambda message: boxes.append(message))  # a real box would wait for a click (it hung the Windows CI)
    monkeypatch.setattr(launcher, "Server", broken)
    assert launcher.main(["--home", str(home.root)]) == 1
    if sys.platform == "win32":  # a windowed program has no console: the user gets a box
        assert boxes == ["detour no pudo arrancar: puerto ocupado"]
    else:
        assert boxes == [] and "puerto ocupado" in capsys.readouterr().err
    assert "puerto ocupado" in (home.logs / "detour.log").read_text(encoding="utf-8")
    assert launcher.main(["--version"]) == 0 and capsys.readouterr().out.strip() == "0.1.0-beta"


def test_on_windows_the_error_goes_to_a_message_box_and_a_failing_box_is_ignored(monkeypatch):
    boxes = []
    monkeypatch.setattr(launcher, "_message_box", lambda message: boxes.append(message))
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    launcher.show_error("no arrancó")
    assert boxes == ["no arrancó"]

    def broken(message):
        raise OSError("no desktop")

    monkeypatch.setattr(launcher, "_message_box", broken)
    launcher.show_error("otra vez")  # must not raise


# ---- the maintenance tools ---------------------------------------------------------------------------------------------------


def test_the_self_test_checks_libraries_screens_and_data_on_a_copy_and_leaves_the_folder_alone(home, capsys):
    before = sorted(str(p.relative_to(home.root)) for p in home.root.rglob("*"))
    assert tools.self_test(home) == 0
    output = capsys.readouterr().out
    for line in ("biblioteca lightgbm", "pantalla principal", "ECharts incluido", "estado", "consumo real"):
        assert f"[ok] {line}" in output
    assert "comprobaciones bien" in output and "FALLA" not in output
    assert sorted(str(p.relative_to(home.root)) for p in home.root.rglob("*")) == before


def test_the_self_test_can_run_a_real_forecast_and_an_excel_download(home, capsys):
    assert tools.self_test(home, forecast=True) == 0
    output = capsys.readouterr().out
    assert "[ok] generar el pronóstico" in output and "[ok] descarga de Excel" in output
    assert not (home.results).exists()  # the forecast went to the temporary copy


def test_the_self_test_fails_when_something_is_missing(tmp_path, capsys):
    empty = AppPaths(tmp_path)
    assert tools.self_test(empty) == 1
    assert "[FALLA] estado" in capsys.readouterr().out


def test_a_missing_library_is_reported_by_name(monkeypatch):
    real = tools.importlib.import_module

    def fake(name):
        if name == "optuna":
            raise ImportError("no hay optuna")
        return real(name)

    monkeypatch.setattr(tools.importlib, "import_module", fake)
    failed = [r for r in tools.check_libraries() if not r[1]]
    assert [r[0] for r in failed] == ["biblioteca optuna"] and "no hay optuna" in failed[0][2]
    assert tools.check_window()[0][1] is True  # the window is informative only


def test_the_password_is_changed_from_the_tools_and_only_the_hash_is_kept(home, monkeypatch, capsys):
    answers = iter(["nueva-clave", "nueva-clave"])
    tools.set_password(home, ask=lambda prompt: next(answers))
    record = json.loads(home.security_json.read_text())["clave"]
    assert verify_password("nueva-clave", record) and "nueva-clave" not in home.security_json.read_text()
    for mine, message in ((["abc", "abc"], "al menos 4"), (["clave-uno", "clave-dos"], "no coinciden")):
        answers = iter(mine)
        with pytest.raises(SystemExit, match=message):
            tools.set_password(home, ask=lambda prompt: next(answers))
    assert verify_password("nueva-clave", json.loads(home.security_json.read_text())["clave"])  # nothing changed


def test_the_tools_commands(home, monkeypatch, capsys):
    assert tools.main(["version"]) == 0 and capsys.readouterr().out.strip() == "0.1.0-beta"
    assert tools.main(["--home", str(home.root), "self-test"]) == 0
    monkeypatch.setattr(tools.getpass, "getpass", lambda prompt: "otra-clave")
    assert tools.main(["--home", str(home.root), "set-password"]) == 0
    assert verify_password("otra-clave", json.loads(home.security_json.read_text())["clave"])

