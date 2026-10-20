"""Maintenance commands of the packaged app (`detour-tools.exe`, a console program): version, self-test and the password of the Ajuste section."""

from __future__ import annotations

import argparse
import getpass
import importlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Callable

from src.app import __version__
from src.app.launcher import Server
from src.app.paths import AppPaths, default_paths
from src.app.security import hash_password, save_record

LIBRARIES = ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "optuna", "openpyxl", "fastapi", "uvicorn", "python_multipart"]
Result = tuple[str, bool, str]


def check_libraries() -> list[Result]:
    r"""Import every library the app needs; a missing one in a packaged build shows up here and not in front of the user."""
    results = []
    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
            results.append((f"biblioteca {name}", True, str(getattr(module, "__version__", ""))))
        except Exception as error:
            results.append((f"biblioteca {name}", False, f"{error.__class__.__name__}: {error}"))
    return results


def check_window() -> list[Result]:
    r"""The window library (pywebview): informative on a machine without a screen, so a failure is reported but does not fail the self-test."""
    try:
        importlib.import_module("webview")
        if sys.platform == "win32":
            importlib.import_module("webview.platforms.edgechromium")
        return [("ventana (pywebview)", True, "disponible")]
    except Exception as error:
        return [("ventana (pywebview)", True, f"no disponible, se usaría el navegador ({error.__class__.__name__})")]


def _get(url: str, timeout: float = 60.0) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as reply:
        return reply.status, reply.read()


def check_app(paths: AppPaths, forecast: bool, wait: float = 600.0) -> list[Result]:
    r"""Start the API on a copy of the folder and use it: screens, data, an Excel download and, optionally, a real forecast."""
    results: list[Result] = []

    def step(name: str, action: Callable[[], str]) -> None:
        try:
            results.append((name, True, action()))
        except Exception as error:
            results.append((name, False, f"{error.__class__.__name__}: {error}"))

    server = Server(paths)
    server.start()

    def home_page() -> str:
        if b"detour" not in _get(server.url + "/")[1]:
            raise ValueError("no es la pantalla de la app")
        return "ok"

    def run_forecast() -> str:
        request = urllib.request.Request(server.url + "/api/pronostico/generar", data=b'{"sobrescribir": true}', headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as reply:
            job = json.loads(reply.read())["trabajo"]
        end = time.time() + wait
        while time.time() < end:
            state = json.loads(_get(f"{server.url}/api/trabajos/{job}")[1])
            if state["estado"] == "listo":
                return f"{state['resultado']['series']} series, {state['resultado']['total_TO']} TO, {state['segundos']} s"
            if state["estado"] == "error":
                raise RuntimeError(state["error"])
            time.sleep(1)
        raise TimeoutError("el pronóstico no terminó a tiempo")

    def excel() -> str:
        status, body = _get(server.url + "/api/descargas/pronostico")
        if status != 200 or body[:2] != b"PK":
            raise ValueError("no es un archivo Excel")
        return f"{len(body) // 1000} KB"

    try:
        step("pantalla principal", home_page)
        step("ECharts incluido", lambda: f"{len(_get(server.url + '/vendor/echarts.min.js')[1]) // 1000} KB")
        step("estado", lambda: str(json.loads(_get(server.url + "/api/estado")[1])["datos"]["ultimo_dia"]))
        step("consumo real", lambda: f"{len(json.loads(_get(server.url + '/api/consumo')[1])['etiquetas'])} meses")
        if forecast:
            step("generar el pronóstico", run_forecast)
            step("descarga de Excel", excel)
    finally:
        server.stop()
    return results


def self_test(paths: AppPaths, forecast: bool = False) -> int:
    r"""Run the checks on a temporary copy of the folder (nothing of the real one changes) and print one line per check; exit code 0 when all pass."""
    results = check_libraries() + check_window()
    with tempfile.TemporaryDirectory(prefix="detour_self_test_") as temp:
        copy = Path(temp)
        for folder in ("data", "config", "results"):
            if (paths.root / folder).is_dir():
                shutil.copytree(paths.root / folder, copy / folder, ignore=shutil.ignore_patterns("staging", "respaldo"))
        results += check_app(AppPaths(copy), forecast)
    for name, ok, detail in results:
        print(f"[{'ok' if ok else 'FALLA'}] {name}: {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)} de {len(results)} comprobaciones bien" + (f"; fallan: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def set_password(paths: AppPaths, ask: Callable[[str], str] | None = None) -> None:
    ask = ask or getpass.getpass
    first = ask("Nueva clave: ")
    if len(first) < 4:
        raise SystemExit("La clave debe tener al menos 4 caracteres.")
    if ask("Repite la clave: ") != first:
        raise SystemExit("Las dos claves no coinciden. No se cambió nada.")
    save_record(paths.security_json, hash_password(first))
    print(f"Listo: se guardó solo el hash en {paths.security_json}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="detour-tools", description="Herramientas de mantenimiento de detour")
    parser.add_argument("--home", type=Path, default=None, help="carpeta con data, results y config")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="versión de la app")
    test = commands.add_parser("self-test", help="comprobar que todo lo necesario funciona")
    test.add_argument("--forecast", action="store_true", help="incluye generar un pronóstico real (unos 30 segundos)")
    commands.add_parser("set-password", help="cambiar la clave de la sección Ajuste")
    args = parser.parse_args(argv)
    paths = AppPaths(args.home.resolve()) if args.home else default_paths()
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "set-password":
        set_password(paths)
        return 0
    return self_test(paths, args.forecast)
