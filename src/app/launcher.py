"""Starts the desktop app: the local API in a thread, then the window (pywebview) or, when there is no window, the default browser."""

from __future__ import annotations

import argparse
import base64
import logging
import logging.handlers
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable

import uvicorn

from src.app import __version__
from src.app.api import create_app
from src.app.paths import AppPaths, default_paths

HOST = "127.0.0.1"
TITLE = "detour"
WINDOW_SIZE = (1360, 860)
MIN_SIZE = (1000, 640)


def free_port() -> int:
    r"""A port nobody is using on this machine."""
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def setup_logging(paths: AppPaths) -> Path:
    r"""Write the log to `logs/detour.log` (1 MB, 3 files). A windowed .exe has no console (stdout and stderr are None), so they go to the same file."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    path = paths.logs / "detour.log"
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    if sys.stdout is None or sys.stderr is None:
        stream = open(path, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    return path


class Server:
    r"""The API in a background thread (uvicorn), listening on this machine only."""

    def __init__(self, paths: AppPaths, port: int | None = None):
        self.port = port or free_port()
        self.url = f"http://{HOST}:{self.port}"
        config = uvicorn.Config(create_app(paths), host=HOST, port=self.port, log_config=None, access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="detour-api", daemon=True)

    def start(self, timeout: float = 30.0) -> None:
        r"""Start the thread and wait until uvicorn says it is listening (asking the port would also succeed if another program had it)."""
        self._thread.start()
        end = time.time() + timeout
        while time.time() < end:
            if self._server.started:
                return
            if not self._thread.is_alive():
                raise RuntimeError("El servidor de la app se detuvo al arrancar (¿el puerto está ocupado?). Revisa logs/detour.log.")
            time.sleep(0.05)
        raise RuntimeError("El servidor de la app no respondió a tiempo.")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


class WindowApi:
    r"""What the screens can ask the window for: a native "save as" dialog (a normal browser download does not exist inside the window)."""

    def __init__(self, window_provider: Callable[[], object], save_dialog: object):
        self._window, self._save_dialog = window_provider, save_dialog

    def save_file(self, name: str, content_base64: str) -> dict:
        chosen = self._window().create_file_dialog(self._save_dialog, save_filename=name)
        if not chosen:
            return {"guardado": False}
        target = Path(chosen if isinstance(chosen, str) else chosen[0])
        target.write_bytes(base64.b64decode(content_base64))
        return {"guardado": True, "ruta": str(target)}


def run_window(url: str) -> bool:
    r"""Open the app in a native window and block until it is closed; False when there is no window system (the caller falls back to the browser)."""
    try:
        import webview
    except Exception:  # not installed, or its GUI libraries are missing
        return False
    holder: list = []
    api = WindowApi(lambda: holder[0], webview.FileDialog.SAVE)
    webview.settings["ALLOW_DOWNLOADS"] = True
    try:
        holder.append(webview.create_window(TITLE, url, js_api=api, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], min_size=MIN_SIZE, text_select=True, zoomable=True))
        webview.start()
    except Exception:
        logging.getLogger(__name__).exception("The window could not be opened")
        return False
    return True


def show_error(message: str) -> None:
    r"""Tell the user the app could not start (a windowed .exe has no console to print to)."""
    logging.getLogger(__name__).error(message)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, TITLE, 0x10)
        except Exception:
            pass
    else:
        print(message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="detour: pronóstico de consumo de papel")
    parser.add_argument("--browser", action="store_true", help="abrir en el navegador en vez de la ventana")
    parser.add_argument("--no-open", action="store_true", help="solo arrancar el servidor (con --browser, sin abrir el navegador)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--home", type=Path, default=None, help="carpeta con data, results y config")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    paths = AppPaths(args.home.resolve()) if args.home else default_paths()
    setup_logging(paths)
    logging.getLogger(__name__).info("detour %s, folder %s", __version__, paths.root)
    try:
        server = Server(paths, args.port)
        server.start()
    except Exception as error:
        show_error(f"detour no pudo arrancar: {error}")
        return 1
    try:
        if not args.browser and run_window(server.url):
            return 0
        if not args.no_open:
            webbrowser.open(server.url)
        print(f"detour está en {server.url} (Ctrl+C para cerrar)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0
    finally:
        server.stop()
