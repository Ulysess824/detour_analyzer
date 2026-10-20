"""Local HTTP API of the desktop app (FastAPI). It listens on 127.0.0.1 only and serves the screens of the app from the services."""

from __future__ import annotations

from datetime import date
from typing import Callable, Literal

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.app.consumption_service import ConsumptionStore, DataMissing, NotFound
from src.app.paths import STATIC, AppPaths, default_paths
from src.app.status_service import app_status


def create_app(paths: AppPaths | None = None, today: Callable[[], date] = date.today) -> FastAPI:
    paths = paths or default_paths()
    store = ConsumptionStore(paths)
    app = FastAPI(title="detour", docs_url=None, redoc_url=None)

    @app.exception_handler(NotFound)
    def _not_found(_, error: NotFound):
        return JSONResponse(status_code=404, content={"detalle": str(error), "campo": error.kind, "sugerencias": error.options})

    @app.exception_handler(DataMissing)
    def _missing(_, error: DataMissing):
        return JSONResponse(status_code=503, content={"detalle": str(error)})

    @app.exception_handler(ValueError)
    def _invalid(_, error: ValueError):
        return JSONResponse(status_code=422, content={"detalle": str(error)})

    @app.get("/api/salud")
    def health():
        return {"ok": True}

    @app.get("/api/estado")
    def status():
        try:
            info = store.info()
        except DataMissing:
            info = None
        return app_status(paths, info, today())

    @app.get("/api/plantas")
    def plantas():
        return {"plantas": store.plantas()}

    @app.get("/api/skus")
    def skus(planta: str | None = None):
        return {"skus": store.skus(store.select(planta, None)[0])}

    @app.get("/api/sugerencias")
    def suggestions(campo: Literal["planta", "sku"], q: str = "", planta: str | None = None, limite: int = Query(8, ge=1, le=50)):
        scope = store.select(planta, None)[0] if planta else None
        return {"campo": campo, "sugerencias": store.suggestions(campo, q, scope, limite)}

    @app.get("/api/consumo")
    def consumption(
        planta: str | None = None,
        sku: str | None = None,
        nivel: Literal["mensual", "diario"] = "mensual",
        desde: str | None = None,
        hasta: str | None = None,
    ):
        if nivel == "diario":
            return store.daily(planta, sku, desde, hasta)
        return store.monthly(planta, sku)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")  # last: the API routes above win
    return app
