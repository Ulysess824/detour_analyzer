"""Local HTTP API of the desktop app (FastAPI). It listens on 127.0.0.1 only and serves the screens of the app from the services."""

from __future__ import annotations

from datetime import date
from typing import Callable, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.app.consumption_service import ConsumptionStore, DataMissing, NotFound
from src.app import forecast_service, tracking_service
from src.app.forecast_service import ForecastMissing
from src.app.jobs import Busy, JobRunner
from src.app.paths import STATIC, AppPaths, default_paths
from src.app.status_service import app_status


class GenerateBody(BaseModel):
    sobrescribir: bool = False


class MeasureBody(BaseModel):
    mes: str


def create_app(paths: AppPaths | None = None, today: Callable[[], date] = date.today) -> FastAPI:
    paths = paths or default_paths()
    store = ConsumptionStore(paths)
    jobs = JobRunner()
    app = FastAPI(title="detour", docs_url=None, redoc_url=None)

    @app.exception_handler(NotFound)
    def _not_found(_, error: NotFound):
        return JSONResponse(status_code=404, content={"detalle": str(error), "campo": error.kind, "sugerencias": error.options})

    @app.exception_handler(DataMissing)
    def _missing(_, error: DataMissing):
        return JSONResponse(status_code=503, content={"detalle": str(error)})

    @app.exception_handler(ForecastMissing)
    def _no_forecast(_, error: ForecastMissing):
        return JSONResponse(status_code=404, content={"detalle": str(error)})

    @app.exception_handler(LookupError)
    def _no_record(_, error: LookupError):
        return JSONResponse(status_code=404, content={"detalle": str(error.args[0]) if error.args else "No encontrado"})

    @app.exception_handler(Busy)
    def _busy(_, error: Busy):
        return JSONResponse(status_code=409, content={"detalle": str(error)})

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

    # ---- forecast -----------------------------------------------------------------------------------------------------------

    @app.get("/api/pronosticos")
    def forecasts():
        return {"pronosticos": forecast_service.list_forecasts(paths)}

    @app.get("/api/pronostico")
    def forecast(mes: str | None = None, planta: str | None = None, sku: str | None = None, limite: int = Query(forecast_service.TABLE_LIMIT, ge=1, le=500)):
        return forecast_service.forecast_detail(paths, store, mes, planta, sku, limite)

    @app.post("/api/pronostico/generar")
    def generate(body: GenerateBody):
        month = forecast_service.next_month(store)
        if paths.forecast_csv(month).exists() and not body.sobrescribir:
            return JSONResponse(status_code=409, content={"detalle": f"Ya existe el pronóstico de {month}.", "existe": True, "mes": month})
        return {"trabajo": jobs.start("pronostico", lambda progress: forecast_service.generate(paths, store, progress)), "mes": month}

    @app.get("/api/trabajos/activo")
    def active_job():
        return {"trabajo": jobs.active()}

    @app.get("/api/trabajos/{job_id}")
    def job(job_id: str):
        state = jobs.get(job_id)
        if state is None:
            raise LookupError("No existe esa tarea.")
        return state

    # ---- tracking -----------------------------------------------------------------------------------------------------------

    @app.get("/api/seguimiento")
    def tracking():
        try:
            return tracking_service.history(paths, store if paths.consumption_csv.exists() else None)
        except DataMissing:
            return tracking_service.history(paths, None)

    @app.get("/api/seguimiento/detalle")
    def tracking_detail(mes: str | None = None, origen: Literal["produccion", "backtest"] | None = None):
        return tracking_service.detail(paths, store if paths.consumption_csv.exists() else None, mes, origen)

    @app.post("/api/seguimiento/medir")
    def measure(body: MeasureBody):
        return tracking_service.measure(paths, store, body.mes)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")  # last: the API routes above win
    return app
