"""Local HTTP API of the desktop app (FastAPI). It listens on 127.0.0.1 only and serves the screens of the app from the services."""

from __future__ import annotations

from datetime import date
from typing import Callable, Literal

from fastapi import Depends, FastAPI, Header, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.app.consumption_service import ConsumptionStore, DataMissing, NotFound
from src.app import __version__, data_service, export_service, forecast_service, history_service, tracking_service, tuning_service
from src.app.forecast_service import ForecastMissing
from src.app.jobs import Busy, JobRunner
from src.app.paths import STATIC, AppPaths, default_paths
from src.app.security import Locked, NoPassword, Sessions, Unauthorized, WrongPassword, load_record
from src.app.status_service import app_status, retune_status


class GenerateBody(BaseModel):
    sobrescribir: bool = False


class MeasureBody(BaseModel):
    mes: str


class UnlockBody(BaseModel):
    clave: str


class RetuneBody(BaseModel):
    tipo: str = "completo"


class ReviewBody(BaseModel):
    id: str
    politica: str = "detener"


class StagedBody(BaseModel):
    id: str


def create_app(paths: AppPaths | None = None, today: Callable[[], date] = date.today) -> FastAPI:
    paths = paths or default_paths()
    store = ConsumptionStore(paths)
    jobs = JobRunner()
    sessions = Sessions(lambda: load_record(paths.security_json))

    def unlocked(x_ajuste_token: str | None = Header(default=None)) -> str:
        sessions.check(x_ajuste_token)
        return x_ajuste_token
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

    @app.exception_handler(Unauthorized)
    def _locked_screen(_, error: Unauthorized):
        return JSONResponse(status_code=401, content={"detalle": str(error), "bloqueado": True})

    @app.exception_handler(WrongPassword)
    def _wrong(_, error: WrongPassword):
        return JSONResponse(status_code=401, content={"detalle": str(error), "intentos": error.remaining})

    @app.exception_handler(Locked)
    def _too_many(_, error: Locked):
        return JSONResponse(status_code=429, content={"detalle": str(error), "segundos": error.seconds})

    @app.exception_handler(NoPassword)
    def _no_password(_, error: NoPassword):
        return JSONResponse(status_code=503, content={"detalle": str(error)})

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

    # ---- hyperparameter retune ----------------------------------------------------------------------------------------------

    @app.get("/api/ajuste")
    def retune_state():
        try:
            partial = store.is_partial()
        except DataMissing:
            partial = False
        return {"estado": retune_status(paths, today()), "historial": tuning_service.history(paths), "candidato": tuning_service.candidate_summary(paths),
                "tipos": tuning_service.presets(), "mes_parcial": partial, "clave_configurada": sessions.configured(), "tarea": jobs.active()}

    @app.post("/api/ajuste/desbloquear")
    def unlock(body: UnlockBody):
        return {"token": sessions.login(body.clave)}

    @app.post("/api/ajuste/bloquear")
    def lock(x_ajuste_token: str | None = Header(default=None)):
        sessions.logout(x_ajuste_token)
        return {"ok": True}

    @app.post("/api/ajuste/reafinar")
    def retune(body: RetuneBody, _: str = Depends(unlocked)):
        return {"trabajo": jobs.start("ajuste", lambda progress: tuning_service.retune(paths, store, body.tipo, progress))}

    @app.post("/api/ajuste/adoptar")
    def adopt(_: str = Depends(unlocked)):
        tuning_service.adopt(paths, today())
        return {"estado": retune_status(paths, today()), "historial": tuning_service.history(paths)}

    @app.post("/api/ajuste/descartar")
    def discard(_: str = Depends(unlocked)):
        tuning_service.discard(paths)
        return {"ok": True}

    # ---- data loading -------------------------------------------------------------------------------------------------------

    @app.get("/api/datos")
    def data_state():
        try:
            current = store.info()
            current = {k: v for k, v in current.items() if k != "meses"}
        except DataMissing:
            current = None
        return {"actual": current, "respaldos": data_service.backups(paths)}

    @app.post("/api/datos/subir")
    async def upload(files: list[UploadFile]):
        return data_service.stage(paths, [(f.filename or "", await f.read()) for f in files])

    @app.post("/api/datos/revisar")
    def review_data(body: ReviewBody):
        return {"trabajo": jobs.start("datos", lambda progress: data_service.review(paths, body.id, body.politica, progress))}

    @app.post("/api/datos/aplicar")
    def apply_data(body: StagedBody):
        result = data_service.apply(paths, body.id)
        store.info()  # rebuilds the panel now, so a bad file shows up here and not in the next screen
        return result

    @app.post("/api/datos/cancelar")
    def cancel_data(body: StagedBody):
        data_service.cancel(paths, body.id)
        return {"ok": True}

    @app.post("/api/datos/deshacer")
    def undo_data():
        return data_service.undo(paths)

    # ---- history, downloads and about ---------------------------------------------------------------------------------------

    @app.get("/api/historial")
    def history():
        try:
            return {"pronosticos": history_service.forecast_history(paths, store if paths.consumption_csv.exists() else None)}
        except DataMissing:
            return {"pronosticos": history_service.forecast_history(paths, None)}

    def workbook(result: tuple[str, bytes]) -> Response:
        name, content = result
        return Response(content, media_type=export_service.XLSX, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/descargas/pronostico")
    def download_forecast(mes: str | None = None):
        return workbook(export_service.forecast_workbook(paths, store, mes))

    @app.get("/api/descargas/consumo")
    def download_consumption(planta: str | None = None, sku: str | None = None, nivel: Literal["mensual", "diario"] = "mensual", desde: str | None = None, hasta: str | None = None):
        return workbook(export_service.consumption_workbook(store, planta, sku, nivel, desde, hasta))

    @app.get("/api/descargas/seguimiento")
    def download_tracking(mes: str | None = None):
        return workbook(export_service.tracking_workbook(paths, store if paths.consumption_csv.exists() else None, mes))

    @app.get("/api/acerca")
    def about():
        return {"version": __version__, "carpeta": str(paths.root), "archivos": {"datos": str(paths.consumption_csv), "pronosticos": str(paths.results), "respaldos": str(paths.backups), "registro": str(paths.logs / "detour.log")}}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")  # last: the API routes above win
    return app
