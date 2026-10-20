"""Long tasks (forecast, retune) run in one background thread so the screen can show progress; one task at a time."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Callable


class Busy(RuntimeError):
    r"""Another task is still running."""


class JobRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._active: str | None = None

    def start(self, kind: str, work: Callable[[Callable[[str, float], None]], dict]) -> str:
        r"""Run `work(progress)` in a thread; `progress(message, fraction)` feeds the state. The dict `work` returns is kept as the result."""
        with self._lock:
            if self._active is not None and self._jobs[self._active]["estado"] == "corriendo":
                raise Busy("Ya hay una tarea en marcha. Espera a que termine.")
            job_id = uuid.uuid4().hex[:8]
            self._jobs[job_id] = {"id": job_id, "tipo": kind, "estado": "corriendo", "mensaje": "Empezando", "progreso": 0.0, "inicio": time.time(), "fin": None,
                                  "resultado": None, "error": None}
            self._active = job_id
        threading.Thread(target=self._run, args=(job_id, work), daemon=True).start()
        return job_id

    def _run(self, job_id: str, work: Callable) -> None:
        job = self._jobs[job_id]

        def progress(message: str, fraction: float) -> None:
            job["mensaje"], job["progreso"] = message, float(fraction)

        try:
            job["resultado"] = work(progress)
            job["estado"], job["progreso"] = "listo", 1.0
        except (Exception, SystemExit) as error:  # the checks of the engine stop with SystemExit and a message
            job["estado"], job["error"] = "error", str(error) or error.__class__.__name__
        finally:
            job["fin"] = time.time()

    def active(self) -> dict | None:
        r"""The task that is still running, or None."""
        job = self.get(self._active) if self._active else None
        return job if job and job["estado"] == "corriendo" else None

    def get(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        end = job["fin"] or time.time()
        return {**job, "segundos": round(end - job["inicio"], 1)}
