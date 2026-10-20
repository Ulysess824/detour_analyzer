"""Where the app reads and writes: one root folder (the repo, or the folder that holds the installed app)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STATIC = Path(__file__).resolve().parent / "static"  # the screens (html, css, js, ECharts, logo) served by the API


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def consumption_csv(self) -> Path:
        return self.root / "data" / "consumos_long.csv"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def tracking_csv(self) -> Path:
        return self.results / "seguimiento.csv"

    @property
    def tuning_json(self) -> Path:
        return self.results / "tuning_trees.json"

    @property
    def production_tuning(self) -> Path:
        return self.results / "tuning_produccion.json"

    @property
    def tuning_candidate(self) -> Path:
        return self.results / "tuning_candidato.json"

    @property
    def tuning_archive(self) -> Path:
        return self.results / "ajustes_anteriores"

    @property
    def security_json(self) -> Path:
        return self.root / "config" / "seguridad.json"

    def tuning_for_forecast(self) -> Path:
        r"""The tuned parameters the app forecasts with: the production file once a retune was adopted, else the research file."""
        return self.production_tuning if self.production_tuning.exists() else self.tuning_json

    @property
    def database(self) -> Path:
        return self.root / "data" / "detour.db"

    def forecast_csv(self, month: str) -> Path:
        return self.results / f"forecast_{month}.csv"

    @property
    def adjustments_json(self) -> Path:
        return self.results / "ajustes.json"


def default_paths() -> AppPaths:
    return AppPaths(Path(__file__).resolve().parents[2])
