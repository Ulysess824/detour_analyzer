"""Where the app reads and writes: one root folder (the repo, or the folder that holds the installed app)."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


STATIC = Path(__file__).resolve().parent / "static"  # the screens (html, css, js, ECharts, logo) served by the API


FORECAST_NAME = re.compile(r"^forecast_(\d{4}-\d{2})(?:_([a-z_]+))?$")


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
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def staging(self) -> Path:
        return self.root / "data" / "staging"

    @property
    def backups(self) -> Path:
        return self.root / "data" / "respaldo"

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

    def forecast_csv(self, month: str, strategy: str = "h1") -> Path:
        r"""`forecast_<month>.csv` for the one-month forecast, `forecast_<month>_<strategy>.csv` for the others (same names as the script)."""
        return self.results / (f"forecast_{month}.csv" if strategy == "h1" else f"forecast_{month}_{strategy}.csv")

    def forecast_files(self) -> list[tuple[str, str, Path]]:
        r"""Saved forecasts as (target month, strategy, csv), newest month first and the one-month forecast first within a month."""
        found = []
        for csv in self.results.glob("forecast_*.csv"):
            match = FORECAST_NAME.match(csv.stem)
            if match:
                found.append((match.group(1), match.group(2) or "h1", csv))
        return sorted(found, key=lambda f: (f[0], f[1] == "h1", f[1]), reverse=True)

    @property
    def settings_json(self) -> Path:
        return self.root / "config" / "ajustes_app.json"

    @property
    def adjustments_json(self) -> Path:
        return self.results / "ajustes.json"


def default_paths() -> AppPaths:
    r"""The folder with data, results and config: `DETOUR_HOME` if set, the folder of the .exe when the app is packaged, else the repository."""
    if os.environ.get("DETOUR_HOME"):
        return AppPaths(Path(os.environ["DETOUR_HOME"]).resolve())
    if getattr(sys, "frozen", False):
        return AppPaths(Path(sys.executable).resolve().parent)
    return AppPaths(Path(__file__).resolve().parents[2])
