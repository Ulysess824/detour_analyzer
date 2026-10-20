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
    def adjustments_json(self) -> Path:
        return self.results / "ajustes.json"


def default_paths() -> AppPaths:
    return AppPaths(Path(__file__).resolve().parents[2])
