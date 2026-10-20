"""Real consumption for the app: filters by planta and SKU, monthly and daily series, summary and breakdown. Reads the master csv once."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.app.paths import AppPaths
from src.app.search import resolve, suggest
from src.engine.data_checks import incomplete_last_month_message
from src.engine.panel_utils import Panel, build_panel, load_consumption

TOP_SERIES = 7  # the chart palette has 8 colours: seven series and one 'Otros'
DEFAULT_DAYS = 90


class NotFound(LookupError):
    r"""The typed planta or SKU does not exist; `options` are the closest names."""

    def __init__(self, kind: str, text: str, options: list[str]):
        super().__init__(f"No existe {kind} '{text}'.")
        self.kind, self.text, self.options = kind, text, options


class DataMissing(RuntimeError):
    r"""The consumption file is not there yet."""


def _values(array: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else round(float(v), 3) for v in array]


class ConsumptionStore:
    r"""The panel of `data/consumos_long.csv`, rebuilt only when the file changes."""

    def __init__(self, paths: AppPaths):
        self.paths = paths
        self._stamp: tuple[float, int] | None = None
        self._panel: Panel | None = None
        self._rows = 0
        self._first_day = ""

    @property
    def panel(self) -> Panel:
        path = self.paths.consumption_csv
        if not path.exists():
            raise DataMissing("Todavía no hay datos de consumo. Carga los exportes de SAP en la sección Datos.")
        stat = path.stat()
        stamp = (stat.st_mtime, stat.st_size)
        if stamp != self._stamp:
            df = load_consumption(path, validate=False)  # showing data is not forecasting: the Datos section validates what is loaded
            self._panel, self._rows, self._stamp = build_panel(df), len(df), stamp
            self._first_day = str(df["fecha"].min().date())
        return self._panel

    # ---- options --------------------------------------------------------------------------------------------------------------

    def plantas(self) -> list[str]:
        return list(self.panel.plantas)

    def skus(self, planta: str | None = None) -> list[str]:
        keys = self.panel.keys
        return sorted({s for p, s in keys if planta is None or p == planta})

    def suggestions(self, field: str, text: str, planta: str | None = None, limit: int = 8) -> list[str]:
        options = self.plantas() if field == "planta" else self.skus(planta)
        return suggest(text, options, limit)

    # ---- selection ------------------------------------------------------------------------------------------------------------

    def _name(self, kind: str, text: str | None, options: list[str]) -> str | None:
        if text is None or not text.strip():
            return None
        found = resolve(text, options)
        if found is None:
            raise NotFound(kind, text, suggest(text, options, 5))
        return found

    def select(self, planta: str | None, sku: str | None) -> tuple[str | None, str | None, np.ndarray]:
        r"""The exact planta and SKU names and the indices of the series they select (all when both are empty)."""
        planta = self._name("la planta", planta, self.plantas())
        sku = self._name("el SKU", sku, self.skus(planta))
        keys = self.panel.keys
        index = np.array([i for i, (p, s) in enumerate(keys) if (planta is None or p == planta) and (sku is None or s == sku)], dtype=int)
        return planta, sku, index

    # ---- series ---------------------------------------------------------------------------------------------------------------

    def _monthly(self, index: np.ndarray) -> np.ndarray:
        S = self.panel
        block = S.monthly[index]
        if len(index) == 1:
            return block[0]  # one series: empty (null) before its first month
        return np.nansum(block, axis=0)

    def _breakdown(self, planta: str | None, sku: str | None, index: np.ndarray) -> dict | None:
        S = self.panel
        if sku is not None:
            return None
        if planta is None:
            names = [(name, np.nansum(S.monthly[S.planta_code == c], axis=0)) for c, name in enumerate(S.plantas)]
            by = "planta"
        else:
            names = [(S.keys[i][1], np.nan_to_num(S.monthly[i])) for i in index]
            by = "sku"
        names.sort(key=lambda pair: -pair[1].sum())
        top, rest = names[:TOP_SERIES], names[TOP_SERIES:]
        series = [{"nombre": n, "valores": _values(v)} for n, v in top]
        if rest:
            series.append({"nombre": "Otros", "valores": _values(np.sum([v for _, v in rest], axis=0))})
        return {"por": by, "series": series}

    def monthly(self, planta: str | None = None, sku: str | None = None, breakdown: bool = True) -> dict:
        r"""Monthly consumption (TO) of the selection, its summary and, when no SKU is chosen, the breakdown by planta or by SKU."""
        planta, sku, index = self.select(planta, sku)
        if not len(index):
            raise NotFound("una combinación de planta y SKU en", f"{planta or 'todas'} / {sku}", [])
        S = self.panel
        values = self._monthly(index)
        last = pd.Timestamp(S.dates[-1])
        partial = incomplete_last_month_message(pd.DataFrame({"fecha": [last]})) is not None
        # the summary is about the months with data: a SKU that appeared later starts at its first month
        seen = ~np.isnan(values)
        complete = seen.copy()
        if partial:
            complete[-1] = False
        done = values[complete]
        last_done = int(np.where(complete)[0][-1]) if complete.any() else None
        previous = values[last_done - 1] if last_done is not None and last_done > 0 else np.nan
        change = None
        if last_done is not None and previous and not np.isnan(previous) and previous > 0:
            change = round(float((values[last_done] / previous - 1) * 100), 1)
        summary = {
            "n_series": int(len(index)),
            "total_TO": round(float(np.nansum(values)), 1),
            "promedio_mensual_TO": round(float(done.mean()), 1) if len(done) else None,
            "ultimo_mes_completo": S.labels[last_done] if last_done is not None else None,
            "ultimo_mes_TO": round(float(values[last_done]), 1) if last_done is not None else None,
            "cambio_vs_mes_anterior_pct": change,
            "primer_mes": S.labels[int(np.argmax(seen))] if seen.any() else None,
            "ultimo_dato": str(last.date()),
            "mes_parcial": S.labels[-1] if partial else None,
        }
        return {
            "filtro": {"planta": planta, "sku": sku},
            "nivel": "mensual",
            "etiquetas": list(S.labels),
            "valores": _values(values),
            "resumen": summary,
            "desglose": self._breakdown(planta, sku, index) if breakdown else None,
        }

    def daily(self, planta: str | None = None, sku: str | None = None, since: str | None = None, until: str | None = None) -> dict:
        r"""Daily consumption (TO) of the selection between two dates (the last 90 days by default)."""
        planta, sku, index = self.select(planta, sku)
        if not len(index):
            raise NotFound("una combinación de planta y SKU en", f"{planta or 'todas'} / {sku}", [])
        S = self.panel
        end = pd.Timestamp(until) if until else pd.Timestamp(S.dates[-1])
        start = pd.Timestamp(since) if since else end - pd.Timedelta(days=DEFAULT_DAYS - 1)
        if start > end:
            raise ValueError("La fecha inicial es posterior a la final.")
        keep = (S.dates >= start) & (S.dates <= end)
        days = S.daily[index][:, keep].sum(axis=0, dtype=float)
        total = float(days.sum())
        return {
            "filtro": {"planta": planta, "sku": sku},
            "nivel": "diario",
            "etiquetas": [str(d.date()) for d in S.dates[keep]],
            "valores": _values(days),
            "resumen": {
                "n_series": int(len(index)),
                "total_TO": round(total, 1),
                "dias_con_consumo": int((days > 0).sum()),
                "desde": str(start.date()),
                "hasta": str(end.date()),
            },
        }

    def info(self) -> dict:
        S = self.panel
        return {
            "filas": self._rows,
            "series": len(S.keys),
            "plantas": len(S.plantas),
            "primer_dia": self._first_day,
            "ultimo_dia": str(S.dates[-1].date()),
            "meses": list(S.labels),
            "mes_parcial": S.labels[-1] if incomplete_last_month_message(pd.DataFrame({"fecha": [S.dates[-1]]})) else None,
        }
