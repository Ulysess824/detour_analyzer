"""The planner / internal model / ml_mean comparison as a simple Excel workbook: one table of predictions and one of excluded rows."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from src.utils.planner_utils import LOW_HISTORY, MISSING_INTERNAL, NO_MODEL, NO_SERIES, SUBGRADE

MISSING_TEXT = "faltante"
COMPARISON_COLUMNS = ["SKU", "Planta", "Mes", "Real (TO)", "Planificador (TO)", "Modelo interno (TO)", "Nuestro modelo ml_mean (TO)"]
EXCLUDED_COLUMNS = ["SKU", "Planta", "Mes", "Planificador (TO)", "Motivo", "Explicación"]

REASON_TEXT = {
    SUBGRADE: ("Subgrado distinto", "El nombre del SKU no coincide con los datos de consumo porque cambia el subgrado ({detail}). No se cruzó: pendiente de confirmar con el planificador."),
    LOW_HISTORY: ("Menos de 3 meses de historia", "El SKU tiene solo {detail} de consumo antes del mes pronosticado y el modelo necesita al menos 3."),
    MISSING_INTERNAL: ("Modelo interno sin pronóstico", "{detail} Se muestra como faltante en la hoja de comparación y no cuenta en las métricas."),
    NO_SERIES: ("SKU sin consumo en los datos", "El SKU nunca tuvo consumo en los datos, no hay serie que pronosticar."),
    NO_MODEL: ("Nuestro modelo sin pronóstico", "El SKU existe pero no estaba activo el mes anterior, así que el modelo no lo pronosticó."),
}


DETAIL_TEXT = {
    "0 en todos los meses": "El modelo interno tiene 0 en todos los meses para este SKU; se toma como sin pronóstico.",
    "no está en el archivo del modelo interno": "El SKU no está en el archivo del modelo interno.",
}


def comparison_sheet(table: pd.DataFrame) -> pd.DataFrame:
    r"""Planta, month, SKU, the real and the three forecasts; the internal model is "faltante" where it has no forecast."""
    out = pd.DataFrame(
        {
            "SKU": table["sku"], "Planta": table["planta"], "Mes": table["mes"], "Real (TO)": table["real"].round(1), "Planificador (TO)": table["planner"].round(1),
            "Modelo interno (TO)": table["internal"].round(1).astype(object).where(table["internal"].notna(), MISSING_TEXT),
            "Nuestro modelo ml_mean (TO)": table["ml_mean"].round(1),
        }
    )
    return out[COMPARISON_COLUMNS].sort_values(["Planta", "SKU"], ignore_index=True)


def excluded_sheet(excluded: pd.DataFrame) -> pd.DataFrame:
    r"""Every left-out row with the reason in plain words and an explanation."""
    rows = []
    for _, r in excluded.iterrows():
        reason, template = REASON_TEXT.get(r["reason"], (r["reason"], "{detail}"))
        detail = "" if pd.isna(r.get("detail")) else str(r["detail"])
        detail = DETAIL_TEXT.get(detail, detail).replace(" antes del origen", "")
        rows.append({"SKU": r["sku"], "Planta": r["planta"], "Mes": r["mes"], "Planificador (TO)": round(float(r["planner"]), 1), "Motivo": reason, "Explicación": template.format(detail=detail).replace(" ()", "")})
    return pd.DataFrame(rows, columns=EXCLUDED_COLUMNS).sort_values(["Motivo", "Planta", "SKU"], ignore_index=True)


def write_workbook(path: str | Path, sheets: dict[str, pd.DataFrame]) -> None:
    r"""One sheet per entry, each an Excel table with a filter, a frozen header, tidy widths and numbers with one decimal."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, frame in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(list(frame.columns))
        for row in frame.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and pd.isna(v)) else v for v in row])
        last = f"{get_column_letter(len(frame.columns))}{len(frame) + 1}"
        table = Table(displayName="T_" + "".join(c for c in name if c.isalnum()), ref=f"A1:{last}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        ws.add_table(table)
        ws.freeze_panes = "A2"
        for j, col in enumerate(frame.columns, start=1):
            letter = get_column_letter(j)
            width = max([len(str(col))] + [len(str(v)) for v in frame[col].head(500)]) + 2
            ws.column_dimensions[letter].width = min(max(width, 10), 70)
            ws[f"{letter}1"].alignment = Alignment(wrap_text=True, vertical="center")
            for cell in ws[letter][1:]:
                if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                    cell.number_format = "#,##0.0"
                elif cell.value == MISSING_TEXT:
                    cell.font = Font(italic=True, color="808080")
                    cell.alignment = Alignment(horizontal="right")
        ws.row_dimensions[1].height = 32
    wb.save(path)
