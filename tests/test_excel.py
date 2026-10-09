"""Excel export of the comparison: two tables, "faltante" for the internal model, a plain-words reason for every excluded row."""

import numpy as np
import openpyxl
import pandas as pd

from src.utils.excel_utils import COMPARISON_COLUMNS, EXCLUDED_COLUMNS, HELPER_FORMULAS, SUMMARY_COLUMNS, add_error_columns, comparison_sheet, excluded_sheet, summary_sheet, write_workbook
from src.utils.planner_utils import LOW_HISTORY, MISSING_INTERNAL, SUBGRADE


def test_comparison_sheet_has_only_the_keys_and_the_four_values_and_marks_the_missing_internal():
    table = pd.DataFrame(
        {"mes": "2026-09", "planta": ["B", "A"], "sku": ["K/01/110gsm/2100mm", "K/01/125gsm/2100mm"], "strategy": "VMI", "real": [10.04, 20.0], "planner": [11.0, 19.0], "internal": [np.nan, 18.26], "ml_mean": [9.0, 21.0], "history": [30, 30]}
    )
    out = comparison_sheet(table)
    assert list(out.columns) == COMPARISON_COLUMNS
    assert out["Planta"].tolist() == ["A", "B"]  # sorted by planta
    assert out["Modelo interno (TO)"].tolist() == [18.26, "faltante"]  # three decimals are kept, the cells show one
    assert out["Real (TO)"].tolist() == [20.0, 10.04]


def test_every_excluded_row_gets_a_reason_and_an_explanation():
    excluded = pd.DataFrame(
        {
            "planta": "A", "mes": "2026-09", "sku": ["S1", "S2", "S3", "S4"], "strategy": "VMI", "planner": [1.0, 2.0, 3.0, 4.0],
            "reason": [SUBGRADE, LOW_HISTORY, MISSING_INTERNAL, MISSING_INTERNAL],
            "detail": ["en los datos: KS/257/215gsm/1800mm", "2 meses antes del origen", "0 en todos los meses", "no está en el archivo del modelo interno"],
        }
    )
    out = excluded_sheet(excluded)
    assert list(out.columns) == EXCLUDED_COLUMNS
    by = out.set_index("SKU")
    assert by.loc["S1", "Motivo"] == "Subgrado distinto" and "KS/257/215gsm/1800mm" in by.loc["S1", "Explicación"]
    assert "solo 2 meses de consumo" in by.loc["S2", "Explicación"]
    assert by.loc["S3", "Motivo"] == by.loc["S4", "Motivo"] == "Modelo interno sin pronóstico"
    assert "0 en todos los meses" in by.loc["S3", "Explicación"] and "no está en el archivo" in by.loc["S4", "Explicación"]
    assert (out["Explicación"].str.len() > 20).all()


def test_workbook_has_two_sheets_each_with_an_excel_table(tmp_path):
    sheets = {"Comparación": pd.DataFrame({"SKU": ["a"], "Real (TO)": [1.5], "Modelo interno (TO)": ["faltante"]}), "Excluidos": pd.DataFrame({"SKU": ["b"], "Motivo": ["x"]})}
    path = tmp_path / "c.xlsx"
    write_workbook(path, sheets)
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["Comparación", "Excluidos"]
    assert all(len(ws.tables) == 1 for ws in wb)
    assert wb["Comparación"]["C2"].value == "faltante" and wb["Comparación"]["B2"].value == 1.5


def test_error_columns_are_formulas_that_survive_a_missing_internal_forecast():
    frame = pd.DataFrame({c: [1.0, 2.0] for c in COMPARISON_COLUMNS})
    out = add_error_columns(frame)
    assert list(out.columns) == COMPARISON_COLUMNS + list(HELPER_FORMULAS)
    assert out.loc[0, "Tiene interno"] == "=ISNUMBER(F2)" and out.loc[1, "Tiene interno"] == "=ISNUMBER(F3)"
    assert out.loc[0, "Error planificador (TO)"] == "=E2-D2"
    # the internal columns return "" instead of an error when the forecast is "faltante", and no percent divides by a real of 0
    assert 'ISNUMBER(F2)' in out.loc[0, "Error modelo interno (TO)"] and out.loc[0, "Error modelo interno (TO)"].endswith(',"")')
    assert all("D2=0" in out.loc[0, c] for c in ("Error % planificador", "Error % modelo interno", "Error % nuestro modelo"))


def test_summary_sheet_has_one_row_per_planta_plus_all_and_uses_only_rows_with_an_internal_forecast():
    out = summary_sheet(["A", "B"], 100)
    assert list(out.columns) == SUMMARY_COLUMNS and out["Planta"].tolist() == ["Todas", "A", "B"]
    assert all("$H$2:$H$101,TRUE" in out.loc[i, "SKU en las tres fuentes"] for i in range(3))  # the "Tiene interno" column filters every row
    assert "$B$2:$B$101,$A3" in out.loc[1, "Real (TO)"] and "$B$2:$B$101" not in out.loc[0, "Real (TO)"]
    assert out.loc[1, "Acierto planificador"] == "=1-G3" and out.loc[1, "Sesgo nuestro modelo"] == "=F3/C3-1"
