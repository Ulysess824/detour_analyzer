"""The example of the raw export shown in Datos: the table on the screen, and that the loader really reads a file laid out like it."""

import datetime

import openpyxl
from fastapi.testclient import TestClient

from src.app.api import create_app
from src.app.example_service import HEADER, ROWS
from src.app.paths import AppPaths
from src.engine.transform_utils import read_export


def example_file(path, rows=ROWS):
    r"""An .xlsx laid out exactly like the example on the screen."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(HEADER)
    for row in rows:
        sheet.append(row)
    book.save(path)
    return path


def test_the_screen_gets_the_example_table(tmp_path):
    table = TestClient(create_app(AppPaths(tmp_path))).get("/api/datos/formato").json()
    assert [c["letra"] for c in table["columnas"]] == list("ABCDEF")
    assert [c["titulo"] for c in table["columnas"]] == ["Customer", "", "Material", "", "Day", "TO"]  # B and D have no title in a real export
    assert table["filas"] == [list(row) for row in ROWS]


def test_the_loader_reads_a_file_laid_out_like_the_example(tmp_path):
    records = read_export(example_file(tmp_path / "ejemplo.xlsx"), drop_zeros=True)
    assert len(records) == len(ROWS) and {r[0] for r in records} == {"DEMO1", "DEMO2"}
    assert records[0] == ("DEMO1", "K/01/125gsm/2000mm/1200-1400", "2026-03-02", 12.345)  # sku = the description, day as yyyy-mm-dd
    assert all(r[3] > 0 for r in records)


def test_only_a_day_written_as_text_dd_mm_yyyy_is_read(tmp_path):
    r"""What the screen warns about: an Excel date or another text format is dropped without any message."""
    rows = [("DEMO1", "CLIENTE", "900001", "K/01/125gsm/2000mm/1200-1400", day, 5.0) for day in ("17.02.2026", datetime.datetime(2026, 2, 18), "2026-02-19")]
    assert [r[2] for r in read_export(example_file(tmp_path / "days.xlsx", rows))] == ["2026-02-17"]
