"""Planner workbooks and the comparison over several plantas: reading, exclusion rules and the planta in every join."""

import openpyxl
import pandas as pd
import pytest

from src.utils.planner_io_utils import OUTPUT_COLUMNS, parse_month, read_assignment
from src.utils.planner_utils import LOW_HISTORY, NO_MODEL, NO_SERIES, SUBGRADE, attach_planner, compare_month_audit, dataset_key, first_consumption_dates

HEADER = ["Customer", "Grade", "Sub-grade", "Grammage", "Width", "Product", "Strategy", "Plant Forecast", "SEPT´26 allocation"]


def write_workbook(path, rows, header=HEADER):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(header)
    for row in rows:
        ws.append(row)
    wb.save(path)


@pytest.mark.parametrize(
    "header, month",
    [("JUN´26 allocation", "2026-06"), ("JUL´26 allocation", "2026-07"), ("AGO´26 allocation", "2026-08"), ("SEPT´26 allocation", "2026-09"), ("dic'27 allocation", "2027-12")],
)
def test_month_is_read_from_the_header(header, month):
    assert parse_month(header) == month


def test_unknown_month_header_is_rejected():
    with pytest.raises(ValueError):
        parse_month("XYZ´26 allocation")


def test_workbook_becomes_the_planner_csv_layout(tmp_path):
    path = tmp_path / "a.xlsx"
    write_workbook(path, [["SCAN", "K", 1, 110, 2100, 111, "VMI", 68, "CDP"], ["PCEL", "KCL", 134, 145, 2000, 222, None, 25.5, "Saica"], [None, None, None, None, None, None, None, None, None]])
    out = read_assignment(path)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out["sku_planner"].tolist() == ["K/01/110gsm/2100mm", "KCL/134/145gsm/2000mm"]
    assert out["mes"].unique().tolist() == ["2026-09"]
    assert out["strategy"].tolist() == ["VMI", ""]  # an empty strategy stays empty, it is not "nan"
    assert out["forecast_to"].tolist() == [68.0, 25.5]


def test_workbook_with_a_repeated_sku_or_a_negative_forecast_is_rejected(tmp_path):
    repeated = tmp_path / "r.xlsx"
    write_workbook(repeated, [["SCAN", "K", 1, 110, 2100, 111, "VMI", 68, "CDP"], ["SCAN", "K", 1, 110, 2100, 112, "VMI", 10, "CDP"]])
    with pytest.raises(ValueError, match="repeat"):
        read_assignment(repeated)
    negative = tmp_path / "n.xlsx"
    write_workbook(negative, [["SCAN", "K", 1, 110, 2100, 111, "VMI", -1, "CDP"]])
    with pytest.raises(ValueError, match="negative"):
        read_assignment(negative)


@pytest.fixture
def world():
    r"""Series 0 (A, old), 1 (A, started at month 4), 2 (A, started at month 8), 3 (B, same SKU as series 0), 4 (A, KS/257)."""
    skus = ["K/01/110gsm/2100mm/1200-1450", "K/01/160gsm/2100mm/1200-1450", "K/01/125gsm/2100mm/1200-1450", "K/01/110gsm/2100mm/1200-1450", "KS/257/215gsm/1800mm/1200-1450"]
    pairs = pd.DataFrame({"planta": ["A", "A", "A", "B", "A"], "sku": skus, "series": range(5), "first_month": [0, 4, 8, 0, 0]})
    pairs["key"] = pairs["sku"].map(dataset_key)
    # month 9 (origin 8): the saved forecasts have no row for series 2, as if it had not been forecast
    res = pd.DataFrame({"series": [0, 1, 3, 4], "t": 9, "y": [10.0, 20.0, 7.0, 5.0], "p_ml": [9.0, 18.0, 8.0, 4.0]})
    return pairs, res


def planner_rows(*rows):
    return pd.DataFrame(rows, columns=["planta", "mes", "sku_planner", "strategy", "forecast_to"])


def test_exclusion_reasons_and_kept_rows(world):
    pairs, res = world
    planner = planner_rows(
        ("A", "2026-09", "K/01/110gsm/2100mm", "VMI", 12.0),  # kept: 9 months of history
        ("A", "2026-09", "K/01/160gsm/2100mm", "VMI", 22.0),  # kept: first month 4, origin 8 -> 5 months
        ("A", "2026-09", "K/01/125gsm/2100mm", "VMI", 3.0),  # first month 8 = origin -> 1 month of history
        ("A", "2026-09", "K/01/999gsm/2100mm", "VMI", 3.0),  # never consumed
        ("A", "2026-09", "KS/01/215gsm/1800mm", "VMI", 6.0),  # exists as KS/257
    )
    table, excluded = compare_month_audit(planner, res, pairs, 9, ["ml"], min_history=3)
    assert table["sku"].tolist() == ["K/01/110gsm/2100mm", "K/01/160gsm/2100mm"]
    assert table["history"].tolist() == [9, 5]
    reasons = dict(zip(excluded["sku"], excluded["reason"]))
    assert reasons == {"K/01/125gsm/2100mm": LOW_HISTORY, "K/01/999gsm/2100mm": NO_SERIES, "KS/01/215gsm/1800mm": SUBGRADE}
    assert "KS/257/215gsm/1800mm" in excluded.set_index("sku").loc["KS/01/215gsm/1800mm", "detail"]


def test_a_sku_the_models_cannot_forecast_is_excluded_not_counted_as_zero(world):
    pairs, res = world
    planner = planner_rows(("A", "2026-09", "K/01/125gsm/2100mm", "VMI", 3.0))  # series 2 has no model row
    table, excluded = compare_month_audit(planner, res, pairs, 9, ["ml"], min_history=0)
    assert table.empty
    assert excluded["reason"].tolist() == [NO_MODEL]


def test_the_same_sku_in_two_plantas_is_not_mixed(world):
    pairs, res = world
    planner = planner_rows(("A", "2026-09", "K/01/110gsm/2100mm", "VMI", 12.0), ("B", "2026-09", "K/01/110gsm/2100mm", "VMI", 6.0))
    table, _ = compare_month_audit(planner, res, pairs, 9, ["ml"])
    by_planta = table.set_index("planta")
    assert (by_planta.loc["A", "real"], by_planta.loc["A", "ml"], by_planta.loc["A", "planner"]) == (10.0, 9.0, 12.0)
    assert (by_planta.loc["B", "real"], by_planta.loc["B", "ml"], by_planta.loc["B", "planner"]) == (7.0, 8.0, 6.0)


def test_strategies_are_a_filter_not_a_code_path(world):
    pairs, res = world
    planner = planner_rows(("A", "2026-09", "K/01/110gsm/2100mm", "VMI", 12.0), ("A", "2026-09", "K/01/160gsm/2100mm", "NO VMI", 22.0))
    assert compare_month_audit(planner, res, pairs, 9, ["ml"], strategies=["VMI"])[0]["sku"].tolist() == ["K/01/110gsm/2100mm"]
    assert len(compare_month_audit(planner, res, pairs, 9, ["ml"], strategies=["VMI", "NO VMI"])[0]) == 2
    assert len(compare_month_audit(planner, res, pairs, 9, ["ml"])[0]) == 2  # None keeps every strategy


def test_planner_forecast_is_attached_by_planta_month_and_sku():
    table = pd.DataFrame({"planta": ["A", "B"], "mes": ["2026-09", "2026-09"], "sku": ["K/01/110gsm/2100mm"] * 2, "internal": [1.0, 2.0]})
    planner = planner_rows(("A", "2026-09", "K/01/110gsm/2100mm", "VMI", 12.0), ("B", "2026-09", "K/01/110gsm/2100mm", "VMI", 6.0), ("A", "2026-08", "K/01/110gsm/2100mm", "VMI", 99.0))
    joined = attach_planner(table, planner)
    assert len(joined) == 2  # no duplicated rows when a SKU exists in several plantas or months
    assert joined["planner"].tolist() == [12.0, 6.0]


def test_first_consumption_date_is_the_first_day_over_all_series_with_the_key():
    df = pd.DataFrame(
        {
            "planta": ["A", "A", "A", "B"],
            "sku": ["K/01/110gsm/2100mm/1200-1450", "K/01/110gsm/2100mm/1200-1250", "K/01/110gsm/2100mm/1200-1450", "K/01/110gsm/2100mm/1200-1450"],
            "fecha": pd.to_datetime(["2026-07-10", "2026-07-03", "2026-08-01", "2026-05-02"]),
            "consumo": 1.0,
        }
    )
    first = first_consumption_dates(df)
    assert first[("A", "k/01/110gsm/2100mm")] == pd.Timestamp("2026-07-03")
    assert first[("B", "k/01/110gsm/2100mm")] == pd.Timestamp("2026-05-02")
