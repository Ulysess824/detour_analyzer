"""Descriptive tables of the consumption and the reading of the internal-model forecast."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.eda_utils import eda_by_planta, eda_by_planta_sku
from src.evaluation.internal_utils import MONTH_COLUMNS, load_internal_forecast, product_to_sku


@pytest.fixture()
def table():
    rows = [
        ("A", "s1", "2024-01-01", 2.0), ("A", "s1", "2024-01-03", 0.0), ("A", "s1", "2024-01-05", 4.0),
        ("A", "s2", "2024-01-02", 10.0), ("A", "s2", "2024-01-04", 10.0),
        ("B", "s3", "2024-01-09", 7.0),
    ]
    return pd.DataFrame(rows, columns=["planta", "sku", "fecha", "consumo"]).assign(fecha=lambda d: pd.to_datetime(d["fecha"]))


def test_series_table_matches_a_hand_calculation(table):
    out = eda_by_planta_sku(table).set_index(["planta", "sku"])
    s1 = out.loc[("A", "s1")]
    assert (s1["n_obs"], s1["span_days"], s1["mean"], s1["std"], s1["min"], s1["max"]) == (3, 5, 2.0, 2.0, 0.0, 4.0)
    assert s1["pct_zero"] == 33.3 and s1["coverage_pct"] == 60.0 and s1["cv"] == 1.0
    assert str(s1["fecha_min"].date()) == "2024-01-01" and str(s1["fecha_max"].date()) == "2024-01-05"
    s2 = out.loc[("A", "s2")]
    assert (s2["span_days"], s2["std"], s2["cv"], s2["pct_zero"]) == (3, 0.0, 0.0, 0.0) and s2["coverage_pct"] == 66.7
    single = out.loc[("B", "s3")]
    assert single["n_obs"] == 1 and single["span_days"] == 1 and np.isnan(single["std"]) and np.isnan(single["cv"])
    assert list(eda_by_planta_sku(table)[["planta", "sku"]].itertuples(index=False, name=None)) == [("A", "s1"), ("A", "s2"), ("B", "s3")]


def test_planta_table_adds_up_the_series(table):
    out = eda_by_planta(table).set_index("planta")
    assert list(eda_by_planta(table)["planta"]) == ["A", "B"]  # sorted by total consumption, largest first
    a = out.loc["A"]
    assert (a["n_sku"], a["n_obs"], a["consumo_total"], a["consumo_mean"], a["consumo_min"], a["consumo_max"]) == (2, 5, 26.0, 5.2, 0.0, 10.0)
    assert a["pct_zero"] == 20.0 and a["obs_por_sku"] == 2.5
    series = eda_by_planta_sku(table)
    assert out["n_obs"].sum() == series["n_obs"].sum() == len(table)
    assert out["n_sku"].to_dict() == series.groupby("planta").size().to_dict()


def write_planner(path, rows):
    pd.DataFrame(rows, columns=["planta", "product_sap", "sku_planner", "forecast_to"]).to_csv(path, index=False)


def test_product_code_maps_to_the_sku_and_the_first_file_wins(tmp_path):
    first, second = tmp_path / "p1.csv", tmp_path / "p2.csv"
    write_planner(first, [("SCAN", 100, "HP2/01/110gsm/1990mm", 5.0), ("SCAN", 101, "HP2/01/120gsm/1990mm", 6.0)])
    write_planner(second, [("SCAN", 101, "OTHER", 1.0), ("SCAN", 102, "K/01/200gsm/1800mm", 2.0)])
    assert product_to_sku([first, second]) == {100: "HP2/01/110gsm/1990mm", 101: "HP2/01/120gsm/1990mm", 102: "K/01/200gsm/1800mm"}


def test_internal_forecast_in_long_layout_and_unmatched_rows_apart(tmp_path):
    planner = tmp_path / "planner.csv"
    write_planner(planner, [("SCAN", 100, "HP2/01/110gsm/1990mm", 5.0), ("SCAN", 101, "HP2/01/120gsm/1990mm", 6.0)])
    internal = pd.DataFrame({"planta": ["SCAN"] * 3, "product_sap": [100, 101, 999], **{m: [float(i + 1) for i in range(3)] for m in MONTH_COLUMNS}})
    internal.loc[0, "2026-07"] = 55.0
    path = tmp_path / "internal.csv"
    internal.to_csv(path, index=False)
    long, unmatched = load_internal_forecast(path, [planner])
    assert list(long.columns) == ["planta", "mes", "sku_planner", "product_sap", "strategy", "forecast_to"]
    assert len(long) == 2 * len(MONTH_COLUMNS) and set(long["mes"]) == set(MONTH_COLUMNS) and (long["strategy"] == "").all()
    assert long.loc[(long["product_sap"] == 100) & (long["mes"] == "2026-07"), "forecast_to"].item() == 55.0
    assert set(long["sku_planner"]) == {"HP2/01/110gsm/1990mm", "HP2/01/120gsm/1990mm"}
    assert unmatched["product_sap"].tolist() == [999]
