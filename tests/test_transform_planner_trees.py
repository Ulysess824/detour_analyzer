"""Reading exports, matching the planner SKUs, and the tree models."""

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src.utils.feature_utils import make_frame
from src.utils.planner_utils import accuracy_table, compare_month, dataset_key, planner_key
from src.utils.transform_utils import combine_exports, read_export
from src.utils.tree_utils import TREE_BASE, fit_predict_trees


def write_long_export(path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Customer", None, "Material", None, "Day", "TO"])
    for planta, sku, day, value in rows:
        ws.append([planta, "client", "123", sku, day, value])
    wb.save(path)


def test_long_export_clips_negatives_and_can_drop_zeros(tmp_path):
    path = tmp_path / "e.xlsx"
    write_long_export(path, [("SCAN", "K/01/110gsm/2100mm/1200-1450", "02.01.2026", -3.0),
                             ("SCAN", "K/01/110gsm/2100mm/1200-1450", "03.01.2026", 0.0),
                             ("SCAN", "K/01/110gsm/2100mm/1200-1450", "05.01.2026", 4.5)])  # fmt: skip
    kept = read_export(path)
    assert [r[3] for r in kept] == [0.0, 0.0, 4.5]  # the negative became 0
    dropped = read_export(path, drop_zeros=True)
    assert [r[3] for r in dropped] == [0.0, 4.5]  # only the raw zero was dropped, not the clipped negative


def test_overlapping_exports_keep_identical_rows_once_and_stop_on_conflicts():
    a = [("SCAN", "S", "2026-01-02", 1.0), ("SCAN", "S", "2026-01-03", 2.0)]
    b = [("SCAN", "S", "2026-01-03", 2.0), ("SCAN", "S", "2026-01-04", 3.0)]
    assert combine_exports([a, b]) == [("SCAN", "S", "2026-01-02", 1.0), ("SCAN", "S", "2026-01-03", 2.0), ("SCAN", "S", "2026-01-04", 3.0)]
    with pytest.raises(SystemExit, match="different values"):
        combine_exports([a, [("SCAN", "S", "2026-01-03", 9.0)]])


@pytest.mark.parametrize(
    "planner_sku, data_sku",
    [
        ("K/01/110gsm/2100mm", "K/01/110gsm/2100mm/1200-1450"),
        ("K/01/1/110gsm/2100mm", "K/01/110gsm/2100mm/1200-1450"),  # old planner format with "/1"
        ("TSL/01/150gsm/2100mm", "TSL/01/150gsm/2100mm/1200-1400/Subst."),  # suffix variant
        ("TW2000/51/115gsm/2100mm", "tw2000/51/115gsm/2100mm/1200-1450"),  # case
    ],
)
def test_planner_and_dataset_skus_share_a_key(planner_sku, data_sku):
    assert planner_key(planner_sku) == dataset_key(data_sku)


def test_planner_comparison_adds_up_series_with_different_core_widths():
    pairs = pd.DataFrame({"planta": ["SCAN"] * 3, "sku": ["K/01/160gsm/2450mm/1200-1250", "K/01/160gsm/2450mm/1200-1450", "K/01/110gsm/2100mm/1200-1450"],
                          "series": [0, 1, 2], "first_month": [0, 0, 0]})  # fmt: skip
    pairs["key"] = pairs["sku"].map(dataset_key)
    res = pd.DataFrame({"series": [0, 1, 2], "t": [5, 5, 5], "y": [10.0, 20.0, 7.0], "p_ml": [9.0, 18.0, 8.0]})
    planner = pd.DataFrame({"planta": ["SCAN", "SCAN", "SCAN"], "sku_planner": ["K/01/160gsm/2450mm", "K/01/110gsm/2100mm", "X/01/9gsm/9mm"],
                            "forecast_to": [32.0, 6.0, 1.0], "strategy": ["VMI", "VMI", "VMI"]})  # fmt: skip
    table, missing = compare_month(planner, res, pairs, 5, ["ml"])
    assert missing == ["X/01/9gsm/9mm"]
    row = table.set_index("sku").loc["K/01/160gsm/2450mm"]
    assert (row["series"], row["real"], row["ml"], row["planner"]) == (2, 30.0, 27.0, 32.0)
    ranking = accuracy_table(table, ["planner", "ml"])
    assert list(ranking["rank"]) == [1, 2]


@pytest.mark.parametrize("kind", ["lgbm", "rf"])
def test_tree_models_are_deterministic_and_never_negative(toy_panel, kind):
    frame = make_frame(toy_panel, 1)
    origin = 24
    train, rows = frame[frame["t"] <= origin], frame[frame["o"] == origin]
    first = fit_predict_trees(kind, TREE_BASE[kind], train, rows, seed=0)
    second = fit_predict_trees(kind, TREE_BASE[kind], train, rows, seed=0)
    assert np.allclose(first, second)
    assert (first >= 0).all() and len(first) == len(rows)
