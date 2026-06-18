"""Synthetic-data smoke test for transform_data (MTD source)."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.data_transformation import transform_data


def build_raw() -> pl.DataFrame:
    """Two plants, one SKU each, two months of MTD readings with a mid-month reset."""
    rows = []
    # Business days across Jan and Feb 2026 for two customers
    fechas = [
        "02.01.2026", "05.01.2026", "06.01.2026", "07.01.2026", "08.01.2026",
        "02.02.2026", "03.02.2026", "04.02.2026", "05.02.2026", "06.02.2026",
    ]
    # MTD cumulative consumption that resets each month
    mtd_a = [10.0, 25.0, 40.0, 38.0, 60.0, 12.0, 30.0, 55.0, 80.0, 110.0]
    mtd_b = [5.0, 9.0, 14.0, 20.0, 27.0, 6.0, 13.0, 21.0, 30.0, 40.0]
    for f, a, b in zip(fechas, mtd_a, mtd_b):
        rows.append({
            "fecha_fichero": f, "Grade": "G1", "Sbgr": "S1", "Gram": 80, "Width": 100,
            "Customer": "PlantaA", "Hist. Month\nto date\n(TO)": a, "Total Fcst M\n(TO)": 200.0,
        })
        rows.append({
            "fecha_fichero": f, "Grade": "G2", "Sbgr": "S2", "Gram": 90, "Width": 120,
            "Customer": "PlantaB", "Hist. Month\nto date\n(TO)": b, "Total Fcst M\n(TO)": 100.0,
        })
    return pl.DataFrame(rows)


def main() -> None:
    df_raw = build_raw()
    out = transform_data(df_raw, umbral_pico=2.0)

    print("Shape:", out.shape)
    print("Columns:", out.columns)
    print(out.sort(["planta", "sku", "fecha"]))

    # Basic invariants
    assert {"planta", "sku", "fecha", "consumo_real", "daily_forecast",
            "target_is_peak", "target_consumo_real"}.issubset(out.columns)
    assert out["consumo_real"].min() >= 0.0, "consumo_real must be non-negative"
    assert out["target_is_peak"].null_count() == 0, "targets must be filtered of nulls"
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
