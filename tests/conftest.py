"""Shared fixtures: a small synthetic consumption table with the same shape as the real one."""

import numpy as np
import pandas as pd
import pytest

from src.utils.panel_utils import build_panel

N_MONTHS = 30


def make_consumption(seed: int = 0, n_months: int = N_MONTHS, extra_sku: bool = False) -> pd.DataFrame:
    r"""3 plantas, 4 SKUs each (one starts late), daily rows without Sundays, 2023-01-02 onwards."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-02", periods=n_months * 31, freq="D")
    days = days[(days.dayofweek != 6) & (days < pd.Timestamp("2023-01-01") + pd.DateOffset(months=n_months))]
    series = [(p, f"SKU{i}/01/100gsm/2100mm/1200-1450") for p in "ABC" for i in range(4)]
    if extra_sku:
        series.append(("A", "AAA/01/100gsm/2100mm/1200-1450"))  # sorts first, shifts every series index
    rows = []
    for k, (planta, sku) in enumerate(series):
        level = 3.0 + k
        start = days[0] if k % 5 else days[250]  # some series appear later
        mine = days[days >= start]
        month_effect = 1 + 0.3 * np.sin(2 * np.pi * mine.month / 12)
        values = rng.poisson(level * month_effect) * rng.binomial(1, 0.7, len(mine))
        rows.append(pd.DataFrame({"planta": planta, "sku": sku, "fecha": mine, "consumo": values.astype(float)}))
    df = pd.concat(rows, ignore_index=True)
    return df[df["consumo"] > 0].reset_index(drop=True)


@pytest.fixture(scope="session")
def toy_df() -> pd.DataFrame:
    return make_consumption()


@pytest.fixture(scope="session")
def toy_panel(toy_df):
    return build_panel(toy_df)
