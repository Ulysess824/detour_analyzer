"""Cached readers of the files in data/ and results/. The dashboard only reads, it never retrains."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CONSUMPTION_CSV = ROOT / "data" / "consumos_long.csv"
DATA_DIR = ROOT / "data"
LEVELS = ["sku", "planta", "total"]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # make `src` importable

from src.utils.econometric_utils import ECONOMETRIC_MODELS  # noqa: E402
from src.utils.ensemble_utils import METHODS, add_ensembles  # noqa: E402
from src.utils.member_utils import ENSEMBLE_GROUPS, ML_MODELS, SIMPLE_MODELS  # noqa: E402

ENSEMBLES = [f"{group}_{method}" for group in ENSEMBLE_GROUPS for method in METHODS]
# The simple rules (naive, means, per-day rate...) are baselines, not fitted models.
MODEL_GROUPS = {
    "Machine learning": ML_MODELS,
    "Econometría clásica": ECONOMETRIC_MODELS,
    "Baselines (reglas simples)": SIMPLE_MODELS,
    "Ensembles": ENSEMBLES,
}
MODELS = [model for group in MODEL_GROUPS.values() for model in group]


@st.cache_data
def load_ensembles() -> dict:
    r"""The accuracy and Model Confidence Set tables written by scripts/run_ensembles.py."""
    return json.loads((RESULTS_DIR / "ensembles.json").read_text())


def best_by_level(ensembles: dict) -> dict[str, dict]:
    r"""The best record (lowest rank, then lowest mae) of the accuracy table of each level."""
    return {level: min(ensembles["accuracy"][level], key=lambda r: (r["rank"], r["mae"])) for level in LEVELS}


@st.cache_data
def load_monthly_actual() -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Monthly consumption per series, and the table that maps a series index to (planta, sku).

    The series index is the position of the (planta, sku) pair in alphabetical order, the same
    index used by results/member_forecasts.csv. Month 0 is the month of the first date.
    """
    df = pd.read_csv(CONSUMPTION_CSV, usecols=["planta", "sku", "fecha", "consumo"], parse_dates=["fecha"])
    start = df["fecha"].min().to_period("M")
    month = (df["fecha"].dt.year - start.year) * 12 + (df["fecha"].dt.month - start.month)
    pairs = df.groupby(["planta", "sku"]).size().index.to_frame(index=False)
    pairs["series"] = range(len(pairs))
    monthly = df.assign(t=month).groupby(["planta", "sku", "t"], as_index=False)["consumo"].sum()
    return monthly.merge(pairs, on=["planta", "sku"]), pairs


@st.cache_data
def month_labels() -> dict[int, str]:
    r"""Label ("2025-07") of every month index."""
    df = pd.read_csv(CONSUMPTION_CSV, usecols=["fecha"], parse_dates=["fecha"])
    periods = pd.period_range(df["fecha"].min(), df["fecha"].max(), freq="M")
    return {i: str(p) for i, p in enumerate(periods)}


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    r"""Member forecasts plus the ensembles (computed as in scripts/run_ensembles.py), with planta and sku."""
    res = pd.read_csv(RESULTS_DIR / "member_forecasts.csv")
    res = add_ensembles(res, ENSEMBLE_GROUPS, horizon=1)
    _, pairs = load_monthly_actual()
    return res.merge(pairs, on="series")


@st.cache_data
def load_planner_months() -> dict[str, pd.DataFrame]:
    r"""
    One table per month the planners forecasted, newest first: real, planner forecast and model forecasts per
    planner SKU (written by scripts/compare_planner.py as results/planner_comparison_<month>.csv).
    Each table has the columns `mes` and `planta`, taken from data/planner_forecast_<month>.csv.
    """
    months = {}
    for path in sorted(RESULTS_DIR.glob("planner_comparison_*.csv"), reverse=True):
        month = path.stem.removeprefix("planner_comparison_")
        table = pd.read_csv(path)
        planta = pd.read_csv(DATA_DIR / f"planner_forecast_{month}.csv", usecols=["planta"])["planta"].iloc[0]
        months[month] = table.assign(mes=month, planta=planta)
    return months
