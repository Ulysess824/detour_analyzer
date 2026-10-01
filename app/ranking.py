"""Tab "RANKING": error metrics of every model at the chosen level, ordered by rank."""

import pandas as pd
import streamlit as st

from loaders import MODEL_GROUPS
from theme import CYAN, MAGENTA, MUTED

GROUP_COLORS = {
    "Machine learning": CYAN,
    "Ensembles": MAGENTA,
    "Econometría clásica": "#8f8bff",
    "Baselines (reglas simples)": MUTED,
}
GROUP_OF = {model: group for group, models in MODEL_GROUPS.items() for model in models}


def ranking_table(ensembles: dict, level: str) -> pd.DataFrame:
    r"""The accuracy records of `level`, ordered by rank (ties by mae), with the group of each model."""
    table = pd.DataFrame(ensembles["accuracy"][level]).sort_values(["rank", "mae"], ignore_index=True)
    table.insert(2, "grupo", table["model"].map(GROUP_OF))
    table["wape"] *= 100
    table["accuracy"] *= 100
    return table[["rank", "model", "grupo", "n", "mae", "wape", "accuracy", "bias_pct"]]


def paint_group(row: pd.Series) -> list[str]:
    r"""Color the model and group cells with the color of the group (the group name is also written)."""
    color = f"color: {GROUP_COLORS[row['grupo']]}"
    return [color if column in ("model", "grupo") else "" for column in row.index]


def render(ensembles: dict, level: str, level_label: str) -> None:
    table = ranking_table(ensembles, level)
    st.markdown(f'<p class="rf-section">Ranking de modelos: nivel {level_label}</p>', unsafe_allow_html=True)
    styled = table.style.apply(paint_group, axis=1).format(
        {"rank": "{:d}", "n": "{:,d}", "mae": "{:,.3f}", "wape": "{:.2f}", "accuracy": "{:.2f}", "bias_pct": "{:.2f}"}
    )
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={
            "wape": st.column_config.Column("wape (%)"),
            "accuracy": st.column_config.Column("accuracy (%)"),
            "bias_pct": st.column_config.Column("bias (%)"),
        },
    )
    st.markdown(
        '<p class="rf-label">Orden por WAPE ascendente (rank 1 = menor error). Un mismo rank indica empate. '
        "Bias positivo: el modelo predice de menos. Test 2025-07 a 2026-06, h=1.</p>",
        unsafe_allow_html=True,
    )
