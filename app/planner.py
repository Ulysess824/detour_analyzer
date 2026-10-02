"""Tab "PLANIFICADOR": the planners' forecast against the models and the real consumption, SKU by SKU."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from loaders import load_planner_comparison
from src.utils.planner_utils import accuracy_table
from theme import AMBER, MAGENTA, MODEL_COLORS, REAL_COLOR, plotly_layout

ORDERS = {
    "Consumo real (mayor a menor)": ("real", False),
    "Error del planificador (mayor a menor)": ("planner_abs_err", False),
    "Nombre del SKU": ("sku", True),
}


def add_real_markers(fig: go.Figure, x, y) -> None:
    r"""The actual as large amber squares with a glow. Joining SKUs with a line would mean nothing."""
    fig.add_scatter(x=x, y=y, mode="markers", showlegend=False, hoverinfo="skip", marker=dict(symbol="square", size=26, color=REAL_COLOR, opacity=0.2))
    fig.add_scatter(
        x=x, y=y, name="real", mode="markers", legendrank=0,
        marker=dict(symbol="square", size=12, color=REAL_COLOR, line=dict(color="#ffffff", width=1.5)),
    )  # fmt: skip


MODEL_TONES = [c for c in MODEL_COLORS if c != MAGENTA]  # magenta is the planner


def build_figure(table: pd.DataFrame, models: list[str]) -> go.Figure:
    r"""One column of points per SKU: models in cyan and violet tones, the planner in magenta, the real on top."""
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_scatter(
            x=table["sku"], y=table[model], name=model, mode="markers", opacity=0.9,
            marker=dict(symbol="circle", size=7, color=MODEL_TONES[i % len(MODEL_TONES)]),
        )  # fmt: skip
    fig.add_scatter(
        x=table["sku"], y=table["planner"], name="planificador", mode="markers",
        marker=dict(symbol="diamond", size=11, color=MAGENTA, line=dict(color="#ffffff", width=1)),
    )  # fmt: skip
    add_real_markers(fig, table["sku"], table["real"])
    fig.update_layout(
        **plotly_layout(height=460, yaxis_title="Consumo del mes (TO)", xaxis_type="category", hovermode="x unified")
    )
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=10))
    return fig


def summary_cards(table: pd.DataFrame, month: str, planta: str) -> None:
    r"""Totals of the planner SKUs: real, planner and its difference."""
    real, planner = table["real"].sum(), table["planner"].sum()
    diff = (planner - real) / real * 100
    cards = [
        ("Consumo real", f"{real:,.0f}", f"{planta}, {month}, {len(table)} SKU"),
        ("Forecast del planificador", f"{planner:,.0f}", f"{diff:+.1f}% frente al real"),
        ("SKU sobreestimados", f"{int((table['planner'] > table['real']).sum())} de {len(table)}", "planificador por encima del real"),
    ]
    for column, (label, value, note) in zip(st.columns(len(cards)), cards):
        column.markdown(
            f'<div class="rf-card"><div class="rf-label">{label}</div>'
            f'<div class="value">{value}</div><div class="rf-label" style="text-align:right">{note}</div></div>',
            unsafe_allow_html=True,
        )


def paint_planner(row: pd.Series) -> list[str]:
    r"""Highlight the planner row (the name is also written, so the color is not the only cue)."""
    return [f"color: {AMBER}; font-weight: 700" if row["model"] == "planificador" else "" for _ in row.index]


def render_ranking(table: pd.DataFrame, models: list[str]) -> None:
    r"""Accuracy of the planner and of the selected models on the same SKUs and month."""
    ranking = accuracy_table(table, ["planner", *models])
    ranking["model"] = ranking["model"].replace({"planner": "planificador"})
    ranking["wape"] *= 100
    ranking["accuracy"] *= 100
    st.markdown('<p class="rf-section">Quién acertó más</p>', unsafe_allow_html=True)
    styled = ranking.style.apply(paint_planner, axis=1).format(
        {"rank": "{:d}", "n": "{:d}", "mae": "{:,.2f}", "wape": "{:.2f}", "accuracy": "{:.2f}", "bias_pct": "{:.2f}"}
    )
    st.dataframe(
        styled, hide_index=True, width="stretch",
        column_config={"wape": st.column_config.Column("wape (%)"), "accuracy": st.column_config.Column("accuracy (%)"), "bias_pct": st.column_config.Column("bias (%)")},
    )  # fmt: skip
    st.markdown(
        '<p class="rf-label">Orden por WAPE ascendente (rank 1 = menor error). Bias positivo: predice de menos; '
        "negativo: predice de más. Mismos SKU y mismo mes para todos.</p>",
        unsafe_allow_html=True,
    )


def render_detail(table: pd.DataFrame, models: list[str]) -> None:
    r"""Table by SKU with the real, the planner, its error and the selected models."""
    detail = table[["sku", "real", "planner", *models]].copy()
    detail.insert(3, "planner_err_%", (table["planner"] - table["real"]) / table["real"] * 100)
    st.markdown('<p class="rf-section">Detalle por SKU</p>', unsafe_allow_html=True)
    st.dataframe(detail.style.format({c: "{:,.1f}" for c in detail.columns if c != "sku"}), hide_index=True, width="stretch")
    st.markdown(
        '<p class="rf-label">planner_err_% = (planificador - real) / real. Positivo: el planificador pidió de más.</p>',
        unsafe_allow_html=True,
    )


def render(models: list[str]) -> None:
    table, planta, month = load_planner_comparison()
    st.markdown(f'<p class="rf-label">Planificador contra modelos y consumo real: planta {planta}, mes {month}</p>', unsafe_allow_html=True)
    summary_cards(table, month, planta)
    if not models:
        st.markdown('<p class="rf-label">Elige al menos un modelo en la barra lateral</p>', unsafe_allow_html=True)
        return

    order = st.selectbox("Orden de los SKU", list(ORDERS), key="planner_order")
    column, ascending = ORDERS[order]
    table = table.assign(planner_abs_err=(table["planner"] - table["real"]).abs()).sort_values(column, ascending=ascending, ignore_index=True)

    st.markdown('<p class="rf-section">Real, planificador y modelos por SKU</p>', unsafe_allow_html=True)
    st.plotly_chart(build_figure(table, models), width="stretch", theme=None)
    render_ranking(table, models)
    render_detail(table, models)
    st.markdown(
        '<p class="rf-label">Un solo mes y una sola planta: sirve como referencia, no como prueba. '
        "El SKU del planificador no trae ancho de núcleo; se suman las series del dataset con igual tipo, gramaje y ancho.</p>",
        unsafe_allow_html=True,
    )
