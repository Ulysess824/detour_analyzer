"""
Retrofuturistic dashboard of the forecasting results (read-only).

Usage:
    streamlit run app/dashboard.py
"""

import streamlit as st

import ranking
import real_vs_pred
from loaders import LEVELS, best_by_level, load_ensembles, load_monthly_actual
from theme import apply_theme

LEVEL_LABELS = {"sku": "SKU", "planta": "PLANTA", "total": "TOTAL"}
HORIZONS = ["h=1 (1 mes)"]  # fixed for now: the stored forecasts are for h=1


def render_header() -> None:
    st.markdown(
        '<p class="rf-title">Detour Analyzer</p>'
        '<p class="rf-subtitle">Pronóstico de consumo mensual por planta y SKU</p>'
        '<span class="rf-status">Resultados cargados</span>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, list[str]]:
    r"""Sidebar controls; returns the selected level and the models chosen for the comparison."""
    with st.sidebar:
        st.markdown('<p class="rf-section">Controles</p>', unsafe_allow_html=True)
        level = st.selectbox("Nivel", LEVELS, format_func=LEVEL_LABELS.get)
        st.selectbox("Horizonte", HORIZONS, disabled=True)
    return level, real_vs_pred.render_model_sidebar()


def render_best_cards(best: dict[str, dict], selected: str) -> None:
    r"""One card per level with the best model and its accuracy; the selected level is highlighted."""
    st.markdown('<p class="rf-section">Mejor modelo por nivel</p>', unsafe_allow_html=True)
    for column, level in zip(st.columns(len(LEVELS)), LEVELS):
        record = best[level]
        active = " active" if level == selected else ""
        column.markdown(
            f'<div class="rf-card{active}">'
            f'<div class="rf-label">Nivel {LEVEL_LABELS[level]}</div>'
            f'<div class="model">{record["model"]}</div>'
            f'<div class="value">{record["accuracy"] * 100:.1f}%</div>'
            f'<div class="rf-label" style="text-align:right">Accuracy</div>'
            "</div>",
            unsafe_allow_html=True,
        )


def render_tabs(ensembles: dict, level: str, planta: str | None, sku: str | None, models: list[str]) -> None:
    real_tab, ranking_tab, mcs_tab = st.tabs(["REAL VS PREDICHO", "RANKING", "CONJUNTO DE CONFIANZA"])
    with real_tab:
        real_vs_pred.render(planta, sku, models)
    with ranking_tab:
        ranking.render(ensembles, level, LEVEL_LABELS[level])
    with mcs_tab:
        st.markdown('<p class="rf-section">Conjunto de confianza</p>', unsafe_allow_html=True)
        st.markdown('<p class="rf-label">Sin contenido todavía</p>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Detour Analyzer", layout="wide")
    apply_theme()
    render_header()
    level, models = render_sidebar()
    ensembles = load_ensembles()
    render_best_cards(best_by_level(ensembles), level)
    planta, sku = real_vs_pred.render_series_picker(load_monthly_actual()[0])
    st.markdown("<br>", unsafe_allow_html=True)
    render_tabs(ensembles, level, planta, sku, models)


main()
