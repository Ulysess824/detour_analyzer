"""Tab "REAL VS PREDICHO": monthly actual against the forecasts of the chosen models."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from loaders import MODEL_GROUPS, MODELS, load_forecasts, load_monthly_actual, month_labels
from theme import MODEL_COLORS, add_model_trace, add_real_trace, plotly_layout

ALL_PLANTAS = "Todas las plantas"
ALL_SKUS = "Todos los SKU"
SEARCH_HINT = "Escribe para buscar"
DEFAULT_MODELS = ["lgbm", "ml_econ_mean"]


def select_view(forecasts: pd.DataFrame, monthly: pd.DataFrame, planta: str | None, sku: str | None):
    r"""Rows of the forecasts and of the actuals of the chosen planta and SKU (None means all)."""
    for column, value in (("planta", planta), ("sku", sku)):
        if value is not None:
            forecasts, monthly = forecasts[forecasts[column] == value], monthly[monthly[column] == value]
    return forecasts, monthly


def describe_selection(planta: str | None, sku: str | None) -> str:
    r"""Plain-language description of what the chart shows."""
    if sku is None:
        return "total de todas las plantas" if planta is None else f"total de la planta {planta}"
    return f"SKU {sku} en " + ("todas las plantas" if planta is None else f"la planta {planta}")


def monthly_series(forecasts: pd.DataFrame, monthly: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    r"""
    One row per test month: the actual (all SKUs of the scope) and the summed forecast of each model.

    The forecast of a month is NaN when the scope has no forecast row for it (a SKU that did not
    exist yet at the forecast origin).
    """
    months = sorted(forecasts["t"].unique())
    actual = monthly.groupby("t")["consumo"].sum().reindex(months, fill_value=0.0)
    pred = forecasts.groupby("t")[[f"p_{m}" for m in models]].sum(min_count=1).reindex(months)
    pred.columns = models
    return pd.concat([actual.rename("real"), pred], axis=1)


def score_models(table: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    r"""Accuracy and bias of each model over the months where it has a forecast, ranked by accuracy."""
    rows = []
    for model in models:
        valid = table[model].notna()
        y, p = table.loc[valid, "real"], table.loc[valid, model]
        total = y.sum()
        wape = (y - p).abs().sum() / total if total > 0 else float("nan")
        bias = (y - p).sum() / total * 100 if total > 0 else float("nan")
        rows.append({"model": model, "accuracy": (1 - wape) * 100, "bias_pct": bias})
    out = pd.DataFrame(rows).sort_values("accuracy", ascending=False, ignore_index=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def build_figure(table: pd.DataFrame, models: list[str], labels: dict[int, str]) -> go.Figure:
    r"""Line chart by month: thin model lines in cyan, magenta and muted tones, the actual highlighted on top."""
    x = [labels[t] for t in table.index]
    fig = go.Figure()
    for i, model in enumerate(models):
        add_model_trace(fig, x, table[model], model, MODEL_COLORS[i % len(MODEL_COLORS)])
    add_real_trace(fig, x, table["real"])
    fig.update_layout(**plotly_layout(height=420, yaxis_title="Consumo mensual", xaxis_type="category"))
    return fig


def render_series_picker(monthly: pd.DataFrame) -> tuple[str | None, str | None]:
    r"""
    Searchable planta and SKU lists (click and type to filter); returns (planta, sku), None meaning all.

    The SKU list only has the SKUs of the chosen planta. Choosing a SKU without a planta adds it up over
    the plantas that have it.
    """
    st.markdown('<p class="rf-section" style="margin-top:2.2rem">Serie a visualizar</p>', unsafe_allow_html=True)
    left, right = st.columns(2)
    planta = left.selectbox("Planta", [ALL_PLANTAS, *sorted(monthly["planta"].unique())], key="rvp_planta", placeholder=SEARCH_HINT)
    in_planta = monthly if planta == ALL_PLANTAS else monthly[monthly["planta"] == planta]
    sku = right.selectbox("SKU", [ALL_SKUS, *sorted(in_planta["sku"].unique())], key=f"rvp_sku_{planta}", placeholder=SEARCH_HINT)
    return (None if planta == ALL_PLANTAS else planta), (None if sku == ALL_SKUS else sku)


def render_model_picker() -> list[str]:
    r"""Toggle buttons grouped by category; returns the chosen models in a fixed order (stable colors)."""
    chosen = []
    for group, models in MODEL_GROUPS.items():
        st.markdown(f'<p class="rf-label rf-group">{group}</p>', unsafe_allow_html=True)
        default = [m for m in models if m in DEFAULT_MODELS]
        picked = st.pills(group, models, selection_mode="multi", default=default, key=f"rvp_{group}", label_visibility="collapsed")
        chosen += picked or []
    return sorted(chosen, key=MODELS.index)


def render_model_sidebar() -> list[str]:
    r"""The model buttons, in the fixed sidebar so they stay visible next to the chart."""
    with st.sidebar:
        st.markdown('<p class="rf-section">Modelos a comparar</p>', unsafe_allow_html=True)
        return render_model_picker()


def render(planta: str | None, sku: str | None, models: list[str]) -> None:
    if not models:
        st.markdown('<p class="rf-label">Elige al menos un modelo en la barra lateral</p>', unsafe_allow_html=True)
        return
    view_forecasts, view_monthly = select_view(load_forecasts(), load_monthly_actual()[0], planta, sku)
    st.markdown(f'<p class="rf-label">Mostrando: {describe_selection(planta, sku)}</p>', unsafe_allow_html=True)
    if view_forecasts.empty:
        st.markdown(
            '<p class="rf-label">Esta serie no tiene pronósticos en el test: aparece después del último origen.</p>',
            unsafe_allow_html=True,
        )
        return

    table = monthly_series(view_forecasts, view_monthly, models)
    st.plotly_chart(build_figure(table, models, month_labels()), width="stretch", theme=None)

    st.markdown('<p class="rf-section">Accuracy y bias por modelo</p>', unsafe_allow_html=True)
    st.dataframe(
        score_models(table, models),
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("rank", format="%d"),
            "model": st.column_config.TextColumn("model"),
            "accuracy": st.column_config.NumberColumn("accuracy (%)", format="%.1f"),
            "bias_pct": st.column_config.NumberColumn("bias (%)", format="%.1f"),
        },
    )
    st.markdown(
        '<p class="rf-label">Bias positivo: el modelo predice de menos. '
        "El pronóstico agregado suma los SKU existentes al origen; los SKU nuevos no se pronostican.</p>",
        unsafe_allow_html=True,
    )
