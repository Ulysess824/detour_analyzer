"""Retrofuturistic theme: palette, fonts and the CSS injected into the Streamlit page."""

import streamlit as st

BG = "#0b0a1a"
PANEL = "#14122b"
PANEL_HI = "#1b1838"
GRID = "#2a2650"
TEXT = "#e6e3ff"
MUTED = "#8a86b8"
MAGENTA = "#ff2e97"
CYAN = "#19e3ff"
AMBER = "#ffb000"
PHOSPHOR = "#39ff88"  # status indicators only

REAL_COLOR = AMBER
MODEL_COLORS = [CYAN, MAGENTA, "#8f8bff", "#5aa3b5", "#c36aa0", "#9aa0d6", "#6f8fbf", "#b58fd0"]

FONT_DATA ="'Share Tech Mono', 'IBM Plex Mono', monospace"
FONT_TITLE = "'Orbitron', 'Audiowide', sans-serif"
FONTS_URL = "https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap"

CSS = f"""
@import url('{FONTS_URL}');

html, body, .stApp, [class*="st-"] {{ font-family: {FONT_DATA}; }}
/* keep the icon font, otherwise icons such as the sidebar arrow show up as their ligature text */
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: {BG}; color: {TEXT}; }}

/* CRT scanlines, subtle and non-interactive */
.stApp::before {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    background: repeating-linear-gradient(to bottom, rgba(255,255,255,0.025) 0, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 3px);
}}

[data-testid="stHeader"] {{ background: transparent; }}
.stAppDeployButton {{ display: none; }}
.block-container {{ padding-top: 2.2rem; max-width: 1200px; }}

/* Decorative text is not selectable, so the browser's caret browsing mode (F7) cannot put a caret in it */
.rf-card, .rf-title, .rf-subtitle, .rf-section, .rf-label, .rf-status {{ user-select: none; caret-color: transparent; }}

/* Titles */
.rf-title {{
    font-family: {FONT_TITLE}; font-weight: 900; font-size: 2.2rem; letter-spacing: 0.35em;
    text-transform: uppercase; color: {TEXT}; margin: 0; text-shadow: 0 0 14px {CYAN}88;
}}
.rf-subtitle {{ color: {MUTED}; letter-spacing: 0.18em; text-transform: uppercase; font-size: 0.8rem; margin: 0.4rem 0 0 0; }}
.rf-section {{
    font-family: {FONT_TITLE}; font-weight: 700; font-size: 0.95rem; letter-spacing: 0.3em;
    text-transform: uppercase; color: {TEXT}; margin: 1.6rem 0 0.8rem 0;
}}
.rf-label {{ color: {MUTED}; font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; }}

/* Status chip */
.rf-status {{ display: inline-block; margin: 0.9rem 0 1.4rem 0; font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; color: {PHOSPHOR}; }}
.rf-status::before {{
    content: ""; display: inline-block; width: 8px; height: 8px; margin-right: 8px;
    background: {PHOSPHOR}; box-shadow: 0 0 8px {PHOSPHOR};
}}

/* Metric cards */
.rf-card {{
    background: {PANEL}; border: 1px solid {GRID}; border-radius: 2px; padding: 1rem 1.2rem; height: 100%;
}}
.rf-card.active {{ border-color: {CYAN}; box-shadow: 0 0 12px {CYAN}55; }}
.rf-card .model {{ font-size: 1.05rem; letter-spacing: 0.08em; color: {TEXT}; margin: 0.5rem 0 0.2rem 0; word-break: break-all; }}
.rf-card .value {{
    font-family: {FONT_TITLE}; font-weight: 700; font-size: 2rem; text-align: right;
    font-variant-numeric: tabular-nums; color: {TEXT};
}}
.rf-card.active .value {{ color: {CYAN}; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {GRID}; }}
[data-testid="stWidgetLabel"] p {{ color: {MUTED}; font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}
/* Streamlit pulls every markdown block up by 1rem to cancel the 1rem bottom margin of its paragraph, so a smaller bottom margin makes the next widget overlap */
[data-testid="stSidebar"] .rf-section {{ font-size: 0.8rem; letter-spacing: 0.18em; margin: 1rem 0 0.9rem 0; }}
[data-testid="stSidebar"] .rf-label {{ font-size: 0.68rem; letter-spacing: 0.08em; margin: 0.3rem 0 0.85rem 0; }}
[data-testid="stSidebar"] [data-baseweb="button-group"] {{ gap: 0.3rem; }}
[data-baseweb="select"] > div {{ background: {PANEL_HI}; border: 1px solid {GRID}; border-radius: 2px; }}

/* Tabs */
[data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {GRID}; }}
[data-baseweb="tab"] {{ background: transparent; }}
[data-baseweb="tab"] p {{ font-family: {FONT_TITLE}; font-size: 0.8rem; letter-spacing: 0.2em; text-transform: uppercase; color: {MUTED}; }}
[data-baseweb="tab"][aria-selected="true"] p {{ color: {CYAN}; text-shadow: 0 0 8px {CYAN}88; }}
[data-baseweb="tab-highlight"] {{ background: {CYAN}; box-shadow: 0 0 8px {CYAN}; }}

/* Tables and numbers */
[data-testid="stDataFrame"] {{ border: 1px solid {GRID}; border-radius: 2px; }}
td, th {{ font-variant-numeric: tabular-nums; }}
"""


def add_model_trace(fig, x, y, name: str, color: str) -> None:
    r"""A model line: thin and slightly transparent, so the actual stands out over it."""
    fig.add_scatter(
        x=x, y=y, name=name, mode="lines+markers", opacity=0.85,
        line=dict(color=color, width=1.6), marker=dict(symbol="square", size=5),
    )  # fmt: skip


def add_real_trace(fig, x, y, name: str = "real") -> None:
    r"""
    The actual series, the most prominent line of every chart: a wide glow underneath, a thick
    line with large white-edged markers on top. Add it after the models so it is drawn over them;
    `legendrank` keeps it first in the legend.
    """
    fig.add_scatter(
        x=x, y=y, mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=REAL_COLOR, width=14), opacity=0.22,
    )  # fmt: skip
    fig.add_scatter(
        x=x, y=y, name=name, mode="lines+markers", legendrank=0,
        line=dict(color=REAL_COLOR, width=4.5),
        marker=dict(symbol="square", size=11, color=REAL_COLOR, line=dict(color="#ffffff", width=1.5)),
    )  # fmt: skip


def plotly_layout(**overrides) -> dict:
    r"""Common plotly layout: transparent paper, panel-colored plot area, faint grid, monospaced text."""
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), automargin=True)
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(family="Share Tech Mono, IBM Plex Mono, monospace", color=TEXT),
        xaxis=axis,
        yaxis=axis,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color=TEXT)),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PANEL_HI, bordercolor=GRID, font=dict(color=TEXT)),
    )
    layout.update(overrides)
    return layout


def apply_theme() -> None:
    r"""Inject the retrofuturistic CSS into the page."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
