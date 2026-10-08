"""Shared page template: header, KPIs, caption, technical expander, footer.

Presentation only. Every page follows the same order: plain-question title, grey
live subtitle, at most four KPIs, one main chart with one caption, one
"Technical details" expander, and a small grey footer. Everything is self-contained:
one system sans-serif font, inline CSS, no remote resources.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError

from app import _theme as theme
from app._common import model_review_flag

MISSING_FILE_HELP = (
    "Run `python run_all.py` from the project root to generate the pipeline "
    "outputs, then reload this page."
)
TECHNICAL_LABEL = "Technical details"
# Shown in Technical details wherever a seller-level count appears.
ITEMS_SOLD_NOTE = (
    "Items sold counts order lines: an order with three products is three items, while "
    "delivered orders counts each order once."
)
SYNTHETIC_TAG = "Synthetic data"

_CSS = f"""
<style>
.block-container {{ padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1100px; }}
h1 {{ font-size: 2rem; font-weight: 650; line-height: 1.2; letter-spacing: -0.01em; }}
h2, h3 {{ font-weight: 600; }}
.subtitle {{ color: {theme.TEXT_MUTED}; font-size: 1.05rem; margin: 0 0 1.5rem 0; }}
[data-testid="stMetricValue"] {{ font-size: 1.8rem; font-weight: 600;
  white-space: normal; overflow: visible; text-overflow: clip; }}
[data-testid="stMetricValue"] > div {{ white-space: normal; overflow: visible;
  text-overflow: clip; }}
[data-testid="stMetricLabel"] p {{ color: {theme.TEXT_MUTED}; }}
.tag {{ color: {theme.TEXT_MUTED}; font-size: 0.75rem; font-weight: 500;
  border: 1px solid {theme.BORDER}; border-radius: 0.3rem; padding: 0.05rem 0.4rem;
  margin-left: 0.6rem; vertical-align: middle; }}
.footer {{ color: {theme.TEXT_MUTED}; font-size: 0.8rem; margin-top: 2.5rem;
  padding-top: 0.75rem; border-top: 1px solid {theme.BORDER}; }}
.note {{ color: {theme.TEXT_MUTED}; font-size: 0.85rem; }}
.action-word {{ font-size: 2.6rem; font-weight: 650; line-height: 1.1; margin: 0.2rem 0 0.4rem 0; }}
.flow-box {{ border: 1px solid {theme.BORDER}; border-radius: 0.4rem;
  padding: 1rem 1.1rem; height: 16rem; box-sizing: border-box; overflow: hidden; }}
.flow-note {{ color: {theme.TEXT_MUTED}; font-size: 0.85rem; margin-top: 1.75rem; }}
.flow-box h4 {{ margin: 0 0 0.5rem 0; font-size: 1.05rem; }}
.flow-box p {{ margin: 0.15rem 0; color: {theme.TEXT_MUTED}; font-size: 0.9rem; }}
[data-testid="stExpander"] {{ border: 1px solid {theme.BORDER}; box-shadow: none; }}
</style>
"""


def _safe(text: str) -> str:
    """HTML-escape `text` and neutralise `$` so Streamlit does not read it as maths."""
    return html.escape(text).replace("$", "&#36;")


def set_page(title: str) -> None:
    """Page config and shared CSS. Must be the first Streamlit call on a page."""
    st.set_page_config(page_title=f"{title} | Olist DSS", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str) -> Any:
    """Plain-question title; returns the slot the page fills with its live subtitle."""
    st.title(title)
    return st.empty()


def subtitle(slot: Any, text: str) -> None:
    """Fill the grey subtitle slot with one live-computed sentence."""
    slot.markdown(f'<div class="subtitle">{_safe(text)}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str | None]]) -> None:
    """At most four `(label, value, tooltip)` numbers in one row."""
    assert len(items) <= 4, "a page shows at most four KPI numbers"
    for col, (label, value, help_text) in zip(st.columns(len(items)), items, strict=True):
        col.metric(label, value, help=help_text)


def section(title: str, synthetic: bool = False) -> None:
    """Section heading; synthetic data gets a small grey tag beside the title."""
    tag = f'<span class="tag">{SYNTHETIC_TAG}</span>' if synthetic else ""
    st.markdown(f"### {title} {tag}" if tag else f"### {title}", unsafe_allow_html=True)


def caption(text: str) -> None:
    """The single short caption under a chart."""
    st.caption(text)


def note(text: str) -> None:
    """Small grey plain-text note."""
    st.markdown(f'<div class="note">{_safe(text)}</div>', unsafe_allow_html=True)


def technical() -> Any:
    """The page's one "Technical details" expander (use as a context manager)."""
    return st.expander(TECHNICAL_LABEL)


def footer(topic: str, analytics: str, level: str) -> None:
    """Small grey footer: syllabus topic · analytics type · management level."""
    st.markdown(
        f'<div class="footer">Syllabus topic: {_safe(topic)} · Analytics type: '
        f"{_safe(analytics)} · Management level: {_safe(level)}</div>",
        unsafe_allow_html=True,
    )


def show(fig: go.Figure) -> None:
    """Render a Plotly figure with the shared template (Streamlit's theme must not override)."""
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def money_short(value: float, currency: str = "R$") -> str:
    """Compact currency that never truncates, e.g. `R$ 13.2M`, `R$ 842K`, `R$ 950`."""
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{currency} {value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{currency} {value / 1_000:.0f}K"
    return f"{currency} {value:,.0f}"


def short_id(seller_id: str, length: int = 8) -> str:
    """Readable seller id for display; the full id stays in downloads and technical tables."""
    return seller_id if len(seller_id) <= length else f"{seller_id[:length]}…"


def page_link_or_label(target: str, label: str) -> None:
    """Clickable link to another page, or a plain label if Streamlit can't resolve it.

    `st.page_link` raises `StreamlitPageNotFoundError` when no multipage
    registry exists (e.g. a page executed standalone under `AppTest`).
    """
    try:
        st.page_link(target, label=label)
    except StreamlitPageNotFoundError:
        st.markdown(label)


def review_line() -> None:
    """One plain line under the subtitle, only if the logs carry the model-review flag."""
    flag = model_review_flag()
    if flag:
        source, reason = flag
        note(
            f"The predictive model is under review (see {source})"
            + (f": {reason}." if reason else ".")
            + " Treat risk scores as provisional."
        )


def run_page(render: Callable[[], None]) -> None:
    """Run a page body, turning a missing pipeline output into a plain message.

    Only `FileNotFoundError` is handled; any other exception is a real bug and
    is left to surface.
    """
    try:
        render()
    except FileNotFoundError as exc:
        missing = exc.filename or str(exc)
        with st.container(border=True):
            st.markdown("**A required pipeline output is missing**")
            st.markdown(f"`{missing}` could not be found. {MISSING_FILE_HELP}")
