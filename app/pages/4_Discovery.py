"""Page 4 — Discovery: quality-vs-popularity quadrants and hidden gems (Unit 4 extension)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import _theme as theme
from app._common import (
    hidden_gems_download,
    load_parquet,
    load_report_csv,
    quadrant_thresholds,
)
from app._components import (
    ITEMS_SOLD_NOTE,
    caption,
    footer,
    note,
    page_header,
    run_page,
    section,
    set_page,
    short_id,
    show,
    subtitle,
    table,
    technical,
)
from app._format import fmt_int, fmt_num
from app._labels import with_tier_labels
from src import config, explain
from src.discovery import TIER_ACTIONABLE, TIER_WATCHLIST, top_hidden_gems

set_page("Discovery")
_SUBTITLE = page_header("Which good sellers are we overlooking?")

_N_CONF = config.DISCOVERY_CONFIDENCE_MIN_ORDERS
_QUADRANTS = tuple(theme.QUADRANTS)
_TOP_N = 10
_SCORE_LABEL = f"Quality score ({config.DISCOVERY_MIN_ORDERS}+ items pool)"
# (quadrant, x, y, xanchor, yanchor, one-line label)
_CORNERS = (
    ("Star", 0.99, 0.99, "right", "top", "Star: good and popular"),
    ("Hidden Gem", 0.01, 0.99, "left", "top", "Hidden Gem: good, under-exposed"),
    ("Overrated", 0.99, 0.01, "right", "bottom", "Overrated: popular, weaker quality"),
    ("Overlooked-Low-Quality", 0.01, 0.01, "left", "bottom",
     "Overlooked-Low-Quality: neither"),
)


def _scatter(quadrants: pd.DataFrame, quality_line: float, popularity_line: float) -> go.Figure:
    fig = go.Figure()
    for quadrant in _QUADRANTS:
        color = theme.QUADRANTS[quadrant].color
        for confidence in ("normal", "low"):
            part = quadrants[
                (quadrants["quadrant"] == quadrant) & (quadrants["confidence"] == confidence)
            ]
            low = confidence == "low"
            fig.add_trace(
                go.Scatter(
                    x=part["order_volume"], y=part["ahp_score"], mode="markers",
                    name=quadrant, legendgroup=quadrant, showlegend=not low,
                    marker={
                        "color": color,
                        "symbol": "circle-open" if low else "circle",
                        "size": 8 if low else 6,
                        "opacity": 0.75 if low else 0.8,
                        "line": {"width": 1.5 if low else 0},
                    },
                    customdata=[
                        (short_id(s), c)
                        for s, c in zip(part["seller_id"], part["confidence"], strict=True)
                    ],
                    hovertemplate=(
                        f"<b>{quadrant}</b><br>Seller %{{customdata[0]}}<br>"
                        "Items sold: %{x}<br>Quality score: %{y:.3f}<br>"
                        "Confidence: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
    # Legend keys for the confidence encoding (not data).
    for symbol, label in (
        ("circle", f"Filled: {_N_CONF}+ items sold"),
        ("circle-open", f"Hollow: under {_N_CONF} items sold (low confidence)"),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker={"color": theme.TEXT_MUTED, "symbol": symbol, "size": 8,
                    "line": {"width": 1.5}},
        ))
    line = {"color": theme.TEXT_MUTED, "dash": "dash", "width": 1}
    fig.add_vline(x=popularity_line, line=line)
    fig.add_hline(y=quality_line, line=line)
    for quadrant, x, y, xanchor, yanchor, text in _CORNERS:
        fig.add_annotation(
            x=x, y=y, xref="paper", yref="paper", showarrow=False, xanchor=xanchor,
            yanchor=yanchor, text=f"<b>{text}</b>", bgcolor="rgba(12,16,21,0.8)",
            font={"size": 12, "color": theme.QUADRANTS[quadrant].text},
        )
    fig.update_layout(
        xaxis={"title": "Popularity: items sold (log scale)", "type": "log", "showgrid": False},
        yaxis={"title": f"Quality score, {config.DISCOVERY_MIN_ORDERS}+ items pool (0 to 1)",
               "showgrid": True, "gridcolor": theme.GRID},
        height=540,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        legend={"y": -0.2},
    )
    return fig


def _counts_line(quadrants: pd.DataFrame) -> None:
    counts = quadrants["quadrant"].value_counts()
    parts = [
        f'<span style="color:{theme.QUADRANTS[q].text}"><b>{q}</b></span> '
        f"{fmt_int(int(counts.get(q, 0)))}"
        for q in _QUADRANTS
    ]
    st.markdown(" · ".join(parts), unsafe_allow_html=True)


def _tier_table(rows: pd.DataFrame) -> None:
    shown = pd.DataFrame({
        "seller_id": rows["seller_id"].map(short_id),
        "order_volume": rows["order_volume"],
        "ahp_score": rows["ahp_score"],
        "avg_review": rows["avg_review"],
        "confidence": rows["confidence"].map(
            {"low": f"low (under {_N_CONF} items sold)", "normal": "normal"}
        ),
    })
    table(shown, {"order_volume": "int", "ahp_score": "num3", "avg_review": "num2"},
          overrides={"ahp_score": _SCORE_LABEL})


def _gems_tables(quadrants: pd.DataFrame, popularity_line: float) -> None:
    everyone = top_hidden_gems(quadrants, n=len(quadrants), tiered=True)
    reviews = load_parquet("SellerFacts.parquet", columns=("seller_id", "avg_review"))
    everyone = everyone.merge(reviews, on="seller_id", how="left")
    counts = everyone["tier"].value_counts()

    section("Hidden gems by confidence")
    caption("Both lists are ranked by quality score (shrunk average review and on-time "
            "delivery, price and items sold), so the review column is not in order.")
    note(f"By definition, hidden gems have under {fmt_num(popularity_line, 0)} items sold, so "
         "most picks are directional rather than proven.")
    for tier, label in ((TIER_ACTIONABLE, f"Stronger evidence ({_N_CONF}+ items sold)"),
                        (TIER_WATCHLIST, f"Early signal (under {_N_CONF} items sold)")):
        total = int(counts.get(tier, 0))
        st.markdown(f"**{label}: {fmt_int(total)} sellers**")
        rows = everyone[everyone["tier"] == tier].head(_TOP_N)
        if tier == TIER_ACTIONABLE and total < _TOP_N:
            st.markdown(f"Only {total} hidden gems have {_N_CONF}+ items sold, so this list is "
                        "short rather than padded.")
        if total:
            _tier_table(rows)
    st.download_button(
        "Download all hidden gems (CSV, full seller ids, with tier)",
        hidden_gems_download(everyone.drop(columns=["avg_review"])),
        file_name="hidden_gems.csv",
        mime="text/csv",
    )


def render() -> None:
    quadrants = load_report_csv("discovery_quadrants.csv")
    quality_line, popularity_line = quadrant_thresholds(quadrants)
    gems = quadrants[quadrants["quadrant"] == "Hidden Gem"]
    subtitle(_SUBTITLE, explain.discovery_answer(
        len(gems), int((gems["confidence"] == "low").sum()), _N_CONF
    ))
    _counts_line(quadrants)
    show(_scatter(quadrants, quality_line, popularity_line))
    caption(
        "This page surfaces sellers with strong quality signals but few items sold. "
        f"Sellers with fewer than {_N_CONF} items sold are flagged low-confidence; a few good "
        "reviews on a handful of items sold is promising, not proof."
    )
    note('Here "good" means sound business quality (reviews and on-time delivery), not food '
         "or nutrition: Olist has no such data.")

    _gems_tables(quadrants, popularity_line)

    with technical():
        definitions = explain.quadrant_definitions(quality_line, popularity_line)
        counts = quadrants["quadrant"].value_counts()
        for q in _QUADRANTS:
            st.markdown(f"- {explain.quadrant_sentence(q, int(counts.get(q, 0)), definitions[q])}")
        st.markdown(
            f"Split lines: quality at least {quality_line:.3f} AHP score and popularity at "
            f"least {popularity_line:,.0f} items sold count as high (medians of the "
            f"{len(quadrants):,} AHP-eligible sellers). Quality is the AHP score recomputed at "
            f"{config.DISCOVERY_MIN_ORDERS} items sold, lower than the Choice page's "
            f"{config.AHP_MIN_ORDERS}."
        )
        note(ITEMS_SOLD_NOTE)
        st.markdown("**Top hidden gems per tier, with full seller ids**")
        table(with_tier_labels(top_hidden_gems(quadrants, n=_TOP_N, tiered=True)),
              {"ahp_score": "num4", "order_volume": "int"}, overrides={"ahp_score": _SCORE_LABEL})
        st.markdown("**Every plotted seller**")
        table(quadrants.sort_values("ahp_score", ascending=False),
              {"ahp_score": "num4", "order_volume": "int"}, overrides={"ahp_score": _SCORE_LABEL})


run_page(render)
footer(
    "Multi-criteria decision making, extended to surface long-tail quality (Unit 4)",
    "Prescriptive",
    "Tactical",
)
