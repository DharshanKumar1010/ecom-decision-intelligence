"""Page 1 — Intelligence: KPIs, clickstream funnel (Unit 1) and call sentiment (Unit 3)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import _theme as theme
from app._common import data_key, load_parquet, marketplace_review, processed_path
from app._components import (
    ITEMS_SOLD_NOTE,
    caption,
    footer,
    kpis,
    money_short,
    note,
    page_header,
    run_page,
    section,
    set_page,
    show,
    subtitle,
    synthetic_tag,
    table,
    technical,
)
from app._format import fmt_int, fmt_num, fmt_pct
from src import clickstream, explain, sentiment

set_page("Intelligence")
_SUBTITLE = page_header("What is happening in the marketplace?")

_STAGES = ["page_view", "add_to_cart", "checkout", "purchase"]
_STAGE_LABELS = {
    "page_view": "Viewed a page",
    "add_to_cart": "Added to cart",
    "checkout": "Reached checkout",
    "purchase": "Purchased",
}


@st.cache_data(show_spinner="Computing clickstream metrics…")
def _clickstream_metrics(
    path: str, mtime: float
) -> tuple[pd.DataFrame, float, float, pd.DataFrame]:
    events = pd.read_parquet(path)
    return (
        clickstream.funnel_conversion(events),
        clickstream.bounce_rate(events),
        clickstream.cart_abandonment_rate(events),
        clickstream.funnel_by_device(events),
    )


def _funnel_chart(funnel: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Funnel(
            y=[_STAGE_LABELS.get(s, s) for s in funnel.index],
            x=funnel["sessions"].tolist(),
            texttemplate="%{value:,}<br>%{percentInitial:.0%}",
            hovertemplate="%{y}: %{value:,} sessions<extra></extra>",
            textfont={"color": theme.FUNNEL_TEXT},
            marker={"color": list(theme.FUNNEL_RAMP)},
            connector={"line": {"color": theme.BORDER}},
        )
    )
    fig.update_layout(margin={"l": 8, "r": 8, "t": 8, "b": 8}, height=340, showlegend=False)
    return fig


def _devices(by_device: pd.DataFrame) -> None:
    long = (
        by_device.reset_index()
        .melt(id_vars=by_device.index.name or "device", var_name="stage", value_name="sessions")
        .rename(columns={by_device.index.name or "device": "device"})
    )
    long["stage"] = long["stage"].map(lambda s: _STAGE_LABELS.get(s, s))
    fig = px.bar(
        long, x="stage", y="sessions", color="device", barmode="group",
        category_orders={"stage": [_STAGE_LABELS[s] for s in _STAGES]},
        color_discrete_sequence=list(theme.DEVICE_COLORS),
        labels={"stage": "", "sessions": "Sessions", "device": "Device"},
    )
    fig.update_layout(height=340)
    fig.update_traces(marker_line_width=0)
    show(fig)


def _agent_summary() -> pd.DataFrame:
    calls = load_parquet("CallSentiment.parquet")
    return sentiment.aggregate_by_agent(calls).reset_index()


def _negative_share_chart(agents: pd.DataFrame) -> go.Figure:
    ordered = agents.sort_values("negative_call_share", ascending=False)
    fig = go.Figure(
        go.Bar(
            x=ordered["agent_id"],
            y=ordered["negative_call_share"] * 100,
            marker={"color": theme.ACCENT},
            hovertemplate="%{x}: %{y:.0f}% negative<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        xaxis={"title": "Agent", "type": "category"},
        yaxis={"title": "Calls labelled negative (%)", "ticksuffix": "%"},
    )
    return fig


def _call_details(agents: pd.DataFrame) -> None:
    means = dict(zip(agents["agent_id"], agents["mean_sentiment"], strict=True))
    st.markdown(explain.sentiment_range_sentence(means))
    agents = agents.assign(silence_pct_display=agents["mean_silence_pct"] * 100)
    limit = float(agents["mean_sentiment"].abs().max()) or 1.0
    fig = px.scatter(
        agents, x="mean_hold_seconds", y="silence_pct_display", size="call_count",
        color="mean_sentiment", color_continuous_scale=theme.DIVERGING_SCALE,
        range_color=(-limit, limit), hover_name="agent_id",
        labels={
            "mean_hold_seconds": "Average time on hold (seconds)",
            "silence_pct_display": "Average silence (% of call)",
            "mean_sentiment": "Mean sentiment",
            "call_count": "Calls",
        },
    )
    fig.update_layout(height=380, coloraxis_colorbar={"title": "Mean sentiment"})
    show(fig)
    caption("One bubble per agent (hover for the agent id). Bubble size is the number of calls "
            "the agent handled; colour is the agent's mean sentiment, from red (negative) "
            "through grey to blue (positive).")
    table(
        agents.drop(columns=["silence_pct_display"]),
        {
            "mean_sentiment": "num3", "mean_hold_seconds": "num1", "mean_silence_pct": "pct",
            "negative_call_share": "pct", "call_count": "int",
        },
    )


def render() -> None:
    fact = load_parquet(
        "FactOrderItems.parquet", columns=("order_id", "price", "is_late", "review_score")
    )
    funnel, bounce, abandonment, by_device = _clickstream_metrics(
        *data_key(processed_path("Clickstream.parquet"))
    )
    late = float(fact["is_late"].mean())
    orders = fact["order_id"].nunique()
    review_per_pair, review_per_item, review_per_order = marketplace_review()
    no_review = float(fact["review_score"].isna().mean())
    subtitle(_SUBTITLE,
             f"{fmt_int(orders)} delivered orders; {fmt_pct(late, 1)} of items arrived late.")
    kpis([
        ("Delivered orders", fmt_int(orders), None),
        ("Item revenue", money_short(float(fact["price"].sum())),
         "Sum of item prices in Brazilian reais, excluding freight."),
        ("Late-delivery rate", fmt_pct(late, 2),
         "Share of order items delivered after the estimated date."),
        ("Average review score", f"{fmt_num(review_per_pair, 2)} / 5",
         "Mean of one review per seller-order pair (an order with items from two sellers counts "
         "once for each seller): the definition behind every seller's average review. Averaged "
         f"over distinct reviewed orders it is {fmt_num(review_per_order, 2)}; counting a review "
         f"once for every item it would be {fmt_num(review_per_item, 2)}. "
         f"{fmt_pct(no_review, 2)} of items have no review."),
    ])
    section("Where visitors drop out", synthetic=True)
    show(_funnel_chart(funnel))
    caption(explain.funnel_sentence(funnel))

    agents = _agent_summary()
    section("How customers sound on calls", synthetic=True)
    show(_negative_share_chart(agents))
    caption("Sentiment is scored from call transcripts only (a VADER score of the text); no "
            "audio and no speech recognition are involved.")

    with technical():
        section("Funnel table", synthetic=True)
        table(
            funnel.rename(index=_STAGE_LABELS).rename_axis("Step").reset_index(),
            {"sessions": "int", "stage_conversion": "pct", "overall_conversion": "pct"},
        )
        st.markdown(
            f"Bounce rate: **{fmt_pct(bounce, 2)}** of sessions view one page and leave. "
            f"Cart abandonment: **{fmt_pct(abandonment, 2)}** of sessions that add to the cart "
            f"never buy. {synthetic_tag()}",
            unsafe_allow_html=True,
        )
        section("Funnel by device", synthetic=True)
        _devices(by_device)
        section("Call sentiment by agent", synthetic=True)
        _call_details(agents)
        note(f"KPIs are computed live from {fmt_int(len(fact))} real Olist order items. "
             + ITEMS_SOLD_NOTE)


run_page(render)
footer(
    "Clickstream analysis (Unit 1) and sentiment analysis / speech analytics (Unit 3)",
    "Descriptive",
    "Operational",
)
