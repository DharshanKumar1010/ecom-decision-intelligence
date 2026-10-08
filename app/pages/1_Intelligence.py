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
from app._common import data_key, load_parquet, processed_path
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
    technical,
)
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
_DEVICE_RAMP = ("#9db8d1", "#40729b", "#1F4E79")


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
        color_discrete_sequence=list(_DEVICE_RAMP),
        labels={"stage": "", "sessions": "Sessions", "device": "Device"},
    )
    fig.update_layout(height=340)
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
    st.dataframe(
        agents.drop(columns=["silence_pct_display"]),
        hide_index=True,
        width="stretch",
        column_config={
            "agent_id": "Agent",
            "mean_sentiment": st.column_config.NumberColumn("Mean sentiment", format="%.3f"),
            "mean_hold_seconds": st.column_config.NumberColumn("Mean hold (s)", format="%.1f"),
            "mean_silence_pct": st.column_config.NumberColumn("Mean silence", format="percent"),
            "negative_call_share": st.column_config.NumberColumn(
                "Negative share", format="percent"
            ),
            "call_count": "Calls",
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
    subtitle(_SUBTITLE, f"{orders:,} delivered orders; {late * 100:.1f}% of items arrived late.")
    kpis([
        ("Delivered orders", f"{orders:,}", None),
        ("Item revenue", money_short(float(fact["price"].sum())),
         "Sum of item prices in Brazilian reais, excluding freight."),
        ("Late-delivery rate", f"{late * 100:.2f}%",
         "Share of order items delivered after the estimated date."),
        ("Average review score", f"{fact['review_score'].mean():.2f} / 5",
         "Mean of orders that have a review (about 0.75% have none)."),
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
        st.markdown("**Funnel table**")
        st.dataframe(
            funnel.rename(columns={
                "sessions": "Sessions",
                "stage_conversion": "Stage conversion",
                "overall_conversion": "Overall conversion",
            }),
            width="stretch",
            column_config={
                "Stage conversion": st.column_config.NumberColumn(format="percent"),
                "Overall conversion": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.markdown(
            f"Bounce rate: **{bounce * 100:.1f}%** of sessions view one page and leave. "
            f"Cart abandonment: **{abandonment * 100:.1f}%** of sessions that add to the cart "
            "never buy."
        )
        section("Funnel by device", synthetic=True)
        _devices(by_device)
        section("Call sentiment by agent", synthetic=True)
        _call_details(agents)
        note(f"KPIs are computed live from {len(fact):,} real Olist order items. "
             + ITEMS_SOLD_NOTE)


run_page(render)
footer(
    "Clickstream analysis (Unit 1) and sentiment analysis / speech analytics (Unit 3)",
    "Descriptive",
    "Operational",
)
