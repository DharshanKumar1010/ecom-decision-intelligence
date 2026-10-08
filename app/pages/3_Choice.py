"""Page 3 — Choice: AHP seller ranking with an editable pairwise matrix (Unit 4)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import _theme as theme
from app._common import load_report_csv
from app._components import (
    caption,
    footer,
    kpis,
    note,
    page_header,
    run_page,
    section,
    set_page,
    short_id,
    show,
    subtitle,
    technical,
)
from src import config, explain
from src.ahp import compute_weights, score_sellers

set_page("Choice")
_SUBTITLE = page_header("Which established sellers are best overall?")

_LABELS = {
    "avg_review_score": "Average review score",
    "on_time_rate": "On-time delivery rate",
    "avg_price": "Average price (lower is better)",
    "order_volume": "Order volume",
}
_N = len(config.AHP_CRITERIA)
_PAIRS = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_SAATY_MIN, _SAATY_MAX = 1 / 9, 9.0
_TOP_N = 10


def _key(i: int, j: int) -> str:
    return f"ahp_matrix_{i}_{j}"


def _reset_matrix() -> None:
    for i, j in _PAIRS:
        st.session_state[_key(i, j)] = float(config.AHP_PAIRWISE_MATRIX[i, j])


def _matrix_editor() -> np.ndarray:
    """Number inputs for the upper triangle; the lower triangle is filled reciprocally."""
    for i, j in _PAIRS:
        st.session_state.setdefault(_key(i, j), float(config.AHP_PAIRWISE_MATRIX[i, j]))
    st.markdown(
        "**Change how much each factor matters.** Each box answers: how much more important "
        "is the first factor than the second? On the Saaty scale, 1 means equal, 3 moderately, "
        "5 strongly and 9 extremely; below 1 means the second factor matters more."
    )
    columns = st.columns(3)
    for index, (i, j) in enumerate(_PAIRS):
        columns[index % 3].number_input(
            f"{_LABELS[config.AHP_CRITERIA[i]]} vs. {_LABELS[config.AHP_CRITERIA[j]]}",
            min_value=_SAATY_MIN, max_value=_SAATY_MAX, step=0.25, format="%.3f",
            key=_key(i, j),
        )
    st.button("Reset to the configured matrix", on_click=_reset_matrix)

    matrix = np.ones((_N, _N), dtype=np.float64)
    for i, j in _PAIRS:
        value = float(st.session_state[_key(i, j)])
        matrix[i, j] = value
        matrix[j, i] = 1.0 / value
    return matrix


def _weights_chart(weights: np.ndarray) -> go.Figure:
    order = np.argsort(weights)
    fig = go.Figure(
        go.Bar(
            x=weights[order],
            y=[_LABELS[config.AHP_CRITERIA[i]] for i in order],
            orientation="h",
            marker={"color": theme.ACCENT},
            text=[f"{weights[i] * 100:.0f}%" for i in order],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        xaxis={"tickformat": ".0%", "showgrid": True, "gridcolor": theme.GRID,
               "range": [0, float(weights.max()) * 1.25]},
        yaxis={"showgrid": False},
        margin={"l": 8, "r": 40, "t": 8, "b": 8},
    )
    return fig


def _top_table(ranking: pd.DataFrame, weights: np.ndarray) -> None:
    rescored = score_sellers(ranking, weights, config.AHP_CRITERIA)
    top = rescored.head(_TOP_N).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["seller"] = top["seller_id"].map(short_id)
    st.dataframe(
        top[["rank", "seller", "seller_state", "ahp_score", "order_volume"]],
        hide_index=True,
        width="stretch",
        column_config={
            "rank": "Rank",
            "seller": "Seller",
            "seller_state": "State",
            "ahp_score": st.column_config.NumberColumn("Score", format="%.3f"),
            "order_volume": st.column_config.NumberColumn("Orders", format="%d"),
        },
    )


def _full_ranking(ranking: pd.DataFrame, weights: np.ndarray, lambda_max: float, ci: float,
                  cr: float) -> None:
    st.markdown(
        f"**Consistency maths:** λ max {lambda_max:.4f} · consistency index {ci:.4f} · "
        f"consistency ratio {cr:.4f}"
    )
    st.dataframe(
        pd.DataFrame({
            "Criterion": list(config.AHP_CRITERIA),
            "Direction": [config.AHP_CRITERION_DIRECTION[c] for c in config.AHP_CRITERIA],
            "Weight (your matrix)": weights,
            "Weight (configured)": compute_weights(config.AHP_PAIRWISE_MATRIX)[0],
        }),
        hide_index=True,
        width="stretch",
        column_config={
            "Weight (your matrix)": st.column_config.NumberColumn(format="percent"),
            "Weight (configured)": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.markdown(f"**All {len(ranking):,} established sellers (configured weights)**")
    st.dataframe(
        ranking,
        hide_index=True,
        width="stretch",
        column_order=[
            "seller_id", "seller_state", "ahp_score",
            "contribution_avg_review_score", "contribution_on_time_rate",
            "contribution_avg_price", "contribution_order_volume",
            "avg_review_score", "on_time_rate", "avg_price", "order_volume",
        ],
        column_config={
            "seller_id": "Seller",
            "seller_state": "State",
            "ahp_score": st.column_config.NumberColumn("AHP score", format="%.4f"),
            "avg_review_score": st.column_config.NumberColumn("Avg review", format="%.2f"),
            "on_time_rate": st.column_config.NumberColumn("On-time rate", format="percent"),
            "avg_price": st.column_config.NumberColumn("Avg price (BRL)", format="%.2f"),
            "order_volume": "Orders",
        },
    )


def render() -> None:
    ranking = load_report_csv("ahp_ranking.csv")
    subtitle(_SUBTITLE, explain.choice_answer(len(ranking), config.AHP_MIN_ORDERS))

    main = st.container()
    with technical():
        matrix = _matrix_editor()
        weights, lambda_max, ci, cr = compute_weights(matrix)
        st.divider()
        _full_ranking(ranking, weights, lambda_max, ci, cr)

    with main:
        kpis([
            ("Established sellers ranked", f"{len(ranking):,}", None),
            ("Minimum orders", f"{config.AHP_MIN_ORDERS}",
             "Only sellers with at least this many orders are ranked here. See the Discovery "
             "page for good sellers with fewer."),
            ("Consistency ratio", f"{cr:.3f}",
             "Below 0.10 means the pairwise judgments do not contradict each other."),
        ])
        left, right = st.columns([1, 1])
        with left:
            section("How much each factor counts")
            show(_weights_chart(weights))
        with right:
            section(f"Top {_TOP_N} sellers")
            _top_table(ranking, weights)
        caption(explain.cr_sentence(cr, config.AHP_CR_THRESHOLD))
        note(f"This ranking only includes sellers with {config.AHP_MIN_ORDERS}+ orders; the "
             "Discovery page covers good sellers with fewer.")


run_page(render)
footer(
    "Multi-criteria decision making and pairwise comparison (AHP) (Unit 4)",
    "Prescriptive",
    "Tactical",
)
