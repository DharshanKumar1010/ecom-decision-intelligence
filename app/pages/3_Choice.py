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
    ITEMS_SOLD_NOTE,
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
    table,
    technical,
)
from app._format import fmt_int, fmt_num
from src import config, explain
from src.ahp import compute_weights, score_sellers

set_page("Choice")
_SUBTITLE = page_header("Which established sellers are best overall?")

_LABELS = {
    "avg_review_score": "Average review score",
    "on_time_rate": "On-time delivery rate",
    "avg_price": "Average price (lower is better)",
    "order_volume": "Items sold",
}
_N = len(config.AHP_CRITERIA)
_PAIRS = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_SAATY_MIN, _SAATY_MAX = 1 / 9, 9.0
_TOP_N = 10
_DIRECTION = {"benefit": "Higher is better", "cost": "Lower is better"}
# The pool this page scores: sellers with at least this many items sold. Discovery scores a
# larger pool (config.DISCOVERY_MIN_ORDERS), so the same seller's number differs slightly.
_SCORE_LABEL = f"AHP score ({config.AHP_MIN_ORDERS}+ items pool)"


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
    top["seller_id"] = top["seller_id"].map(short_id)
    table(
        top[["rank", "seller_id", "seller_state", "ahp_score", "order_volume"]],
        {"ahp_score": "num3", "order_volume": "int"},
        overrides={"ahp_score": _SCORE_LABEL},
    )


def _full_ranking(ranking: pd.DataFrame, weights: np.ndarray, lambda_max: float, ci: float,
                  cr: float) -> None:
    st.markdown(
        f"**Consistency maths:** λ max {fmt_num(lambda_max, 4)} · consistency index "
        f"{fmt_num(ci, 4)} · consistency ratio {fmt_num(cr, 4)}"
    )
    table(
        pd.DataFrame({
            "Criterion": [_LABELS[c] for c in config.AHP_CRITERIA],
            "Direction": [_DIRECTION[config.AHP_CRITERION_DIRECTION[c]]
                          for c in config.AHP_CRITERIA],
            "Weight (your matrix)": weights,
            "Weight (configured)": compute_weights(config.AHP_PAIRWISE_MATRIX)[0],
        }),
        {"Weight (your matrix)": "pct", "Weight (configured)": "pct"},
    )
    note(
        "Average review and on-time rate are pulled toward the marketplace mean before "
        "scoring (empirical-Bayes shrinkage counted in distinct orders: prior strength "
        f"{config.AHP_SHRINKAGE_K_REVIEW} for reviews and {config.AHP_SHRINKAGE_K_ON_TIME} "
        "for on-time rate), so a few perfect reviews cannot outrank a long, strong record. "
        "The table lists each seller's raw average review and on-time rate (one value per "
        "order) with the number of reviewed and delivered orders behind them, then the shrunk "
        "values that are used for scoring (headed shrunk), then the four contributions that "
        "add up to the score. "
        + ITEMS_SOLD_NOTE
    )
    note(
        "The same seller can show a slightly different score here and on the Discovery page: "
        "each page scales every criterion from its lowest to its highest value within its own "
        f"pool of sellers ({config.AHP_MIN_ORDERS}+ items sold here, "
        f"{config.DISCOVERY_MIN_ORDERS}+ on Discovery). The shrunk review and on-time "
        "values themselves are the same on both."
    )
    st.markdown(f"**All {fmt_int(len(ranking))} established sellers (configured weights)**")
    table(
        ranking[[
            "seller_id", "seller_state", "ahp_score", "order_volume", "n_reviewed", "n_orders",
            "avg_review_raw", "on_time_rate_raw", "avg_review_score", "on_time_rate", "avg_price",
            "contribution_avg_review_score", "contribution_on_time_rate",
            "contribution_avg_price", "contribution_order_volume",
        ]],
        {
            "ahp_score": "num4", "order_volume": "int", "n_reviewed": "int", "n_orders": "int",
            "avg_review_raw": "num2", "on_time_rate_raw": "pct", "avg_review_score": "num3",
            "on_time_rate": "pct", "avg_price": "num2",
            "contribution_avg_review_score": "num4", "contribution_on_time_rate": "num4",
            "contribution_avg_price": "num4", "contribution_order_volume": "num4",
        },
        overrides={"ahp_score": _SCORE_LABEL},
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
            ("Established sellers ranked", fmt_int(len(ranking)), None),
            ("Minimum items sold", f"{config.AHP_MIN_ORDERS}",
             "Only sellers with at least this many items sold are ranked here. See the Discovery "
             "page for good sellers with fewer."),
            ("Consistency ratio", fmt_num(cr, 3),
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
        note(f"This ranking only includes sellers with {config.AHP_MIN_ORDERS}+ items sold; the "
             "Discovery page covers good sellers with fewer.")


run_page(render)
footer(
    "Multi-criteria decision making and pairwise comparison (AHP) (Unit 4)",
    "Prescriptive",
    "Tactical",
)
