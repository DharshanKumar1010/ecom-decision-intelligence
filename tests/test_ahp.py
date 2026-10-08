"""Tests for src.ahp: eigenvector weights, consistency ratio, reciprocity, scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ahp import build_seller_criteria, compute_weights, normalize_criteria, score_sellers


def test_compute_weights_consistent_matrix_gives_low_cr_and_correct_weights() -> None:
    # Perfectly consistent by construction: true weight ratio a:b:c = 4:2:1.
    matrix = np.array(
        [
            [1.0, 2.0, 4.0],
            [1 / 2, 1.0, 2.0],
            [1 / 4, 1 / 2, 1.0],
        ]
    )
    weights, _, _, cr = compute_weights(matrix)
    assert cr == pytest.approx(0.0, abs=1e-6)
    assert weights == pytest.approx([4 / 7, 2 / 7, 1 / 7], abs=1e-6)
    assert weights.sum() == pytest.approx(1.0)


def test_compute_weights_inconsistent_matrix_gives_high_cr() -> None:
    # Classic maximally-inconsistent 3x3: a >> b, b >> c, but c >> a.
    matrix = np.array(
        [
            [1.0, 9.0, 1.0],
            [1 / 9, 1.0, 9.0],
            [1.0, 1 / 9, 1.0],
        ]
    )
    _, _, _, cr = compute_weights(matrix)
    assert cr > 0.10


def test_compute_weights_rejects_non_reciprocal_matrix() -> None:
    bad_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # should be [[1, 2], [0.5, 1]]
    with pytest.raises(AssertionError):
        compute_weights(bad_matrix)


def test_build_seller_criteria_aggregation_and_min_orders_filter() -> None:
    fact = pd.DataFrame(
        {
            "item_key": ["i1", "i2", "i3", "i4", "i5"],
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "seller_id": ["S1", "S1", "S1", "S2", "S2"],
            "review_score": [5.0, 4.0, 3.0, 5.0, 5.0],
            "is_late": [0, 0, 1, 0, 0],
            "price": [100.0, 200.0, 300.0, 50.0, 50.0],
        }
    )
    dim_seller = pd.DataFrame({"seller_id": ["S1", "S2"], "seller_state": ["SP", "RJ"]})

    criteria = build_seller_criteria(fact, dim_seller, min_orders=3, shrink=False)
    assert set(criteria["seller_id"]) == {"S1"}  # S2 has only 2 orders, below min_orders=3

    row = criteria.loc[criteria["seller_id"] == "S1"].iloc[0]
    assert row["avg_review_score"] == pytest.approx(4.0)
    assert row["on_time_rate"] == pytest.approx(2 / 3)
    assert row["avg_price"] == pytest.approx(200.0)
    assert row["order_volume"] == 3


def test_normalize_criteria_min_max_and_cost_inversion_preserving_raw_columns() -> None:
    df = pd.DataFrame(
        {
            "seller_id": ["A", "B", "C"],
            "avg_review_score": [3.0, 4.0, 5.0],
            "avg_price": [10.0, 20.0, 30.0],
        }
    )
    direction_map = {"avg_review_score": "benefit", "avg_price": "cost"}
    normalized = normalize_criteria(df, direction_map)

    assert normalized["norm_avg_review_score"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert normalized["norm_avg_price"].tolist() == pytest.approx([1.0, 0.5, 0.0])
    # Raw criterion columns must survive unchanged for callers like src.discovery
    # that need the seller's actual order count/price, not a normalized proxy.
    assert normalized["avg_price"].tolist() == [10.0, 20.0, 30.0]


def test_score_sellers_weighted_sum_and_sort_order() -> None:
    normalized = pd.DataFrame(
        {
            "seller_id": ["A", "B"],
            "norm_x": [1.0, 0.0],
            "norm_y": [0.0, 1.0],
        }
    )
    weights = np.array([0.7, 0.3])
    scored = score_sellers(normalized, weights, ("x", "y"))

    by_seller = scored.set_index("seller_id")
    assert by_seller.loc["A", "ahp_score"] == pytest.approx(0.7)
    assert by_seller.loc["B", "ahp_score"] == pytest.approx(0.3)
    assert scored.iloc[0]["seller_id"] == "A"  # sorted by ahp_score descending
