"""Tests for src.discovery: quadrant classification, confidence flag, hidden gems, data contract."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DISCOVERY_CONFIDENCE_MIN_ORDERS
from src.discovery import (
    TIER_ACTIONABLE,
    TIER_WATCHLIST,
    build_seller_facts,
    compute_quadrants,
    confidence_flag,
    hidden_gem_tier,
    top_hidden_gems,
)


def test_confidence_flag_boundary_14_vs_15_orders() -> None:
    assert confidence_flag(DISCOVERY_CONFIDENCE_MIN_ORDERS - 1) == "low"  # 14
    assert confidence_flag(DISCOVERY_CONFIDENCE_MIN_ORDERS) == "normal"  # 15


def test_compute_quadrants_assigns_all_four_labels_on_known_fixture() -> None:
    ahp_scores = pd.Series({"star": 0.9, "gem": 0.9, "overrated": 0.2, "poor": 0.2})
    order_counts = pd.Series({"star": 100, "gem": 5, "overrated": 100, "poor": 5})

    result = compute_quadrants(
        ahp_scores, order_counts, quality_threshold=0.5, popularity_threshold=50
    )
    labels = result.set_index("seller_id")["quadrant"]

    assert labels["star"] == "Star"
    assert labels["gem"] == "Hidden Gem"
    assert labels["overrated"] == "Overrated"
    assert labels["poor"] == "Overlooked-Low-Quality"


def test_compute_quadrants_threshold_override_changes_classification() -> None:
    ahp_scores = pd.Series({"a": 0.6, "b": 0.4})
    order_counts = pd.Series({"a": 10, "b": 10})

    lenient = compute_quadrants(ahp_scores, order_counts, quality_threshold=0.5,
                                 popularity_threshold=5)
    assert lenient.set_index("seller_id").loc["a", "quadrant"] == "Star"

    strict = compute_quadrants(ahp_scores, order_counts, quality_threshold=0.9,
                                popularity_threshold=5)
    assert strict.set_index("seller_id").loc["a", "quadrant"] == "Overrated"


def test_compute_quadrants_defaults_to_median_when_no_override_given() -> None:
    ahp_scores = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    order_counts = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})

    result = compute_quadrants(ahp_scores, order_counts)  # median (2.0) not hard-coded
    labels = result.set_index("seller_id")["quadrant"]

    assert labels["a"] == "Overlooked-Low-Quality"  # below median on both axes
    assert labels["b"] == "Star"  # exactly at median counts as high (inclusive >=)
    assert labels["c"] == "Star"  # above median on both axes


def test_top_hidden_gems_never_includes_other_quadrants() -> None:
    quadrant_df = pd.DataFrame(
        {
            "seller_id": ["a", "b", "c", "d"],
            "ahp_score": [0.9, 0.8, 0.95, 0.1],
            "order_volume": [5, 5, 100, 5],
            "quadrant": ["Hidden Gem", "Hidden Gem", "Star", "Overlooked-Low-Quality"],
            "confidence": ["low", "low", "normal", "low"],
        }
    )
    gems = top_hidden_gems(quadrant_df, n=10)
    assert set(gems["quadrant"]) == {"Hidden Gem"}
    assert list(gems["seller_id"]) == ["a", "b"]  # sorted by ahp_score descending


def test_top_hidden_gems_respects_n() -> None:
    quadrant_df = pd.DataFrame(
        {
            "seller_id": [f"s{i}" for i in range(5)],
            "ahp_score": [0.9, 0.8, 0.7, 0.6, 0.5],
            "order_volume": [5] * 5,
            "quadrant": ["Hidden Gem"] * 5,
            "confidence": ["low"] * 5,
        }
    )
    assert len(top_hidden_gems(quadrant_df, n=3)) == 3


def _gem_frame() -> pd.DataFrame:
    n = DISCOVERY_CONFIDENCE_MIN_ORDERS
    return pd.DataFrame(
        {
            "seller_id": ["low_hi", "low_mid", "ok_lo", "ok_hi", "tie_a", "tie_b", "star"],
            "ahp_score": [0.99, 0.90, 0.60, 0.80, 0.70, 0.70, 0.95],
            "order_volume": [5, n - 1, n, n + 2, n + 1, n, 100],
            "quadrant": ["Hidden Gem"] * 6 + ["Star"],
            "confidence": ["low", "low", "normal", "normal", "normal", "normal", "normal"],
        }
    )


def test_default_top_hidden_gems_is_unchanged_and_has_no_tier_column() -> None:
    gems = top_hidden_gems(_gem_frame(), n=10)
    assert "tier" not in gems.columns
    assert list(gems["seller_id"]) == ["low_hi", "low_mid", "ok_hi", "tie_a", "tie_b", "ok_lo"]


def test_tiered_top_hidden_gems_lists_actionable_first_ranked_by_score_then_orders() -> None:
    gems = top_hidden_gems(_gem_frame(), n=10, tiered=True)
    assert list(gems["tier"]) == [TIER_ACTIONABLE] * 4 + [TIER_WATCHLIST] * 2
    # Actionable by AHP descending; the 0.70 tie is broken by more orders first.
    assert list(gems["seller_id"][:4]) == ["ok_hi", "tie_a", "tie_b", "ok_lo"]
    assert list(gems["seller_id"][4:]) == ["low_hi", "low_mid"]


def test_tier_boundary_at_14_versus_15_orders() -> None:
    n = DISCOVERY_CONFIDENCE_MIN_ORDERS
    quadrants = compute_quadrants(
        pd.Series({"a": 0.9, "b": 0.9, "c": 0.1, "d": 0.1}),
        pd.Series({"a": n - 1, "b": n, "c": 100, "d": 100}),
        quality_threshold=0.5,
        popularity_threshold=50,
    )
    gems = top_hidden_gems(quadrants, tiered=True).set_index("seller_id")
    assert gems.loc["a", "tier"] == TIER_WATCHLIST  # 14 orders
    assert gems.loc["b", "tier"] == TIER_ACTIONABLE  # 15 orders
    assert hidden_gem_tier(confidence_flag(n - 1)) == TIER_WATCHLIST
    assert hidden_gem_tier(confidence_flag(n)) == TIER_ACTIONABLE


def test_tiered_top_hidden_gems_never_includes_other_quadrants_and_n_is_per_tier() -> None:
    gems = top_hidden_gems(_gem_frame(), n=1, tiered=True)
    assert set(gems["quadrant"]) == {"Hidden Gem"}
    assert "star" not in set(gems["seller_id"])  # the high-scoring Star is never returned
    assert list(gems["tier"]) == [TIER_ACTIONABLE, TIER_WATCHLIST]
    assert list(gems["seller_id"]) == ["ok_hi", "low_hi"]


def test_build_seller_facts_marks_missing_ahp_score_and_hidden_gem_explicitly() -> None:
    fact = pd.DataFrame(
        {
            "item_key": ["i1", "i2", "i3"],
            "seller_id": ["S1", "S1", "S2"],
            "review_score": [5.0, 4.0, 3.0],
            "is_late": [0, 1, 0],
        }
    )
    late_predictions = pd.DataFrame(
        {"item_key": ["i1", "i2", "i3"], "late_risk": [0.2, 0.3, 0.5]}
    )
    # S2 has only 1 order (below DISCOVERY_MIN_ORDERS) so it is absent from
    # quadrant_df, the AHP-eligible sellers table.
    quadrant_df = pd.DataFrame(
        {
            "seller_id": ["S1"],
            "ahp_score": [0.8],
            "quadrant": ["Hidden Gem"],
            "confidence": ["low"],
        }
    )

    seller_facts = build_seller_facts(quadrant_df, late_predictions, fact)
    by_seller = seller_facts.set_index("seller_id")

    assert by_seller.loc["S1", "ahp_score"] == pytest.approx(0.8)
    assert bool(by_seller.loc["S1", "is_hidden_gem"]) is True
    assert by_seller.loc["S1", "avg_late_risk"] == pytest.approx(0.25)
    assert by_seller.loc["S1", "order_volume"] == 2

    assert pd.isna(by_seller.loc["S2", "ahp_score"])
    assert bool(by_seller.loc["S2", "is_hidden_gem"]) is False
    assert by_seller.loc["S2", "avg_late_risk"] == pytest.approx(0.5)
