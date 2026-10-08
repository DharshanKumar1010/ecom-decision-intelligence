"""Tests for the order-level empirical-Bayes shrinkage in src.ahp (Stage 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import ahp, config
from src.ahp import (
    build_seller_criteria,
    compute_weights,
    estimate_prior_strengths,
    normalize_criteria,
    score_sellers,
    select_prior_strength,
    seller_order_table,
    shrink_toward_mean,
)
from src.config import AHP_CRITERIA, AHP_CRITERION_DIRECTION, AHP_PAIRWISE_MATRIX


def _fact(rows: list[tuple[str, float, int, int]], items_per_order: int = 1) -> pd.DataFrame:
    """Fact table from (seller, review_score, is_late, n_orders) groups.

    Every order has `items_per_order` items that share the order's review and lateness.
    """
    frames = []
    for seller, score, late, n_orders in rows:
        n_items = n_orders * items_per_order
        frames.append(
            pd.DataFrame(
                {
                    "item_key": [f"{seller}-{i}" for i in range(n_items)],
                    "order_id": [f"{seller}-o{i // items_per_order}" for i in range(n_items)],
                    "seller_id": seller,
                    "review_score": score,
                    "is_late": late,
                    "price": 100.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _dim(sellers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"seller_id": sellers, "seller_state": ["SP"] * len(sellers)})


def test_shrink_toward_mean_weights_the_seller_by_n_over_n_plus_k() -> None:
    observed = pd.Series([5.0, 4.0, 1.0])
    n = pd.Series([3, 300, 0])
    shrunk = shrink_toward_mean(observed, n, global_mean=4.0, k=10)
    assert shrunk.iloc[0] == pytest.approx((3 / 13) * 5.0 + (10 / 13) * 4.0)
    assert shrunk.iloc[1] == pytest.approx((300 / 310) * 4.0 + (10 / 310) * 4.0)
    assert shrunk.iloc[2] == pytest.approx(4.0)  # no data: the marketplace mean
    assert shrink_toward_mean(pd.Series([np.nan]), pd.Series([0]), 4.2, 5).iloc[0] == 4.2


def test_n_reviewed_counts_distinct_reviewed_orders_not_items() -> None:
    # One order with 3 items and 1 review (5 stars), one late 3-star order, one unreviewed.
    fact = pd.DataFrame(
        {
            "item_key": ["a", "b", "c", "d", "e"],
            "order_id": ["o1", "o1", "o1", "o2", "o3"],
            "seller_id": "S",
            "review_score": [5.0, 5.0, 5.0, 3.0, np.nan],
            "is_late": [0, 0, 0, 1, 0],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    row = build_seller_criteria(fact, _dim(["S"]), min_orders=1, shrink=False).iloc[0]
    assert row["order_volume"] == 5  # items sold
    assert row["n_orders"] == 3  # distinct delivered orders
    assert row["n_reviewed"] == 2  # distinct reviewed orders (items would give 4)
    assert row["avg_review_raw"] == pytest.approx(4.0)  # (5 + 3) / 2, one review per order
    assert row["on_time_rate_raw"] == pytest.approx(2 / 3)  # one of three orders late
    table = seller_order_table(fact)
    assert len(table) == 3 and table["review"].count() == 2


def test_build_seller_criteria_shrinks_with_distinct_order_counts_and_keeps_raw_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ahp, "AHP_SHRINKAGE_K_REVIEW", 10)
    monkeypatch.setattr(ahp, "AHP_SHRINKAGE_K_ON_TIME", 20)
    # "small": 3 orders of 10 items each (30 items sold, but only 3 reviews of evidence).
    fact = pd.concat(
        [
            _fact([("small", 5.0, 0, 3)], items_per_order=10),
            _fact([("big", 4.0, 1, 30), ("mid", 3.0, 0, 7)]),
        ],
        ignore_index=True,
    )
    by_id = build_seller_criteria(
        fact, _dim(["small", "big", "mid"]), min_orders=1
    ).set_index("seller_id")
    orders = seller_order_table(fact)
    global_review = float(orders["review"].mean())
    global_on_time = 1.0 - float(orders["late"].mean())

    small = by_id.loc["small"]
    assert small["order_volume"] == 30 and small["n_orders"] == 3 and small["n_reviewed"] == 3
    assert small["avg_review_raw"] == 5.0 and small["on_time_rate_raw"] == 1.0  # raw kept
    # The weight is n_orders / (n_orders + k) = 3 / 13 -- NOT 30 / 40 as items sold would give.
    assert small["avg_review_score"] == pytest.approx((3 * 5.0 + 10 * global_review) / 13)
    assert small["on_time_rate"] == pytest.approx((3 * 1.0 + 20 * global_on_time) / 23)
    assert global_review < small["avg_review_score"] < 5.0
    assert small["avg_price"] == 100.0  # untouched
    raw = build_seller_criteria(fact, _dim(["small", "big", "mid"]), 1, shrink=False)
    assert raw.set_index("seller_id").loc["small", "avg_review_score"] == 5.0


def test_three_perfect_reviewed_orders_score_below_three_hundred_orders_at_4_8() -> None:
    fact = _fact([("lucky", 5.0, 0, 3), ("proven", 4.8, 0, 300), ("filler", 3.5, 0, 50)])
    dim = _dim(["lucky", "proven", "filler"])
    weights, _, _, _ = compute_weights(AHP_PAIRWISE_MATRIX)

    def rank(frame: pd.DataFrame, shrink: bool) -> list[str]:
        criteria = build_seller_criteria(frame, dim, min_orders=1, shrink=shrink)
        # Neutralise popularity and price so only review and on-time quality differ.
        criteria["order_volume"] = 100
        criteria["avg_price"] = 100.0
        normalized = normalize_criteria(criteria, AHP_CRITERION_DIRECTION)
        return list(score_sellers(normalized, weights, AHP_CRITERIA)["seller_id"])

    assert rank(fact, shrink=False)[0] == "lucky"  # raw: three 5-star orders win
    shrunk = rank(fact, shrink=True)
    assert shrunk.index("proven") < shrunk.index("lucky")  # shrunk: 300 orders at 4.8 win
    # Padding the lucky seller's orders with extra items must not make it look stronger.
    padded = pd.concat(
        [_fact([("lucky", 5.0, 0, 3)], items_per_order=40), fact[fact["seller_id"] != "lucky"]],
        ignore_index=True,
    )
    reshrunk = rank(padded, shrink=True)
    assert reshrunk.index("proven") < reshrunk.index("lucky")


def test_shrinkage_k_is_read_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    fact = _fact([("a", 5.0, 0, 5), ("b", 3.0, 0, 5)])
    dim = _dim(["a", "b"])
    monkeypatch.setattr(ahp, "AHP_SHRINKAGE_K_REVIEW", 5)
    low_k = build_seller_criteria(fact, dim, 1).set_index("seller_id").loc["a", "avg_review_score"]
    monkeypatch.setattr(ahp, "AHP_SHRINKAGE_K_REVIEW", 50)
    high_k = build_seller_criteria(fact, dim, 1).set_index("seller_id").loc["a", "avg_review_score"]
    assert low_k > high_k > 4.0  # a stronger prior pulls the 5-star seller closer to the mean 4.0


def test_select_prior_strength_uses_the_estimate_inside_the_range_else_the_fallback() -> None:
    assert select_prior_strength(17.82) == 18
    assert select_prior_strength(25.91) == 26
    assert select_prior_strength(5.0) == 5 and select_prior_strength(30.0) == 30
    assert select_prior_strength(3.0) == config.AHP_SHRINKAGE_K_FALLBACK  # below the range
    assert select_prior_strength(41.8) == config.AHP_SHRINKAGE_K_FALLBACK  # above the range
    assert select_prior_strength(float("inf")) == config.AHP_SHRINKAGE_K_FALLBACK


def _simulated_fact(items_per_order: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED)
    n_sellers, orders = 1500, 40
    true_mean = rng.normal(4.0, 0.3, n_sellers)  # between-seller variance 0.09
    scores = rng.normal(true_mean[:, None], 1.0, (n_sellers, orders))  # within variance 1.0
    p_true = np.clip(rng.normal(0.9, 0.05, n_sellers), 0.5, 0.995)  # between variance ~0.0025
    late = (rng.random((n_sellers, orders)) > p_true[:, None]).astype(int)
    base = pd.DataFrame(
        {
            "seller_id": np.repeat([f"s{i}" for i in range(n_sellers)], orders),
            "order_id": [f"o{i}" for i in range(n_sellers * orders)],
            "review_score": scores.ravel(),
            "is_late": late.ravel(),
        }
    )
    fact = base.loc[base.index.repeat(items_per_order)].reset_index(drop=True)
    fact["item_key"] = [f"i{i}" for i in range(len(fact))]
    return fact


def test_estimate_prior_strengths_recovers_simulated_variances() -> None:
    estimates = estimate_prior_strengths(_simulated_fact(), min_orders=5)
    review = estimates["avg_review_score"]
    assert review.within_variance == pytest.approx(1.0, rel=0.05)
    assert review.between_variance == pytest.approx(0.09, rel=0.15)
    assert review.k == pytest.approx(1.0 / 0.09, rel=0.2)
    assert review.k == pytest.approx(review.within_variance / review.between_variance)
    on_time = estimates["on_time_rate"]
    assert on_time.between_variance == pytest.approx(0.0025, rel=0.3)
    assert on_time.observed_variance - on_time.noise_variance == pytest.approx(
        on_time.between_variance
    )


def test_estimate_is_unchanged_when_orders_are_padded_with_extra_items() -> None:
    one = estimate_prior_strengths(_simulated_fact(1), min_orders=5)
    three = estimate_prior_strengths(_simulated_fact(3), min_orders=5)
    for name in ("avg_review_score", "on_time_rate"):
        assert three[name].k == pytest.approx(one[name].k)
        assert three[name].sellers == one[name].sellers


def test_estimate_prior_strengths_is_infinite_when_sellers_do_not_differ() -> None:
    # Every seller alternates 3 and 5 stars: no true between-seller variance.
    fact = _fact([(f"s{i}", 4.0, 0, 10) for i in range(20)])
    fact["review_score"] = np.tile([3.0, 5.0], len(fact) // 2)
    assert estimate_prior_strengths(fact, min_orders=5)["avg_review_score"].k == float("inf")


def test_configured_k_matches_the_order_level_estimate_from_the_data() -> None:
    path = config.PROCESSED_DATA_DIR / "FactOrderItems.parquet"
    if not path.exists():
        pytest.skip("FactOrderItems.parquet not available; run run_all.py first")
    estimates = estimate_prior_strengths(pd.read_parquet(path))
    assert select_prior_strength(estimates["avg_review_score"].k) == config.AHP_SHRINKAGE_K_REVIEW
    assert select_prior_strength(estimates["on_time_rate"].k) == config.AHP_SHRINKAGE_K_ON_TIME
    low, high = config.AHP_SHRINKAGE_K_RANGE
    assert low <= config.AHP_SHRINKAGE_K_REVIEW <= high
    assert low <= config.AHP_SHRINKAGE_K_ON_TIME <= high


def test_real_data_review_and_lateness_are_constant_within_a_seller_order() -> None:
    path = config.PROCESSED_DATA_DIR / "FactOrderItems.parquet"
    if not path.exists():
        pytest.skip("FactOrderItems.parquet not available; run run_all.py first")
    fact = pd.read_parquet(path, columns=["seller_id", "order_id", "review_score", "is_late"])
    grouped = fact.groupby(["seller_id", "order_id"])
    assert (grouped["review_score"].nunique() <= 1).all()
    assert (grouped["is_late"].nunique() == 1).all()
