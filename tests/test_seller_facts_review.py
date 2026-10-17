"""The expert facts' `avg_review` is one definition everywhere: mean of one review per order."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.ahp import seller_order_table
from src.discovery import build_seller_facts

_FACT_PATH = config.PROCESSED_DATA_DIR / "FactOrderItems.parquet"
_FACTS_PATH = config.PROCESSED_DATA_DIR / "SellerFacts.parquet"
_RANKING_PATH = config.REPORTS_DIR / "ahp_ranking.csv"


def test_build_seller_facts_averages_one_review_per_order_not_per_item() -> None:
    # Seller S: a 3-item order reviewed 5, a 1-item order reviewed 3, an unreviewed order.
    # Seller T: no reviewed order at all.
    fact = pd.DataFrame(
        {
            "item_key": ["a", "b", "c", "d", "e", "f"],
            "order_id": ["o1", "o1", "o1", "o2", "o3", "o4"],
            "seller_id": ["S", "S", "S", "S", "S", "T"],
            "review_score": [5.0, 5.0, 5.0, 3.0, np.nan, np.nan],
            "is_late": [0, 0, 0, 0, 0, 0],
        }
    )
    late = pd.DataFrame({"item_key": fact["item_key"], "late_risk": 0.1})
    quadrants = pd.DataFrame(
        {"seller_id": ["S"], "ahp_score": [0.7], "quadrant": ["Star"], "confidence": ["low"]}
    )
    facts = build_seller_facts(quadrants, late, fact).set_index("seller_id")
    assert facts.loc["S", "avg_review"] == pytest.approx(4.0)  # (5 + 3) / 2; per item: 4.5
    assert pd.isna(facts.loc["T", "avg_review"])  # no reviews stays missing
    assert facts.loc["S", "order_volume"] == 5  # items sold is still a count of items


def test_real_avg_review_fact_equals_the_per_order_raw_mean_for_every_seller() -> None:
    if not (_FACT_PATH.exists() and _FACTS_PATH.exists()):
        pytest.skip("pipeline outputs not generated; run the pipeline first")
    fact = pd.read_parquet(_FACT_PATH)
    facts = pd.read_parquet(_FACTS_PATH).set_index("seller_id")
    expected = seller_order_table(fact).groupby("seller_id")["review"].mean()
    assert len(facts) == fact["seller_id"].nunique()  # ALL sellers, not only AHP-eligible
    assert set(facts.index) == set(expected.index)
    pd.testing.assert_series_equal(
        facts["avg_review"].sort_index(), expected.sort_index(), check_names=False
    )
    assert facts["avg_review"].isna().sum() == expected.isna().sum()  # no-review handling kept


def test_choice_raw_review_and_expert_fact_are_the_same_number() -> None:
    if not (_FACTS_PATH.exists() and _RANKING_PATH.exists()):
        pytest.skip("pipeline outputs not generated; run the pipeline first")
    facts = pd.read_parquet(_FACTS_PATH).set_index("seller_id")
    ranking = pd.read_csv(_RANKING_PATH).set_index("seller_id")
    shared = ranking.index.intersection(facts.index)
    assert len(shared) == len(ranking)
    diff = (ranking.loc[shared, "avg_review_raw"] - facts.loc[shared, "avg_review"]).abs()
    assert diff.max() < 1e-9


def test_r10_review_cutoff_stays_at_the_median_its_text_promises() -> None:
    """R10 is percentile-intent ("at-or-above-median reviews"); R01/R04/R07 are fixed levels."""
    if not _FACTS_PATH.exists():
        pytest.skip("pipeline outputs not generated; run the pipeline first")
    from src.expert import KnowledgeBase

    rules = {r.id: r for r in KnowledgeBase.from_yaml(config.RULES_DIR / "seller_rules.yaml").rules}
    cutoff = next(c.value for c in rules["R10"].conditions if c.fact == "avg_review")
    reviews = pd.read_parquet(_FACTS_PATH)["avg_review"].dropna()
    share_below = float((reviews < cutoff).mean())
    assert abs(share_below - 0.5) < 0.01, f"R10 cutoff {cutoff} is the {share_below:.1%} percentile"


def _evidence_fixture() -> pd.DataFrame:
    # Seller A: one order with 3 items and one review.
    # Seller B: 4 items in 2 orders, only one of them reviewed.
    # Seller C: 2 orders, none reviewed.
    return pd.DataFrame(
        {
            "item_key": list("abcdefghi"),
            "order_id": ["o1", "o1", "o1", "o2", "o2", "o3", "o3", "o4", "o5"],
            "seller_id": ["A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "review_score": [4.0, 4.0, 4.0, 5.0, 5.0, np.nan, np.nan, np.nan, np.nan],
            "is_late": [0] * 9,
        }
    )


def test_build_seller_facts_counts_distinct_orders_and_reviewed_orders() -> None:
    fact = _evidence_fixture()
    late = pd.DataFrame({"item_key": fact["item_key"], "late_risk": 0.1})
    quadrants = pd.DataFrame(columns=["seller_id", "ahp_score", "quadrant", "confidence"])
    facts = build_seller_facts(quadrants, late, fact).set_index("seller_id")
    assert facts.loc["A", ["order_volume", "n_orders", "n_reviewed"]].tolist() == [3, 1, 1]
    assert facts.loc["B", ["order_volume", "n_orders", "n_reviewed"]].tolist() == [4, 2, 1]
    assert facts.loc["C", ["order_volume", "n_orders", "n_reviewed"]].tolist() == [2, 2, 0]
    assert facts[["n_orders", "n_reviewed"]].dtypes.astype(str).tolist() == ["int64", "int64"]
    # existing columns keep their names and order; the two new ones are appended
    assert list(facts.reset_index().columns) == [
        "seller_id", "ahp_score", "avg_late_risk", "avg_review", "order_volume", "late_rate",
        "is_hidden_gem", "n_orders", "n_reviewed",
    ]


def test_real_evidence_counts_equal_the_seller_order_table_for_every_seller() -> None:
    if not (_FACT_PATH.exists() and _FACTS_PATH.exists()):
        pytest.skip("pipeline outputs not generated; run the pipeline first")
    fact = pd.read_parquet(_FACT_PATH)
    facts = pd.read_parquet(_FACTS_PATH).set_index("seller_id")
    table = seller_order_table(fact).groupby("seller_id")
    expected_orders = table["order_id"].size()
    expected_reviewed = table["review"].count()
    assert facts[["n_orders", "n_reviewed"]].notna().all().all()
    assert (facts["n_orders"].sort_index() == expected_orders.sort_index()).all()
    assert (facts["n_reviewed"].sort_index() == expected_reviewed.sort_index()).all()
    assert (facts["n_orders"] >= 1).all()
    assert (facts["n_reviewed"] <= facts["n_orders"]).all()
    assert (facts["n_orders"] <= facts["order_volume"]).all()
    assert facts["n_orders"].sum() == len(seller_order_table(fact))
