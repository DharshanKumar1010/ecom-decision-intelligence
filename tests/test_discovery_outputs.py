"""Consistency of the real discovery outputs (counts are computed from the files, never typed)."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src import config
from src.discovery import TIER_ACTIONABLE, TIER_WATCHLIST, top_hidden_gems

_QUADRANTS_PATH = config.REPORTS_DIR / "discovery_quadrants.csv"
_FACTS_PATH = config.PROCESSED_DATA_DIR / "SellerFacts.parquet"
_SUMMARY_PATH = config.REPORTS_DIR / "discovery_summary.md"


@pytest.fixture(scope="module")
def quadrants() -> pd.DataFrame:
    if not _QUADRANTS_PATH.exists():
        pytest.skip("discovery_quadrants.csv not generated; run the pipeline first")
    return pd.read_csv(_QUADRANTS_PATH)


def test_is_hidden_gem_fact_equals_the_hidden_gem_quadrant(quadrants: pd.DataFrame) -> None:
    if not _FACTS_PATH.exists():
        pytest.skip("SellerFacts.parquet not generated")
    facts = pd.read_parquet(_FACTS_PATH)
    gems = set(quadrants.loc[quadrants["quadrant"] == "Hidden Gem", "seller_id"])
    assert set(facts.loc[facts["is_hidden_gem"], "seller_id"]) == gems
    assert int(facts["is_hidden_gem"].sum()) == len(gems) > 0


def test_tiered_gems_are_only_hidden_gems_and_tiers_follow_confidence(
    quadrants: pd.DataFrame,
) -> None:
    gems = quadrants[quadrants["quadrant"] == "Hidden Gem"]
    tiered = top_hidden_gems(quadrants, n=len(quadrants), tiered=True)
    assert set(tiered["seller_id"]) == set(gems["seller_id"])  # all gems, nothing else
    assert (tiered["quadrant"] == "Hidden Gem").all()
    tier_of = {"normal": TIER_ACTIONABLE, "low": TIER_WATCHLIST}
    assert (tiered["tier"] == tiered["confidence"].map(tier_of)).all()
    first_watch = (tiered["tier"] == TIER_WATCHLIST).idxmax()
    assert (tiered.loc[:first_watch - 1, "tier"] == TIER_ACTIONABLE).all()  # Actionable first


def test_quadrant_labels_follow_the_median_split_and_confidence_the_threshold(
    quadrants: pd.DataFrame,
) -> None:
    quality, popularity = quadrants["ahp_score"].median(), quadrants["order_volume"].median()
    high_q = quadrants["ahp_score"] >= quality
    high_p = quadrants["order_volume"] >= popularity
    expected = pd.Series("Overlooked-Low-Quality", index=quadrants.index)
    expected[high_q & high_p] = "Star"
    expected[high_q & ~high_p] = "Hidden Gem"
    expected[~high_q & high_p] = "Overrated"
    assert (quadrants["quadrant"] == expected).all()
    low = quadrants["order_volume"] < config.DISCOVERY_CONFIDENCE_MIN_ORDERS
    assert (quadrants["confidence"] == low.map({True: "low", False: "normal"})).all()


def test_summary_counts_equal_the_quadrant_file(quadrants: pd.DataFrame) -> None:
    if not _SUMMARY_PATH.exists():
        pytest.skip("discovery_summary.md not generated")
    text = _SUMMARY_PATH.read_text(encoding="utf-8")
    for label, count in quadrants["quadrant"].value_counts().items():
        match = re.search(rf"^- {re.escape(str(label))}: (\d+)$", text, re.MULTILINE)
        assert match and int(match.group(1)) == count, label
    for label, count in quadrants["confidence"].value_counts().items():
        match = re.search(rf"^- {label}: (\d+)$", text, re.MULTILINE)
        assert match and int(match.group(1)) == count, label
