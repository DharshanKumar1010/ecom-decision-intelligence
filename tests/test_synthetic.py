"""Tests for src.synthetic: causal consistency and seed reproducibility."""

from __future__ import annotations

import pandas as pd

from src.synthetic import (
    N_CLICKSTREAM_SESSIONS,
    N_REACH_CART,
    N_REACH_CHECKOUT,
    N_REACH_PURCHASE,
    generate_call_transcripts,
    generate_clickstream,
)


def _fake_fact() -> pd.DataFrame:
    return pd.DataFrame({"product_id": [f"P{i:03d}" for i in range(50)]})


def test_clickstream_row_count_and_stage_counts() -> None:
    events = generate_clickstream(_fake_fact(), seed=42)
    assert len(events) == 100_000
    counts = events.groupby("event_type")["session_id"].nunique()
    assert counts["page_view"] == N_CLICKSTREAM_SESSIONS
    assert counts["add_to_cart"] == N_REACH_CART
    assert counts["checkout"] == N_REACH_CHECKOUT
    assert counts["purchase"] == N_REACH_PURCHASE


def test_clickstream_causal_consistency() -> None:
    events = generate_clickstream(_fake_fact(), seed=42)
    for _, group in events.groupby("session_id"):
        ordered = group.sort_values("event_time")
        assert ordered["event_time"].is_monotonic_increasing
        assert ordered["event_time"].duplicated().sum() == 0
        stages_present = set(ordered["event_type"])
        if "purchase" in stages_present:
            assert {"checkout", "add_to_cart", "page_view"} <= stages_present
        if "checkout" in stages_present:
            assert {"add_to_cart", "page_view"} <= stages_present
        if "add_to_cart" in stages_present:
            assert "page_view" in stages_present


def test_clickstream_reproducibility() -> None:
    e1 = generate_clickstream(_fake_fact(), seed=42)
    e2 = generate_clickstream(_fake_fact(), seed=42)
    pd.testing.assert_frame_equal(e1, e2)


def test_call_transcripts_shape() -> None:
    calls = generate_call_transcripts(seed=42)
    assert len(calls) == 300
    assert set(calls.columns) == {
        "call_id",
        "agent_id",
        "transcript",
        "hold_seconds",
        "silence_pct",
    }
    assert calls["silence_pct"].between(0, 1).all()
    assert calls["hold_seconds"].gt(0).all()


def test_call_transcripts_reproducibility() -> None:
    c1 = generate_call_transcripts(seed=42)
    c2 = generate_call_transcripts(seed=42)
    pd.testing.assert_frame_equal(c1, c2)
