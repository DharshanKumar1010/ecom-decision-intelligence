"""Tests for src.clickstream: funnel monotonicity and metric ranges."""

from __future__ import annotations

import pandas as pd
import pytest

from src.clickstream import bounce_rate, cart_abandonment_rate, funnel_by_device, funnel_conversion
from src.synthetic import generate_clickstream


def _fake_fact() -> pd.DataFrame:
    return pd.DataFrame({"product_id": ["P1", "P2", "P3"]})


def _toy_events() -> pd.DataFrame:
    rows = [
        ("s1", "page_view", pd.Timestamp("2020-01-01 00:00:00")),
        ("s2", "page_view", pd.Timestamp("2020-01-01 00:00:00")),
        ("s2", "add_to_cart", pd.Timestamp("2020-01-01 00:01:00")),
        ("s3", "page_view", pd.Timestamp("2020-01-01 00:00:00")),
        ("s3", "add_to_cart", pd.Timestamp("2020-01-01 00:01:00")),
        ("s3", "checkout", pd.Timestamp("2020-01-01 00:02:00")),
        ("s3", "purchase", pd.Timestamp("2020-01-01 00:03:00")),
    ]
    return pd.DataFrame(rows, columns=["session_id", "event_type", "event_time"])


def test_bounce_rate_known_value() -> None:
    assert bounce_rate(_toy_events()) == pytest.approx(1 / 3)


def test_cart_abandonment_rate_known_value() -> None:
    assert cart_abandonment_rate(_toy_events()) == pytest.approx(1 / 2)


def test_funnel_conversion_known_counts() -> None:
    result = funnel_conversion(_toy_events())
    assert result.loc["page_view", "sessions"] == 3
    assert result.loc["add_to_cart", "sessions"] == 2
    assert result.loc["checkout", "sessions"] == 1
    assert result.loc["purchase", "sessions"] == 1


def test_funnel_monotonic_on_generated_data() -> None:
    events = generate_clickstream(_fake_fact(), seed=42)
    sessions = funnel_conversion(events)["sessions"]
    assert (sessions.diff().dropna() <= 0).all()


def test_bounce_and_abandonment_in_unit_interval() -> None:
    events = generate_clickstream(_fake_fact(), seed=42)
    assert 0.0 <= bounce_rate(events) <= 1.0
    assert 0.0 <= cart_abandonment_rate(events) <= 1.0


def test_funnel_by_device_monotonic_rows() -> None:
    events = generate_clickstream(_fake_fact(), seed=42)
    result = funnel_by_device(events)
    assert set(result.columns) == {"page_view", "add_to_cart", "checkout", "purchase"}
    for _, row in result.iterrows():
        assert (row.diff().dropna() <= 0).all()
