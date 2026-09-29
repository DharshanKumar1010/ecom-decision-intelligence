"""Tests for src.sentiment: process stages and per-agent aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.sentiment import (
    aggregate_by_agent,
    classify,
    classify_calls,
    extract_features,
    preprocess,
    score,
)


def test_preprocess_removes_stopwords_and_punctuation() -> None:
    tokens = preprocess("This is NOT a great product, is it?")
    assert "is" not in tokens
    assert "a" not in tokens
    assert all(char not in "".join(tokens) for char in ",?")
    assert "not" in tokens


def test_preprocess_lemmatizes_simple_suffixes() -> None:
    tokens = preprocess("The boxes were delivered quickly")
    assert "box" in tokens
    assert "deliver" in tokens


def test_extract_features_counts_and_negation() -> None:
    features = extract_features(["not", "great", "product"])
    assert features["token_count"] == 3
    assert features["negation_count"] == 1
    assert features["has_negation"] is True

    no_negation = extract_features(["great", "product"])
    assert no_negation["negation_count"] == 0
    assert no_negation["has_negation"] is False


def test_classify_known_positive_sentence() -> None:
    compound = classify("This was a wonderful experience, thank you so much!")
    assert compound >= 0.05
    assert score(compound) == "Positive"


def test_classify_known_negative_sentence() -> None:
    compound = classify("This is terrible, I am furious and want a refund immediately.")
    assert compound <= -0.05
    assert score(compound) == "Negative"


def test_classify_known_neutral_sentence() -> None:
    compound = classify("I am calling to check the status of my order shipment.")
    assert score(compound) == "Neutral"


def test_score_boundaries() -> None:
    assert score(0.05) == "Positive"
    assert score(-0.05) == "Negative"
    assert score(0.0) == "Neutral"
    assert score(0.049) == "Neutral"
    assert score(-0.049) == "Neutral"


def test_aggregate_by_agent_hand_computed() -> None:
    calls = pd.DataFrame(
        {
            "call_id": ["C1", "C2", "C3"],
            "agent_id": ["AG01", "AG01", "AG02"],
            "compound": [0.8, -0.2, 0.0],
            "sentiment_label": ["Positive", "Negative", "Neutral"],
            "hold_seconds": [60.0, 120.0, 90.0],
            "silence_pct": [0.05, 0.15, 0.10],
        }
    )
    summary = aggregate_by_agent(calls)

    assert summary.loc["AG01", "mean_sentiment"] == pytest.approx(0.3)
    assert summary.loc["AG01", "mean_hold_seconds"] == pytest.approx(90.0)
    assert summary.loc["AG01", "mean_silence_pct"] == pytest.approx(0.10)
    assert summary.loc["AG01", "negative_call_share"] == pytest.approx(0.5)
    assert summary.loc["AG01", "call_count"] == 2

    assert summary.loc["AG02", "mean_sentiment"] == pytest.approx(0.0)
    assert summary.loc["AG02", "negative_call_share"] == pytest.approx(0.0)
    assert summary.loc["AG02", "call_count"] == 1


def test_classify_calls_adds_expected_columns() -> None:
    transcripts = pd.DataFrame(
        {
            "call_id": ["C1"],
            "agent_id": ["AG01"],
            "transcript": ["Thank you, everything was perfect!"],
            "hold_seconds": [45.0],
            "silence_pct": [0.02],
        }
    )
    labeled = classify_calls(transcripts)
    assert "compound" in labeled.columns
    assert "sentiment_label" in labeled.columns
    assert labeled.loc[0, "sentiment_label"] == "Positive"
