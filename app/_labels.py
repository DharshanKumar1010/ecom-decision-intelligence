"""Friendly names shown to people: table headers and evidence-tier names.

One mapping from the raw column names the pipeline writes (`order_volume`, `ahp_score`, ...)
to the headers every Technical-details table shows. Raw names appear only in help text and in
CSV downloads, which keep the pipeline's column names (so a file stays joinable), except that
the evidence tier uses its user-facing name. Presentation only.
"""

from __future__ import annotations

import pandas as pd

COLUMN_LABELS: dict[str, str] = {
    # sellers
    "rank": "Rank",
    "seller_id": "Seller",
    "seller_state": "State",
    "order_volume": "Items sold",
    "n_reviewed": "Reviewed orders",
    "n_orders": "Delivered orders",
    "avg_review": "Average review (per order)",
    "avg_review_raw": "Average review (per order)",
    "on_time_rate_raw": "On-time rate (per order)",
    "avg_review_score": "Average review (shrunk)",
    "on_time_rate": "On-time rate (shrunk)",
    "avg_price": "Average price (BRL)",
    "avg_late_risk": "Predicted late risk",
    "late_rate": "Late-delivery rate",
    "is_hidden_gem": "Hidden gem",
    "ahp_score": "Quality score",
    "quadrant": "Group",
    "confidence": "Confidence",
    "tier": "Evidence",
    # AHP pieces
    "norm_avg_review_score": "Scaled: reviews",
    "norm_on_time_rate": "Scaled: on-time rate",
    "norm_avg_price": "Scaled: price",
    "norm_order_volume": "Scaled: items sold",
    "contribution_avg_review_score": "Contribution: reviews",
    "contribution_on_time_rate": "Contribution: on-time rate",
    "contribution_avg_price": "Contribution: price",
    "contribution_order_volume": "Contribution: items sold",
    # clickstream and calls
    "sessions": "Sessions",
    "stage_conversion": "From the previous step",
    "overall_conversion": "From the first step",
    "agent_id": "Agent",
    "call_count": "Calls",
    "mean_sentiment": "Mean sentiment",
    "mean_hold_seconds": "Mean hold (s)",
    "mean_silence_pct": "Mean silence",
    "negative_call_share": "Negative calls",
    # models
    "feature": "Input",
    "importance_mean": "Drop in AUC when scrambled (mean)",
    "importance_std": "Drop in AUC when scrambled (std)",
    "model": "Model",
    "perturbation": "Nudge",
    "mean_late_risk_change": "Change in late risk",
    "standardized_coefficient": "Weight",
}

# src/discovery.py names the two tiers "Actionable" and "Watchlist"; people see these.
TIER_LABELS: dict[str, str] = {"Actionable": "Stronger evidence", "Watchlist": "Early signal"}


def friendly(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Copy of `df` with raw column names replaced by their friendly headers.

    `overrides` lets a page name one column differently (for example the quality score of its
    own seller pool). Columns without an entry keep their name.
    """
    names = {**COLUMN_LABELS, **(overrides or {})}
    return df.rename(columns={c: names[c] for c in df.columns if c in names})


def tier_label(raw: str) -> str:
    """User-facing name of an evidence tier (`Actionable` -> `Stronger evidence`)."""
    return TIER_LABELS.get(raw, raw)


def with_tier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` whose `tier` column (if any) uses the user-facing names."""
    if "tier" not in df.columns:
        return df
    return df.assign(tier=df["tier"].map(tier_label))
