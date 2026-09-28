"""Clickstream funnel metrics (CLAUDE.md section 7.1, Unit 1: clickstream analysis).

All functions are pure: they take an events DataFrame in and return a metrics
DataFrame out, with no file I/O. The event log this module expects is the
synthetic clickstream produced by `src.synthetic.generate_clickstream`
(columns: session_id, event_type, product_id, device, source, event_time).
"""

from __future__ import annotations

import pandas as pd

_FUNNEL_STAGES = ["page_view", "add_to_cart", "checkout", "purchase"]


def funnel_conversion(events: pd.DataFrame) -> pd.DataFrame:
    """Compute session counts and conversion rates at each funnel stage.

    Args:
        events: clickstream events with columns session_id, event_type.

    Returns:
        DataFrame indexed by stage (in funnel order) with columns:
        `sessions` (distinct sessions reaching the stage), `stage_conversion`
        (sessions at this stage / sessions at the previous stage), and
        `overall_conversion` (sessions at this stage / sessions at the first
        stage). Each stage's `sessions` count is <= the previous stage's.
    """
    sessions_per_stage = {
        stage: events.loc[events["event_type"] == stage, "session_id"].nunique()
        for stage in _FUNNEL_STAGES
    }
    counts = pd.Series(sessions_per_stage, name="sessions").reindex(_FUNNEL_STAGES)

    first_stage_count = counts.iloc[0]
    stage_conversion = counts / counts.shift(1)
    stage_conversion.iloc[0] = 1.0
    overall_conversion = counts / first_stage_count if first_stage_count else counts * 0.0

    result = pd.DataFrame(
        {
            "sessions": counts,
            "stage_conversion": stage_conversion,
            "overall_conversion": overall_conversion,
        }
    )
    result.index.name = "stage"
    return result


def bounce_rate(events: pd.DataFrame) -> float:
    """Share of sessions that contain exactly one event, and it is a page_view.

    Args:
        events: clickstream events with columns session_id, event_type.

    Returns:
        Bounce rate in [0, 1]; 0.0 if there are no sessions.
    """
    per_session = events.groupby("session_id")["event_type"].agg(list)
    if len(per_session) == 0:
        return 0.0
    is_bounce = per_session.apply(lambda types: len(types) == 1 and types[0] == "page_view")
    return float(is_bounce.mean())


def cart_abandonment_rate(events: pd.DataFrame) -> float:
    """Share of add_to_cart sessions that never reach purchase.

    Args:
        events: clickstream events with columns session_id, event_type.

    Returns:
        Cart abandonment rate in [0, 1]; 0.0 if no session added to cart.
    """
    cart_sessions = set(events.loc[events["event_type"] == "add_to_cart", "session_id"])
    if not cart_sessions:
        return 0.0
    purchase_sessions = set(events.loc[events["event_type"] == "purchase", "session_id"])
    abandoned = cart_sessions - purchase_sessions
    return len(abandoned) / len(cart_sessions)


def funnel_by_device(events: pd.DataFrame, by: str = "device") -> pd.DataFrame:
    """Compute session counts reaching each funnel stage, broken out by a dimension.

    Args:
        events: clickstream events with columns session_id, event_type, and `by`.
        by: column to group by in addition to funnel stage (e.g. "device" or "source").

    Returns:
        DataFrame indexed by `by` value with one column per funnel stage,
        holding the count of distinct sessions reaching that stage for that
        group. Within each row, stage columns are monotonically non-increasing
        left to right, following `_FUNNEL_STAGES` order.
    """
    subset = events[events["event_type"].isin(_FUNNEL_STAGES)]
    grouped = (
        subset.groupby([by, "event_type"])["session_id"]
        .nunique()
        .unstack("event_type")
        .reindex(columns=_FUNNEL_STAGES, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    grouped.columns.name = None
    return grouped
