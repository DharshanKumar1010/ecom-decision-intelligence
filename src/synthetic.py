"""Synthetic clickstream and call-transcript generators (CLAUDE.md section 6.3).

SYNTHETIC DATA: Olist has no real clickstream or call-center data. Both
datasets produced here (`generate_clickstream`, `generate_call_transcripts`)
are entirely synthetic and must be labelled as such everywhere they are
shown (dashboard captions, README, reports). They exist only to demonstrate
clickstream analytics and sentiment/speech-analytics techniques on top of a
plausible, internally consistent e-commerce narrative.

Both generators are deterministic: each seeds its own
`numpy.random.default_rng(seed)` (default ``SEED`` from `src.config`) and
produces identical output across runs given the same seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, SEED, get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Clickstream
# --------------------------------------------------------------------------

N_CLICKSTREAM_SESSIONS = 70_000
N_REACH_CART = 15_000
N_REACH_CHECKOUT = 9_000
N_REACH_PURCHASE = 6_000

_DEVICES = np.array(["mobile", "desktop", "tablet"])
_DEVICE_PROBS = np.array([0.60, 0.30, 0.10])
_SOURCES = np.array(["organic", "ads", "email", "social"])
_SOURCE_PROBS = np.array([0.40, 0.30, 0.20, 0.10])

# Session start times spread across Olist's real order date range.
_SESSION_START = pd.Timestamp("2018-01-01")
_SESSION_END = pd.Timestamp("2018-08-01")


def _session_attrs(
    session_ids: np.ndarray, rng: np.random.Generator, product_pool: np.ndarray
) -> pd.DataFrame:
    """Sample per-session device, source, product, and start time.

    Args:
        session_ids: array of session id strings for this group.
        rng: seeded numpy random Generator.
        product_pool: pool of real product_id values to sample from.

    Returns:
        DataFrame indexed by position with columns session_id, device, source,
        product_id, start_time.
    """
    n = len(session_ids)
    span_seconds = int((_SESSION_END - _SESSION_START).total_seconds())
    start_offsets = rng.integers(0, span_seconds, size=n)
    return pd.DataFrame(
        {
            "session_id": session_ids,
            "device": rng.choice(_DEVICES, size=n, p=_DEVICE_PROBS),
            "source": rng.choice(_SOURCES, size=n, p=_SOURCE_PROBS),
            "product_id": rng.choice(product_pool, size=n),
            "start_time": _SESSION_START + pd.to_timedelta(start_offsets, unit="s"),
        }
    )


def _stage_rows(
    attrs: pd.DataFrame, event_type: str, elapsed_seconds: np.ndarray
) -> pd.DataFrame:
    """Build event rows for one funnel stage, given cumulative elapsed seconds.

    Args:
        attrs: per-session attribute DataFrame from `_session_attrs`.
        event_type: one of page_view/add_to_cart/checkout/purchase.
        elapsed_seconds: seconds since `start_time` for this stage's event.

    Returns:
        DataFrame with columns session_id, event_type, product_id, device,
        source, event_time.
    """
    return pd.DataFrame(
        {
            "session_id": attrs["session_id"].to_numpy(),
            "event_type": event_type,
            "product_id": attrs["product_id"].to_numpy(),
            "device": attrs["device"].to_numpy(),
            "source": attrs["source"].to_numpy(),
            "event_time": attrs["start_time"] + pd.to_timedelta(elapsed_seconds, unit="s"),
        }
    )


def generate_clickstream(fact_order_items: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic, causally consistent clickstream event log.

    Sessions are partitioned by the deepest funnel stage they reach so that
    stage-reach counts are exactly monotonic (view >= cart >= checkout >=
    purchase) and match the target overall proportions (~70/15/9/6%): 70,000
    sessions view, of which 15,000 also add-to-cart, of which 9,000 also
    checkout, of which 6,000 also purchase. `purchase` therefore always
    implies an earlier `checkout` and `add_to_cart` in the same session, and
    event_time strictly increases within a session by construction.

    Args:
        fact_order_items: the real FactOrderItems table, used only as a pool
            of valid `product_id` values.
        seed: seed for the internal `numpy.random.default_rng`.

    Returns:
        DataFrame with columns: session_id, event_type, product_id, device,
        source, event_time (exactly 100,000 rows).
    """
    rng = np.random.default_rng(seed)
    product_pool = fact_order_items["product_id"].dropna().unique()

    session_ids = np.array([f"S{i:06d}" for i in rng.permutation(N_CLICKSTREAM_SESSIONS)])
    purchase_ids = session_ids[:N_REACH_PURCHASE]
    checkout_only_ids = session_ids[N_REACH_PURCHASE:N_REACH_CHECKOUT]
    cart_only_ids = session_ids[N_REACH_CHECKOUT:N_REACH_CART]
    view_only_ids = session_ids[N_REACH_CART:]

    frames: list[pd.DataFrame] = []
    for ids, depth in (
        (purchase_ids, 4),
        (checkout_only_ids, 3),
        (cart_only_ids, 2),
        (view_only_ids, 1),
    ):
        attrs = _session_attrs(ids, rng, product_pool)
        n = len(ids)
        elapsed = np.zeros(n, dtype=np.int64)
        frames.append(_stage_rows(attrs, "page_view", elapsed))
        if depth >= 2:
            elapsed = elapsed + rng.integers(30, 1800, size=n)
            frames.append(_stage_rows(attrs, "add_to_cart", elapsed))
        if depth >= 3:
            elapsed = elapsed + rng.integers(30, 1800, size=n)
            frames.append(_stage_rows(attrs, "checkout", elapsed))
        if depth >= 4:
            elapsed = elapsed + rng.integers(10, 900, size=n)
            frames.append(_stage_rows(attrs, "purchase", elapsed))

    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(["session_id", "event_time"]).reset_index(drop=True)
    logger.info(
        "Generated synthetic clickstream: %d events, %d sessions",
        len(events),
        N_CLICKSTREAM_SESSIONS,
    )
    return events


# --------------------------------------------------------------------------
# Call transcripts
# --------------------------------------------------------------------------

N_CALLS = 300
_AGENTS = np.array([f"AG{i:02d}" for i in range(1, 11)])

# Negative calls are generated with higher mean hold_seconds and silence_pct
# than neutral/positive calls (longer holds and more dead air correlate with
# unresolved, frustrating interactions in this synthetic model). Values are
# drawn from clipped normal distributions:
#   negative: hold ~ N(180, 40)s clipped [30, 600];  silence ~ N(0.25, 0.08) clipped [0, 1]
#   neutral:  hold ~ N(90, 25)s  clipped [30, 600];  silence ~ N(0.12, 0.05) clipped [0, 1]
#   positive: hold ~ N(60, 20)s  clipped [30, 600];  silence ~ N(0.05, 0.03) clipped [0, 1]
_HOLD_PARAMS = {"negative": (180.0, 40.0), "neutral": (90.0, 25.0), "positive": (60.0, 20.0)}
_SILENCE_PARAMS = {"negative": (0.25, 0.08), "neutral": (0.12, 0.05), "positive": (0.05, 0.03)}

_NEGATIVE_TEMPLATES = [
    "This is the third time I'm calling about my late delivery and nobody has helped me.",
    "My package arrived damaged and the seller is refusing to issue a refund.",
    "I have been waiting three weeks and the tracking still shows no movement at all.",
    "The product I received is completely different from what was advertised on the site.",
    "I am extremely frustrated, this delay has cost me money and nobody seems to care.",
    "Your courier left my order at the wrong address and now it is lost.",
    "I want to cancel my order, the seller never responded to any of my messages.",
    "This is unacceptable, I paid for express shipping and it still hasn't arrived.",
    "The item broke after one day and customer service keeps ignoring my emails.",
    "I am very upset because the seller sent the wrong item and won't fix it.",
    "Nobody explained why my refund was denied and I am tired of being transferred around.",
    "This experience has been a disaster from start to finish, I want my money back.",
]

_NEUTRAL_TEMPLATES = [
    "I'm calling to check the current status of my order shipment.",
    "Can you confirm the estimated delivery date for my recent purchase?",
    "I would like to update the shipping address on an order I placed yesterday.",
    "I have a question about the return policy for electronics items.",
    "Could you tell me which carrier is handling the delivery of my package?",
    "I want to know if partial refunds are possible for a multi-item order.",
    "I'm following up on a support ticket I opened earlier this week.",
    "Can you clarify how installment payments are processed on this platform?",
    "I need to know if I can change the payment method after checkout.",
    "I'm asking whether the product comes with a manufacturer warranty.",
    "Please let me know the process for exchanging an item for a different size.",
    "I wanted to verify that my last payment was received correctly.",
]

_POSITIVE_TEMPLATES = [
    "I just wanted to say the delivery arrived early and the product is great.",
    "Thank you so much, the seller resolved my issue quickly and I'm very happy.",
    "The packaging was excellent and the item works perfectly, thanks for the help.",
    "I appreciate how fast your team responded, this was a smooth experience.",
    "Everything about this order exceeded my expectations, great service.",
    "The support agent was very helpful and solved my problem in minutes.",
    "I'm calling just to compliment the seller on the quality of the product.",
    "Delivery was fast and the item matched the description perfectly, thank you.",
    "This was a wonderful shopping experience and I will definitely order again.",
    "I love how easy it was to track my order and get updates the whole way.",
    "Thanks for the quick refund, your team handled everything very professionally.",
    "The product quality is fantastic and it arrived well ahead of schedule.",
]

_MODIFIERS = ["", " Really.", " Honestly.", " Just so you know.", " Thanks for listening."]

_TEMPLATES_BY_CLASS = {
    "negative": _NEGATIVE_TEMPLATES,
    "neutral": _NEUTRAL_TEMPLATES,
    "positive": _POSITIVE_TEMPLATES,
}


def generate_call_transcripts(n_calls: int = N_CALLS, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic call-center transcripts with sentiment-correlated metrics.

    Each call is assigned a target sentiment class (negative/neutral/positive,
    evenly split), a transcript sampled from >=10 hand-written templates for
    that class with a light random modifier appended for lexical variation,
    and `hold_seconds`/`silence_pct` drawn from class-specific clipped normal
    distributions (see module-level `_HOLD_PARAMS`/`_SILENCE_PARAMS`) such
    that negative calls have higher hold time and silence percentage on
    average than neutral or positive calls.

    Args:
        n_calls: number of synthetic calls to generate.
        seed: seed for the internal `numpy.random.default_rng`.

    Returns:
        DataFrame with columns: call_id, agent_id, transcript, hold_seconds,
        silence_pct.
    """
    rng = np.random.default_rng(seed)

    classes = np.array(["negative", "neutral", "positive"])
    class_assignment = rng.choice(classes, size=n_calls)

    call_ids = [f"C{i:04d}" for i in range(1, n_calls + 1)]
    agent_ids = rng.choice(_AGENTS, size=n_calls)

    transcripts = []
    hold_seconds = np.empty(n_calls)
    silence_pct = np.empty(n_calls)
    for i, cls in enumerate(class_assignment):
        template = rng.choice(_TEMPLATES_BY_CLASS[cls])
        modifier = rng.choice(_MODIFIERS)
        transcripts.append(f"{template}{modifier}")

        hold_mean, hold_std = _HOLD_PARAMS[cls]
        hold_seconds[i] = np.clip(rng.normal(hold_mean, hold_std), 30, 600)

        sil_mean, sil_std = _SILENCE_PARAMS[cls]
        silence_pct[i] = np.clip(rng.normal(sil_mean, sil_std), 0.0, 1.0)

    calls = pd.DataFrame(
        {
            "call_id": call_ids,
            "agent_id": agent_ids,
            "transcript": transcripts,
            "hold_seconds": hold_seconds.round(1),
            "silence_pct": silence_pct.round(4),
        }
    )
    logger.info("Generated %d synthetic call transcripts", len(calls))
    return calls


def main() -> None:
    """Generate and persist both synthetic datasets to data/processed/."""
    fact_path = PROCESSED_DATA_DIR / "FactOrderItems.parquet"
    fact_order_items = pd.read_parquet(fact_path, columns=["product_id"])

    clickstream = generate_clickstream(fact_order_items)
    calls = generate_call_transcripts()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clickstream.to_parquet(PROCESSED_DATA_DIR / "Clickstream.parquet", index=False)
    calls.to_parquet(PROCESSED_DATA_DIR / "CallTranscripts.parquet", index=False)
    logger.info(
        "Wrote synthetic Clickstream (%d rows) and CallTranscripts (%d rows)",
        len(clickstream),
        len(calls),
    )


if __name__ == "__main__":
    main()
