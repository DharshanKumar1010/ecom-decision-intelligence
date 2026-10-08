"""Quality-vs-popularity "hidden gems" discovery (CLAUDE.md section 7.7).

Surfaces sellers that are high quality but low visibility — the opposite of
a naive "sort by orders" view. "Quality" here reuses the AHP composite score
from `src.ahp` (`compute_weights`, `normalize_criteria`, `score_sellers`),
recomputed at a lower `config.DISCOVERY_MIN_ORDERS` threshold so smaller,
newer sellers are eligible; no second, parallel quality metric is built.
"Popularity" is total order volume per seller.

"Healthier"/"good" in this module means sound business quality — good
reviews, reliable delivery — never literal product nutrition or food
health; Olist has no such data (CLAUDE.md section 7.7.1).

Every row is annotated with a `confidence` flag (`confidence_flag`), and a
seller below `config.DISCOVERY_CONFIDENCE_MIN_ORDERS` orders must never be
presented with the same visual weight as an established one — enforced in
the dashboard (a later stage), not just noted here as a column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ahp import build_seller_criteria, compute_weights, normalize_criteria, score_sellers
from src.config import (
    AHP_CRITERIA,
    AHP_CRITERION_DIRECTION,
    AHP_PAIRWISE_MATRIX,
    DISCOVERY_CONFIDENCE_MIN_ORDERS,
    DISCOVERY_MIN_ORDERS,
    DISCOVERY_POPULARITY_THRESHOLD,
    DISCOVERY_QUALITY_THRESHOLD,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    get_logger,
)

logger = get_logger(__name__)

# Confidence tiers for presenting Hidden Gems. By construction a Hidden Gem has fewer orders
# than the popularity median, so most gems fall under the confidence threshold; the tiers
# keep the few well-supported picks ("Actionable") apart from the directional ones
# ("Watchlist"). Tier follows `confidence_flag`, so the boundary is defined in one place.
TIER_ACTIONABLE = "Actionable"
TIER_WATCHLIST = "Watchlist"

_QUADRANT_LABELS: tuple[str, str, str, str] = (
    "Star",
    "Hidden Gem",
    "Overrated",
    "Overlooked-Low-Quality",
)


def confidence_flag(order_count: float) -> str:
    """Flag whether a seller's order count supports normal-confidence display.

    Args:
        order_count: the seller's total order volume.

    Returns:
        `"low"` if `order_count < config.DISCOVERY_CONFIDENCE_MIN_ORDERS`,
        else `"normal"`.
    """
    return "low" if order_count < DISCOVERY_CONFIDENCE_MIN_ORDERS else "normal"


def hidden_gem_tier(confidence: str) -> str:
    """Map a `confidence` flag to a presentation tier.

    Args:
        confidence: `"normal"` or `"low"` (as produced by `confidence_flag`).

    Returns:
        `"Actionable"` for normal confidence, `"Watchlist"` for low confidence.
    """
    return TIER_ACTIONABLE if confidence == "normal" else TIER_WATCHLIST


def compute_quadrants(
    ahp_scores: pd.Series,
    order_counts: pd.Series,
    quality_threshold: float | None = None,
    popularity_threshold: float | None = None,
) -> pd.DataFrame:
    """Classify sellers into quality-vs-popularity quadrants.

    Args:
        ahp_scores: AHP composite score per seller, indexed by `seller_id`.
        order_counts: total order volume per seller, indexed by `seller_id`.
        quality_threshold: split point on the quality axis. Defaults to
            `config.DISCOVERY_QUALITY_THRESHOLD` if set, else the median of
            `ahp_scores`.
        popularity_threshold: split point on the popularity axis. Defaults
            to `config.DISCOVERY_POPULARITY_THRESHOLD` if set, else the
            median of `order_counts`.

    Returns:
        DataFrame with columns `seller_id`, `ahp_score`, `order_volume`,
        `quadrant` (one of Star/Hidden Gem/Overrated/
        Overlooked-Low-Quality), `confidence` (`low`/`normal`). Only
        sellers present in both `ahp_scores` and `order_counts` are
        included.
    """
    if quality_threshold is None:
        quality_threshold = DISCOVERY_QUALITY_THRESHOLD
    if quality_threshold is None:
        quality_threshold = float(ahp_scores.median())

    if popularity_threshold is None:
        popularity_threshold = DISCOVERY_POPULARITY_THRESHOLD
    if popularity_threshold is None:
        popularity_threshold = float(order_counts.median())

    combined = pd.DataFrame(
        {"ahp_score": ahp_scores, "order_volume": order_counts}
    ).dropna(subset=["ahp_score", "order_volume"])

    high_quality = combined["ahp_score"] >= quality_threshold
    high_popularity = combined["order_volume"] >= popularity_threshold

    combined["quadrant"] = np.select(
        [
            high_quality & high_popularity,
            high_quality & ~high_popularity,
            ~high_quality & high_popularity,
            ~high_quality & ~high_popularity,
        ],
        _QUADRANT_LABELS,
        default="",
    )
    combined["confidence"] = combined["order_volume"].map(confidence_flag)
    combined.index.name = "seller_id"
    return combined.reset_index()


def top_hidden_gems(quadrant_df: pd.DataFrame, n: int = 10, tiered: bool = False) -> pd.DataFrame:
    """Return the top Hidden Gem sellers.

    Args:
        quadrant_df: output of `compute_quadrants`.
        n: number of sellers to return (per tier when `tiered`).
        tiered: if False (default), rank all gems together by AHP score. If True, add a
            `tier` column and list `Actionable` (normal confidence) gems first, then
            `Watchlist` (low confidence) gems, each ranked by `ahp_score` descending with
            ties broken by `order_volume` descending, up to `n` rows per tier.

    Returns:
        Rows from `quadrant_df` where `quadrant == "Hidden Gem"`, with `confidence`
        retained. Never includes an `Overrated` or `Overlooked-Low-Quality` seller.
    """
    gems = quadrant_df[quadrant_df["quadrant"] == "Hidden Gem"]
    if not tiered:
        return gems.sort_values("ahp_score", ascending=False).head(n).reset_index(drop=True)

    gems = gems.assign(tier=gems["confidence"].map(hidden_gem_tier))
    ranked = gems.sort_values(["ahp_score", "order_volume"], ascending=[False, False])
    parts = [ranked[ranked["tier"] == tier].head(n) for tier in (TIER_ACTIONABLE, TIER_WATCHLIST)]
    return pd.concat(parts).reset_index(drop=True)


def build_seller_facts(
    quadrant_df: pd.DataFrame, late_predictions: pd.DataFrame, fact: pd.DataFrame
) -> pd.DataFrame:
    """Build the per-seller fact table consumed by src.expert.WorkingMemory.

    Facts are computed over ALL sellers in `fact` (not just the AHP-eligible
    ones in `quadrant_df`), so every seller gets a recommendation later. A
    seller with fewer than `config.DISCOVERY_MIN_ORDERS` orders has no AHP
    score and gets `ahp_score = NaN`, `is_hidden_gem = False` — left
    explicit via a left join, never silently coerced to a default score.

    Args:
        quadrant_df: output of `compute_quadrants` (AHP-eligible sellers
            only, i.e. >= `config.DISCOVERY_MIN_ORDERS` orders).
        late_predictions: `LatePredictions.parquet` (`item_key`, `late_risk`).
        fact: FactOrderItems (must contain item_key, seller_id,
            review_score, is_late).

    Returns:
        DataFrame with columns `seller_id`, `ahp_score`, `avg_late_risk`,
        `avg_review`, `order_volume`, `late_rate`, `is_hidden_gem`.
    """
    item_level = fact[["item_key", "seller_id", "review_score", "is_late"]].merge(
        late_predictions, on="item_key", how="left"
    )
    seller_stats = item_level.groupby("seller_id").agg(
        avg_late_risk=("late_risk", "mean"),
        avg_review=("review_score", "mean"),
        late_rate=("is_late", "mean"),
        order_volume=("item_key", "count"),
    )
    seller_stats = seller_stats.reset_index()

    merged = seller_stats.merge(
        quadrant_df[["seller_id", "ahp_score", "quadrant"]], on="seller_id", how="left"
    )
    merged["is_hidden_gem"] = merged["quadrant"] == "Hidden Gem"
    merged = merged.drop(columns="quadrant")
    return merged[
        ["seller_id", "ahp_score", "avg_late_risk", "avg_review", "order_volume",
         "late_rate", "is_hidden_gem"]
    ]


def _write_summary(quadrant_df: pd.DataFrame, gems: pd.DataFrame) -> None:
    """Write reports/discovery_summary.md: quadrant counts, gems, confidence breakdown."""
    quadrant_counts = quadrant_df["quadrant"].value_counts()
    confidence_counts = quadrant_df["confidence"].value_counts()

    lines = ["# Discovery Summary", "", "## Quadrant counts"]
    for label in _QUADRANT_LABELS:
        lines.append(f"- {label}: {int(quadrant_counts.get(label, 0))}")

    lines += ["", "## Confidence flag breakdown (all AHP-eligible sellers)"]
    for label in ("low", "normal"):
        lines.append(f"- {label}: {int(confidence_counts.get(label, 0))}")

    lines += [
        "",
        "## Top hidden gems (top 10 per tier)",
        "",
        "| tier | seller_id | ahp_score | order_volume | confidence |",
        "|---|---|---|---|---|",
    ]
    for _, row in gems.iterrows():
        lines.append(
            f"| {row['tier']} | {row['seller_id']} | {row['ahp_score']:.4f} "
            f"| {int(row['order_volume'])} | {row['confidence']} |"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "discovery_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the discovery pipeline and write the quadrant summary + seller facts."""
    fact = pd.read_parquet(PROCESSED_DATA_DIR / "FactOrderItems.parquet")
    dim_seller = pd.read_parquet(PROCESSED_DATA_DIR / "DimSeller.parquet")
    late_predictions = pd.read_parquet(PROCESSED_DATA_DIR / "LatePredictions.parquet")

    criteria = build_seller_criteria(fact, dim_seller, min_orders=DISCOVERY_MIN_ORDERS)
    weights, _, _, _ = compute_weights(AHP_PAIRWISE_MATRIX)
    normalized = normalize_criteria(criteria, AHP_CRITERION_DIRECTION)
    scored = score_sellers(normalized, weights, AHP_CRITERIA)

    ahp_scores = scored.set_index("seller_id")["ahp_score"]
    order_counts = scored.set_index("seller_id")["order_volume"]
    quadrant_df = compute_quadrants(ahp_scores, order_counts)

    gems = top_hidden_gems(quadrant_df, tiered=True)
    seller_facts = build_seller_facts(quadrant_df, late_predictions, fact)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    seller_facts.to_parquet(PROCESSED_DATA_DIR / "SellerFacts.parquet", index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    quadrant_df.to_csv(REPORTS_DIR / "discovery_quadrants.csv", index=False)

    _write_summary(quadrant_df, gems)
    logger.info(
        "Discovery: %d AHP-eligible sellers (>= %d orders), %d Hidden Gems, "
        "%d low-confidence rows",
        len(quadrant_df),
        DISCOVERY_MIN_ORDERS,
        int((quadrant_df["quadrant"] == "Hidden Gem").sum()),
        int((quadrant_df["confidence"] == "low").sum()),
    )


if __name__ == "__main__":
    main()
