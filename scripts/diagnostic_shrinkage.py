"""One-off diagnostic: would shrinkage or a rate-based popularity fix the hidden-gem ranking?

NOT part of the pipeline: not wired into `run_all.py`, nothing adopted, no model fitted.
Run directly: `python scripts/diagnostic_shrinkage.py` (a few seconds).

Problem. By construction a Hidden Gem has fewer orders than the popularity median, and
"low confidence" is under `DISCOVERY_CONFIDENCE_MIN_ORDERS`, so the top of the AHP ranking
inside the gem quadrant is dominated by tiny-sample sellers with perfect averages.

Part 1 (shrinkage). For each prior strength k, the two noisy per-seller criteria are pulled
toward the global mean with weight k pseudo-observations:
    shrunk = (n * seller_value + k * global_value) / (n + k)
average review uses the seller's reviewed-item count as n and the global review mean;
on-time rate uses the seller's order count as n and the global on-time rate. Average
price and order volume are left alone. The unchanged AHP functions (`normalize_criteria`,
`score_sellers`, configured weights) then re-score the same AHP-eligible sellers, and the
unchanged `compute_quadrants` re-splits them at the new medians.

Part 2 (rate popularity). Popularity as orders per active month (first to last
`order_purchase_timestamp` per seller, floored at one month) instead of raw volume, with
the CURRENT quality score and quality split line; only the popularity axis changes.

Reuses `src.ahp` and `src.discovery` functions unmodified. Writes
`reports/diagnostic_shrinkage.txt`. "Order" counts here are order items, as everywhere in
the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ahp import (  # noqa: E402
    compute_weights,
    normalize_criteria,
    score_sellers,
)
from src.config import (  # noqa: E402
    AHP_CRITERIA,
    AHP_CRITERION_DIRECTION,
    AHP_PAIRWISE_MATRIX,
    DISCOVERY_CONFIDENCE_MIN_ORDERS,
    DISCOVERY_MIN_ORDERS,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from src.discovery import compute_quadrants  # noqa: E402

K_VALUES = (5, 10, 15, 30)
TOP_N = 10


def item_level_criteria(
    fact: pd.DataFrame, dim_seller: pd.DataFrame, min_orders: int
) -> pd.DataFrame:
    """The raw ITEM-level criteria this diagnostic was written against (frozen on purpose).

    `src.ahp.build_seller_criteria` now works at order level; this historical diagnostic keeps
    its original item-level definition so its report stays reproducible.
    """
    grouped = fact.groupby("seller_id").agg(
        avg_review_score=("review_score", "mean"),
        on_time_rate=("is_late", lambda s: 1.0 - s.mean()),
        avg_price=("price", "mean"),
        order_volume=("item_key", "count"),
        n_reviewed=("review_score", "count"),
    ).reset_index()
    grouped = grouped.merge(dim_seller[["seller_id", "seller_state"]], on="seller_id", how="left")
    return grouped[grouped["order_volume"] >= min_orders].reset_index(drop=True)


def _score(criteria: pd.DataFrame) -> pd.DataFrame:
    weights, _, _, _ = compute_weights(AHP_PAIRWISE_MATRIX)
    normalized = normalize_criteria(criteria, AHP_CRITERION_DIRECTION)
    return score_sellers(normalized, weights, AHP_CRITERIA)


def _quadrants(scored: pd.DataFrame, orders: pd.Series | None = None,
               quality_threshold: float | None = None) -> pd.DataFrame:
    """Quadrants from a scored frame; `orders` overrides the popularity axis if given."""
    ahp = scored.set_index("seller_id")["ahp_score"]
    volume = scored.set_index("seller_id")["order_volume"]
    popularity = volume if orders is None else orders
    quadrants = compute_quadrants(ahp, popularity, quality_threshold=quality_threshold)
    # Confidence always uses real order counts, whatever the popularity axis is.
    quadrants["order_volume"] = quadrants["seller_id"].map(volume)
    quadrants["confidence"] = np.where(
        quadrants["order_volume"] < DISCOVERY_CONFIDENCE_MIN_ORDERS, "low", "normal"
    )
    return quadrants


def _gems(quadrants: pd.DataFrame) -> pd.DataFrame:
    gems = quadrants[quadrants["quadrant"] == "Hidden Gem"]
    return gems.sort_values(["ahp_score", "order_volume"], ascending=[False, False])


def _top_normal(gems: pd.DataFrame) -> int:
    return int((gems.head(TOP_N)["confidence"] == "normal").sum())


def shrunk_criteria(criteria: pd.DataFrame, fact: pd.DataFrame, k: int) -> pd.DataFrame:
    """Criteria with average review and on-time rate shrunk toward the global mean."""
    out = criteria.copy()
    global_review = float(fact["review_score"].mean())
    global_on_time = 1.0 - float(fact["is_late"].mean())
    n_rev = out["n_reviewed"].fillna(0)
    out["avg_review_score"] = (
        n_rev * out["avg_review_score"].fillna(global_review) + k * global_review
    ) / (n_rev + k)
    n_ord = out["order_volume"]
    out["on_time_rate"] = (n_ord * out["on_time_rate"] + k * global_on_time) / (n_ord + k)
    return out


def _active_months(fact: pd.DataFrame) -> pd.Series:
    span = fact.groupby("seller_id")["order_purchase_timestamp"].agg(["min", "max"])
    months = (span["max"] - span["min"]).dt.total_seconds() / (30.44 * 86400)
    return months.clip(lower=1.0)


def build_report() -> str:
    """Run both experiments and return the report text."""
    fact = pd.read_parquet(PROCESSED_DATA_DIR / "FactOrderItems.parquet")
    dim_seller = pd.read_parquet(PROCESSED_DATA_DIR / "DimSeller.parquet")
    criteria = item_level_criteria(fact, dim_seller, DISCOVERY_MIN_ORDERS)
    base_scored = _score(criteria)
    base = _quadrants(base_scored)
    base_gems = _gems(base)
    base_ids = set(base_gems["seller_id"])
    base_score = base_scored.set_index("seller_id")["ahp_score"]
    quality_line = float(base_score.median())

    lines = [
        "Diagnostic: shrinkage and rate-based popularity for the hidden-gem ranking",
        "=" * 78,
        "",
        f"AHP-eligible sellers (>= {DISCOVERY_MIN_ORDERS} orders): {len(base)}. "
        f"Current Hidden Gems: {len(base_ids)} "
        f"({int((base_gems['confidence'] == 'normal').sum())} normal / "
        f"{int((base_gems['confidence'] == 'low').sum())} low confidence).",
        f"Current top {TOP_N} gems by AHP score: {_top_normal(base_gems)} normal-confidence.",
        "",
        "Part 1: Bayesian shrinkage of average review and on-time rate (prior strength k)",
        "-" * 78,
        f"{'k':>4} {'gems':>5} {'stay':>5} {'enter':>6} {'top10 normal':>13} "
        f"{'rho all':>8} {'rho gems':>9} {'normal gems':>12}",
    ]
    for k in K_VALUES:
        scored = _score(shrunk_criteria(criteria, fact, k))
        quadrants = _quadrants(scored)
        gems = _gems(quadrants)
        ids = set(gems["seller_id"])
        score_k = scored.set_index("seller_id")["ahp_score"]
        common = base_score.index.intersection(score_k.index)
        rho_all = spearmanr(base_score[common], score_k[common]).statistic
        in_both = [s for s in common if s in base_ids]
        rho_gems = spearmanr(base_score[in_both], score_k[in_both]).statistic
        lines.append(
            f"{k:>4} {len(ids):>5} {len(ids & base_ids):>5} {len(ids - base_ids):>6} "
            f"{_top_normal(gems):>13} {rho_all:>8.3f} {rho_gems:>9.3f} "
            f"{int((gems['confidence'] == 'normal').sum()):>12}"
        )
    lines += [
        "",
        "  stay/enter: gems that remain/join relative to the current 478; top10 normal: "
        "normal-confidence",
        "  sellers among the 10 highest-scoring gems; rho: Spearman correlation of the "
        "AHP score with",
        "  the current score (all eligible sellers / current gems only).",
        "",
        "Part 2: orders per active month as the popularity axis (current quality score)",
        "-" * 78,
    ]
    volume = base_scored.set_index("seller_id")["order_volume"]
    rate = volume / _active_months(fact).reindex(volume.index)
    alt = _quadrants(base_scored, orders=rate, quality_threshold=quality_line)
    alt_gems = _gems(alt)
    alt_ids = set(alt_gems["seller_id"])
    rho_vol = spearmanr(volume, rate).statistic
    lines += [
        f"Median orders/active month: {float(rate.median()):.3f} "
        f"(median raw volume: {float(volume.median()):.0f}); Spearman(volume, rate) = "
        f"{rho_vol:.3f}",
        f"Gems under the rate measure: {len(alt_ids)}  stay: {len(alt_ids & base_ids)}  "
        f"enter: {len(alt_ids - base_ids)}  leave: {len(base_ids - alt_ids)}",
        f"Normal-confidence gems: {int((alt_gems['confidence'] == 'normal').sum())} of "
        f"{len(alt_ids)} (current: {int((base_gems['confidence'] == 'normal').sum())} of "
        f"{len(base_ids)}); top {TOP_N} normal-confidence: {_top_normal(alt_gems)}",
        f"Order-count range of rate-based gems: {int(alt_gems['order_volume'].min())} to "
        f"{int(alt_gems['order_volume'].max())} (current: "
        f"{int(base_gems['order_volume'].min())} to {int(base_gems['order_volume'].max())})",
    ]
    return "\n".join(lines)


def main() -> None:
    """Print the report and write it to `reports/diagnostic_shrinkage.txt`."""
    report = build_report()
    print(report)  # noqa: T201 - one-off diagnostic script
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "diagnostic_shrinkage.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
