"""AHP multi-criteria seller scoring (CLAUDE.md section 7.4).

Criteria (order fixed by `config.AHP_CRITERIA`): average review score,
on-time rate, average price, order volume — computed per seller from
`FactOrderItems`, restricted to sellers with at least `min_orders` orders.
The main ("established sellers") ranking uses `config.AHP_MIN_ORDERS` (30);
`src.discovery` reuses `build_seller_criteria`/`compute_weights`/
`normalize_criteria`/`score_sellers` at a lower threshold rather than
reimplementing any of this scoring logic.

Average review and on-time rate are shrunk toward the marketplace mean before scoring
(empirical Bayes at ORDER level, `estimate_prior_strengths`/`shrink_toward_mean`, k in
`config`), so
the Choice ranking and the Discovery quadrants share one definition of quality and a few
perfect reviews cannot outrank a long, strong record. The raw values stay available as
`avg_review_raw`/`on_time_rate_raw` (order level: reviews and lateness belong to orders,
so independent evidence is distinct orders, not items sold).

`compute_weights` is exposed standalone (not buried in a larger pipeline
function) so a future UI can recompute weights live from an edited pairwise
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import (
    AHP_CR_THRESHOLD,
    AHP_CRITERIA,
    AHP_CRITERION_DIRECTION,
    AHP_MIN_ORDERS,
    AHP_PAIRWISE_MATRIX,
    AHP_RANDOM_INDEX,
    AHP_SHRINKAGE_K_FALLBACK,
    AHP_SHRINKAGE_K_ON_TIME,
    AHP_SHRINKAGE_K_RANGE,
    AHP_SHRINKAGE_K_REVIEW,
    DISCOVERY_MIN_ORDERS,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    get_logger,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class PriorEstimate:
    """Method-of-moments empirical-Bayes estimate of one criterion's prior strength.

    `within_variance` is the variance of a single order's observation around its seller's own
    true value; `observed_variance` is the variance of the sellers' observed values;
    `noise_variance` is the part of that spread that is just sampling noise (the mean of
    `within_variance / n`); `between_variance` is what is left, the variance of the sellers'
    TRUE values; `k = within_variance / between_variance`.
    """

    criterion: str
    sellers: int
    within_variance: float
    observed_variance: float
    noise_variance: float
    between_variance: float
    k: float


def seller_order_table(fact: pd.DataFrame) -> pd.DataFrame:
    """One row per (seller, distinct order): the review and lateness that seller's order got.

    A review belongs to an ORDER, and so does lateness (every item of an order shares both),
    so independent evidence about a seller is its distinct orders, not its items sold. An
    order that contains items from two sellers counts once for each of them.

    Args:
        fact: FactOrderItems (seller_id, order_id, review_score, is_late).

    Returns:
        DataFrame with `seller_id`, `order_id`, `review` (NaN if the order has no review) and
        `late` (0/1).
    """
    return (
        fact.groupby(["seller_id", "order_id"], sort=False)
        .agg(review=("review_score", "first"), late=("is_late", "max"))
        .reset_index()
    )


def estimate_prior_strengths(
    fact: pd.DataFrame, min_orders: int = DISCOVERY_MIN_ORDERS
) -> dict[str, PriorEstimate]:
    """Estimate the shrinkage prior strength k for average review and on-time rate.

    Everything is measured at ORDER level (`seller_order_table`): one review and one
    lateness flag per distinct order. Average review: within-seller variance is the pooled
    variance of order review scores (weighted by n - 1, sellers with >= 2 reviewed orders);
    the between-seller variance of TRUE means is the variance of the observed seller means
    minus the mean sampling noise `within / n`. On-time rate is the binomial analogue:
    within variance `p(1 - p)` at the marketplace on-time rate, between variance from the
    spread of observed rates. `k = within / between`: the number of orders a seller needs
    before its own record counts as much as the marketplace average.

    Args:
        fact: FactOrderItems (seller_id, order_id, review_score, is_late, item_key).
        min_orders: only sellers with at least this many ITEMS SOLD are used (the pool the
            quality score is computed for, so the pools keep their definition).

    Returns:
        `{"avg_review_score": PriorEstimate, "on_time_rate": PriorEstimate}`. `k` is
        `inf` if the estimated between-seller variance is not positive.
    """
    sold = fact.groupby("seller_id")["item_key"].count()
    eligible = sold[sold >= min_orders].index
    orders = seller_order_table(fact)
    orders = orders[orders["seller_id"].isin(eligible)]
    by_seller = orders.groupby("seller_id")

    reviews = by_seller["review"].agg(n="count", mean="mean", var=lambda s: s.var(ddof=1))
    reviews = reviews.dropna()
    reviews = reviews[reviews["n"] >= 2]
    within_r = float((reviews["var"] * (reviews["n"] - 1)).sum() / (reviews["n"] - 1).sum())
    observed_r = float(reviews["mean"].var(ddof=1))
    noise_r = float((within_r / reviews["n"]).mean())

    rates = pd.DataFrame({"n": by_seller["order_id"].count(), "p": 1.0 - by_seller["late"].mean()})
    p_bar = 1.0 - float(seller_order_table(fact)["late"].mean())
    within_p = p_bar * (1.0 - p_bar)
    observed_p = float(rates["p"].var(ddof=1))
    noise_p = float((within_p / rates["n"]).mean())

    def _estimate(
        name: str, sellers: int, within: float, observed: float, noise: float
    ) -> PriorEstimate:
        between = observed - noise
        k = within / between if between > 0 else float("inf")
        return PriorEstimate(name, sellers, within, observed, noise, between, k)

    return {
        "avg_review_score": _estimate(
            "avg_review_score", len(reviews), within_r, observed_r, noise_r
        ),
        "on_time_rate": _estimate("on_time_rate", len(rates), within_p, observed_p, noise_p),
    }


def select_prior_strength(
    k_estimate: float,
    k_range: tuple[int, int] = AHP_SHRINKAGE_K_RANGE,
    fallback: int = AHP_SHRINKAGE_K_FALLBACK,
) -> int:
    """Whole-number k to use: the rounded estimate if it lies in `k_range`, else `fallback`."""
    low, high = k_range
    if np.isfinite(k_estimate) and low <= k_estimate <= high:
        return int(round(k_estimate))
    return fallback


def shrink_toward_mean(
    observed: pd.Series, n: pd.Series, global_mean: float, k: float
) -> pd.Series:
    """Empirical-Bayes shrinkage `(n * observed + k * global_mean) / (n + k)`.

    The weight on the seller's own value is `n / (n + k)`. A missing observation (n = 0)
    becomes the global mean.
    """
    own = observed.fillna(global_mean)
    return (n * own + k * global_mean) / (n + k)


def build_seller_criteria(
    fact: pd.DataFrame, dim_seller: pd.DataFrame, min_orders: int, shrink: bool = True
) -> pd.DataFrame:
    """Aggregate per-seller AHP criteria from FactOrderItems.

    Review and lateness are properties of an ORDER, shared by all of its items, so the
    review and on-time statistics are computed over each seller's distinct orders
    (`seller_order_table`), and `n_reviewed` / `n_orders` count distinct orders. Average
    review and on-time rate are noisy for small sellers, so by default they are shrunk
    toward the marketplace mean (`shrink_toward_mean`, k from `config`) with weight
    `n / (n + k)`, n being distinct reviewed orders for reviews and distinct delivered
    orders for on-time rate. Price and items sold (`order_volume`) are untouched, and
    eligibility is still defined by items sold. The raw (unshrunk) order-level values stay
    available as `avg_review_raw` and `on_time_rate_raw`.

    Args:
        fact: FactOrderItems (must contain seller_id, order_id, item_key, review_score,
            is_late, price).
        dim_seller: DimSeller (must contain seller_id, seller_state).
        min_orders: minimum order_volume (ITEMS SOLD) for a seller to be included.
        shrink: if False, `avg_review_score` and `on_time_rate` hold the raw values.

    Returns:
        DataFrame with columns `seller_id`, `seller_state`, `avg_review_score`,
        `on_time_rate` (shrunk unless `shrink=False`), `avg_price`, `order_volume`,
        `avg_review_raw`, `on_time_rate_raw`, `n_reviewed` (distinct reviewed orders),
        `n_orders` (distinct delivered orders); one row per eligible seller.
    """
    items = fact.groupby("seller_id").agg(
        avg_price=("price", "mean"),
        order_volume=("item_key", "count"),
    )
    orders = seller_order_table(fact)
    by_order = orders.groupby("seller_id").agg(
        avg_review_raw=("review", "mean"),
        late_share=("late", "mean"),
        n_reviewed=("review", "count"),
        n_orders=("order_id", "count"),
    )
    grouped = items.join(by_order).reset_index()
    grouped["on_time_rate_raw"] = 1.0 - grouped.pop("late_share")
    grouped["avg_review_score"] = grouped["avg_review_raw"]
    grouped["on_time_rate"] = grouped["on_time_rate_raw"]
    if shrink:
        grouped["avg_review_score"] = shrink_toward_mean(
            grouped["avg_review_raw"],
            grouped["n_reviewed"],
            float(orders["review"].mean()),
            AHP_SHRINKAGE_K_REVIEW,
        )
        grouped["on_time_rate"] = shrink_toward_mean(
            grouped["on_time_rate_raw"],
            grouped["n_orders"],
            1.0 - float(orders["late"].mean()),
            AHP_SHRINKAGE_K_ON_TIME,
        )
    grouped = grouped.merge(dim_seller[["seller_id", "seller_state"]], on="seller_id", how="left")
    eligible = grouped[grouped["order_volume"] >= min_orders].reset_index(drop=True)
    logger.info(
        "AHP seller criteria: %d sellers with >= %d items sold (of %d total), "
        "order-level shrinkage %s",
        len(eligible),
        min_orders,
        len(grouped),
        "on" if shrink else "off",
    )
    return eligible[
        [
            "seller_id", "seller_state", "avg_review_score", "on_time_rate", "avg_price",
            "order_volume", "avg_review_raw", "on_time_rate_raw", "n_reviewed", "n_orders",
        ]
    ]


def _validate_reciprocity(matrix: NDArray[np.float64]) -> None:
    """Validate that a pairwise comparison matrix is reciprocal.

    Args:
        matrix: square pairwise comparison matrix.

    Raises:
        AssertionError: if `matrix` is not square, has a non-1 diagonal, or
            `matrix[j][i] != 1 / matrix[i][j]` for any i, j.
    """
    n = matrix.shape[0]
    assert matrix.shape == (n, n), f"pairwise matrix must be square, got {matrix.shape}"
    assert np.allclose(np.diag(matrix), 1.0), "pairwise matrix diagonal must be all 1s"
    assert np.allclose(matrix, 1.0 / matrix.T), "pairwise matrix is not reciprocal"


def compute_weights(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float, float]:
    """Compute AHP criterion weights via the principal right eigenvector.

    Args:
        matrix: reciprocal Saaty pairwise comparison matrix (validated for
            reciprocity before use).

    Returns:
        Tuple `(weights, lambda_max, ci, cr)`: `weights` is the principal
        right eigenvector normalized to sum to 1; `lambda_max` is the
        largest real eigenvalue; `ci` is the consistency index
        `(lambda_max - n) / (n - 1)`; `cr` is the consistency ratio
        `ci / RI(n)` (0.0 when `RI(n) == 0`, i.e. n <= 2).
    """
    _validate_reciprocity(matrix)
    n = matrix.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues[max_index].real)
    principal_vector = eigenvectors[:, max_index].real
    weights = np.abs(principal_vector) / np.abs(principal_vector).sum()

    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = AHP_RANDOM_INDEX.get(n, AHP_RANDOM_INDEX[max(AHP_RANDOM_INDEX)])
    cr = ci / ri if ri > 0 else 0.0

    return weights, lambda_max, ci, cr


def normalize_criteria(criteria_df: pd.DataFrame, direction_map: dict[str, str]) -> pd.DataFrame:
    """Min-max normalize each criterion into a new `norm_<criterion>` column.

    The raw criterion columns (e.g. `order_volume`, `avg_price`) are left
    untouched — callers that need the seller's actual order count or price
    (e.g. `src.discovery`'s popularity axis) read the original column, not
    a normalized proxy for it.

    Args:
        criteria_df: DataFrame with one column per criterion in
            `direction_map` (plus any identifying columns, left untouched).
        direction_map: maps criterion name to `"benefit"` (higher is
            better, normalized to [0, 1] as-is) or `"cost"` (lower is
            better, normalized then inverted as `1 - x`).

    Returns:
        A copy of `criteria_df` with an added `norm_<criterion>` column in
        [0, 1] for each criterion in `direction_map`. A constant column
        normalizes to 1.0 for every row (no information to differentiate
        sellers on it).
    """
    normalized = criteria_df.copy()
    for criterion, direction in direction_map.items():
        col = criteria_df[criterion]
        col_min, col_max = col.min(), col.max()
        if col_max == col_min:
            normalized[f"norm_{criterion}"] = 1.0
            continue
        scaled = (col - col_min) / (col_max - col_min)
        normalized[f"norm_{criterion}"] = 1.0 - scaled if direction == "cost" else scaled
    return normalized


def score_sellers(
    normalized_df: pd.DataFrame, weights: NDArray[np.float64], criteria_order: tuple[str, ...]
) -> pd.DataFrame:
    """Compute the weighted AHP composite score and per-criterion contributions.

    Args:
        normalized_df: output of `normalize_criteria` (raw criterion
            columns plus `norm_<criterion>` columns in [0, 1], plus
            identifying columns such as `seller_id`).
        weights: criterion weights aligned to `criteria_order`, summing to 1.
        criteria_order: criterion names, in the same order as `weights`.

    Returns:
        A copy of `normalized_df` with one `contribution_<criterion>` column
        per criterion (`norm_<criterion> * weight`) and a final `ahp_score`
        column (their sum), sorted by `ahp_score` descending. Raw criterion
        columns (e.g. `order_volume`) are preserved unchanged.
    """
    result = normalized_df.copy()
    contributions = []
    for criterion, weight in zip(criteria_order, weights, strict=True):
        contrib_col = f"contribution_{criterion}"
        result[contrib_col] = normalized_df[f"norm_{criterion}"] * weight
        contributions.append(contrib_col)
    result["ahp_score"] = result[contributions].sum(axis=1)
    return result.sort_values("ahp_score", ascending=False).reset_index(drop=True)


def _write_prior_strength(fact: pd.DataFrame) -> None:
    """Write reports/ahp_prior_strength.csv: the estimates behind the configured k."""
    configured = {
        "avg_review_score": AHP_SHRINKAGE_K_REVIEW,
        "on_time_rate": AHP_SHRINKAGE_K_ON_TIME,
    }
    estimates = estimate_prior_strengths(fact)
    rows = []
    for name, est in estimates.items():
        rows.append(
            {
                "criterion": name,
                "sellers": est.sellers,
                "within_variance": est.within_variance,
                "observed_variance": est.observed_variance,
                "noise_variance": est.noise_variance,
                "between_variance": est.between_variance,
                "k_estimate": est.k,
                "k_selected_by_rule": select_prior_strength(est.k),
                "k_configured": configured[name],
            }
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "ahp_prior_strength.csv", index=False)
    logger.info(
        "Wrote ahp_prior_strength.csv: %s",
        {name: round(est.k, 2) for name, est in estimates.items()},
    )


def main() -> None:
    """Run the AHP pipeline for established sellers and write the ranking."""
    fact = pd.read_parquet(PROCESSED_DATA_DIR / "FactOrderItems.parquet")
    dim_seller = pd.read_parquet(PROCESSED_DATA_DIR / "DimSeller.parquet")
    _write_prior_strength(fact)

    criteria = build_seller_criteria(fact, dim_seller, min_orders=AHP_MIN_ORDERS)

    weights, lambda_max, ci, cr = compute_weights(AHP_PAIRWISE_MATRIX)
    logger.info(
        "AHP weights: %s (lambda_max=%.4f, CI=%.4f, CR=%.4f)",
        dict(zip(AHP_CRITERIA, weights.round(4), strict=True)),
        lambda_max,
        ci,
        cr,
    )
    if cr > AHP_CR_THRESHOLD:
        logger.warning(
            "AHP pairwise matrix consistency ratio %.4f exceeds threshold %.2f; "
            "judgments should be revisited.",
            cr,
            AHP_CR_THRESHOLD,
        )

    normalized = normalize_criteria(criteria, AHP_CRITERION_DIRECTION)
    ranked = score_sellers(normalized, weights, AHP_CRITERIA)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(REPORTS_DIR / "ahp_ranking.csv", index=False)
    logger.info("Wrote ahp_ranking.csv (%d established sellers, CR=%.4f)", len(ranked), cr)


if __name__ == "__main__":
    main()
