"""AHP multi-criteria seller scoring (CLAUDE.md section 7.4).

Criteria (order fixed by `config.AHP_CRITERIA`): average review score,
on-time rate, average price, order volume — computed per seller from
`FactOrderItems`, restricted to sellers with at least `min_orders` orders.
The main ("established sellers") ranking uses `config.AHP_MIN_ORDERS` (30);
`src.discovery` reuses `build_seller_criteria`/`compute_weights`/
`normalize_criteria`/`score_sellers` at a lower threshold rather than
reimplementing any of this scoring logic.

`compute_weights` is exposed standalone (not buried in a larger pipeline
function) so a future UI can recompute weights live from an edited pairwise
matrix.
"""

from __future__ import annotations

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
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    get_logger,
)

logger = get_logger(__name__)


def build_seller_criteria(
    fact: pd.DataFrame, dim_seller: pd.DataFrame, min_orders: int
) -> pd.DataFrame:
    """Aggregate per-seller AHP criteria from FactOrderItems.

    Args:
        fact: FactOrderItems (must contain seller_id, review_score, is_late,
            price).
        dim_seller: DimSeller (must contain seller_id, seller_state).
        min_orders: minimum order_volume for a seller to be included.

    Returns:
        DataFrame with columns `seller_id`, `seller_state`,
        `avg_review_score`, `on_time_rate`, `avg_price`, `order_volume`,
        one row per eligible seller.
    """
    grouped = fact.groupby("seller_id").agg(
        avg_review_score=("review_score", "mean"),
        on_time_rate=("is_late", lambda s: 1.0 - s.mean()),
        avg_price=("price", "mean"),
        order_volume=("item_key", "count"),
    )
    grouped = grouped.reset_index()
    grouped = grouped.merge(dim_seller[["seller_id", "seller_state"]], on="seller_id", how="left")
    eligible = grouped[grouped["order_volume"] >= min_orders].reset_index(drop=True)
    logger.info(
        "AHP seller criteria: %d sellers with >= %d orders (of %d total)",
        len(eligible),
        min_orders,
        len(grouped),
    )
    return eligible


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


def main() -> None:
    """Run the AHP pipeline for established sellers and write the ranking."""
    fact = pd.read_parquet(PROCESSED_DATA_DIR / "FactOrderItems.parquet")
    dim_seller = pd.read_parquet(PROCESSED_DATA_DIR / "DimSeller.parquet")

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
