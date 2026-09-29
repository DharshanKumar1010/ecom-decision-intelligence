"""Central configuration: paths, seed, logging, and shared constants.

Per CLAUDE.md section 5: all paths, constants, the seed, and the AHP matrix
live here. No magic numbers or hard-coded paths in other modules.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RULES_DIR: Path = PROJECT_ROOT / "rules"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

SEED: int = 42

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with a consistent format.

    Args:
        name: usually ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# Raw data contract
# --------------------------------------------------------------------------

EXPECTED_RAW_FILES: tuple[str, ...] = (
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
)


def verify_raw_files(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Verify all expected Olist CSVs are present in ``raw_dir``.

    Args:
        raw_dir: directory expected to contain the 9 raw Olist CSVs.

    Raises:
        FileNotFoundError: listing every missing expected file, if any are missing.
    """
    missing = [name for name in EXPECTED_RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} expected raw data file(s) in {raw_dir}: "
            + ", ".join(missing)
        )


# --------------------------------------------------------------------------
# Predictive model feature contract (src/predict.py)
#
# Stage 3 (CLAUDE.md section 17) expands this list one feature at a time,
# each independently leakage-checked; this is the CURRENT, real feature set
# after that process, not the original Stage 2 4-feature baseline. See
# BUILD_LOG.md for the before/after AUC of each addition, including any
# that were tried and backed out.
# --------------------------------------------------------------------------

ALLOWED_FEATURES: tuple[str, ...] = (
    "price",
    "freight_value",
    "product_weight_g",
    "same_state",
    "product_category_freq",
    "order_month",
    "day_of_week",
    "n_items_in_order",
    "payment_installments",
)

# --------------------------------------------------------------------------
# AHP placeholder (src/ahp.py, built in a later stage)
#
# Criteria order: avg_review_score, on_time_rate, avg_price, order_volume.
# Direction: whether higher is better ("benefit") or lower is better ("cost").
# avg_price is provisionally "cost" (lower average price favors accessibility);
# this judgment is not yet validated against data and must be reviewed when
# ahp.py is implemented.
#
# The pairwise matrix below is a PLACEHOLDER Saaty judgment (reciprocal by
# construction: matrix[j][i] == 1 / matrix[i][j]) so config.py has the shape
# ahp.py will consume. It is not yet justified against real distributions and
# must be revisited in the AHP stage.
# --------------------------------------------------------------------------

AHP_CRITERIA: tuple[str, ...] = (
    "avg_review_score",
    "on_time_rate",
    "avg_price",
    "order_volume",
)

AHP_CRITERION_DIRECTION: dict[str, str] = {
    "avg_review_score": "benefit",
    "on_time_rate": "benefit",
    "avg_price": "cost",
    "order_volume": "benefit",
}

AHP_PAIRWISE_MATRIX: NDArray[np.float64] = np.array(
    [
        [1.0, 2.0, 5.0, 3.0],
        [1 / 2, 1.0, 4.0, 2.0],
        [1 / 5, 1 / 4, 1.0, 1 / 3],
        [1 / 3, 1 / 2, 3.0, 1.0],
    ],
    dtype=np.float64,
)

# Minimum orders for a seller to enter the main ("established sellers") AHP
# ranking (src/ahp.py). Saaty's Random Index, keyed by matrix size n, used to
# compute the consistency ratio CR = CI / RI(n). CR is flagged, not treated
# as a hard failure, if it exceeds AHP_CR_THRESHOLD.
AHP_MIN_ORDERS: int = 30

AHP_RANDOM_INDEX: dict[int, float] = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

AHP_CR_THRESHOLD: float = 0.10

# --------------------------------------------------------------------------
# Sentiment thresholds (src/sentiment.py, Stage 2)
#
# VADER compound-score cutoffs for the three-way label. These are VADER's
# own published defaults, not tuned on this project's data.
# --------------------------------------------------------------------------

SENTIMENT_POSITIVE_THRESHOLD: float = 0.05
SENTIMENT_NEGATIVE_THRESHOLD: float = -0.05

# --------------------------------------------------------------------------
# Discovery / hidden-gems thresholds (src/discovery.py, Stage 2)
#
# DISCOVERY_MIN_ORDERS is the lower AHP eligibility threshold (vs.
# AHP_MIN_ORDERS above) so smaller/newer sellers can appear in the
# quality-vs-popularity quadrant at all. DISCOVERY_CONFIDENCE_MIN_ORDERS is
# the separate "how sure are we" cutoff below which a seller's row is
# flagged low-confidence regardless of which quadrant it lands in. Both are
# justified against real order-count distributions in
# docs/knowledge_engineering.md, not chosen arbitrarily.
# DISCOVERY_QUALITY_THRESHOLD / DISCOVERY_POPULARITY_THRESHOLD are override
# hooks: None means "use the median of that axis," computed at call time in
# discovery.compute_quadrants, so the split is never hard-coded into the
# function itself.
# --------------------------------------------------------------------------

DISCOVERY_MIN_ORDERS: int = 5
DISCOVERY_CONFIDENCE_MIN_ORDERS: int = 15
DISCOVERY_QUALITY_THRESHOLD: float | None = None
DISCOVERY_POPULARITY_THRESHOLD: float | None = None

# --------------------------------------------------------------------------
# Predictive model guardrails (src/predict.py, Stage 2)
#
# PREDICT_AUC_LEAK_GUARDRAIL: a test-set ROC-AUC above this is treated as a
# probable target leak given the deliberately small, non-date feature set in
# ALLOWED_FEATURES, and predict.py raises rather than silently reporting it.
# PREDICT_SENSITIVITY_PCTS: perturbation sizes used in the sensitivity
# analysis (binary same_state is flipped 0/1 instead of scaled).
# --------------------------------------------------------------------------

PREDICT_AUC_LEAK_GUARDRAIL: float = 0.95
PREDICT_SENSITIVITY_PCTS: tuple[float, ...] = (-0.25, -0.10, 0.10, 0.25)

# --------------------------------------------------------------------------
# Expert system (src/expert.py + rules/seller_rules.yaml, Stage 2)
#
# PROMOTE_LATE_RISK_THRESHOLD is used by the Promote rules in
# rules/seller_rules.yaml (kept here, not just inline in the YAML, since it
# is also referenced directly by CLAUDE.md section 7.7.2's rule spec and by
# discovery.py's docs).
# --------------------------------------------------------------------------

PROMOTE_LATE_RISK_THRESHOLD: float = 0.3
