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
EXPORT_DATA_DIR: Path = DATA_DIR / "export"
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
    "seller_historical_late_rate",
    "geo_distance",
    "geo_distance_missing",
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

# Empirical-Bayes shrinkage of the two rate-like AHP criteria (src/ahp.py): a seller's
# average review and on-time rate are pulled toward the marketplace mean with weight
# k / (n + k), so a handful of perfect reviews cannot outrank a long, strong record.
# n counts distinct ORDERS (a review and lateness belong to an order and repeat across its
# items): reviewed orders for the review, delivered orders for the on-time rate.
# k is the method-of-moments prior strength (within-seller variance / between-seller
# variance of true values) at ORDER level, estimated by src.ahp.estimate_prior_strengths on
# the pool of sellers with >= DISCOVERY_MIN_ORDERS items sold and rounded:
# review 17.8 -> 18, on-time 25.9 -> 26.
# Both estimates fall inside AHP_SHRINKAGE_K_RANGE (the range tested in
# scripts/diagnostics/diagnostic_shrinkage.py); an estimate outside it would use the fallback
# instead.
# Derivation, intermediate variances and sensitivities: docs/knowledge_engineering.md
# section 8. A test fails if these constants stop matching the data-derived estimate.
AHP_SHRINKAGE_K_REVIEW: int = 18
AHP_SHRINKAGE_K_ON_TIME: int = 26
AHP_SHRINKAGE_K_RANGE: tuple[int, int] = (5, 30)
AHP_SHRINKAGE_K_FALLBACK: int = 10

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

# Order-grouped evaluation (src/predict.py). Items of one order share is_late and most
# features, so rows are NEVER split at random: StratifiedGroupKFold over order_id makes
# PREDICT_N_FOLDS folds. Fold 0 is the train/test split (1 / PREDICT_N_FOLDS = 20% test) used
# for metrics, permutation importance, sensitivity and the persisted models; all folds
# together give the out-of-fold late_risk written to LatePredictions.parquet.
PREDICT_N_FOLDS: int = 5

# RandomForest leaf size. Chosen from {1, 5, 20} under the order-grouped out-of-fold protocol
# (reports/diagnostics/diagnostic_rf_leaf.txt): 20 gave the highest pooled OOF AUC (0.7367 vs
# 0.7233 for fully grown trees) and a fitted model about 16% of the size. It was chosen on the
# same folds it is reported on, so the gain is slightly optimistic.
PREDICT_RF_MIN_SAMPLES_LEAF: int = 20

# --------------------------------------------------------------------------
# Expert system (src/expert.py + rules/seller_rules.yaml)
#
# Every rule threshold lives in rules/seller_rules.yaml and nowhere else. Two unused
# copies of the Promote cutoffs used to sit here; one had drifted (0.05 against the
# file's current value), so both were removed. The reasoning behind each cutoff is in
# docs/knowledge_engineering.md; the current values and fire counts are in
# reports/RESULTS.md.
# --------------------------------------------------------------------------
