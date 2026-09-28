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
# Predictive model feature contract (src/predict.py, built in a later stage)
# --------------------------------------------------------------------------

ALLOWED_FEATURES: tuple[str, ...] = (
    "price",
    "freight_value",
    "product_weight_g",
    "same_state",
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
