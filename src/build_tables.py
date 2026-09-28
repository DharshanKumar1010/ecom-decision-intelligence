"""Build cleaned fact/dimension tables from raw Olist CSVs (CLAUDE.md section 6.2).

Reads the 9 raw Olist CSVs, applies the delivered-only filter, derives
delivery/lateness fields, dedupes reviews to the latest per order, joins the
English category translation, and writes FactOrderItems/DimSeller/DimProduct/
DimCustomer as Parquet to ``data/processed/``. Writes a data-quality summary
to ``reports/dq_report.md``.
"""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR, get_logger, verify_raw_files

logger = get_logger(__name__)

# Only the columns named in CLAUDE.md section 6.1 are selected from each raw
# file, even though the real CSVs contain additional columns (e.g. order
# approval/carrier timestamps, shipping_limit_date, product dimensions, the
# dataset's own "product_name_lenght"/"product_description_lenght" typos).
_ORDERS_COLS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
_ITEMS_COLS = ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]
_REVIEWS_COLS = ["order_id", "review_score", "review_creation_date"]
_PRODUCTS_COLS = ["product_id", "product_category_name", "product_weight_g"]
_SELLERS_COLS = ["seller_id", "seller_state"]
_CUSTOMERS_COLS = ["customer_id", "customer_state"]
_TRANSLATION_COLS = ["product_category_name", "product_category_name_english"]

_TIMESTAMP_COLS = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

_FACT_ROW_MIN = 100_000
_FACT_ROW_MAX = 115_000


def _load_csv(filename: str, usecols: list[str]) -> pd.DataFrame:
    """Load a raw Olist CSV, selecting only the given columns.

    Args:
        filename: name of the CSV under ``RAW_DATA_DIR``.
        usecols: columns to select from the file.

    Returns:
        DataFrame containing only ``usecols``.
    """
    path = RAW_DATA_DIR / filename
    # product_category_name_translation.csv ships with a UTF-8 BOM; reading it
    # with plain utf-8 would turn the first column into "﻿product_category_name"
    # and silently break the category join.
    encoding = "utf-8-sig" if filename == "product_category_name_translation.csv" else "utf-8"
    return pd.read_csv(path, usecols=usecols, encoding=encoding)


def _parse_timestamps(orders: pd.DataFrame) -> pd.DataFrame:
    """Parse order timestamp columns, coercing invalid values to NaT.

    Args:
        orders: orders DataFrame with raw timestamp columns.

    Returns:
        A copy of ``orders`` with timestamp columns parsed to datetime.
    """
    orders = orders.copy()
    for col in _TIMESTAMP_COLS:
        before_na = orders[col].isna().sum()
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
        new_na = int(orders[col].isna().sum() - before_na)
        if new_na > 0:
            logger.info("Column %s: %d values coerced to NaT during parsing", col, new_na)
    return orders


def _filter_delivered(orders: pd.DataFrame) -> pd.DataFrame:
    """Keep only delivered orders with a non-null delivered-customer date.

    Args:
        orders: orders DataFrame with parsed timestamps.

    Returns:
        The filtered DataFrame.
    """
    before = len(orders)
    filtered = orders[
        (orders["order_status"] == "delivered") & orders["order_delivered_customer_date"].notna()
    ].copy()
    logger.info(
        "Delivered-only filter: %d -> %d rows (%d dropped)",
        before,
        len(filtered),
        before - len(filtered),
    )
    return filtered


def _derive_order_fields(orders: pd.DataFrame) -> pd.DataFrame:
    """Derive order_date, delivery_days, and is_late on delivered orders.

    Args:
        orders: delivered-only orders DataFrame with parsed timestamps.

    Returns:
        A copy of ``orders`` with the derived columns added.
    """
    orders = orders.copy()
    orders["order_date"] = orders["order_purchase_timestamp"].dt.normalize()
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.days
    orders["is_late"] = (
        orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
    ).astype(int)
    return orders


def _dedupe_latest_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest review per order_id by review_creation_date.

    Args:
        reviews: raw reviews DataFrame (may contain multiple reviews per order).

    Returns:
        Deduplicated reviews DataFrame, one row per ``order_id``.
    """
    reviews = reviews.copy()
    before = len(reviews)
    reviews["review_creation_date"] = pd.to_datetime(
        reviews["review_creation_date"], errors="coerce"
    )
    reviews = reviews.sort_values("review_creation_date", ascending=False, na_position="last")
    reviews = reviews.drop_duplicates(subset="order_id", keep="first")
    logger.info("Review dedupe (latest per order_id): %d -> %d rows", before, len(reviews))
    return reviews


def _build_fact_order_items(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble FactOrderItems by joining delivered orders, items, and reviews.

    Args:
        orders: delivered-only orders with derived fields.
        items: raw order items.
        reviews: deduplicated (latest-only) reviews.

    Returns:
        The FactOrderItems DataFrame with an ``item_key`` primary key.
    """
    fact = items.merge(orders, on="order_id", how="inner")
    fact = fact.merge(reviews, on="order_id", how="left")
    fact["item_key"] = fact["order_id"] + "_" + fact["order_item_id"].astype(str)
    return fact


def _join_category_translation(
    products: pd.DataFrame, translation: pd.DataFrame
) -> pd.DataFrame:
    """Join English category names onto products, filling missing with 'unknown'.

    Args:
        products: raw products DataFrame.
        translation: category translation table.

    Returns:
        Products DataFrame with a ``product_category_name_english`` column.
    """
    products = products.merge(translation, on="product_category_name", how="left")
    products["product_category_name_english"] = products["product_category_name_english"].fillna(
        "unknown"
    )
    return products


def _run_assertions(fact: pd.DataFrame) -> None:
    """Validate FactOrderItems invariants, raising loudly on violation.

    Args:
        fact: the assembled FactOrderItems DataFrame.

    Raises:
        AssertionError: if any invariant is violated.
    """
    assert fact["item_key"].is_unique, "item_key is not unique in FactOrderItems"
    n = len(fact)
    assert _FACT_ROW_MIN <= n <= _FACT_ROW_MAX, (
        f"FactOrderItems row count {n} outside expected range "
        f"[{_FACT_ROW_MIN}, {_FACT_ROW_MAX}]"
    )
    assert fact["seller_id"].notna().all(), "FactOrderItems has null seller_id"
    assert set(fact["is_late"].unique()) <= {0, 1}, "is_late contains values outside {0, 1}"


def _write_dq_report(
    orders_raw_count: int,
    orders_delivered_count: int,
    fact: pd.DataFrame,
) -> None:
    """Write the data-quality summary report.

    Args:
        orders_raw_count: row count of raw orders before filtering.
        orders_delivered_count: row count of orders after the delivered-only filter.
        fact: the final FactOrderItems DataFrame.
    """
    null_rates = fact.isna().mean().sort_values(ascending=False)
    late_rate = fact["is_late"].mean()

    lines = [
        "# Data Quality Report",
        "",
        "## Row counts",
        f"- Raw orders: {orders_raw_count}",
        f"- Delivered orders (after filter): {orders_delivered_count}",
        f"- FactOrderItems rows: {len(fact)}",
        "",
        "## Null rates (FactOrderItems columns)",
    ]
    for col, rate in null_rates.items():
        lines.append(f"- {col}: {rate:.4%}")
    lines += [
        "",
        "## Late delivery rate",
        f"- is_late == 1 share: {late_rate:.4%}",
        "",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dq_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote data-quality report to %s", REPORTS_DIR / "dq_report.md")


def main() -> None:
    """Run the full build_tables pipeline: load, clean, derive, assert, write."""
    verify_raw_files()

    orders_raw = _load_csv("olist_orders_dataset.csv", _ORDERS_COLS)
    orders_raw_count = len(orders_raw)

    orders = _parse_timestamps(orders_raw)
    orders = _filter_delivered(orders)
    orders_delivered_count = len(orders)
    orders = _derive_order_fields(orders)

    items = _load_csv("olist_order_items_dataset.csv", _ITEMS_COLS)
    reviews_raw = _load_csv("olist_order_reviews_dataset.csv", _REVIEWS_COLS)
    reviews = _dedupe_latest_reviews(reviews_raw)

    products = _load_csv("olist_products_dataset.csv", _PRODUCTS_COLS).drop_duplicates(
        subset="product_id"
    )
    translation = _load_csv("product_category_name_translation.csv", _TRANSLATION_COLS)
    products = _join_category_translation(products, translation)

    sellers = _load_csv("olist_sellers_dataset.csv", _SELLERS_COLS).drop_duplicates(
        subset="seller_id"
    )
    customers = _load_csv("olist_customers_dataset.csv", _CUSTOMERS_COLS).drop_duplicates(
        subset="customer_id"
    )

    fact = _build_fact_order_items(orders, items, reviews)

    _run_assertions(fact)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fact.to_parquet(PROCESSED_DATA_DIR / "FactOrderItems.parquet", index=False)
    sellers.to_parquet(PROCESSED_DATA_DIR / "DimSeller.parquet", index=False)
    products.to_parquet(PROCESSED_DATA_DIR / "DimProduct.parquet", index=False)
    customers.to_parquet(PROCESSED_DATA_DIR / "DimCustomer.parquet", index=False)
    logger.info("Wrote FactOrderItems (%d rows) and 3 dimension tables", len(fact))

    _write_dq_report(orders_raw_count, orders_delivered_count, fact)


if __name__ == "__main__":
    main()
