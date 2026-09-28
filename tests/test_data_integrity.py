"""Integration tests for src.build_tables output. Skipped if raw data is absent."""

from __future__ import annotations

import pandas as pd


def test_item_key_unique(fact_order_items: pd.DataFrame) -> None:
    assert fact_order_items["item_key"].is_unique


def test_row_count_in_range(fact_order_items: pd.DataFrame) -> None:
    assert 100_000 <= len(fact_order_items) <= 115_000


def test_no_null_seller_id(fact_order_items: pd.DataFrame) -> None:
    assert fact_order_items["seller_id"].notna().all()


def test_is_late_binary(fact_order_items: pd.DataFrame) -> None:
    assert set(fact_order_items["is_late"].unique()) <= {0, 1}


def test_no_orphan_seller_id(fact_order_items: pd.DataFrame, dim_seller: pd.DataFrame) -> None:
    orphan = set(fact_order_items["seller_id"]) - set(dim_seller["seller_id"])
    assert not orphan


def test_no_orphan_product_id(fact_order_items: pd.DataFrame, dim_product: pd.DataFrame) -> None:
    orphan = set(fact_order_items["product_id"]) - set(dim_product["product_id"])
    assert not orphan
