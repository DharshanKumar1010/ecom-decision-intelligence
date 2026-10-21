"""Unit tests for the table-building rules in `src.build_tables` (hand-built fixtures, no files).

These pin the definitions the whole project rests on: a delivered order has status "delivered" and
a delivered date; late means strictly after the estimate; one review per order, the latest by
creation date; an item key is order id plus item number.
"""

from __future__ import annotations

import pandas as pd

from src.build_tables import (
    _build_fact_order_items,
    _dedupe_latest_reviews,
    _derive_order_fields,
    _filter_delivered,
)


def _orders() -> pd.DataFrame:
    stamp = pd.Timestamp
    return pd.DataFrame(
        {
            "order_id": ["on_time", "exactly_on_the_day", "late", "canceled", "no_date"],
            "order_status": ["delivered", "delivered", "delivered", "canceled", "delivered"],
            "order_purchase_timestamp": [stamp("2018-01-01 10:00:00")] * 5,
            "order_delivered_customer_date": [
                stamp("2018-01-05 09:00:00"),
                stamp("2018-01-10 12:00:00"),
                stamp("2018-01-12 08:00:00"),
                stamp("2018-01-03 08:00:00"),
                pd.NaT,
            ],
            "order_estimated_delivery_date": [
                stamp("2018-01-10 00:00:00"),
                stamp("2018-01-10 12:00:00"),
                stamp("2018-01-10 00:00:00"),
                stamp("2018-01-10 00:00:00"),
                stamp("2018-01-10 00:00:00"),
            ],
        }
    )


def test_only_delivered_orders_with_a_delivered_date_are_kept() -> None:
    kept = _filter_delivered(_orders())
    assert sorted(kept["order_id"]) == ["exactly_on_the_day", "late", "on_time"]


def test_late_means_strictly_after_the_estimated_date() -> None:
    derived = _derive_order_fields(_filter_delivered(_orders())).set_index("order_id")
    assert derived.loc["late", "is_late"] == 1
    assert derived.loc["on_time", "is_late"] == 0
    assert derived.loc["exactly_on_the_day", "is_late"] == 0  # equal is not late
    assert set(derived["is_late"].unique()) <= {0, 1}


def test_delivery_days_and_order_date_come_from_the_purchase_time() -> None:
    derived = _derive_order_fields(_filter_delivered(_orders())).set_index("order_id")
    assert derived.loc["on_time", "delivery_days"] == 3  # 4 days minus one hour, whole days
    assert derived.loc["late", "delivery_days"] == 10
    assert (derived["order_date"] == pd.Timestamp("2018-01-01")).all()


def test_the_latest_review_of_an_order_is_kept_and_orders_stay_unique() -> None:
    reviews = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3"],
            "review_score": [1, 5, 3, 4],
            "review_creation_date": ["2018-01-01", "2018-02-01", "2018-01-15", None],
        }
    )
    kept = _dedupe_latest_reviews(reviews).set_index("order_id")
    assert len(kept) == 3 and kept.index.is_unique
    assert kept.loc["o1", "review_score"] == 5  # the later review wins
    assert kept.loc["o3", "review_score"] == 4  # an undated review is kept when it is the only one


def test_item_key_is_order_id_plus_item_number_and_undelivered_items_drop_out() -> None:
    orders = _derive_order_fields(_filter_delivered(_orders()))
    items = pd.DataFrame(
        {
            "order_id": ["late", "late", "canceled", "on_time"],
            "order_item_id": [1, 2, 1, 1],
            "product_id": ["p1", "p2", "p3", "p4"],
            "seller_id": ["s1", "s1", "s2", "s3"],
            "price": [10.0, 20.0, 30.0, 40.0],
            "freight_value": [1.0, 1.0, 1.0, 1.0],
        }
    )
    reviews = pd.DataFrame(
        {"order_id": ["late"], "review_score": [2], "review_creation_date": [pd.NaT]}
    )
    fact = _build_fact_order_items(orders, items, reviews)
    assert sorted(fact["item_key"]) == ["late_1", "late_2", "on_time_1"]
    assert fact.loc[fact["order_id"] == "late", "review_score"].tolist() == [
        2,
        2,
    ]  # repeated per item
    assert fact.loc[fact["order_id"] == "on_time", "review_score"].isna().all()
