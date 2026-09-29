"""Tests for src.predict: feature contract, leakage guardrail, reproducibility.

Uses the real processed tables (via conftest fixtures) since the feature
table depends on joins across Fact/Dim tables that would be tedious and
less convincing to hand-build; each test still runs a fast, deterministic
train/test split rather than the full model-metrics pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import ALLOWED_FEATURES, PREDICT_AUC_LEAK_GUARDRAIL
from src.predict import (
    _build_pipelines,
    _evaluate,
    _split_and_encode,
    build_feature_table,
    encode_product_category_frequency,
    fit_product_category_frequency,
)

_FORBIDDEN_COLUMNS = {
    "delivery_days",
    "order_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_purchase_timestamp",
    "is_late",
}


@pytest.fixture(scope="module")
def feature_table(
    fact_order_items: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_seller: pd.DataFrame,
    dim_customer: pd.DataFrame,
) -> pd.DataFrame:
    return build_feature_table(fact_order_items, dim_product, dim_seller, dim_customer)


def test_feature_columns_equal_allowed_set_exactly(feature_table: pd.DataFrame) -> None:
    # build_feature_table returns every precomputable ALLOWED_FEATURES column
    # plus the raw product_category string (encoded post-split, not itself a
    # feature) and item_key/is_late.
    columns = set(feature_table.columns) - {"item_key", "is_late", "product_category"}
    assert columns == set(ALLOWED_FEATURES) - {"product_category_freq"}


def test_product_category_frequency_fit_and_encode() -> None:
    train_categories = pd.Series(["toys", "toys", "toys", "books", "books", "games"])
    freq_map = fit_product_category_frequency(train_categories)

    assert freq_map["toys"] == pytest.approx(3 / 6)
    assert freq_map["books"] == pytest.approx(2 / 6)
    assert freq_map["games"] == pytest.approx(1 / 6)
    assert sum(freq_map.values()) == pytest.approx(1.0)

    encoded = encode_product_category_frequency(
        pd.Series(["toys", "books", "unseen_category"]), freq_map
    )
    assert encoded.iloc[0] == pytest.approx(3 / 6)
    assert encoded.iloc[1] == pytest.approx(2 / 6)
    assert encoded.iloc[2] == pytest.approx(0.0)  # never observed in training


def test_order_month_and_day_of_week_derived_correctly() -> None:
    fact = pd.DataFrame(
        {
            "item_key": ["i1", "i2"],
            "seller_id": ["S1", "S1"],
            "product_id": ["P1", "P1"],
            "customer_id": ["C1", "C1"],
            "price": [10.0, 10.0],
            "freight_value": [1.0, 1.0],
            "order_purchase_timestamp": [
                pd.Timestamp("2018-03-05 10:00:00"),  # Monday, March
                pd.Timestamp("2017-12-25 08:00:00"),  # Monday, December
            ],
            "is_late": [0, 1],
        }
    )
    dim_product = pd.DataFrame({
        "product_id": ["P1"], "product_weight_g": [500.0],
        "product_category_name_english": ["toys"],
    })
    dim_seller = pd.DataFrame({"seller_id": ["S1"], "seller_state": ["SP"]})
    dim_customer = pd.DataFrame({"customer_id": ["C1"], "customer_state": ["SP"]})

    result = build_feature_table(fact, dim_product, dim_seller, dim_customer)

    assert result.loc[0, "order_month"] == 3.0
    assert result.loc[0, "day_of_week"] == 0.0  # Monday == 0 (pandas dayofweek convention)
    assert result.loc[1, "order_month"] == 12.0
    assert result.loc[1, "day_of_week"] == 0.0


def test_split_and_encode_uses_train_only_frequencies(feature_table: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test, freq_map = _split_and_encode(feature_table)
    assert set(x_train.columns) == set(ALLOWED_FEATURES)
    assert set(x_test.columns) == set(ALLOWED_FEATURES)
    assert set(x_train.index).isdisjoint(set(x_test.index))
    assert x_train["product_category_freq"].between(0.0, 1.0).all()
    assert x_test["product_category_freq"].between(0.0, 1.0).all()
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)
    assert freq_map  # non-empty: fit on a non-trivial train fold


def test_no_forbidden_leakage_columns(feature_table: pd.DataFrame) -> None:
    assert _FORBIDDEN_COLUMNS.isdisjoint(set(ALLOWED_FEATURES))
    # is_late is the target and is expected in feature_table; every other
    # forbidden (date-derived) column must be absent from it.
    assert (_FORBIDDEN_COLUMNS - {"is_late"}).isdisjoint(feature_table.columns)


def test_train_test_disjoint(feature_table: pd.DataFrame) -> None:
    x_train, x_test, _, _, _ = _split_and_encode(feature_table)
    assert set(x_train.index).isdisjoint(set(x_test.index))


def test_late_risk_in_unit_interval(feature_table: pd.DataFrame) -> None:
    x_train, x_test, y_train, _, _ = _split_and_encode(feature_table)
    lr_pipeline, _ = _build_pipelines()
    lr_pipeline.fit(x_train, y_train)
    late_risk = lr_pipeline.predict_proba(x_test)[:, 1]
    assert ((late_risk >= 0.0) & (late_risk <= 1.0)).all()


def test_auc_below_leak_guardrail(feature_table: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test, _ = _split_and_encode(feature_table)
    lr_pipeline, dt_pipeline = _build_pipelines()
    lr_pipeline.fit(x_train, y_train)
    dt_pipeline.fit(x_train, y_train)
    assert _evaluate(lr_pipeline, x_test, y_test)["roc_auc"] < PREDICT_AUC_LEAK_GUARDRAIL
    assert _evaluate(dt_pipeline, x_test, y_test)["roc_auc"] < PREDICT_AUC_LEAK_GUARDRAIL


def test_seed_reproducibility(feature_table: pd.DataFrame) -> None:
    def _fit_predict() -> pd.Series:
        x_train, x_test, y_train, _, _ = _split_and_encode(feature_table)
        lr_pipeline, _ = _build_pipelines()
        lr_pipeline.fit(x_train, y_train)
        return lr_pipeline.predict_proba(x_test)[:, 1]

    first = _fit_predict()
    second = _fit_predict()
    assert (first == second).all()
