"""Tests for src.predict: feature contract, leakage guardrail, reproducibility.

Uses the real processed tables (via conftest fixtures) since the feature
table depends on joins across Fact/Dim tables that would be tedious and
less convincing to hand-build; each test still runs a fast, deterministic
train/test split rather than the full model-metrics pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.config import ALLOWED_FEATURES, PREDICT_AUC_LEAK_GUARDRAIL, SEED
from src.predict import _build_pipelines, _evaluate, build_feature_table

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
    columns = set(feature_table.columns) - {"item_key", "is_late"}
    assert columns == set(ALLOWED_FEATURES)


def test_no_forbidden_leakage_columns(feature_table: pd.DataFrame) -> None:
    assert _FORBIDDEN_COLUMNS.isdisjoint(set(ALLOWED_FEATURES))
    # is_late is the target and is expected in feature_table; every other
    # forbidden (date-derived) column must be absent from it.
    assert (_FORBIDDEN_COLUMNS - {"is_late"}).isdisjoint(feature_table.columns)


def test_train_test_disjoint(feature_table: pd.DataFrame) -> None:
    x = feature_table[list(ALLOWED_FEATURES)]
    y = feature_table["is_late"]
    x_train, x_test, _, _ = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    assert set(x_train.index).isdisjoint(set(x_test.index))


def test_late_risk_in_unit_interval(feature_table: pd.DataFrame) -> None:
    x = feature_table[list(ALLOWED_FEATURES)]
    y = feature_table["is_late"]
    x_train, x_test, y_train, _ = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    lr_pipeline, _ = _build_pipelines()
    lr_pipeline.fit(x_train, y_train)
    late_risk = lr_pipeline.predict_proba(x_test)[:, 1]
    assert ((late_risk >= 0.0) & (late_risk <= 1.0)).all()


def test_auc_below_leak_guardrail(feature_table: pd.DataFrame) -> None:
    x = feature_table[list(ALLOWED_FEATURES)]
    y = feature_table["is_late"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    lr_pipeline, dt_pipeline = _build_pipelines()
    lr_pipeline.fit(x_train, y_train)
    dt_pipeline.fit(x_train, y_train)
    assert _evaluate(lr_pipeline, x_test, y_test)["roc_auc"] < PREDICT_AUC_LEAK_GUARDRAIL
    assert _evaluate(dt_pipeline, x_test, y_test)["roc_auc"] < PREDICT_AUC_LEAK_GUARDRAIL


def test_seed_reproducibility(feature_table: pd.DataFrame) -> None:
    x = feature_table[list(ALLOWED_FEATURES)]
    y = feature_table["is_late"]

    def _fit_predict() -> pd.Series:
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=SEED
        )
        lr_pipeline, _ = _build_pipelines()
        lr_pipeline.fit(x_train, y_train)
        return lr_pipeline.predict_proba(x_test)[:, 1]

    first = _fit_predict()
    second = _fit_predict()
    assert (first == second).all()
