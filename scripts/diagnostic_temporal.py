"""One-off diagnostic: does RandomForest's lead survive a forward-looking (temporal) split?

NOT part of the pipeline: not wired into `run_all.py`; the only file written is
`reports/diagnostic_temporal.txt`. Run directly: `python scripts/diagnostic_temporal.py`
(slow: two RandomForest fits).

Train on the earliest 80% of ORDERS by `order_purchase_timestamp` and test on the latest 20%
(all items of an order follow their order, so no order is in both sets and no test order is
earlier than a train order). Logistic regression, decision tree and random forest keep their
committed configs. Run twice: with all 12 features, and without `order_month` and
`day_of_week`, because a model that leans on the calendar can only learn the periods it was
trained on and a forward-looking split removes that crutch.

The product-category frequency encoding is fitted on the train rows only, and the imputers
sit inside the pipelines, exactly as in `src.predict`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALLOWED_FEATURES, PROCESSED_DATA_DIR, REPORTS_DIR  # noqa: E402
from src.predict import (  # noqa: E402
    _build_pipelines,
    _build_rf_pipeline,
    _load_geo_centroids,
    _load_payments,
    _load_seller_customer_zips,
    build_feature_table,
    encode_product_category_frequency,
    fit_product_category_frequency,
)

_CALENDAR = ("order_month", "day_of_week")
_TRAIN_SHARE = 0.8


def temporal_masks(fact: pd.DataFrame, table_keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Boolean train/test masks over `table_keys` (item keys): earliest 80% of orders train."""
    orders = fact.drop_duplicates("order_id")[["order_id", "order_purchase_timestamp"]]
    orders = orders.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    cutoff = int(len(orders) * _TRAIN_SHARE)
    train_orders = set(orders["order_id"].iloc[:cutoff])
    order_of_item = table_keys.map(fact.set_index("item_key")["order_id"])
    train = order_of_item.isin(train_orders).to_numpy()
    return train, ~train


def evaluate(table: pd.DataFrame, train: np.ndarray, test: np.ndarray,
             features: list[str]) -> dict[str, float]:
    """Fit the three models on the train rows with `features`; return test ROC-AUCs."""
    train_part, test_part = table[train], table[test]
    freq = fit_product_category_frequency(train_part["product_category"])
    train_part = train_part.assign(product_category_freq=encode_product_category_frequency(
        train_part["product_category"], freq))
    test_part = test_part.assign(product_category_freq=encode_product_category_frequency(
        test_part["product_category"], freq))
    lr, dt = _build_pipelines()
    rf = _build_rf_pipeline()
    scores = {}
    for name, pipeline in (("logistic_regression", lr), ("decision_tree", dt),
                           ("random_forest", rf)):
        pipeline.fit(train_part[features], train_part["is_late"])
        proba = pipeline.predict_proba(test_part[features])[:, 1]
        scores[name] = float(roc_auc_score(test_part["is_late"], proba))
    return scores


def main() -> None:
    """Run both feature settings, print and save the report."""
    fact = pd.read_parquet(PROCESSED_DATA_DIR / "FactOrderItems.parquet")
    table = build_feature_table(
        fact,
        pd.read_parquet(PROCESSED_DATA_DIR / "DimProduct.parquet"),
        pd.read_parquet(PROCESSED_DATA_DIR / "DimSeller.parquet"),
        pd.read_parquet(PROCESSED_DATA_DIR / "DimCustomer.parquet"),
        _load_payments(),
        _load_geo_centroids(),
        *_load_seller_customer_zips(),
    ).reset_index(drop=True)
    train, test = temporal_masks(fact, table["item_key"])
    stamps = table["item_key"].map(fact.set_index("item_key")["order_purchase_timestamp"])
    assert stamps[train].max() <= stamps[test].min(), "temporal split is not forward-looking"

    all_features = list(ALLOWED_FEATURES)
    no_calendar = [f for f in all_features if f not in _CALENDAR]
    results = {
        "all 12 features": evaluate(table, train, test, all_features),
        "without order_month and day_of_week (10 features)": evaluate(
            table, train, test, no_calendar),
    }

    y = table["is_late"]
    lines = [
        "Diagnostic: temporal (forward-looking) split",
        "=" * 78,
        f"train = earliest {_TRAIN_SHARE:.0%} of orders: {int(train.sum()):,} items, "
        f"{stamps[train].min():%Y-%m-%d} to {stamps[train].max():%Y-%m-%d}, "
        f"late rate {y[train].mean():.4%}",
        f"test  = latest {1 - _TRAIN_SHARE:.0%} of orders: {int(test.sum()):,} items, "
        f"{stamps[test].min():%Y-%m-%d} to {stamps[test].max():%Y-%m-%d}, "
        f"late rate {y[test].mean():.4%}",
        "",
        f"{'features':<52}{'LR':>8}{'DT':>8}{'RF':>8}{'RF-LR':>8}{'RF-DT':>8}",
    ]
    for label, s in results.items():
        lr, dt, rf = (s["logistic_regression"], s["decision_tree"], s["random_forest"])
        lines.append(f"{label:<52}{lr:>8.4f}{dt:>8.4f}{rf:>8.4f}{rf - lr:>+8.4f}{rf - dt:>+8.4f}")
    report = "\n".join(lines)
    print(report)  # noqa: T201 - one-off diagnostic script
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "diagnostic_temporal.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
