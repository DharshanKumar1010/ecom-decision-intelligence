"""One-off diagnostic: sibling leakage in the item-level split, and in-sample late_risk.

NOT part of the pipeline: not wired into `run_all.py`, no committed file is changed other than
the report this script writes (`reports/diagnostic_group_leak.txt`). Run directly:
`python scripts/diagnostic_group_leak.py` (slow: two RandomForest fits on ~88k rows).

Items of one order share `is_late` and most features (110,189 items = 97,811 seller-order
pairs = 96,470 orders), but the committed split put individual item rows at random into train
and test, so siblings land on both sides and an unlimited-depth forest can memorise them.

Measured, with the current model configs and the 12 features (nothing is retuned):
  (a) test AUC under the random stratified item split;
  (b) test AUC under an 80/20 split grouped by `order_id` (StratifiedGroupKFold, fold 0);
  (c) the share of test items (and test orders) whose order also has an item in train under (a);
  (d) the AUC of the COMMITTED `late_risk` (LatePredictions.parquet) over all rows, over the
      random-split test rows only and over its train rows only, and the seller-level
      correlation between `avg_late_risk` and the actual `late_rate`.

The random split is re-implemented here (not imported) so the script stays valid after
`src.predict` switches to the grouped split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALLOWED_FEATURES, PROCESSED_DATA_DIR, REPORTS_DIR, SEED  # noqa: E402
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


def fit_and_score(
    table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> dict[str, float]:
    """Fit LR, DT and RF on the train rows and return each one's test ROC-AUC."""
    train, test = table.iloc[train_idx], table.iloc[test_idx]
    freq = fit_product_category_frequency(train["product_category"])
    train = train.assign(product_category_freq=encode_product_category_frequency(
        train["product_category"], freq))
    test = test.assign(product_category_freq=encode_product_category_frequency(
        test["product_category"], freq))
    x_train, x_test = train[list(ALLOWED_FEATURES)], test[list(ALLOWED_FEATURES)]
    lr, dt = _build_pipelines()
    rf = _build_rf_pipeline()
    scores = {}
    for name, pipeline in (("logistic_regression", lr), ("decision_tree", dt),
                           ("random_forest", rf)):
        pipeline.fit(x_train, train["is_late"])
        scores[name] = float(roc_auc_score(test["is_late"], pipeline.predict_proba(x_test)[:, 1]))
    return scores


def main() -> None:
    """Run the measurements, print them and write the report."""
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
    orders = table["item_key"].map(fact.set_index("item_key")["order_id"])
    y = table["is_late"].to_numpy()
    lines = [
        "Diagnostic: sibling leakage and in-sample late_risk",
        "=" * 78,
        f"items {len(table):,}; distinct orders {orders.nunique():,}; "
        f"overall late rate {y.mean():.4%}",
        "",
    ]

    # (a) random stratified item split (the committed protocol, re-implemented)
    all_idx = np.arange(len(table))
    r_train, r_test = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=SEED)
    random_scores = fit_and_score(table, r_train, r_test)

    # (c) how many random-split test items have a sibling in train
    train_orders = set(orders.iloc[r_train])
    test_order_series = orders.iloc[r_test]
    item_share = float(test_order_series.isin(train_orders).mean())
    order_share = float(pd.Series(test_order_series.unique()).isin(train_orders).mean())

    # (b) grouped split: StratifiedGroupKFold fold 0 is the test set
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    g_train, g_test = next(iter(splitter.split(all_idx, y, groups=orders.to_numpy())))
    assert set(orders.iloc[g_train]).isdisjoint(set(orders.iloc[g_test])), "orders overlap"
    grouped_scores = fit_and_score(table, g_train, g_test)

    lines += [
        "(a) vs (b): test ROC-AUC, current configs, 12 features",
        f"{'model':<22}{'random item split':>20}{'grouped by order':>20}{'drop':>10}",
    ]
    for name in random_scores:
        lines.append(f"{name:<22}{random_scores[name]:>20.4f}{grouped_scores[name]:>20.4f}"
                     f"{random_scores[name] - grouped_scores[name]:>10.4f}")
    lines += [
        "",
        f"random split : train {len(r_train):,} items / test {len(r_test):,} items; "
        f"base rate train {y[r_train].mean():.4%}, test {y[r_test].mean():.4%}",
        f"grouped split: train {len(g_train):,} items / test {len(g_test):,} items "
        f"({len(g_test) / len(table):.2%}); orders train {orders.iloc[g_train].nunique():,} / "
        f"test {orders.iloc[g_test].nunique():,}; base rate train {y[g_train].mean():.4%}, "
        f"test {y[g_test].mean():.4%}",
        "",
        "(c) siblings across the random split",
        f"share of test ITEMS whose order also has an item in train: {item_share:.2%}",
        f"share of test ORDERS that also appear in train:             {order_share:.2%}",
        "",
    ]

    # (d) committed late_risk: in-sample vs held-out, and the seller-level echo of late_rate
    committed = pd.read_parquet(PROCESSED_DATA_DIR / "LatePredictions.parquet")
    aligned = committed.set_index("item_key").loc[table["item_key"]].reset_index(drop=True)
    lines += ["(d) AUC of the committed LatePredictions (late_risk = RandomForest, "
              "late_risk_lr, late_risk_dt)",
              f"{'column':<14}{'all rows':>12}{'random-test rows':>20}{'random-train rows':>20}"]
    for col in ("late_risk", "late_risk_lr", "late_risk_dt"):
        p = aligned[col].to_numpy()
        lines.append(
            f"{col:<14}{roc_auc_score(y, p):>12.4f}{roc_auc_score(y[r_test], p[r_test]):>20.4f}"
            f"{roc_auc_score(y[r_train], p[r_train]):>20.4f}"
        )
    facts = pd.read_parquet(PROCESSED_DATA_DIR / "SellerFacts.parquet")
    lines += ["", "seller-level correlation: avg_late_risk (committed) vs actual late_rate"]
    for label, subset in (("all sellers", facts), ("sellers with >= 5 items sold",
                                                    facts[facts["order_volume"] >= 5])):
        pearson = pearsonr(subset["avg_late_risk"], subset["late_rate"])[0]
        spearman = spearmanr(subset["avg_late_risk"], subset["late_rate"])[0]
        lines.append(
            f"{label:<32} n={len(subset):>5,}  Pearson {pearson:.3f}  Spearman {spearman:.3f}"
        )
    report = "\n".join(lines)
    print(report)  # noqa: T201 - one-off diagnostic script
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "diagnostic_group_leak.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
