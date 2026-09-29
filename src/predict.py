"""Late-delivery risk prediction + sensitivity analysis (CLAUDE.md section 7.3).

Target: `is_late`. Features: EXACTLY `config.ALLOWED_FEATURES` — no
`delivery_days`, delivered/estimated dates, or anything derived from them,
since those define the target and would leak it. `build_feature_table`'s
output columns (minus the one exception below) are asserted against
`ALLOWED_FEATURES` and tested.

Three models are trained on the identical split and feature set:
`LogisticRegression` and `DecisionTreeClassifier` (kept for interpretability
— readable coefficients / a printable rule path) and `RandomForestClassifier`
as the PRIMARY/production model, adopted based on confirmed cross-validated
evidence (5-fold CV mean AUC 0.7928 vs. DT's 0.6870 and LR's 0.6247, stable
low-variance advantage, identical top-feature ranking in every fold — see
`reports/diagnostic_rf_cv.txt` and `BUILD_LOG.md`). This is an explicit,
confirmed override of CLAUDE.md's historical "LR/DT only" constraint — see
CLAUDE.md section 1 and 7.3, updated alongside this change.

Missing feature values are never dropped or imputed against the full
dataset: all three models are `sklearn.pipeline.Pipeline`s whose first step
is a `SimpleImputer(strategy="median")`, so the median is always fit on the
train fold only (`Pipeline.fit` on `X_train`) and merely applied
(`.transform`) everywhere else, including the full-table `LatePredictions`
pass.

`product_category_freq` is the one feature that can't be computed inside
`build_feature_table` (which runs before the train/test split exists): it's
a frequency encoding that must be FIT on the train fold only, exactly like
the imputer's median. `build_feature_table` instead returns the raw
`product_category` string column, and `_split_and_encode` fits/applies the
encoding immediately after splitting, before any model sees it — see that
function's docstring.

Guardrail: `ALLOWED_FEATURES` is a deliberately small, non-date feature set,
so a test ROC-AUC above `config.PREDICT_AUC_LEAK_GUARDRAIL` is treated as a
probable leak — `main()` raises rather than silently reporting such a
number (CLAUDE.md section 7.3).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import (
    ALLOWED_FEATURES,
    MODELS_DIR,
    PREDICT_AUC_LEAK_GUARDRAIL,
    PREDICT_SENSITIVITY_PCTS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SEED,
    get_logger,
)

logger = get_logger(__name__)

_TARGET = "is_late"


_PRECOMPUTED_FEATURES: tuple[str, ...] = ("price", "freight_value", "product_weight_g",
                                           "same_state", "order_month", "day_of_week",
                                           "n_items_in_order", "payment_installments",
                                           "seller_historical_late_rate", "geo_distance",
                                           "geo_distance_missing")


def _aggregate_geo_centroids(geo: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw geolocation rows to one median lat/lng centroid per zip prefix.

    Centroid = MEDIAN lat/lng within each zip_code_prefix — more robust to
    occasional outlier/bad geocodes in this crowd-sourced data than a mean.

    Args:
        geo: raw geolocation rows with columns `geolocation_zip_code_prefix`,
            `geolocation_lat`, `geolocation_lng`.

    Returns:
        DataFrame with columns `zip_prefix`, `lat`, `lng`.
    """
    return (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .reset_index()
        .rename(
            columns={
                "geolocation_zip_code_prefix": "zip_prefix",
                "geolocation_lat": "lat",
                "geolocation_lng": "lng",
            }
        )
    )


def _load_geo_centroids() -> pd.DataFrame:
    """Load raw geolocation and aggregate via `_aggregate_geo_centroids`.

    Returns:
        DataFrame with columns `zip_prefix`, `lat`, `lng`.
    """
    geo = pd.read_csv(
        RAW_DATA_DIR / "olist_geolocation_dataset.csv",
        usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
    )
    return _aggregate_geo_centroids(geo)


def _load_seller_customer_zips() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load seller/customer zip-code prefixes directly from the raw CSVs.

    Not read via `build_tables.py`, which doesn't ingest these columns into
    DimSeller/DimCustomer — see the module docstring.

    Returns:
        `(seller_zips, customer_zips)`: DataFrames with columns
        (`seller_id`, `seller_zip_prefix`) and (`customer_id`,
        `customer_zip_prefix`) respectively.
    """
    seller_zips = pd.read_csv(
        RAW_DATA_DIR / "olist_sellers_dataset.csv",
        usecols=["seller_id", "seller_zip_code_prefix"],
    ).rename(columns={"seller_zip_code_prefix": "seller_zip_prefix"})
    customer_zips = pd.read_csv(
        RAW_DATA_DIR / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_zip_code_prefix"],
    ).rename(columns={"customer_zip_code_prefix": "customer_zip_prefix"})
    return seller_zips, customer_zips


def _haversine_km(
    lat1: pd.Series, lng1: pd.Series, lat2: pd.Series, lng2: pd.Series
) -> pd.Series:
    """Vectorized great-circle distance in kilometers.

    Args:
        lat1, lng1, lat2, lng2: coordinate series in decimal degrees, same
            index and length.

    Returns:
        Distance in kilometers, aligned to `lat1`'s index; `NaN` wherever
        any input is `NaN`.
    """
    earth_radius_km = 6371.0
    lat1_rad, lng1_rad = np.radians(lat1), np.radians(lng1)
    lat2_rad, lng2_rad = np.radians(lat2), np.radians(lng2)
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2) ** 2
    return pd.Series(earth_radius_km * 2 * np.arcsin(np.sqrt(a)), index=lat1.index)


def compute_geo_distance(
    fact: pd.DataFrame,
    geo_centroids: pd.DataFrame,
    seller_zips: pd.DataFrame,
    customer_zips: pd.DataFrame,
) -> pd.DataFrame:
    """Haversine distance between each item's seller and customer zip centroids.

    A seller or customer zip prefix with no geolocation match (7 of 2,246
    seller zips, 157 of 14,994 customer zips in the real data) is never
    silently dropped: `geo_distance` is left `NaN` for the pipeline's
    train-median imputer, and `geo_distance_missing` flags it explicitly —
    the same "don't drop, impute and flag" pattern used elsewhere.

    Args:
        fact: FactOrderItems (must contain item_key, seller_id, customer_id).
        geo_centroids: output of `_load_geo_centroids`.
        seller_zips: first element of `_load_seller_customer_zips`.
        customer_zips: second element of `_load_seller_customer_zips`.

    Returns:
        DataFrame with columns `item_key`, `geo_distance`, `geo_distance_missing`.
    """
    table = fact[["item_key", "seller_id", "customer_id"]].merge(
        seller_zips, on="seller_id", how="left"
    )
    table = table.merge(customer_zips, on="customer_id", how="left")

    seller_coords = geo_centroids.rename(
        columns={"zip_prefix": "seller_zip_prefix", "lat": "seller_lat", "lng": "seller_lng"}
    )
    customer_coords = geo_centroids.rename(
        columns={"zip_prefix": "customer_zip_prefix", "lat": "customer_lat", "lng": "customer_lng"}
    )
    table = table.merge(seller_coords, on="seller_zip_prefix", how="left")
    table = table.merge(customer_coords, on="customer_zip_prefix", how="left")

    table["geo_distance"] = _haversine_km(
        table["seller_lat"], table["seller_lng"], table["customer_lat"], table["customer_lng"]
    )
    table["geo_distance_missing"] = table["geo_distance"].isna().astype(float)
    return table[["item_key", "geo_distance", "geo_distance_missing"]]


def compute_seller_historical_late_rate(fact: pd.DataFrame) -> pd.DataFrame:
    """Per-item expanding mean of the seller's own PRIOR order outcomes.

    Never uses the seller's overall/global late rate, which would leak
    future information into past predictions. Computed in three steps,
    each addressing a real tie found in this data (verified empirically,
    not assumed away):

    1. Collapse to one row per (seller_id, order_id). Items sharing an
       order and seller share `order_purchase_timestamp` and `is_late` by
       construction, so computing history at the item level would let a
       sibling item of the SAME order leak into its own "prior" feature.
    2. Further collapse to one row per (seller_id, order_purchase_timestamp).
       112 of 97,697 real (seller_id, order_id) pairs share a seller and an
       exact-second timestamp across TWO OR THREE DIFFERENT orders; these
       are simultaneous, not strictly ordered relative to each other, and
       must not see each other either.
    3. Within each seller (sorted by timestamp), take a cumulative
       sum/count of `is_late` and then shift it by one row — via
       `groupby("seller_id")[...].cumsum()` followed by a SEPARATE
       `groupby("seller_id")[...].shift(1)` call. Both are standalone
       groupby operations, which correctly reset at every seller boundary.
       This is deliberately NOT `groupby(...).expanding().shift()`: shifting
       the result of an `.expanding()` chain does not reset at group
       boundaries and would silently leak the last row of one seller into
       the first row of the next.

    The per-(seller, timestamp) rate is then broadcast back to every item
    row via a merge. A seller's earliest timestamp has no prior history and
    is left `NaN`, for the pipeline's train-median imputer.

    Args:
        fact: FactOrderItems (must contain item_key, seller_id, order_id,
            order_purchase_timestamp, is_late).

    Returns:
        DataFrame with columns `item_key`, `seller_historical_late_rate`.
    """
    seller_orders = fact[
        ["seller_id", "order_id", "order_purchase_timestamp", "is_late"]
    ].drop_duplicates(subset=["seller_id", "order_id"])

    per_timestamp = (
        seller_orders.groupby(["seller_id", "order_purchase_timestamp"])["is_late"]
        .agg(late_sum="sum", order_count="count")
        .reset_index()
        .sort_values(["seller_id", "order_purchase_timestamp"])
        .reset_index(drop=True)
    )
    per_timestamp["cum_late"] = per_timestamp.groupby("seller_id")["late_sum"].cumsum()
    per_timestamp["cum_orders"] = per_timestamp.groupby("seller_id")["order_count"].cumsum()
    per_timestamp["cum_late_before"] = per_timestamp.groupby("seller_id")["cum_late"].shift(1)
    per_timestamp["cum_orders_before"] = per_timestamp.groupby("seller_id")["cum_orders"].shift(1)
    per_timestamp["seller_historical_late_rate"] = (
        per_timestamp["cum_late_before"] / per_timestamp["cum_orders_before"]
    )

    item_lookup = fact[["item_key", "seller_id", "order_purchase_timestamp"]].merge(
        per_timestamp[["seller_id", "order_purchase_timestamp", "seller_historical_late_rate"]],
        on=["seller_id", "order_purchase_timestamp"],
        how="left",
    )
    return item_lookup[["item_key", "seller_historical_late_rate"]]


def _load_payments() -> pd.DataFrame:
    """Load raw order payments, aggregated to one row per order_id.

    Multiple payment rows per order (2,961 of 99,440 orders in the raw
    data) reflect split payments across payment methods (e.g. a voucher
    plus a credit card in the same order). `payment_installments` is
    aggregated via MAX across an order's payment rows: it captures the
    longest real installment commitment on the order and is deterministic
    regardless of row order, unlike taking the first row; summing
    installment counts across different payment methods isn't a
    meaningful quantity. Not read via `build_tables.py`, which doesn't
    ingest this raw file — see the module docstring.

    Returns:
        DataFrame with columns `order_id`, `payment_installments`.
    """
    payments = pd.read_csv(
        RAW_DATA_DIR / "olist_order_payments_dataset.csv",
        usecols=["order_id", "payment_installments"],
    )
    return payments.groupby("order_id", as_index=False)["payment_installments"].max()


def build_feature_table(
    fact: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_seller: pd.DataFrame,
    dim_customer: pd.DataFrame,
    payments: pd.DataFrame,
    geo_centroids: pd.DataFrame,
    seller_zips: pd.DataFrame,
    customer_zips: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the modeling table: item_key, precomputed features, is_late.

    `same_state` is derived as `seller_state == customer_state` after
    joining `dim_seller`/`dim_customer` onto `fact`. `product_weight_g`
    comes from `dim_product`. No date or delivery_days column is read or
    joined here.

    `order_month`/`day_of_week` come from `order_purchase_timestamp` — the
    time the customer placed the order, known well before any delivery
    outcome exists, not a delivered/estimated date. `n_items_in_order` is
    the count of item rows sharing an `order_id`, also fixed at order
    placement.

    `payment_installments` is left-joined from `payments` (see
    `_load_payments`) on `order_id`; an order absent from the raw payments
    file is left `NaN`, for the pipeline's train-median imputer.
    `seller_historical_late_rate` comes from `compute_seller_historical_late_rate`.
    `geo_distance`/`geo_distance_missing` come from `compute_geo_distance`.

    Args:
        fact: FactOrderItems (must contain item_key, order_id, seller_id,
            product_id, customer_id, price, freight_value,
            order_purchase_timestamp, is_late).
        dim_product: DimProduct (must contain product_id, product_weight_g,
            product_category_name_english).
        dim_seller: DimSeller (must contain seller_id, seller_state).
        dim_customer: DimCustomer (must contain customer_id, customer_state).
        payments: output of `_load_payments` (order_id, payment_installments).
        geo_centroids: output of `_load_geo_centroids`.
        seller_zips: first element of `_load_seller_customer_zips`.
        customer_zips: second element of `_load_seller_customer_zips`.

    Returns:
        DataFrame with `item_key`, every precomputable feature in
        `config.ALLOWED_FEATURES` (i.e. everything except
        `product_category_freq`), the raw `product_category` string (not
        itself a model feature — encoded post-split by `_split_and_encode`),
        and `is_late`.
    """
    table = fact[["item_key", "order_id", "seller_id", "product_id", "customer_id", "price",
                  "freight_value", "order_purchase_timestamp", "is_late"]].copy()
    table["n_items_in_order"] = table.groupby("order_id")["item_key"].transform("size").astype(
        float
    )
    table = table.merge(
        dim_product[["product_id", "product_weight_g", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    table = table.rename(columns={"product_category_name_english": "product_category"})
    table = table.merge(dim_seller[["seller_id", "seller_state"]], on="seller_id", how="left")
    table = table.merge(
        dim_customer[["customer_id", "customer_state"]], on="customer_id", how="left"
    )
    table["same_state"] = (table["seller_state"] == table["customer_state"]).astype(float)
    table["order_month"] = table["order_purchase_timestamp"].dt.month.astype(float)
    table["day_of_week"] = table["order_purchase_timestamp"].dt.dayofweek.astype(float)
    table = table.merge(payments, on="order_id", how="left")
    table["payment_installments"] = table["payment_installments"].astype(float)
    table = table.merge(compute_seller_historical_late_rate(fact), on="item_key", how="left")
    table = table.merge(
        compute_geo_distance(fact, geo_centroids, seller_zips, customer_zips),
        on="item_key",
        how="left",
    )

    result = table[["item_key", *_PRECOMPUTED_FEATURES, "product_category", "is_late"]].copy()

    na_counts = result[list(_PRECOMPUTED_FEATURES)].isna().sum()
    for feature, count in na_counts.items():
        if count > 0:
            logger.info("Feature %s: %d missing value(s), left for train-median imputation",
                        feature, count)
    return result


def fit_product_category_frequency(categories: pd.Series) -> dict[str, float]:
    """Fit a train-only frequency encoding: category -> share of train items in it.

    Must be fit on the TRAIN fold only (mirrors the pipeline imputer's
    train-median-only rule) and applied via `encode_product_category_frequency`
    to train, test, and the full table alike.

    Args:
        categories: `product_category` values from the TRAIN fold only.

    Returns:
        Dict mapping each observed category to its proportion of train rows
        (sums to 1.0 across all keys).
    """
    counts: dict[str, float] = categories.value_counts(normalize=True).to_dict()
    return counts


def encode_product_category_frequency(
    categories: pd.Series, frequency_map: dict[str, float]
) -> pd.Series:
    """Apply a fitted frequency encoding, mapping unseen categories to 0.0.

    0.0 for an unseen category means "never observed in training," which is
    a distinct and more honest signal than mapping it to the rarest known
    category's frequency.

    Args:
        categories: `product_category` values to encode (train, test, or
            the full table).
        frequency_map: output of `fit_product_category_frequency`.

    Returns:
        Float series of encoded frequencies, same index as `categories`.
    """
    return categories.map(frequency_map).fillna(0.0)


def _split_and_encode(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, float]]:
    """Stratified 80/20 split, then fit/apply product_category encoding on train only.

    Args:
        table: output of `build_feature_table` (must contain every column in
            `_PRECOMPUTED_FEATURES`, `product_category`, and `is_late`).

    Returns:
        `(x_train, x_test, y_train, y_test, category_freq_map)`: the first
        four have exactly `config.ALLOWED_FEATURES` columns;
        `category_freq_map` is returned so callers (e.g. the full-table
        `LatePredictions` pass) can apply the same train-fitted encoding
        elsewhere without refitting it.
    """
    table_train, table_test = train_test_split(
        table, test_size=0.2, stratify=table[_TARGET], random_state=SEED
    )
    assert set(table_train.index).isdisjoint(set(table_test.index)), "train/test overlap"

    category_freq_map = fit_product_category_frequency(table_train["product_category"])
    table_train = table_train.assign(
        product_category_freq=encode_product_category_frequency(
            table_train["product_category"], category_freq_map
        )
    )
    table_test = table_test.assign(
        product_category_freq=encode_product_category_frequency(
            table_test["product_category"], category_freq_map
        )
    )

    x_train = table_train[list(ALLOWED_FEATURES)]
    x_test = table_test[list(ALLOWED_FEATURES)]
    y_train = table_train[_TARGET]
    y_test = table_test[_TARGET]
    return x_train, x_test, y_train, y_test, category_freq_map


def _build_pipelines() -> tuple[Pipeline, Pipeline]:
    """Construct the (unfitted) logistic regression and decision tree pipelines.

    Returns:
        A tuple `(logistic_regression_pipeline, decision_tree_pipeline)`,
        each starting with a `SimpleImputer(strategy="median")` step.
    """
    lr_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]
    )
    dt_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=SEED),
            ),
        ]
    )
    return lr_pipeline, dt_pipeline


def _build_rf_pipeline() -> Pipeline:
    """Construct the (unfitted) RandomForest pipeline — the primary model.

    Hyperparameters are hardcoded inline, matching how `_build_pipelines`
    already hardcodes LR's/DT's own hyperparameters (existing convention,
    not a new one). Adopted based on confirmed cross-validated evidence —
    see the module docstring.

    Returns:
        An unfitted `Pipeline` starting with a `SimpleImputer(strategy="median")`
        step.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, max_depth=None, class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def _evaluate(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute test-set-only metrics for a fitted pipeline.

    Args:
        pipeline: a fitted sklearn Pipeline with `predict`/`predict_proba`.
        x_test: held-out feature matrix.
        y_test: held-out true labels.

    Returns:
        Dict with `roc_auc`, `precision`/`recall`/`f1` per class (0 and 1),
        `confusion_matrix` (nested list, rows=true, cols=predicted), and
        `base_rate` (share of `y_test` equal to 1).
    """
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = pipeline.predict(x_test)

    auc = float(roc_auc_score(y_test, y_proba))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "roc_auc": auc,
        "precision": {"0": float(precision[0]), "1": float(precision[1])},
        "recall": {"0": float(recall[0]), "1": float(recall[1])},
        "f1": {"0": float(f1[0]), "1": float(f1[1])},
        "confusion_matrix": cm.tolist(),
        "base_rate": float(y_test.mean()),
    }


def _sensitivity_analysis(
    pipeline: Pipeline, x_test: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Perturb each feature and record the mean change in predicted late_risk.

    Continuous features (`price`, `freight_value`, `product_weight_g`) are
    scaled by each factor in `config.PREDICT_SENSITIVITY_PCTS`, holding the
    other features fixed per row. The binary `same_state` feature is instead
    flipped 0<->1 (a percentage scaling is meaningless for a 0/1 flag).

    Args:
        pipeline: fitted pipeline used to predict `late_risk`.
        x_test: held-out feature matrix (only ALLOWED_FEATURES columns).
        model_name: label written into every output row's `model` column,
            so results for multiple models can be concatenated into one file.

    Returns:
        DataFrame with columns `model`, `feature`, `perturbation`,
        `mean_late_risk_change`.
    """
    baseline = pipeline.predict_proba(x_test)[:, 1]
    rows: list[dict[str, Any]] = []

    for feature in ALLOWED_FEATURES:
        if feature == "same_state":
            perturbed = x_test.copy()
            perturbed[feature] = 1.0 - perturbed[feature]
            new_pred = pipeline.predict_proba(perturbed)[:, 1]
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "perturbation": "flip_0_1",
                    "mean_late_risk_change": float((new_pred - baseline).mean()),
                }
            )
        else:
            for pct in PREDICT_SENSITIVITY_PCTS:
                perturbed = x_test.copy()
                perturbed[feature] = perturbed[feature] * (1.0 + pct)
                new_pred = pipeline.predict_proba(perturbed)[:, 1]
                rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "perturbation": f"{pct:+.0%}",
                        "mean_late_risk_change": float((new_pred - baseline).mean()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the full predict pipeline: build features, train, evaluate, write outputs."""
    fact = pd.read_parquet(PROCESSED_DATA_DIR / "FactOrderItems.parquet")
    dim_product = pd.read_parquet(PROCESSED_DATA_DIR / "DimProduct.parquet")
    dim_seller = pd.read_parquet(PROCESSED_DATA_DIR / "DimSeller.parquet")
    dim_customer = pd.read_parquet(PROCESSED_DATA_DIR / "DimCustomer.parquet")
    payments = _load_payments()
    geo_centroids = _load_geo_centroids()
    seller_zips, customer_zips = _load_seller_customer_zips()

    table = build_feature_table(
        fact, dim_product, dim_seller, dim_customer, payments,
        geo_centroids, seller_zips, customer_zips,
    )
    x_train, x_test, y_train, y_test, category_freq_map = _split_and_encode(table)
    logger.info("Train/test split: %d / %d rows, base rate %.4f%%", len(x_train), len(x_test),
                float(y_test.mean()) * 100)

    lr_pipeline, dt_pipeline = _build_pipelines()
    rf_pipeline = _build_rf_pipeline()
    lr_pipeline.fit(x_train, y_train)
    dt_pipeline.fit(x_train, y_train)
    rf_pipeline.fit(x_train, y_train)

    lr_metrics = _evaluate(lr_pipeline, x_test, y_test)
    dt_metrics = _evaluate(dt_pipeline, x_test, y_test)
    rf_metrics = _evaluate(rf_pipeline, x_test, y_test)

    all_metrics = (
        ("logistic_regression", lr_metrics),
        ("decision_tree", dt_metrics),
        ("random_forest", rf_metrics),
    )
    for name, metrics in all_metrics:
        if metrics["roc_auc"] > PREDICT_AUC_LEAK_GUARDRAIL:
            raise AssertionError(
                f"{name} test ROC-AUC {metrics['roc_auc']:.4f} exceeds the leak guardrail "
                f"{PREDICT_AUC_LEAK_GUARDRAIL}; investigate for target leakage before reporting."
            )
        logger.info("%s test ROC-AUC: %.4f", name, metrics["roc_auc"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(dict(all_metrics), fh, indent=2)

    tree_text = export_text(dt_pipeline.named_steps["clf"], feature_names=list(ALLOWED_FEATURES))
    (REPORTS_DIR / "decision_tree.txt").write_text(tree_text, encoding="utf-8")

    coefficients = pd.DataFrame(
        {
            "feature": list(ALLOWED_FEATURES),
            "standardized_coefficient": lr_pipeline.named_steps["clf"].coef_[0],
        }
    )
    coefficients.to_csv(REPORTS_DIR / "coefficients.csv", index=False)

    importance_rows = []
    for name, pipeline in (("logistic_regression", lr_pipeline), ("decision_tree", dt_pipeline)):
        result = permutation_importance(
            pipeline, x_test, y_test, n_repeats=20, random_state=SEED, scoring="roc_auc"
        )
        for i, feature in enumerate(ALLOWED_FEATURES):
            importance_rows.append(
                {
                    "feature": feature,
                    "model": name,
                    "importance_mean": float(result.importances_mean[i]),
                    "importance_std": float(result.importances_std[i]),
                }
            )
    pd.DataFrame(importance_rows).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    rf_importance = permutation_importance(
        rf_pipeline, x_test, y_test, n_repeats=20, random_state=SEED, scoring="roc_auc"
    )
    rf_importance_df = (
        pd.DataFrame(
            {
                "feature": list(ALLOWED_FEATURES),
                "importance_mean": rf_importance.importances_mean,
                "importance_std": rf_importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    rf_importance_df.to_csv(REPORTS_DIR / "rf_feature_importance.csv", index=False)

    sensitivity = pd.concat(
        [
            _sensitivity_analysis(lr_pipeline, x_test, "logistic_regression"),
            _sensitivity_analysis(dt_pipeline, x_test, "decision_tree"),
            _sensitivity_analysis(rf_pipeline, x_test, "random_forest"),
        ],
        ignore_index=True,
    )
    sensitivity.to_csv(REPORTS_DIR / "sensitivity.csv", index=False)

    full_table = table.assign(
        product_category_freq=encode_product_category_frequency(
            table["product_category"], category_freq_map
        )
    )
    full_x = full_table[list(ALLOWED_FEATURES)]
    late_predictions = table[["item_key"]].copy()
    # Primary: RandomForest, adopted based on confirmed CV evidence (module
    # docstring). LR's/DT's predictions are kept alongside for the
    # interpretability story, not discarded.
    late_predictions["late_risk"] = rf_pipeline.predict_proba(full_x)[:, 1]
    late_predictions["late_risk_lr"] = lr_pipeline.predict_proba(full_x)[:, 1]
    late_predictions["late_risk_dt"] = dt_pipeline.predict_proba(full_x)[:, 1]
    for col in ("late_risk", "late_risk_lr", "late_risk_dt"):
        assert late_predictions[col].between(0, 1).all(), f"{col} outside [0, 1]"
    late_predictions.to_parquet(PROCESSED_DATA_DIR / "LatePredictions.parquet", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(dt_pipeline, MODELS_DIR / "decision_tree.joblib")
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest.joblib")

    timestamp = datetime.now(UTC).isoformat()
    model_files = {
        "logistic_regression": ("logistic_regression.joblib", lr_metrics),
        "decision_tree": ("decision_tree.joblib", dt_metrics),
        "random_forest": ("random_forest.joblib", rf_metrics),
    }
    for name, (_joblib_file, metrics) in model_files.items():
        per_model_metadata = {
            "algorithm": name,
            "features": list(ALLOWED_FEATURES),
            "seed": SEED,
            "timestamp": timestamp,
            "metrics": metrics,
        }
        with (MODELS_DIR / f"{name}_metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(per_model_metadata, fh, indent=2)

    registry = {
        "primary_model": "random_forest",
        "updated": timestamp,
        "models": {
            name: {
                "role": "primary" if name == "random_forest" else "retained_for_interpretability",
                "file": joblib_file,
                "metadata_file": f"{name}_metadata.json",
                "test_roc_auc": metrics["roc_auc"],
            }
            for name, (joblib_file, metrics) in model_files.items()
        },
    }
    with (MODELS_DIR / "registry.json").open("w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)

    logger.info(
        "Wrote model_metrics.json, decision_tree.txt, coefficients.csv, "
        "feature_importance.csv, rf_feature_importance.csv, sensitivity.csv, "
        "LatePredictions.parquet (%d rows), 3 joblib models with per-model metadata, "
        "and models/registry.json (primary=random_forest)",
        len(late_predictions),
    )


if __name__ == "__main__":
    main()
