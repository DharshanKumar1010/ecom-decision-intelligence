"""Late-delivery risk prediction + sensitivity analysis (CLAUDE.md section 7.3).

Target: `is_late`. Features: EXACTLY `config.ALLOWED_FEATURES`
(`price`, `freight_value`, `product_weight_g`, `same_state`) — no
`delivery_days`, delivered/estimated dates, or anything derived from them,
since those define the target and would leak it. `build_feature_table`'s
output columns are asserted against `ALLOWED_FEATURES` and tested.

Missing feature values are never dropped or imputed against the full
dataset: both models are `sklearn.pipeline.Pipeline`s whose first step is a
`SimpleImputer(strategy="median")`, so the median is always fit on the
train fold only (`Pipeline.fit` on `X_train`) and merely applied
(`.transform`) everywhere else, including the full-table `LatePredictions`
pass.

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
import pandas as pd
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
    REPORTS_DIR,
    SEED,
    get_logger,
)

logger = get_logger(__name__)

_TARGET = "is_late"


def build_feature_table(
    fact: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_seller: pd.DataFrame,
    dim_customer: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the modeling table: item_key, ALLOWED_FEATURES, is_late.

    `same_state` is derived as `seller_state == customer_state` after
    joining `dim_seller`/`dim_customer` onto `fact`. `product_weight_g`
    comes from `dim_product`. No date or delivery_days column is read or
    joined here.

    Args:
        fact: FactOrderItems (must contain item_key, seller_id, product_id,
            customer_id, price, freight_value, is_late).
        dim_product: DimProduct (must contain product_id, product_weight_g).
        dim_seller: DimSeller (must contain seller_id, seller_state).
        dim_customer: DimCustomer (must contain customer_id, customer_state).

    Returns:
        DataFrame with columns `item_key`, every column in
        `config.ALLOWED_FEATURES`, and `is_late`.
    """
    table = fact[["item_key", "seller_id", "product_id", "customer_id", "price",
                  "freight_value", "is_late"]].copy()
    table = table.merge(
        dim_product[["product_id", "product_weight_g"]], on="product_id", how="left"
    )
    table = table.merge(dim_seller[["seller_id", "seller_state"]], on="seller_id", how="left")
    table = table.merge(
        dim_customer[["customer_id", "customer_state"]], on="customer_id", how="left"
    )
    table["same_state"] = (table["seller_state"] == table["customer_state"]).astype(float)

    result = table[["item_key", *ALLOWED_FEATURES, "is_late"]].copy()
    assert set(ALLOWED_FEATURES) <= set(result.columns), "missing an allowed feature column"

    na_counts = result[list(ALLOWED_FEATURES)].isna().sum()
    for feature, count in na_counts.items():
        if count > 0:
            logger.info("Feature %s: %d missing value(s), left for train-median imputation",
                        feature, count)
    return result


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


def _sensitivity_analysis(pipeline: Pipeline, x_test: pd.DataFrame) -> pd.DataFrame:
    """Perturb each feature and record the mean change in predicted late_risk.

    Continuous features (`price`, `freight_value`, `product_weight_g`) are
    scaled by each factor in `config.PREDICT_SENSITIVITY_PCTS`, holding the
    other features fixed per row. The binary `same_state` feature is instead
    flipped 0<->1 (a percentage scaling is meaningless for a 0/1 flag).

    Args:
        pipeline: fitted pipeline used to predict `late_risk`.
        x_test: held-out feature matrix (only ALLOWED_FEATURES columns).

    Returns:
        DataFrame with columns `feature`, `perturbation`,
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

    table = build_feature_table(fact, dim_product, dim_seller, dim_customer)
    x = table[list(ALLOWED_FEATURES)]
    y = table[_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    assert set(x_train.index).isdisjoint(set(x_test.index)), "train/test overlap"
    logger.info("Train/test split: %d / %d rows, base rate %.4f%%", len(x_train), len(x_test),
                float(y_test.mean()) * 100)

    lr_pipeline, dt_pipeline = _build_pipelines()
    lr_pipeline.fit(x_train, y_train)
    dt_pipeline.fit(x_train, y_train)

    lr_metrics = _evaluate(lr_pipeline, x_test, y_test)
    dt_metrics = _evaluate(dt_pipeline, x_test, y_test)

    for name, metrics in (("logistic_regression", lr_metrics), ("decision_tree", dt_metrics)):
        if metrics["roc_auc"] > PREDICT_AUC_LEAK_GUARDRAIL:
            raise AssertionError(
                f"{name} test ROC-AUC {metrics['roc_auc']:.4f} exceeds the leak guardrail "
                f"{PREDICT_AUC_LEAK_GUARDRAIL}; investigate for target leakage before reporting."
            )
        logger.info("%s test ROC-AUC: %.4f", name, metrics["roc_auc"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"logistic_regression": lr_metrics, "decision_tree": dt_metrics}, fh, indent=2)

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

    sensitivity = _sensitivity_analysis(lr_pipeline, x_test)
    sensitivity.to_csv(REPORTS_DIR / "sensitivity.csv", index=False)

    late_predictions = table[["item_key"]].copy()
    late_predictions["late_risk"] = lr_pipeline.predict_proba(table[list(ALLOWED_FEATURES)])[:, 1]
    assert late_predictions["late_risk"].between(0, 1).all(), "late_risk outside [0, 1]"
    late_predictions.to_parquet(PROCESSED_DATA_DIR / "LatePredictions.parquet", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(dt_pipeline, MODELS_DIR / "decision_tree.joblib")
    metadata = {
        "features": list(ALLOWED_FEATURES),
        "seed": SEED,
        "timestamp": datetime.now(UTC).isoformat(),
        "metrics": {"logistic_regression": lr_metrics, "decision_tree": dt_metrics},
    }
    with (MODELS_DIR / "model_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info(
        "Wrote model_metrics.json, decision_tree.txt, coefficients.csv, "
        "feature_importance.csv, sensitivity.csv, LatePredictions.parquet (%d rows), "
        "and 2 joblib models with metadata",
        len(late_predictions),
    )


if __name__ == "__main__":
    main()
