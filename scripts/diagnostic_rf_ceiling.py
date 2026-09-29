"""One-off diagnostic: is there a signal ceiling above LR/DT? NOT part of the pipeline.

Trains a `RandomForestClassifier` on the EXACT same train/test split and
12-feature set as `src.predict`'s final Stage 3 model (reusing
`build_feature_table`/`_split_and_encode` directly, not a reimplementation,
so the split is guaranteed identical), purely to answer whether
meaningfully more signal exists in these features than the committed
LogisticRegression/DecisionTreeClassifier models extracted.

Explicitly outside CLAUDE.md's "predictive models are limited to
LogisticRegression and DecisionTreeClassifier" constraint: RandomForest is
used here ONLY for this isolated, throwaway diagnostic, confirmed with the
user as an exception. It is NOT wired into `run_all.py`, NOT persisted to
`models/`, and has NO test — `src/predict.py` and `src/config.py` are read
from, never modified. Run directly: `python scripts/diagnostic_rf_ceiling.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALLOWED_FEATURES, PROCESSED_DATA_DIR, REPORTS_DIR, SEED  # noqa: E402
from src.predict import (  # noqa: E402
    _load_geo_centroids,
    _load_payments,
    _load_seller_customer_zips,
    _split_and_encode,
    build_feature_table,
)

# Committed Stage 3 results (BUILD_LOG.md), quoted for direct comparison —
# not recomputed here, since this script must not touch src/predict.py.
_EXISTING_LR_AUC = 0.6347
_EXISTING_DT_AUC = 0.6738


def main() -> None:
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
    # Identical split to src.predict.main(): same function, same SEED.
    x_train, x_test, y_train, y_test, _ = _split_and_encode(table)
    assert set(x_train.columns) == set(ALLOWED_FEATURES) == set(x_test.columns)

    # SimpleImputer(median), fit on train only -- same missing-value
    # treatment as the committed LR/DT pipelines, for a fair comparison.
    rf_pipeline = Pipeline(
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
    rf_pipeline.fit(x_train, y_train)

    y_proba = rf_pipeline.predict_proba(x_test)[:, 1]
    rf_auc = float(roc_auc_score(y_test, y_proba))

    importance = permutation_importance(
        rf_pipeline, x_test, y_test, n_repeats=20, random_state=SEED, scoring="roc_auc"
    )
    importance_df = (
        pd.DataFrame(
            {
                "feature": list(ALLOWED_FEATURES),
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    gap = rf_auc - _EXISTING_DT_AUC
    if gap > 0.03:
        interpretation = (
            f"RandomForest beats the best committed model (DT) by {gap:+.4f} AUC -- "
            "meaningfully more signal exists in these 12 features than LR/DT extracted; "
            "a more expressive model captures real interactions the simpler models miss."
        )
    elif gap > 0.01:
        interpretation = (
            f"RandomForest edges out DT by {gap:+.4f} AUC -- a modest amount of extra "
            "signal exists, but the committed models are already close to the ceiling "
            "for this feature set."
        )
    else:
        interpretation = (
            f"RandomForest is within {gap:+.4f} AUC of the best committed model (DT) -- "
            "the committed LR/DT models are already near the ceiling for this feature "
            "set; more model complexity does not meaningfully help."
        )

    lines = [
        "Diagnostic: RandomForest signal-ceiling check (one-off, NOT part of the pipeline)",
        "=" * 84,
        "",
        "Test-set ROC-AUC comparison (identical train/test split, identical 12 features):",
        f"  LogisticRegression (committed, Stage 3):   {_EXISTING_LR_AUC:.4f}",
        f"  DecisionTreeClassifier (committed, Stage 3): {_EXISTING_DT_AUC:.4f}",
        f"  RandomForestClassifier (this diagnostic, n_estimators=300): {rf_auc:.4f}",
        "",
        "Permutation feature importance (RandomForest, test set, scoring=roc_auc):",
    ]
    for _, row in importance_df.iterrows():
        lines.append(
            f"  {row['feature']:<28s} {row['importance_mean']:+.4f} "
            f"(+/- {row['importance_std']:.4f})"
        )
    lines += ["", f"Interpretation: {interpretation}"]

    output = "\n".join(lines)
    print(output)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "diagnostic_rf_ceiling.txt").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
