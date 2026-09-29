"""One-off diagnostic: is RF's AUC edge stable under 5-fold CV? NOT part of the pipeline.

Follow-up to `diagnostic_rf_ceiling.py`, which compared a single held-out
split. This script runs 5-fold stratified cross-validation on the SAME
`RandomForestClassifier(n_estimators=300, max_depth=None,
class_weight="balanced", random_state=42)` config, and on the COMMITTED
LogisticRegression/DecisionTreeClassifier pipelines (via
`src.predict._build_pipelines`, reused unmodified — not reimplemented) so
all three are compared under identical CV conditions rather than one split.

The `product_category_freq` frequency encoding is the one feature that
requires a train-only fit (like the pipeline imputer's median); it is
refit separately INSIDE EACH FOLD on that fold's training rows only via
`src.predict.fit_product_category_frequency`/`encode_product_category_frequency`
(also reused unmodified), so no fold's encoding leaks information from its
own held-out rows.

Explicitly outside CLAUDE.md's "predictive models are limited to
LogisticRegression and DecisionTreeClassifier" constraint (RandomForest,
confirmed as an exception for this isolated diagnostic — see
`diagnostic_rf_ceiling.py`'s docstring). NOT wired into `run_all.py`, NOT
persisted to `models/`, NO test, and `src/predict.py`/`config.py` are read
from, never modified. Run directly: `python scripts/diagnostic_rf_cv.py`
(slow: 5 folds x 3 models, RandomForest with unlimited depth on ~88-97k
rows per fold dominates the runtime — expect this to take a while).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALLOWED_FEATURES, PROCESSED_DATA_DIR, REPORTS_DIR, SEED  # noqa: E402
from src.predict import (  # noqa: E402
    _build_pipelines,
    _load_geo_centroids,
    _load_payments,
    _load_seller_customer_zips,
    build_feature_table,
    encode_product_category_frequency,
    fit_product_category_frequency,
)

_TARGET = "is_late"
_N_SPLITS = 5
_PERM_N_REPEATS = 5  # lighter than diagnostic_rf_ceiling.py's 20: 5 folds x 3 models is
                      # already expensive; enough repeats to see a rough per-fold ranking.
_UNUSUAL_FOLD_THRESHOLD_STD = 1.5


def _build_rf_pipeline() -> Pipeline:
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


def _encode_fold(
    table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split by position and fit/apply product_category_freq on this fold's train rows only."""
    table_train = table.iloc[train_idx]
    table_test = table.iloc[test_idx]

    freq_map = fit_product_category_frequency(table_train["product_category"])
    table_train = table_train.assign(
        product_category_freq=encode_product_category_frequency(
            table_train["product_category"], freq_map
        )
    )
    table_test = table_test.assign(
        product_category_freq=encode_product_category_frequency(
            table_test["product_category"], freq_map
        )
    )

    x_train = table_train[list(ALLOWED_FEATURES)]
    x_test = table_test[list(ALLOWED_FEATURES)]
    y_train = table_train[_TARGET]
    y_test = table_test[_TARGET]
    return x_train, x_test, y_train, y_test


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
    y_all = table[_TARGET]

    skf = StratifiedKFold(n_splits=_N_SPLITS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(table, y_all))

    fold_aucs: dict[str, list[float]] = {
        "LogisticRegression": [], "DecisionTree": [], "RandomForest": [],
    }
    rf_importances: list[np.ndarray] = []

    for fold_num, (train_idx, test_idx) in enumerate(fold_splits, start=1):
        print(f"Running fold {fold_num}/{_N_SPLITS}...", flush=True)
        x_train, x_test, y_train, y_test = _encode_fold(table, train_idx, test_idx)
        assert set(x_train.columns) == set(ALLOWED_FEATURES) == set(x_test.columns)

        lr_pipeline, dt_pipeline = _build_pipelines()
        rf_pipeline = _build_rf_pipeline()
        models = {
            "LogisticRegression": lr_pipeline,
            "DecisionTree": dt_pipeline,
            "RandomForest": rf_pipeline,
        }

        for name, model in models.items():
            model.fit(x_train, y_train)
            y_proba = model.predict_proba(x_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
            fold_aucs[name].append(auc)
            print(f"  {name:<20s} fold {fold_num} AUC = {auc:.4f}", flush=True)

            if name == "RandomForest":
                imp = permutation_importance(
                    model, x_test, y_test, n_repeats=_PERM_N_REPEATS,
                    random_state=SEED, scoring="roc_auc",
                )
                rf_importances.append(imp.importances_mean)

    # --- Aggregate per-model fold statistics ---
    stats: dict[str, tuple[float, float]] = {
        name: (float(np.mean(aucs)), float(np.std(aucs))) for name, aucs in fold_aucs.items()
    }
    lr_mean, lr_std = stats["LogisticRegression"]
    dt_mean, dt_std = stats["DecisionTree"]
    rf_mean, rf_std = stats["RandomForest"]

    # --- Unusual RF folds ---
    unusual_folds = [
        (i + 1, auc)
        for i, auc in enumerate(fold_aucs["RandomForest"])
        if abs(auc - rf_mean) > _UNUSUAL_FOLD_THRESHOLD_STD * rf_std
    ]

    # --- RF feature importance stability across folds ---
    importance_matrix = np.array(rf_importances)  # shape (n_folds, n_features)
    per_fold_top_feature = [
        ALLOWED_FEATURES[int(np.argmax(importance_matrix[f]))] for f in range(_N_SPLITS)
    ]
    order_month_top_count = sum(1 for feat in per_fold_top_feature if feat == "order_month")
    mean_importance = importance_matrix.mean(axis=0)
    std_importance = importance_matrix.std(axis=0)
    importance_rank_df = (
        pd.DataFrame(
            {
                "feature": list(ALLOWED_FEATURES),
                "mean_importance": mean_importance,
                "std_importance": std_importance,
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )

    # --- Verdict ---
    gap = rf_mean - dt_mean
    if gap > 0.03 and rf_std < 0.02:
        verdict = (
            f"RandomForest's advantage over the committed models is STABLE under 5-fold CV: "
            f"mean AUC {rf_mean:.4f} +/- {rf_std:.4f} vs. DecisionTree's {dt_mean:.4f} +/- "
            f"{dt_std:.4f} and LogisticRegression's {lr_mean:.4f} +/- {lr_std:.4f} -- the gap "
            f"({gap:+.4f}) holds up across folds with low fold-to-fold variance, so this is "
            f"real, generalizable signal the simpler committed models leave on the table, not "
            f"a single-split artifact."
        )
    elif gap > 0.01:
        verdict = (
            f"RandomForest's advantage over the committed models is MODEST and PARTIALLY "
            f"stable under 5-fold CV: mean AUC {rf_mean:.4f} +/- {rf_std:.4f} vs. "
            f"DecisionTree's {dt_mean:.4f} +/- {dt_std:.4f} and LogisticRegression's "
            f"{lr_mean:.4f} +/- {lr_std:.4f} -- the gap ({gap:+.4f}) is real but smaller "
            f"and/or noisier than the single-split comparison suggested."
        )
    else:
        verdict = (
            f"RandomForest's advantage over the committed models SHRINKS/VANISHES under "
            f"5-fold CV: mean AUC {rf_mean:.4f} +/- {rf_std:.4f} vs. DecisionTree's "
            f"{dt_mean:.4f} +/- {dt_std:.4f} and LogisticRegression's {lr_mean:.4f} +/- "
            f"{lr_std:.4f} -- the single-split gap ({gap:+.4f}) does not hold up as real, "
            f"generalizable signal; the committed models are close to the practical ceiling "
            f"for this feature set."
        )

    # --- Build report text ---
    lines: list[str] = [
        "Diagnostic: RandomForest 5-fold CV stability check (one-off, NOT part of the pipeline)",
        "=" * 90,
        "",
        f"StratifiedKFold(n_splits={_N_SPLITS}, shuffle=True, random_state={SEED}), "
        "identical 12 features and is_late target for all three models.",
        "",
        "Per-fold ROC-AUC:",
    ]
    for name in ("LogisticRegression", "DecisionTree", "RandomForest"):
        fold_str = ", ".join(f"{auc:.4f}" for auc in fold_aucs[name])
        lines.append(f"  {name:<20s} [{fold_str}]")

    lines += [
        "",
        "Mean +/- std across folds:",
    ]
    for name in ("LogisticRegression", "DecisionTree", "RandomForest"):
        mean, std = stats[name]
        lines.append(f"  {name:<20s} {mean:.4f} +/- {std:.4f}")

    lines += [
        "",
        f"Unusual RandomForest folds (|AUC - mean| > {_UNUSUAL_FOLD_THRESHOLD_STD} x std, "
        f"std={rf_std:.4f}):",
    ]
    if unusual_folds:
        for fold_num, auc in unusual_folds:
            lines.append(f"  Fold {fold_num}: AUC={auc:.4f} (mean={rf_mean:.4f})")
    else:
        lines.append("  None -- all 5 folds fall within 1.5 std of the mean.")

    lines += [
        "",
        "RandomForest top feature by permutation importance, per fold:",
    ]
    for fold_num, feat in enumerate(per_fold_top_feature, start=1):
        lines.append(f"  Fold {fold_num}: {feat}")
    stability_note = (
        "STABLE ranking." if order_month_top_count == _N_SPLITS else "ranking MOVES between folds."
    )
    lines.append(
        f"  order_month is the top feature in {order_month_top_count}/{_N_SPLITS} folds -- "
        f"{stability_note}"
    )

    lines += [
        "",
        "RandomForest permutation importance, mean +/- std across folds "
        f"(n_repeats={_PERM_N_REPEATS} per fold):",
    ]
    for _, row in importance_rank_df.iterrows():
        lines.append(
            f"  {row['feature']:<28s} {row['mean_importance']:+.4f} "
            f"(+/- {row['std_importance']:.4f})"
        )

    lines += ["", f"Verdict: {verdict}"]

    output = "\n".join(lines)
    print("\n" + output)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "diagnostic_rf_cv.txt").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
