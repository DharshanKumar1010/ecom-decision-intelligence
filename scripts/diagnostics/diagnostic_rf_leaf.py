"""One-off diagnostic: how much does RandomForest `min_samples_leaf` matter, and what does it cost?

CURRENT: re-running it reproduces the committed report exactly (checked in Stage 8); it is the
evidence for the forest leaf size of 20 (config.PREDICT_RF_MIN_SAMPLES_LEAF).

NOT part of the pipeline: not wired into `run_all.py`; the only file written is
`reports/diagnostics/diagnostic_rf_leaf.txt`. Run directly:
`python scripts/diagnostics/diagnostic_rf_leaf.py`
(slow: 5 RandomForest fits for each of 3 leaf sizes).

Protocol: the same order-grouped 5-fold out-of-fold protocol `src.predict` uses
(`grouped_stratified_folds` + `out_of_fold_predictions`), 12 features, committed forest config
(300 trees, unlimited depth, balanced class weights) with only `min_samples_leaf` varied over
{1, 5, 20}. Reports the pooled out-of-fold ROC-AUC, the per-fold mean and std, and the pickled
size of the model fitted on the fold-0 training rows (the same split that is persisted).

Decision rule written before looking at the numbers: adopt a larger leaf only if its pooled
OOF AUC is within 0.005 of leaf 1 AND its fitted size is materially smaller. The leaf is
chosen on the same folds it is reported on, so any gain would be slightly optimistic.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR  # noqa: E402
from src.predict import (  # noqa: E402
    _build_rf_pipeline,
    _load_geo_centroids,
    _load_payments,
    _load_seller_customer_zips,
    build_feature_table,
    grouped_stratified_folds,
    out_of_fold_predictions,
)

LEAF_SIZES = (1, 5, 20)
AUC_TOLERANCE = 0.005
_sizes: dict[int, int] = {}


class _SizedPipeline(Pipeline):
    """Pipeline that records its pickled size the first time it is fitted (fold 0)."""

    leaf = 0

    def fit(self, X: Any, y: Any = None, **params: Any) -> _SizedPipeline:  # noqa: N803
        super().fit(X, y, **params)
        if self.leaf not in _sizes:
            buffer = io.BytesIO()
            joblib.dump(self, buffer)
            _sizes[self.leaf] = buffer.getbuffer().nbytes
        return self


def _builder(leaf: int):  # type: ignore[no-untyped-def]
    def build() -> Pipeline:
        base = _build_rf_pipeline(min_samples_leaf=leaf)
        sized = _SizedPipeline(base.steps)
        sized.leaf = leaf
        return sized

    return build


def main() -> None:
    """Run the protocol for each leaf size, print and save the report."""
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
    folds = grouped_stratified_folds(table)
    y = table["is_late"].to_numpy()

    rows = []
    for leaf in LEAF_SIZES:
        started = time.time()
        oof = out_of_fold_predictions(table, {"p": _builder(leaf)}, folds)
        proba = oof["p"].to_numpy()
        fold_aucs = [float(roc_auc_score(y[test], proba[test])) for _, test in folds]
        rows.append({
            "leaf": leaf,
            "pooled": float(roc_auc_score(y, proba)),
            "mean": float(np.mean(fold_aucs)),
            "std": float(np.std(fold_aucs)),
            "mb": _sizes[leaf] / 1e6,
            "minutes": (time.time() - started) / 60,
        })
        print(rows[-1], flush=True)  # noqa: T201 - one-off diagnostic script

    base = rows[0]
    lines = [
        "Diagnostic: RandomForest min_samples_leaf under the order-grouped OOF protocol",
        "=" * 78,
        f"{'leaf':>5}{'pooled OOF AUC':>16}{'fold mean':>11}{'fold std':>10}{'vs leaf 1':>11}"
        f"{'fitted size MB':>16}{'size vs leaf 1':>16}{'minutes':>9}",
    ]
    for r in rows:
        lines.append(
            f"{r['leaf']:>5}{r['pooled']:>16.4f}{r['mean']:>11.4f}{r['std']:>10.4f}"
            f"{r['pooled'] - base['pooled']:>+11.4f}{r['mb']:>16.1f}"
            f"{r['mb'] / base['mb']:>15.1%} {r['minutes']:>8.1f}"
        )
    lines += ["", f"Rule: adopt only if pooled OOF AUC is within {AUC_TOLERANCE} of leaf 1 and "
              "the fitted size is materially smaller."]
    verdict = [r["leaf"] for r in rows[1:]
               if base["pooled"] - r["pooled"] <= AUC_TOLERANCE and r["mb"] < 0.5 * base["mb"]]
    lines.append(f"Leaf sizes meeting the rule (AUC within tolerance and under half the size): "
                 f"{verdict or 'none'}")
    report = "\n".join(lines)
    print(report)  # noqa: T201
    report_path = REPORTS_DIR / "diagnostics" / "diagnostic_rf_leaf.txt"
    report_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
