"""Run the full Stage 1 + Stage 2 pipeline in order and print a summary
(CLAUDE.md section 9).

Stages: build_tables -> synthetic -> clickstream sanity checks -> sentiment
-> predict -> ahp -> discovery -> expert. Each stage is timed and logged;
the pipeline stops on the first failure (no exception is swallowed). After
the pipeline, `pytest`, `ruff check .`, and `mypy src` are run as
subprocesses and their pass/fail status is included in the final summary.

Every number in the final summary is read back from a file a stage just
wrote (or, for the AHP consistency ratio, recomputed via `ahp.compute_weights`
on the same `AHP_PAIRWISE_MATRIX` `ahp.main()` used) — nothing here is
recomputed independently or guessed. Per CLAUDE.md section 5, this is the
only module allowed to use bare `print` (the final summary table).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src import ahp, build_tables, clickstream, discovery, expert, predict, sentiment, synthetic
from src.config import (
    AHP_PAIRWISE_MATRIX,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    get_logger,
)

logger = get_logger(__name__)

_STAGE_ROW_COUNT_FILES: tuple[tuple[str, Path], ...] = (
    ("FactOrderItems", PROCESSED_DATA_DIR / "FactOrderItems.parquet"),
    ("DimSeller", PROCESSED_DATA_DIR / "DimSeller.parquet"),
    ("DimProduct", PROCESSED_DATA_DIR / "DimProduct.parquet"),
    ("DimCustomer", PROCESSED_DATA_DIR / "DimCustomer.parquet"),
    ("Clickstream (synthetic)", PROCESSED_DATA_DIR / "Clickstream.parquet"),
    ("CallTranscripts (synthetic)", PROCESSED_DATA_DIR / "CallTranscripts.parquet"),
    ("CallSentiment", PROCESSED_DATA_DIR / "CallSentiment.parquet"),
    ("LatePredictions", PROCESSED_DATA_DIR / "LatePredictions.parquet"),
    ("SellerFacts", PROCESSED_DATA_DIR / "SellerFacts.parquet"),
    ("ahp_ranking.csv (established sellers)", REPORTS_DIR / "ahp_ranking.csv"),
    ("discovery_quadrants.csv", REPORTS_DIR / "discovery_quadrants.csv"),
    ("seller_recommendations.csv", REPORTS_DIR / "seller_recommendations.csv"),
)


def _run_stage(name: str, fn: Callable[[], None]) -> float:
    """Run one pipeline stage, logging its start/end and timing it.

    Args:
        name: human-readable stage name for logging.
        fn: zero-argument callable that runs the stage (raises on failure).

    Returns:
        Elapsed wall-clock time in seconds.
    """
    logger.info("Starting stage: %s", name)
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    logger.info("Finished stage: %s (%.2fs)", name, elapsed)
    return elapsed


def _clickstream_checks() -> None:
    """Sanity-check the synthetic clickstream funnel (Stage 1 metrics, re-verified)."""
    events = pd.read_parquet(PROCESSED_DATA_DIR / "Clickstream.parquet")
    funnel = clickstream.funnel_conversion(events)
    bounce = clickstream.bounce_rate(events)
    abandonment = clickstream.cart_abandonment_rate(events)
    assert (funnel["sessions"].diff().dropna() <= 0).all(), "funnel stage counts not monotonic"
    assert 0.0 <= bounce <= 1.0, "bounce_rate outside [0, 1]"
    assert 0.0 <= abandonment <= 1.0, "cart_abandonment_rate outside [0, 1]"
    logger.info(
        "Clickstream sanity check OK: bounce_rate=%.4f, cart_abandonment_rate=%.4f",
        bounce,
        abandonment,
    )


def _run_quality_gate(name: str, args: list[str]) -> bool:
    """Run one quality-gate subprocess (pytest/ruff/mypy) and report pass/fail.

    Args:
        name: label for logging.
        args: full command, e.g. `[sys.executable, "-m", "pytest", "-q"]`.

    Returns:
        True if the subprocess exited 0.
    """
    result = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    passed = result.returncode == 0
    logger.info("%s: %s", name, "PASSED" if passed else "FAILED")
    if not passed:
        logger.info("%s stdout (tail):\n%s", name, result.stdout[-4000:])
        logger.info("%s stderr (tail):\n%s", name, result.stderr[-4000:])
    return passed


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _row_count(path: Path) -> int:
    if path.suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    return len(pd.read_csv(path))


def main() -> None:
    """Run every pipeline stage in order, then the quality gates, then print the summary."""
    stages: list[tuple[str, Callable[[], None]]] = [
        ("build_tables", build_tables.main),
        ("synthetic", synthetic.main),
        ("clickstream_checks", _clickstream_checks),
        ("sentiment", sentiment.main),
        ("predict", predict.main),
        ("ahp", ahp.main),
        ("discovery", discovery.main),
        ("expert", expert.main),
    ]

    timings: dict[str, float] = {}
    for name, fn in stages:
        timings[name] = _run_stage(name, fn)

    gates = {
        "pytest": _run_quality_gate("pytest", [sys.executable, "-m", "pytest", "-q"]),
        "ruff": _run_quality_gate("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        "mypy": _run_quality_gate("mypy", [sys.executable, "-m", "mypy", "src"]),
    }

    metrics = _read_json(REPORTS_DIR / "model_metrics.json")
    lr_auc = metrics["logistic_regression"]["roc_auc"]
    dt_auc = metrics["decision_tree"]["roc_auc"]

    ahp_ranking = pd.read_csv(REPORTS_DIR / "ahp_ranking.csv")
    _, _, _, ahp_cr = ahp.compute_weights(AHP_PAIRWISE_MATRIX)

    quadrants = pd.read_csv(REPORTS_DIR / "discovery_quadrants.csv")
    quadrant_counts = quadrants["quadrant"].value_counts().to_dict()

    print("\n" + "=" * 72)
    print("STAGE 1 + STAGE 2 PIPELINE SUMMARY")
    print("=" * 72)

    print("\nStage timings:")
    for name, elapsed in timings.items():
        print(f"  {name:<20s} {elapsed:>8.2f}s")

    print("\nFiles written (row counts):")
    for label, path in _STAGE_ROW_COUNT_FILES:
        count = _row_count(path) if path.exists() else "MISSING"
        print(f"  {label:<40s} {count}")

    print("\nLate-delivery models (test-set ROC-AUC):")
    print(f"  logistic_regression: {lr_auc:.4f}")
    print(f"  decision_tree:       {dt_auc:.4f}")

    print("\nAHP (established sellers, >= 30 orders):")
    print(f"  seller count: {len(ahp_ranking)}")
    print(f"  consistency ratio (CR): {ahp_cr:.4f}")

    print("\nDiscovery quadrant counts:")
    for label in ("Star", "Hidden Gem", "Overrated", "Overlooked-Low-Quality"):
        print(f"  {label:<25s} {quadrant_counts.get(label, 0)}")

    print("\nQuality gates:")
    for name, passed in gates.items():
        print(f"  {name:<10s} {'PASSED' if passed else 'FAILED'}")

    print("=" * 72 + "\n")

    if not all(gates.values()):
        raise SystemExit("One or more quality gates failed; see log output above.")


if __name__ == "__main__":
    main()
