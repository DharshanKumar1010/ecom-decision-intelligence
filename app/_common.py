"""Shared UI plumbing for the Streamlit dashboard.

Presentation only: cached file loaders, the data catalog and small parsing
helpers (page chrome lives in `app/_components.py`, colours in `app/_theme.py`).
No business logic lives here or in any page — pages call existing `src/` functions and
read files the pipeline already produced (CLAUDE.md section 7.8).

Every loader resolves its path at CALL time through `src.config` and puts the
resolved path plus the file's mtime into the `st.cache_data` key. That makes
the cache correct when a test redirects a data directory, and makes a re-run
pipeline show fresh numbers without restarting the app.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from src import config
from src.expert import KnowledgeBase

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------


def data_key(path: Path) -> tuple[str, float]:
    """Cache key for a file: its path and mtime (raises `FileNotFoundError` if absent)."""
    return str(path), path.stat().st_mtime


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, mtime: float, columns: tuple[str, ...] | None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_text(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as fh:
        loaded: dict[str, object] = json.load(fh)
    return loaded


@st.cache_data(show_spinner=False)
def _read_rules(path: str, mtime: float) -> KnowledgeBase:
    return KnowledgeBase.from_yaml(Path(path))


@st.cache_data(show_spinner=False)
def _count_rows(path: str, mtime: float) -> int:
    if path.endswith(".parquet"):
        return int(pq.ParquetFile(path).metadata.num_rows)
    return len(pd.read_csv(path))


def processed_path(name: str) -> Path:
    """Path of a `data/processed/` file, resolved at call time."""
    return config.PROCESSED_DATA_DIR / name


def report_path(name: str) -> Path:
    """Path of a `reports/` file, resolved at call time."""
    return config.REPORTS_DIR / name


def load_parquet(name: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Load a `data/processed/` parquet file (optionally only some columns)."""
    return _read_parquet(*data_key(processed_path(name)), columns)


def load_report_csv(name: str) -> pd.DataFrame:
    """Load a `reports/` CSV."""
    return _read_csv(*data_key(report_path(name)))


def load_report_text(name: str) -> str:
    """Load a `reports/` text file."""
    return _read_text(*data_key(report_path(name)))


def load_report_json(name: str) -> dict[str, object]:
    """Load a `reports/` JSON file."""
    return _read_json(*data_key(report_path(name)))


def load_knowledge_base() -> KnowledgeBase:
    """Load and validate `rules/seller_rules.yaml` (via `KnowledgeBase.from_yaml`)."""
    return _read_rules(*data_key(config.RULES_DIR / "seller_rules.yaml"))


# --------------------------------------------------------------------------
# Small presentation helpers
# --------------------------------------------------------------------------

# A model-quality review is announced by an anchored line such as "FLAG: <token>" or
# "STATUS: <token>" (optionally followed by a reason) in BUILD_LOG.md or a reports/ text
# file. Anchoring means prose that merely mentions the token never triggers the note.
REVIEW_TOKEN = "MODEL_UNDER_REVIEW"
_REVIEW_LINE = re.compile(
    rf"^\s*(?:FLAG|STATUS):\s*{REVIEW_TOKEN}\b[\s:.\-]*(?P<reason>.*)$", re.MULTILINE
)


def model_review_flag() -> tuple[str, str] | None:
    """`(source file name, reason)` if a model-review flag exists, else `None`.

    Looks at `BUILD_LOG.md` and every `*.md`/`*.txt` under `reports/`, resolved at call
    time. Nothing in the repo sets this flag today, so the UI shows no note until it exists.
    """
    candidates = [config.PROJECT_ROOT / "BUILD_LOG.md"]
    if config.REPORTS_DIR.exists():
        candidates += sorted(config.REPORTS_DIR.glob("*.md")) + sorted(
            config.REPORTS_DIR.glob("*.txt")
        )
    for path in candidates:
        if not path.exists():
            continue
        match = _REVIEW_LINE.search(path.read_text(encoding="utf-8", errors="replace"))
        if match:
            return path.name, match.group("reason").strip()
    return None


_CV_LINE = re.compile(r"^\s*(\w+)\s+([0-9.]+)\s*\+/-\s*([0-9.]+)\s*$")


def parse_cv_summary(text: str) -> dict[str, tuple[float, float]]:
    """Read the "Mean +/- std across folds" block of `diagnostic_rf_cv.txt`.

    Returns `{model_name: (mean_auc, std_auc)}`, or `{}` when the block is not
    found (the caller falls back to single-split numbers).
    """
    marker = "Mean +/- std across folds:"
    if marker not in text:
        return {}
    block = text.split(marker, 1)[1].split("\n\n", 1)[0]
    parsed: dict[str, tuple[float, float]] = {}
    for line in block.splitlines():
        match = _CV_LINE.match(line)
        if match:
            parsed[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return parsed


def quadrant_thresholds(quadrants: pd.DataFrame) -> tuple[float, float]:
    """Quality and popularity split lines for the Discovery scatter.

    Mirrors `discovery.compute_quadrants`' documented default: the
    `config.DISCOVERY_*_THRESHOLD` override when set, otherwise the median of
    that axis (both overrides are currently `None`).
    """
    quality = config.DISCOVERY_QUALITY_THRESHOLD
    popularity = config.DISCOVERY_POPULARITY_THRESHOLD
    return (
        float(quadrants["ahp_score"].median()) if quality is None else quality,
        float(quadrants["order_volume"].median()) if popularity is None else popularity,
    )


# --------------------------------------------------------------------------
# Data catalog (DSS Architecture page, Home status check)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    """One pipeline output: where it lives and what it is."""

    name: str
    kind: str  # "parquet" (data/processed) or "csv" (reports)
    description: str
    synthetic: bool = False

    @property
    def path(self) -> Path:
        """Resolved at call time so tests can redirect the data directories."""
        if self.kind == "parquet":
            return processed_path(f"{self.name}.parquet")
        return report_path(f"{self.name}.csv")

    def row_count(self) -> int | None:
        """Live row count, or `None` if the file is missing."""
        try:
            return _count_rows(*data_key(self.path))
        except FileNotFoundError:
            return None


DATA_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry("FactOrderItems", "parquet", "Fact table: one row per delivered order item "
                 "(price, freight, review score, is_late target, timestamps)."),
    CatalogEntry("DimSeller", "parquet", "Seller dimension: seller id and state."),
    CatalogEntry("DimProduct", "parquet", "Product dimension: category, weight, English "
                 "category name."),
    CatalogEntry("DimCustomer", "parquet", "Customer dimension: customer id and state (built "
                 "from all raw customers, not only delivered orders)."),
    CatalogEntry("Clickstream", "parquet", "Web events (page_view / add_to_cart / checkout / "
                 "purchase) with device and source.", synthetic=True),
    CatalogEntry("CallTranscripts", "parquet", "Call-center transcripts with hold seconds and "
                 "silence %.", synthetic=True),
    CatalogEntry("CallSentiment", "parquet", "VADER compound score and label per synthetic "
                 "call transcript.", synthetic=True),
    CatalogEntry("LatePredictions", "parquet", "Per-item predicted late-delivery risk: "
                 "late_risk (RandomForest, primary), late_risk_lr, late_risk_dt."),
    CatalogEntry("SellerFacts", "parquet", "Per-seller facts consumed by the expert system "
                 "(AHP score, avg late risk, reviews, volume, late rate, hidden-gem flag)."),
    CatalogEntry("ahp_ranking", "csv", "AHP ranking of established sellers (>= 30 items sold) "
                 "with per-criterion contributions."),
    CatalogEntry("discovery_quadrants", "csv", "Quality-vs-popularity quadrant and confidence "
                 "flag per AHP-eligible seller (>= 5 items sold)."),
    CatalogEntry("seller_recommendations", "csv", "Batch expert-system output: action, fired "
                 "rule and rationale per seller."),
    CatalogEntry("sentiment_agent_summary", "csv", "Per-agent aggregates of the synthetic "
                 "call sentiment.", synthetic=True),
    CatalogEntry("coefficients", "csv", "Standardized logistic-regression coefficients."),
    CatalogEntry("rf_feature_importance", "csv", "RandomForest permutation importance "
                 "(test set)."),
    CatalogEntry("sensitivity", "csv", "Mean change in predicted late risk under feature "
                 "perturbations, per model."),
)
