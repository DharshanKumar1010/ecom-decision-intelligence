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

from app._labels import with_tier_labels
from src import config
from src.ahp import seller_order_table
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
# Order-level counts and figures read from files
# --------------------------------------------------------------------------

_FACT_COLUMNS = ("order_id", "seller_id", "review_score", "is_late")


@st.cache_data(show_spinner=False)
def _marketplace_review(path: str, mtime: float) -> tuple[float, float]:
    fact = pd.read_parquet(path, columns=list(_FACT_COLUMNS))
    per_order = float(seller_order_table(fact)["review"].mean())
    return per_order, float(fact["review_score"].mean())


def marketplace_review() -> tuple[float, float]:
    """`(mean review per reviewed order, mean review per item)` for the whole marketplace.

    The first uses one review per order (the definition behind every seller's average review);
    the second counts a review once for each item of its order and is shown only for contrast.
    """
    return _marketplace_review(*data_key(processed_path("FactOrderItems.parquet")))


_TEMPORAL_ROW = re.compile(
    r"^(?P<label>all 12 features|without order_month.*?)\s+(?P<lr>\d\.\d{4})\s+(?P<dt>\d\.\d{4})\s+"
    r"(?P<rf>\d\.\d{4})",
    re.MULTILINE,
)


def temporal_results() -> dict[str, dict[str, float]]:
    """AUCs of the forward-looking check from `reports/diagnostics/diagnostic_temporal.txt`.

    Returns `{"all": {"lr", "dt", "rf"}, "no_calendar": {...}}`, or `{}` if the report is absent
    or in a different format (the page then leaves the sentence out rather than guess).
    """
    path = config.REPORTS_DIR / "diagnostics" / "diagnostic_temporal.txt"
    if not path.exists():
        return {}
    found: dict[str, dict[str, float]] = {}
    for match in _TEMPORAL_ROW.finditer(_read_text(*data_key(path))):
        key = "all" if match["label"].startswith("all") else "no_calendar"
        found[key] = {m: float(match[m]) for m in ("lr", "dt", "rf")}
    return found if {"all", "no_calendar"} <= set(found) else {}


def schema_table_count() -> int:
    """Number of tables `sql/schema.sql` creates (the files that are loaded into SQL Server)."""
    text = (config.PROJECT_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    return len(re.findall(r"^\s*CREATE\s+TABLE\s", text, flags=re.MULTILINE | re.IGNORECASE))


def hidden_gems_download(gems: pd.DataFrame) -> bytes:
    """CSV bytes of the hidden-gem list: pipeline column names, user-facing tier names."""
    return with_tier_labels(gems).to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------
# Small presentation helpers
# --------------------------------------------------------------------------

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
