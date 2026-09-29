"""SQL Server CSV export (CLAUDE.md section 15).

Converts every current Python output (Parquet in `data/processed/`, CSV
reports in `reports/`) into clean, flat CSVs in `data/export/`, one file per
intended SQL Server table — CLAUDE.md section 16's table mapping is the
source of truth for filenames. No database connection, no SQL, no ORM: this
module's entire job is producing importable flat files for SSMS's Import
Flat File wizard; everything past that is manual (section 16).

SSMS-import-friendly formatting:
- Datetime columns are explicitly formatted as ISO 8601
  (`YYYY-MM-DDTHH:MM:SS`); a missing (`NaT`) value is written as an empty
  string, never the literal text `"NaT"` pandas would otherwise write
  (which SSMS would import as a bad, non-null string instead of NULL).
- A fixed `float_format` keeps decimal formatting consistent across every
  exported file, including the ones already CSV (`ahp_ranking`,
  `discovery_quadrants`, `seller_recommendations`), not just the
  Parquet-sourced ones.
- UTF-8 encoding, header row always on, no nested/list-type columns
  (asserted defensively; none exist in the current schemas).
"""

from __future__ import annotations

import pandas as pd

from src.config import EXPORT_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, get_logger

logger = get_logger(__name__)

_FLOAT_FORMAT = "%.6f"

# CLAUDE.md section 16's CSV -> SQL Server table mapping, by source kind.
_PARQUET_SOURCES: tuple[str, ...] = (
    "FactOrderItems", "DimSeller", "DimProduct", "DimCustomer",
    "LatePredictions", "CallSentiment", "Clickstream",
)
_CSV_REPORT_SOURCES: tuple[str, ...] = (
    "ahp_ranking", "discovery_quadrants", "seller_recommendations",
)
_DATETIME_COLUMNS: dict[str, tuple[str, ...]] = {
    "FactOrderItems": (
        "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date", "order_date", "review_creation_date",
    ),
    "Clickstream": ("event_time",),
}


def _format_datetime_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Explicitly format datetime columns as ISO 8601, `NaT` -> empty string.

    Args:
        df: DataFrame to format (not mutated; a copy is returned).
        columns: datetime column names to format.

    Returns:
        Copy of `df` with each column in `columns` replaced by an ISO 8601
        string column (empty string, not the literal text "NaT", where the
        original value was missing).
    """
    formatted = df.copy()
    for col in columns:
        formatted[col] = formatted[col].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    return formatted


def _assert_no_nested_columns(df: pd.DataFrame, name: str) -> None:
    """Defensively assert no column holds list/dict-typed values.

    Args:
        df: DataFrame to check.
        name: table name, for the error message.

    Raises:
        AssertionError: if any object-dtype column's first non-null value
            is a list or dict.
    """
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if not sample.empty and isinstance(sample.iloc[0], list | dict):
                raise AssertionError(f"{name}.{col} holds nested (list/dict) values")


def _export_table(df: pd.DataFrame, name: str, datetime_columns: tuple[str, ...] = ()) -> int:
    """Format and write one table to `data/export/<name>.csv`.

    Args:
        df: table to export.
        name: base filename (without extension), matching CLAUDE.md section
            16's CSV column.
        datetime_columns: columns to format as ISO 8601 strings.

    Returns:
        Row count written.
    """
    _assert_no_nested_columns(df, name)
    formatted = _format_datetime_columns(df, datetime_columns) if datetime_columns else df
    EXPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DATA_DIR / f"{name}.csv"
    formatted.to_csv(path, index=False, encoding="utf-8", float_format=_FLOAT_FORMAT)
    logger.info("Exported %s (%d rows) -> %s", name, len(formatted), path)
    return len(formatted)


def main() -> dict[str, int]:
    """Export every current Python output to `data/export/*.csv`.

    Returns:
        Dict mapping table name to the real row count exported.
    """
    row_counts: dict[str, int] = {}
    for name in _PARQUET_SOURCES:
        df = pd.read_parquet(PROCESSED_DATA_DIR / f"{name}.parquet")
        row_counts[name] = _export_table(df, name, _DATETIME_COLUMNS.get(name, ()))

    for name in _CSV_REPORT_SOURCES:
        df = pd.read_csv(REPORTS_DIR / f"{name}.csv")
        row_counts[name] = _export_table(df, name)

    logger.info("Exported %d tables to %s: %s", len(row_counts), EXPORT_DATA_DIR, row_counts)
    return row_counts


if __name__ == "__main__":
    main()
