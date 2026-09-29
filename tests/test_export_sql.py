"""Tests for src.export_sql: file existence, columns, datetime formatting, row counts."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src import config
from src.export_sql import (
    _CSV_REPORT_SOURCES,
    _PARQUET_SOURCES,
    _assert_no_nested_columns,
    _format_datetime_columns,
    main,
)


def _pipeline_outputs_available() -> bool:
    parquet_ok = all(
        (config.PROCESSED_DATA_DIR / f"{name}.parquet").exists() for name in _PARQUET_SOURCES
    )
    reports_ok = all(
        (config.REPORTS_DIR / f"{name}.csv").exists() for name in _CSV_REPORT_SOURCES
    )
    return parquet_ok and reports_ok


def _source_row_count(name: str) -> int:
    parquet_path = config.PROCESSED_DATA_DIR / f"{name}.parquet"
    if parquet_path.exists():
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    return len(pd.read_csv(config.REPORTS_DIR / f"{name}.csv"))


def _source_columns(name: str) -> set[str]:
    parquet_path = config.PROCESSED_DATA_DIR / f"{name}.parquet"
    if parquet_path.exists():
        return set(pq.ParquetFile(parquet_path).schema.names)
    return set(pd.read_csv(config.REPORTS_DIR / f"{name}.csv", nrows=0).columns)


@pytest.fixture(scope="module")
def export_row_counts() -> dict[str, int]:
    if not _pipeline_outputs_available():
        pytest.skip("data/processed and reports outputs not available; run run_all.py first")
    return main()


def test_format_datetime_columns_iso8601_and_empty_for_nat() -> None:
    df = pd.DataFrame({"ts": [pd.Timestamp("2020-01-01 10:00:00"), pd.NaT]})
    formatted = _format_datetime_columns(df, ("ts",))
    assert formatted["ts"].iloc[0] == "2020-01-01T10:00:00"
    assert formatted["ts"].iloc[1] == ""


def test_assert_no_nested_columns_raises_on_list_values() -> None:
    df = pd.DataFrame({"col": [[1, 2, 3]]})
    with pytest.raises(AssertionError):
        _assert_no_nested_columns(df, "test_table")


def test_assert_no_nested_columns_passes_on_plain_values() -> None:
    df = pd.DataFrame({"col": ["a", "b", None]})
    _assert_no_nested_columns(df, "test_table")  # must not raise


def test_every_expected_file_exists(export_row_counts: dict[str, int]) -> None:
    for name in (*_PARQUET_SOURCES, *_CSV_REPORT_SOURCES):
        assert (config.EXPORT_DATA_DIR / f"{name}.csv").exists()


def test_row_counts_match_source_exactly(export_row_counts: dict[str, int]) -> None:
    for name in (*_PARQUET_SOURCES, *_CSV_REPORT_SOURCES):
        exported = len(pd.read_csv(config.EXPORT_DATA_DIR / f"{name}.csv"))
        assert exported == _source_row_count(name) == export_row_counts[name]


def test_exported_columns_match_source(export_row_counts: dict[str, int]) -> None:
    for name in (*_PARQUET_SOURCES, *_CSV_REPORT_SOURCES):
        exported_cols = set(pd.read_csv(config.EXPORT_DATA_DIR / f"{name}.csv", nrows=0).columns)
        assert exported_cols == _source_columns(name)


def test_no_literal_nat_string_in_any_export(export_row_counts: dict[str, int]) -> None:
    for name in (*_PARQUET_SOURCES, *_CSV_REPORT_SOURCES):
        raw_text = (config.EXPORT_DATA_DIR / f"{name}.csv").read_text(encoding="utf-8")
        assert "NaT" not in raw_text


def test_fact_order_items_datetime_columns_are_iso8601(export_row_counts: dict[str, int]) -> None:
    df = pd.read_csv(
        config.EXPORT_DATA_DIR / "FactOrderItems.csv", dtype=str, keep_default_na=False
    )
    non_null_sample = df.loc[df["order_purchase_timestamp"] != "", "order_purchase_timestamp"]
    assert non_null_sample.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$").all()
    # review_creation_date has real nulls (~0.75% of orders have no review):
    # confirm they're empty strings, never dropped rows or the literal "NaT".
    assert (df["review_creation_date"] == "").sum() > 0
