"""Tests for src.config: raw-file verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import EXPECTED_RAW_FILES, RAW_DATA_DIR, verify_raw_files


def test_verify_raw_files_passes_on_real_raw_dir() -> None:
    # The raw Olist CSVs are input data, not part of the repository. With none present (a fresh
    # checkout, CI) there is nothing to verify; a directory holding SOME but not all of the
    # files still fails, which is what this test is for.
    if not RAW_DATA_DIR.is_dir() or not any(RAW_DATA_DIR.glob("*.csv")):
        pytest.skip("data/raw has no CSV files; the Olist input data is not in this checkout")
    verify_raw_files()


def test_verify_raw_files_lists_missing(tmp_path: Path) -> None:
    # Create all but one expected file in an empty temp dir.
    missing_file = EXPECTED_RAW_FILES[0]
    for name in EXPECTED_RAW_FILES[1:]:
        (tmp_path / name).touch()

    with pytest.raises(FileNotFoundError) as exc_info:
        verify_raw_files(tmp_path)
    assert missing_file in str(exc_info.value)


def test_verify_raw_files_empty_dir_lists_all(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        verify_raw_files(tmp_path)
    message = str(exc_info.value)
    for name in EXPECTED_RAW_FILES:
        assert name in message
