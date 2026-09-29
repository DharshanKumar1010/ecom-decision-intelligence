"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from src import build_tables, config


def _raw_data_available() -> bool:
    """Return True iff all 9 expected raw Olist CSVs are present."""
    try:
        config.verify_raw_files()
    except FileNotFoundError:
        return False
    return True


@pytest.fixture(scope="session")
def fact_order_items() -> pd.DataFrame:
    """Load FactOrderItems, building it first if needed. Skips if raw data is absent."""
    if not _raw_data_available():
        pytest.skip("raw data not available in data/raw/")
    fact_path = config.PROCESSED_DATA_DIR / "FactOrderItems.parquet"
    if not fact_path.exists():
        build_tables.main()
    return pd.read_parquet(fact_path)


@pytest.fixture(scope="session")
def dim_seller() -> pd.DataFrame:
    """Load DimSeller, building it first if needed. Skips if raw data is absent."""
    if not _raw_data_available():
        pytest.skip("raw data not available in data/raw/")
    path = config.PROCESSED_DATA_DIR / "DimSeller.parquet"
    if not path.exists():
        build_tables.main()
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def dim_product() -> pd.DataFrame:
    """Load DimProduct, building it first if needed. Skips if raw data is absent."""
    if not _raw_data_available():
        pytest.skip("raw data not available in data/raw/")
    path = config.PROCESSED_DATA_DIR / "DimProduct.parquet"
    if not path.exists():
        build_tables.main()
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def dim_customer() -> pd.DataFrame:
    """Load DimCustomer, building it first if needed. Skips if raw data is absent."""
    if not _raw_data_available():
        pytest.skip("raw data not available in data/raw/")
    path = config.PROCESSED_DATA_DIR / "DimCustomer.parquet"
    if not path.exists():
        build_tables.main()
    return pd.read_parquet(path)
