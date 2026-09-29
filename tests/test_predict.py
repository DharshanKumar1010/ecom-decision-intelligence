"""Tests for src.predict: feature contract, leakage guardrail, reproducibility.

Uses the real processed tables (via conftest fixtures) since the feature
table depends on joins across Fact/Dim tables that would be tedious and
less convincing to hand-build; each test still runs a fast, deterministic
train/test split rather than the full model-metrics pipeline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.config import ALLOWED_FEATURES, PREDICT_AUC_LEAK_GUARDRAIL, RAW_DATA_DIR, verify_raw_files
from src.predict import (
    _aggregate_geo_centroids,
    _build_pipelines,
    _build_rf_pipeline,
    _evaluate,
    _haversine_km,
    _load_geo_centroids,
    _load_payments,
    _load_seller_customer_zips,
    _split_and_encode,
    build_feature_table,
    compute_geo_distance,
    compute_seller_historical_late_rate,
    encode_product_category_frequency,
    fit_product_category_frequency,
)

_FORBIDDEN_COLUMNS = {
    "delivery_days",
    "order_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_purchase_timestamp",
    "is_late",
}


@pytest.fixture(scope="module")
def feature_table(
    fact_order_items: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_seller: pd.DataFrame,
    dim_customer: pd.DataFrame,
) -> pd.DataFrame:
    payments = _load_payments()
    geo_centroids = _load_geo_centroids()
    seller_zips, customer_zips = _load_seller_customer_zips()
    return build_feature_table(
        fact_order_items, dim_product, dim_seller, dim_customer, payments,
        geo_centroids, seller_zips, customer_zips,
    )


@pytest.fixture(scope="module")
def fitted_models(feature_table: pd.DataFrame) -> dict[str, Any]:
    """Fit LR/DT/RF exactly once per test module and share across tests.

    RandomForest(300 trees, unlimited depth) on ~88K rows is slow to fit
    (multiple minutes, confirmed by reports/diagnostic_rf_*.txt); fitting it
    fresh in every test would make the suite impractically slow.
    """
    x_train, x_test, y_train, y_test, _ = _split_and_encode(feature_table)
    lr_pipeline, dt_pipeline = _build_pipelines()
    rf_pipeline = _build_rf_pipeline()
    lr_pipeline.fit(x_train, y_train)
    dt_pipeline.fit(x_train, y_train)
    rf_pipeline.fit(x_train, y_train)
    return {
        "lr": lr_pipeline,
        "dt": dt_pipeline,
        "rf": rf_pipeline,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


def test_feature_columns_equal_allowed_set_exactly(feature_table: pd.DataFrame) -> None:
    # build_feature_table returns every precomputable ALLOWED_FEATURES column
    # plus the raw product_category string (encoded post-split, not itself a
    # feature) and item_key/is_late.
    columns = set(feature_table.columns) - {"item_key", "is_late", "product_category"}
    assert columns == set(ALLOWED_FEATURES) - {"product_category_freq"}


def test_product_category_frequency_fit_and_encode() -> None:
    train_categories = pd.Series(["toys", "toys", "toys", "books", "books", "games"])
    freq_map = fit_product_category_frequency(train_categories)

    assert freq_map["toys"] == pytest.approx(3 / 6)
    assert freq_map["books"] == pytest.approx(2 / 6)
    assert freq_map["games"] == pytest.approx(1 / 6)
    assert sum(freq_map.values()) == pytest.approx(1.0)

    encoded = encode_product_category_frequency(
        pd.Series(["toys", "books", "unseen_category"]), freq_map
    )
    assert encoded.iloc[0] == pytest.approx(3 / 6)
    assert encoded.iloc[1] == pytest.approx(2 / 6)
    assert encoded.iloc[2] == pytest.approx(0.0)  # never observed in training


def test_order_month_and_day_of_week_derived_correctly() -> None:
    fact = pd.DataFrame(
        {
            "item_key": ["i1", "i2"],
            "order_id": ["o1", "o2"],
            "seller_id": ["S1", "S1"],
            "product_id": ["P1", "P1"],
            "customer_id": ["C1", "C1"],
            "price": [10.0, 10.0],
            "freight_value": [1.0, 1.0],
            "order_purchase_timestamp": [
                pd.Timestamp("2018-03-05 10:00:00"),  # Monday, March
                pd.Timestamp("2017-12-25 08:00:00"),  # Monday, December
            ],
            "is_late": [0, 1],
        }
    )
    dim_product = pd.DataFrame({
        "product_id": ["P1"], "product_weight_g": [500.0],
        "product_category_name_english": ["toys"],
    })
    dim_seller = pd.DataFrame({"seller_id": ["S1"], "seller_state": ["SP"]})
    dim_customer = pd.DataFrame({"customer_id": ["C1"], "customer_state": ["SP"]})
    payments = pd.DataFrame({"order_id": ["o1", "o2"], "payment_installments": [1, 3]})
    geo_centroids = pd.DataFrame({"zip_prefix": [], "lat": [], "lng": []})
    seller_zips = pd.DataFrame({"seller_id": ["S1"], "seller_zip_prefix": [1000]})
    customer_zips = pd.DataFrame({"customer_id": ["C1"], "customer_zip_prefix": [2000]})

    result = build_feature_table(
        fact, dim_product, dim_seller, dim_customer, payments,
        geo_centroids, seller_zips, customer_zips,
    )

    assert result.loc[0, "order_month"] == 3.0
    assert result.loc[0, "day_of_week"] == 0.0  # Monday == 0 (pandas dayofweek convention)
    assert result.loc[1, "order_month"] == 12.0
    assert result.loc[1, "day_of_week"] == 0.0


def test_n_items_in_order_counts_items_sharing_an_order_id() -> None:
    fact = pd.DataFrame(
        {
            "item_key": ["o1_1", "o1_2", "o1_3", "o2_1"],
            "order_id": ["o1", "o1", "o1", "o2"],
            "seller_id": ["S1", "S1", "S1", "S1"],
            "product_id": ["P1", "P1", "P1", "P1"],
            "customer_id": ["C1", "C1", "C1", "C1"],
            "price": [10.0, 10.0, 10.0, 10.0],
            "freight_value": [1.0, 1.0, 1.0, 1.0],
            "order_purchase_timestamp": [pd.Timestamp("2018-01-01")] * 4,
            "is_late": [0, 0, 0, 1],
        }
    )
    dim_product = pd.DataFrame({
        "product_id": ["P1"], "product_weight_g": [500.0],
        "product_category_name_english": ["toys"],
    })
    dim_seller = pd.DataFrame({"seller_id": ["S1"], "seller_state": ["SP"]})
    dim_customer = pd.DataFrame({"customer_id": ["C1"], "customer_state": ["SP"]})
    payments = pd.DataFrame({"order_id": ["o1", "o2"], "payment_installments": [2, 1]})
    geo_centroids = pd.DataFrame({"zip_prefix": [], "lat": [], "lng": []})
    seller_zips = pd.DataFrame({"seller_id": ["S1"], "seller_zip_prefix": [1000]})
    customer_zips = pd.DataFrame({"customer_id": ["C1"], "customer_zip_prefix": [2000]})

    result = build_feature_table(
        fact, dim_product, dim_seller, dim_customer, payments,
        geo_centroids, seller_zips, customer_zips,
    )
    by_key = result.set_index("item_key")

    assert by_key.loc["o1_1", "n_items_in_order"] == 3.0
    assert by_key.loc["o1_2", "n_items_in_order"] == 3.0
    assert by_key.loc["o1_3", "n_items_in_order"] == 3.0
    assert by_key.loc["o2_1", "n_items_in_order"] == 1.0


def test_load_payments_aggregates_multi_row_orders_via_max() -> None:
    try:
        verify_raw_files()
    except FileNotFoundError:
        pytest.skip("raw data not available in data/raw/")

    payments = _load_payments()
    assert set(payments.columns) == {"order_id", "payment_installments"}
    assert payments["order_id"].is_unique

    # Some orders have >1 raw payment row (split payments across methods);
    # confirm the aggregate is the MAX, not the first row or sum.
    raw = pd.read_csv(
        RAW_DATA_DIR / "olist_order_payments_dataset.csv",
        usecols=["order_id", "payment_installments"],
    )
    multi_row_order = raw.groupby("order_id").size().loc[lambda s: s > 1].index[0]
    expected_max = raw.loc[raw["order_id"] == multi_row_order, "payment_installments"].max()
    actual = payments.loc[payments["order_id"] == multi_row_order, "payment_installments"].iloc[0]
    assert actual == expected_max


def test_seller_historical_late_rate_basic_expanding_correctness() -> None:
    fact = pd.DataFrame(
        {
            "item_key": ["i1", "i2", "i3"],
            "order_id": ["o1", "o2", "o3"],
            "seller_id": ["S1", "S1", "S1"],
            "order_purchase_timestamp": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-02-01"),
                pd.Timestamp("2018-03-01"),
            ],
            "is_late": [1, 0, 1],
        }
    )
    result = compute_seller_historical_late_rate(fact).set_index("item_key")[
        "seller_historical_late_rate"
    ]
    assert pd.isna(result["i1"])  # no prior orders
    assert result["i2"] == pytest.approx(1.0)  # only prior order (o1) was late
    assert result["i3"] == pytest.approx(0.5)  # prior orders o1,o2 -> mean([1, 0])


def test_seller_historical_late_rate_isolated_per_seller() -> None:
    fact = pd.DataFrame(
        {
            "item_key": ["i1", "i2", "i3", "i4"],
            "order_id": ["o1", "o2", "o3", "o4"],
            "seller_id": ["S1", "S2", "S1", "S2"],
            "order_purchase_timestamp": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-01-02"),
                pd.Timestamp("2018-01-03"),
                pd.Timestamp("2018-01-04"),
            ],
            "is_late": [1, 0, 0, 1],
        }
    )
    result = compute_seller_historical_late_rate(fact).set_index("item_key")[
        "seller_historical_late_rate"
    ]
    assert pd.isna(result["i1"])  # S1's first order
    assert pd.isna(result["i2"])  # S2's first order
    assert result["i3"] == pytest.approx(1.0)  # S1's only prior order (o1) was late
    assert result["i4"] == pytest.approx(0.0)  # S2's only prior order (o2) was on-time


def test_seller_historical_late_rate_same_order_items_do_not_see_each_other() -> None:
    # o1 has two items from the same seller (same timestamp, same order-level
    # is_late by construction) -- neither may count as "prior" to the other.
    fact = pd.DataFrame(
        {
            "item_key": ["o1_1", "o1_2", "o2_1"],
            "order_id": ["o1", "o1", "o2"],
            "seller_id": ["S1", "S1", "S1"],
            "order_purchase_timestamp": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-02-01"),
            ],
            "is_late": [1, 1, 0],
        }
    )
    result = compute_seller_historical_late_rate(fact).set_index("item_key")[
        "seller_historical_late_rate"
    ]
    assert pd.isna(result["o1_1"])
    assert pd.isna(result["o1_2"])
    assert result["o2_1"] == pytest.approx(1.0)  # o1 (both items) counts as ONE prior order


def test_seller_historical_late_rate_simultaneous_different_orders_do_not_see_each_other() -> None:
    # Two DIFFERENT orders for the same seller at the exact same timestamp:
    # simultaneous, not strictly ordered relative to each other.
    same_ts = pd.Timestamp("2018-01-01")
    fact = pd.DataFrame(
        {
            "item_key": ["i1", "i2"],
            "order_id": ["o1", "o2"],
            "seller_id": ["S1", "S1"],
            "order_purchase_timestamp": [same_ts, same_ts],
            "is_late": [1, 0],
        }
    )
    result = compute_seller_historical_late_rate(fact).set_index("item_key")[
        "seller_historical_late_rate"
    ]
    assert pd.isna(result["i1"])
    assert pd.isna(result["i2"])


def test_seller_historical_late_rate_ignores_future_outcomes() -> None:
    """The dedicated no-leakage test: mutate the LATEST order's outcome and
    confirm every earlier order's computed feature value is bit-for-bit
    unchanged."""
    fact = pd.DataFrame(
        {
            "item_key": ["i1", "i2", "i3", "i4"],
            "order_id": ["o1", "o2", "o3", "o4"],
            "seller_id": ["S1", "S1", "S1", "S1"],
            "order_purchase_timestamp": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-02-01"),
                pd.Timestamp("2018-03-01"),
                pd.Timestamp("2018-04-01"),
            ],
            "is_late": [0, 1, 0, 0],
        }
    )
    before = compute_seller_historical_late_rate(fact).set_index("item_key")[
        "seller_historical_late_rate"
    ]

    mutated = fact.copy()
    mutated.loc[mutated["order_id"] == "o4", "is_late"] = 1  # flip the LATEST order's outcome
    after = compute_seller_historical_late_rate(mutated).set_index("item_key")[
        "seller_historical_late_rate"
    ]

    # i1, i2, i3 depend only on STRICTLY PRIOR orders, never on o4 (which
    # comes after all of them) -- must be bit-for-bit unchanged.
    for key in ("i1", "i2", "i3"):
        if pd.isna(before[key]):
            assert pd.isna(after[key])
        else:
            assert before[key] == after[key]

    assert before["i3"] == pytest.approx(0.5)  # sanity: mean of o1,o2 = mean([0, 1])
    # i4 depends on o1,o2,o3 (its own is_late is not part of its own feature)
    # so it too is unchanged by flipping o4's own outcome.
    assert before["i4"] == after["i4"] == pytest.approx(1 / 3)


def test_haversine_km_known_distances() -> None:
    zero = pd.Series([0.0])
    same_point = _haversine_km(zero, zero, zero, zero)
    assert same_point.iloc[0] == pytest.approx(0.0, abs=1e-6)

    # 1 degree of latitude is ~111.19 km along a meridian.
    one_degree_lat = _haversine_km(
        pd.Series([0.0]), pd.Series([0.0]), pd.Series([1.0]), pd.Series([0.0])
    )
    assert one_degree_lat.iloc[0] == pytest.approx(111.19, abs=0.5)

    # Sao Paulo (-23.5505, -46.6333) to Rio de Janeiro (-22.9068, -43.1729):
    # real great-circle distance is ~357 km.
    sp_to_rj = _haversine_km(
        pd.Series([-23.5505]), pd.Series([-46.6333]),
        pd.Series([-22.9068]), pd.Series([-43.1729]),
    )
    assert sp_to_rj.iloc[0] == pytest.approx(357, abs=5)


def test_aggregate_geo_centroids_uses_median_not_mean() -> None:
    # An outlier point should barely move the median but would visibly move
    # the mean -- this proves the aggregation is really MEDIAN, not MEAN.
    geo = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [1000, 1000, 1000, 1000, 1000],
            "geolocation_lat": [-23.50, -23.51, -23.49, -23.50, -50.00],  # last is an outlier
            "geolocation_lng": [-46.60, -46.61, -46.59, -46.60, -46.60],
        }
    )
    centroids = _aggregate_geo_centroids(geo).set_index("zip_prefix")
    assert centroids.loc[1000, "lat"] == pytest.approx(-23.50, abs=0.01)  # median, unmoved
    assert centroids.loc[1000, "lat"] != pytest.approx(geo["geolocation_lat"].mean())


def test_compute_geo_distance_flags_missing_zip_without_dropping_row() -> None:
    fact = pd.DataFrame(
        {"item_key": ["i1", "i2"], "seller_id": ["S1", "S2"], "customer_id": ["C1", "C1"]}
    )
    geo_centroids = pd.DataFrame(
        {"zip_prefix": [1000, 2000], "lat": [-23.5, -22.9], "lng": [-46.6, -43.2]}
    )
    seller_zips = pd.DataFrame(
        {"seller_id": ["S1", "S2"], "seller_zip_prefix": [1000, 9999]}  # S2's zip has no geo match
    )
    customer_zips = pd.DataFrame({"customer_id": ["C1"], "customer_zip_prefix": [2000]})

    result = compute_geo_distance(fact, geo_centroids, seller_zips, customer_zips).set_index(
        "item_key"
    )

    assert result.loc["i1", "geo_distance_missing"] == 0.0
    assert result.loc["i1", "geo_distance"] > 0.0
    # i2's row is present (not dropped), geo_distance is NaN, and the flag says so.
    assert "i2" in result.index
    assert result.loc["i2", "geo_distance_missing"] == 1.0
    assert pd.isna(result.loc["i2", "geo_distance"])


def test_split_and_encode_uses_train_only_frequencies(feature_table: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test, freq_map = _split_and_encode(feature_table)
    assert set(x_train.columns) == set(ALLOWED_FEATURES)
    assert set(x_test.columns) == set(ALLOWED_FEATURES)
    assert set(x_train.index).isdisjoint(set(x_test.index))
    assert x_train["product_category_freq"].between(0.0, 1.0).all()
    assert x_test["product_category_freq"].between(0.0, 1.0).all()
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)
    assert freq_map  # non-empty: fit on a non-trivial train fold


def test_no_forbidden_leakage_columns(feature_table: pd.DataFrame) -> None:
    assert _FORBIDDEN_COLUMNS.isdisjoint(set(ALLOWED_FEATURES))
    # is_late is the target and is expected in feature_table; every other
    # forbidden (date-derived) column must be absent from it.
    assert (_FORBIDDEN_COLUMNS - {"is_late"}).isdisjoint(feature_table.columns)


def test_train_test_disjoint(feature_table: pd.DataFrame) -> None:
    x_train, x_test, _, _, _ = _split_and_encode(feature_table)
    assert set(x_train.index).isdisjoint(set(x_test.index))


def test_late_risk_in_unit_interval(fitted_models: dict[str, Any]) -> None:
    x_test = fitted_models["x_test"]
    for key in ("lr", "dt", "rf"):
        late_risk = fitted_models[key].predict_proba(x_test)[:, 1]
        assert ((late_risk >= 0.0) & (late_risk <= 1.0)).all()


def test_auc_below_leak_guardrail(fitted_models: dict[str, Any]) -> None:
    x_test, y_test = fitted_models["x_test"], fitted_models["y_test"]
    for key in ("lr", "dt", "rf"):
        auc = _evaluate(fitted_models[key], x_test, y_test)["roc_auc"]
        assert auc < PREDICT_AUC_LEAK_GUARDRAIL


def test_seed_reproducibility(fitted_models: dict[str, Any]) -> None:
    x_train, x_test = fitted_models["x_train"], fitted_models["x_test"]
    y_train = fitted_models["y_train"]

    lr_pipeline_2, _ = _build_pipelines()
    lr_pipeline_2.fit(x_train, y_train)
    lr_first = fitted_models["lr"].predict_proba(x_test)[:, 1]
    lr_second = lr_pipeline_2.predict_proba(x_test)[:, 1]
    assert (lr_first == lr_second).all()

    # RF is the primary model, so its reproducibility matters most: one more
    # fit (reusing the fixture's fit as the first) rather than two fresh
    # fits, to avoid a third slow RF fit in this test file.
    rf_pipeline_2 = _build_rf_pipeline()
    rf_pipeline_2.fit(x_train, y_train)
    rf_first = fitted_models["rf"].predict_proba(x_test)[:, 1]
    rf_second = rf_pipeline_2.predict_proba(x_test)[:, 1]
    assert (rf_first == rf_second).all()


def test_random_forest_uses_allowed_features_exactly(fitted_models: dict[str, Any]) -> None:
    rf_pipeline = fitted_models["rf"]
    assert list(rf_pipeline.feature_names_in_) == list(ALLOWED_FEATURES)


def test_random_forest_outperforms_committed_models(fitted_models: dict[str, Any]) -> None:
    x_test, y_test = fitted_models["x_test"], fitted_models["y_test"]
    lr_auc = _evaluate(fitted_models["lr"], x_test, y_test)["roc_auc"]
    dt_auc = _evaluate(fitted_models["dt"], x_test, y_test)["roc_auc"]
    rf_auc = _evaluate(fitted_models["rf"], x_test, y_test)["roc_auc"]
    # Regression guard: if a future change erodes RF's real, confirmed
    # advantage (reports/diagnostic_rf_cv.txt), this test should fail.
    assert rf_auc > dt_auc > lr_auc
