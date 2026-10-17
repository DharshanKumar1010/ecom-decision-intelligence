"""Tests for the page polish: number formats, friendly labels, tier names, shared colours.

Format and label helpers are tested directly; page behaviour runs the real pages under
`streamlit.testing.v1.AppTest` against the real pipeline outputs (skipped when they have not been
generated), like `tests/test_app.py`.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import _theme as theme
from app._common import (
    DATA_CATALOG,
    hidden_gems_download,
    schema_table_count,
    temporal_results,
)
from app._format import clean, fmt_int, fmt_num, fmt_pct
from app._labels import COLUMN_LABELS, TIER_LABELS, friendly, tier_label, with_tier_labels
from src import config, explain

APP_DIR = config.PROJECT_ROOT / "app"
PAGES = {
    "Intelligence": APP_DIR / "pages" / "1_Intelligence.py",
    "Design": APP_DIR / "pages" / "2_Design.py",
    "Choice": APP_DIR / "pages" / "3_Choice.py",
    "Discovery": APP_DIR / "pages" / "4_Discovery.py",
    "Implementation": APP_DIR / "pages" / "5_Implementation.py",
    "DSS": APP_DIR / "pages" / "6_DSS_Architecture.py",
}
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")


def _run(name: str) -> AppTest:
    return AppTest.from_file(str(PAGES[name]), default_timeout=120).run()


def _markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture(scope="module")
def real_outputs() -> None:
    missing = [e.name for e in DATA_CATALOG if not e.path.exists()]
    needed = config.REPORTS_DIR / "diagnostics" / "diagnostic_temporal.txt"
    if missing or not needed.exists() or not (config.RULES_DIR / "seller_rules.yaml").exists():
        pytest.skip(f"pipeline outputs not generated (missing: {missing}); run run_all.py")


# --------------------------------------------------------------------------
# Number formats
# --------------------------------------------------------------------------


def test_fmt_int_uses_thousands_separators() -> None:
    assert fmt_int(110189) == "110,189"
    assert fmt_int(96470.0) == "96,470"
    assert fmt_int(0) == "0"


def test_fmt_num_has_fixed_decimals_and_never_a_negative_zero() -> None:
    assert fmt_num(0.74096, 4) == "0.7410"
    assert fmt_num(1234.5, 2) == "1,234.50"
    assert fmt_num(-0.0001, 3) == "0.000"  # not "-0.000"
    assert fmt_num(-0.0, 2) == "0.00"
    assert fmt_num(-0.0004, 3, signed=True) == "+0.000"
    assert fmt_num(0.0366, 3, signed=True) == "+0.037"
    assert fmt_num(-0.0366, 3, signed=True) == "-0.037"
    assert fmt_num(None) == "–" and fmt_num(float("nan")) == "–"


def test_fmt_pct_takes_a_share_and_keeps_the_requested_decimals() -> None:
    assert fmt_pct(0.0791, 1) == "7.9%"
    assert fmt_pct(0.079082, 2) == "7.91%"
    assert fmt_pct(-0.00001, 2) == "0.00%"
    assert fmt_pct(None) == "–"


def test_clean_turns_negative_zero_into_zero() -> None:
    assert math.copysign(1.0, clean(-0.0001, 3)) == 1.0
    assert clean(-0.0001, 3) == 0.0
    assert clean(2.34567, 2) == 2.35


# --------------------------------------------------------------------------
# Labels and tier names
# --------------------------------------------------------------------------


def test_friendly_renames_known_columns_and_leaves_the_rest() -> None:
    frame = pd.DataFrame(columns=["order_volume", "ahp_score", "avg_review", "mystery"])
    assert list(friendly(frame).columns) == [
        "Items sold",
        "Quality score",
        "Average review (per order)",
        "mystery",
    ]
    renamed = friendly(frame, {"ahp_score": "AHP score (30+ items pool)"})
    assert "AHP score (30+ items pool)" in renamed.columns


def test_no_friendly_label_is_itself_a_raw_column_name() -> None:
    assert not [label for label in COLUMN_LABELS.values() if _SNAKE.match(label)]


def test_tier_names_shown_to_people_are_stronger_evidence_and_early_signal() -> None:
    assert TIER_LABELS == {"Actionable": "Stronger evidence", "Watchlist": "Early signal"}
    assert tier_label("Actionable") == "Stronger evidence" and tier_label("other") == "other"
    shown = with_tier_labels(pd.DataFrame({"tier": ["Actionable", "Watchlist"], "x": [1, 2]}))
    assert shown["tier"].tolist() == ["Stronger evidence", "Early signal"]
    assert with_tier_labels(pd.DataFrame({"x": [1]})).columns.tolist() == ["x"]


def test_hidden_gems_download_uses_tier_names_and_pipeline_column_names() -> None:
    gems = pd.DataFrame(
        {
            "seller_id": ["a" * 32, "b" * 32],
            "ahp_score": [0.7, 0.6],
            "order_volume": [20, 6],
            "tier": ["Actionable", "Watchlist"],
        }
    )
    text = hidden_gems_download(gems).decode("utf-8")
    assert "Actionable" not in text and "Watchlist" not in text
    frame = pd.read_csv(io.StringIO(text))
    assert frame["tier"].tolist() == ["Stronger evidence", "Early signal"]
    assert list(frame.columns) == ["seller_id", "ahp_score", "order_volume", "tier"]


# --------------------------------------------------------------------------
# Small file readers used by the pages
# --------------------------------------------------------------------------


def test_schema_table_count_matches_the_create_table_statements() -> None:
    text = (config.PROJECT_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    assert schema_table_count() == len(re.findall(r"^CREATE TABLE ", text, flags=re.MULTILINE))
    assert schema_table_count() == 10


def test_temporal_results_come_from_the_diagnostic_report() -> None:
    path = config.REPORTS_DIR / "diagnostics" / "diagnostic_temporal.txt"
    if not path.exists():
        pytest.skip("temporal diagnostic report not present")
    results = temporal_results()
    text = path.read_text(encoding="utf-8")
    assert set(results) == {"all", "no_calendar"}
    assert f"{results['all']['rf']:.4f}" in text and f"{results['no_calendar']['lr']:.4f}" in text
    assert all(0 < v < 1 for group in results.values() for v in group.values())


def test_fact_label_is_plain_words() -> None:
    assert explain.fact_label("order_volume") == "number of items sold"
    assert explain.fact_label("something_new") == "Something new"


# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------


def _lab(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    lin = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    ]
    x = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def test_device_colours_are_visible_and_not_any_meaning_colour() -> None:
    devices = list(theme.DEVICE_COLORS)
    assert len(devices) == 3
    for colour in devices:
        for surface in (theme.PAGE, theme.SURFACE):
            assert theme.contrast_ratio(colour, surface) >= 3.0, colour
    meaning = (
        [s.color for s in theme.QUADRANTS.values()]
        + [s.color for s in theme.ACTIONS.values()]
        + [theme.ACCENT]
    )
    assert not {c.lower() for c in devices} & {m.lower() for m in meaning}
    for a, b in itertools.combinations(devices, 2):
        assert math.dist(_lab(a), _lab(b)) >= 25, (a, b)
    for colour in devices:
        assert min(math.dist(_lab(colour), _lab(m)) for m in meaning) >= 12, colour


def test_risk_direction_pair_reuses_existing_colours() -> None:
    assert theme.RISK_RAISES == theme.ACTIONS["Suspend"].color
    assert theme.RISK_LOWERS == theme.QUADRANTS["Star"].color
    assert (
        theme.RISK_SCALE[0][1] == theme.RISK_LOWERS and theme.RISK_SCALE[-1][1] == theme.RISK_RAISES
    )


def test_weights_chart_and_heatmap_use_the_same_risk_direction_colours(real_outputs: None) -> None:
    at = _run("Design")
    assert not at.exception
    figures = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    weights = next(
        f
        for f in figures
        if {t.get("name") for t in f["data"]} >= {"Raises late risk", "Lowers late risk"}
    )
    colours = {t["name"]: t["marker"]["color"] for t in weights["data"]}
    assert colours["Raises late risk"] == theme.RISK_RAISES
    assert colours["Lowers late risk"] == theme.RISK_LOWERS
    heat = next(f for f in figures if f["data"][0]["type"] == "heatmap")
    scale = heat["data"][0]["colorscale"]
    assert scale[0][1] == theme.RISK_LOWERS and scale[-1][1] == theme.RISK_RAISES
    assert "red = higher risk" in json.dumps(heat["data"][0]["colorbar"])
    source = (APP_DIR / "pages" / "2_Design.py").read_text(encoding="utf-8")
    assert source.count("theme.RISK_RAISES") >= 2 and "theme.RISK_SCALE" in source


def test_heatmap_text_never_shows_a_negative_zero(real_outputs: None) -> None:
    at = _run("Design")
    figures = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    heat = next(f for f in figures if f["data"][0]["type"] == "heatmap")
    cells = [c for row in heat["data"][0]["text"] for c in row]
    assert cells and "-0.000" not in cells and all(re.fullmatch(r"[+-]\d\.\d{3}", c) for c in cells)


def test_device_chart_uses_the_device_palette(real_outputs: None) -> None:
    at = _run("Intelligence")
    figures = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    bars = [t for f in figures for t in f["data"] if t.get("type") == "bar" and "marker" in t]
    device_colours = {
        t["marker"]["color"]
        for t in bars
        if isinstance(t["marker"].get("color"), str)
        and t["marker"]["color"].lower() in {c.lower() for c in theme.DEVICE_COLORS}
    }
    assert device_colours == {c.lower() for c in theme.DEVICE_COLORS}


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def _all_headers(at: AppTest) -> list[tuple[str, str]]:
    headers = [("dataframe", str(c)) for df in at.dataframe for c in df.value.columns]
    headers += [("table", str(c)) for t in at.table for c in t.value.columns]
    headers += [("table-index", str(t.value.index.name)) for t in at.table]
    return headers


@pytest.mark.parametrize("page", list(PAGES))
def test_technical_details_tables_have_no_raw_snake_case_headers(
    page: str, real_outputs: None
) -> None:
    at = _run(page)
    assert not at.exception
    raw = [(kind, header) for kind, header in _all_headers(at) if _SNAKE.match(header)]
    assert raw == []


def test_discovery_shows_tier_names_and_never_the_internal_ones(real_outputs: None) -> None:
    at = _run("Discovery")
    text = _markdown(at) + "\n".join(c.value for c in at.caption)
    assert "Stronger evidence" in text and "Early signal" in text
    cells = [str(v) for df in at.dataframe for col in df.value.columns for v in df.value[col]]
    everything = text + "\n".join(cells) + "\n".join(str(c) for _, c in _all_headers(at))
    assert "Actionable" not in everything and "Watchlist" not in everything
    tiers = {
        v for df in at.dataframe if "Evidence" in df.value.columns for v in df.value["Evidence"]
    }
    assert tiers == {"Stronger evidence", "Early signal"}


def test_intelligence_technical_details_carry_the_synthetic_tag(real_outputs: None) -> None:
    at = _run("Intelligence")
    text = _markdown(at)
    tag = 'class="tag">Synthetic data'
    funnel_heading = next(m.value for m in at.markdown if "Funnel table" in m.value)
    bounce = next(m.value for m in at.markdown if "Bounce rate" in m.value)
    assert tag in funnel_heading and tag in bounce
    assert text.count(tag) >= 5  # two sections, the funnel table, the bounce line, devices, agents
    headings = [m.value for m in at.markdown if m.value.startswith("###")]
    assert all(tag in h for h in headings if "Funnel" in h or "Call sentiment by agent" in h)
    funnel = next(df.value for df in at.dataframe if "From the first step" in df.value.columns)
    assert {"Viewed a page", "Added to cart", "Reached checkout", "Purchased"} <= set(
        funnel["Step"]
    )
    assert any("Bubble size" in c.value and "colour" in c.value for c in at.caption)


def test_design_states_split_threshold_and_forward_looking_result_from_files(
    real_outputs: None,
) -> None:
    at = _run("Design")
    text = _markdown(at)
    assert "Stratified" not in text
    assert "no order appears in both training and test" in text
    assert "fold 0 of 5" in text and "default decision threshold" in text and "0.5 or more" in text
    results = temporal_results()
    best = max(results["all"].values())
    assert f"the best ROC-AUC was {best:.2f}" in text and "not a forecast" in text
    importance = pd.read_csv(config.REPORTS_DIR / "rf_feature_importance.csv")
    top = importance.sort_values("importance_mean").iloc[-1]["feature"]
    assert f"{explain.feature_label(top)} is the strongest input" in text
    assert "stable across the 5 order-grouped folds" in text
    assert "does not cover the forward-looking test" in text
    assert "Class 1 = late, class 0 = on time." in text
    confusion = [df.value for df in at.dataframe if "Predicted late" in df.value.columns]
    assert len(confusion) == 1 and len(confusion[0]) == 6  # 3 models x actual on time / late
    assert list(confusion[0].columns) == ["Model", "Actual", "Predicted on time", "Predicted late"]
    importance_table = next(
        df.value
        for df in at.dataframe
        if "Input" in df.value.columns and "Drop in AUC when scrambled (mean)" in df.value.columns
    )
    assert "Drop in AUC when scrambled (std)" in importance_table.columns


def test_implementation_default_seller_has_five_reviewed_orders_and_a_fired_rule(
    real_outputs: None,
) -> None:
    at = _run("Implementation")
    seller = at.session_state["impl_seller"]
    recs = pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv").set_index("seller_id")
    fact = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "FactOrderItems.parquet",
        columns=["order_id", "seller_id", "review_score"],
    )
    mine = fact[fact["seller_id"] == seller].groupby("order_id")["review_score"].first()
    assert int(mine.notna().sum()) >= config.DISCOVERY_MIN_ORDERS
    assert recs.loc[seller, "rule_id"] != "R12"
    text = _markdown(at)
    assert "Reviewed orders:" in text and "Delivered orders:" in text and "Items sold:" in text


def test_implementation_shows_the_recommendation_sentence_once(real_outputs: None) -> None:
    at = _run("Implementation")
    text = _markdown(at)
    action = recs_action = (
        pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv")
        .set_index("seller_id")
        .loc[at.session_state["impl_seller"], "action"]
    )
    meaning = explain.ACTION_MEANING[action]
    assert text.count(meaning) == 1
    assert f"{recs_action}: {meaning}" not in text
    assert not re.search(r"(\b[A-Z][a-z]+): \1 ", text)


def test_implementation_explanation_is_wrapped_clean_and_the_search_has_a_placeholder(
    real_outputs: None,
) -> None:
    at = _run("Implementation")
    block = next(c for c in at.code if "Recommendation:" in c.value)
    assert block.proto.wrap_lines is True
    assert (
        max(len(line) for line in block.value.splitlines() if not line.startswith("Because")) < 80
    )
    assert not re.search(r"\d\.0\b|\d{6,}", block.value.split("Matched on:")[1])
    assert at.text_input[0].placeholder.startswith("Paste a full")


def test_dss_architecture_lists_the_loader_the_schema_and_the_explanation_layer(
    real_outputs: None,
) -> None:
    at = _run("DSS")
    modules = next(t.value for t in at.table if "Modules" in t.value.columns)
    joined = " ".join(modules["Modules"])
    for path in ("src/load_sql.py", "sql/schema.sql", "src/explain.py"):
        assert path in joined, path
        assert (config.PROJECT_ROOT / path).exists()
    assert all(v == "Yes" for v in modules["All present"])
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Data files"] == str(len(DATA_CATALOG))
    assert metrics["Loaded into SQL Server"] == str(schema_table_count())
    assert (
        f"{len(DATA_CATALOG)} data files ({schema_table_count()} of them loaded into SQL Server)"
        in (_markdown(at))
    )
    assert "data tables" not in _markdown(at)


def test_design_threshold_text_matches_how_the_metrics_are_computed() -> None:
    source = (config.PROJECT_ROOT / "src" / "predict.py").read_text(encoding="utf-8")
    body = source.split("def _evaluate(", 1)[1].split("def _sensitivity_analysis", 1)[0]
    assert "y_pred = pipeline.predict(x_test)" in body  # library default: probability 0.5
    assert "threshold" not in body.lower()


def test_funnel_percentages_show_two_decimals_and_counts_have_separators(
    real_outputs: None,
) -> None:
    source = (APP_DIR / "_components.py").read_text(encoding="utf-8")
    assert 'format="percent"' in source and 'format="localized"' in source
    path = Path(PAGES["Intelligence"])
    assert "fmt_pct(bounce, 2)" in path.read_text(encoding="utf-8")


def test_intelligence_average_review_kpi_equals_results_md(real_outputs: None) -> None:
    results = (config.REPORTS_DIR / "RESULTS.md").read_text(encoding="utf-8")
    match = re.search(r"Average review per reviewed order: \*\*(\d\.\d\d)\*\*", results)
    assert match, "RESULTS.md has no marketplace average review"
    kpi = {m.label: m.value for m in _run("Intelligence").metric}["Average review score"]
    assert kpi == f"{match.group(1)} / 5"
