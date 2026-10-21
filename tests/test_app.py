"""Tests for the Streamlit dashboard (`app/`): template, content, honesty, missing files.

Smoke and content tests run the real pages with `streamlit.testing.v1.AppTest`
against the real pipeline outputs (skipped if they have not been generated).
Missing-file tests never touch real files: they redirect the `src.config`
data directories to empty temp dirs, which the app resolves at call time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.io as pio
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import _theme as theme
from app._common import (
    DATA_CATALOG,
    quadrant_thresholds,
)
from app._components import money_short, short_id
from src import config
from src.ahp import compute_weights

APP_DIR = config.PROJECT_ROOT / "app"
PAGE_FILES = {
    "Home": APP_DIR / "Home.py",
    "Intelligence": APP_DIR / "pages" / "1_Intelligence.py",
    "Design": APP_DIR / "pages" / "2_Design.py",
    "Choice": APP_DIR / "pages" / "3_Choice.py",
    "Discovery": APP_DIR / "pages" / "4_Discovery.py",
    "Implementation": APP_DIR / "pages" / "5_Implementation.py",
    "DSS_Architecture": APP_DIR / "pages" / "6_DSS_Architecture.py",
}
PAGE_TITLES = {
    "Intelligence": "What is happening in the marketplace?",
    "Design": "How likely is an order to arrive late, and why?",
    "Choice": "Which established sellers are best overall?",
    "Discovery": "Which good sellers are we overlooking?",
    "Implementation": "What should we do about this seller?",
    "DSS_Architecture": "How does the system fit together?",
}
CONTENT_PAGES = list(PAGE_TITLES)
_TIMEOUT = 120


def _run(name: str) -> AppTest:
    return AppTest.from_file(str(PAGE_FILES[name]), default_timeout=_TIMEOUT).run()


def _markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _links_and_text(at: AppTest) -> str:
    """Markdown plus page-link labels (a link renders as markdown only without a registry)."""
    labels = [str(link.proto.label) for link in at.get("page_link")]
    return "\n".join([_markdown(at), *labels])


def _metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def _displayed_action(at: AppTest) -> str:
    for md in at.markdown:
        match = re.search(r'data-action="(\w+)"', md.value)
        if match:
            return match.group(1)
    raise AssertionError("no recommendation rendered")


def _live_matches_batch(at: AppTest) -> str:
    for md in at.markdown:
        if "Live result = batch output" in md.value:
            return md.value
    raise AssertionError("no live-vs-batch line rendered")


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture(scope="module")
def real_outputs() -> None:
    missing = [e.name for e in DATA_CATALOG if not e.path.exists()]
    if missing or not (config.RULES_DIR / "seller_rules.yaml").exists():
        pytest.skip(f"pipeline outputs not generated (missing: {missing}); run run_all.py")


# --------------------------------------------------------------------------
# Smoke and the shared page template
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PAGE_FILES))
def test_page_renders_without_exception(name: str, real_outputs: None) -> None:
    at = _run(name)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("name", CONTENT_PAGES)
def test_content_pages_follow_the_template(name: str, real_outputs: None) -> None:
    at = _run(name)
    text = _markdown(at)
    assert [t.value for t in at.title] == [PAGE_TITLES[name]]
    assert 'class="subtitle"' in text  # the grey live subtitle was filled
    assert len(at.metric) <= 4
    assert not at.tabs
    assert [e.label for e in at.expander] == ["Technical details"]
    assert re.search(
        r"Syllabus topic: .+ · Analytics type: \w+ · Management level: \w+", text
    )


def test_removed_chrome_is_gone(real_outputs: None) -> None:
    for name in PAGE_FILES:
        at = _run(name)
        assert not at.sidebar.radio and not at.sidebar.toggle
        assert not [e for e in at.sidebar.expander]
        text = _links_and_text(at)
        for gone in ("Reading mode", "Presentation mode", "Glossary", "How to read",
                     "The question this page answers", "ANSWER", "guided tour"):
            assert gone not in text, (name, gone)


def test_one_sans_serif_family_and_no_serif_anywhere() -> None:
    for path in [*APP_DIR.rglob("*.py"), *(config.PROJECT_ROOT / ".streamlit").glob("*")]:
        source = path.read_text(encoding="utf-8").lower()
        for banned in ("georgia", "palatino", "iowan", "times new roman", "font_serif"):
            assert banned not in source, (path.name, banned)
    assert "sans-serif" in theme.FONT
    toml = (config.PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'font = "sans serif"' in toml


def test_streamlit_config_is_dark_and_has_no_telemetry() -> None:
    text = (config.PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'base = "dark"' in text and 'backgroundColor = "#0C1015"' in text
    assert 'secondaryBackgroundColor = "#141A22"' in text
    assert "gatherUsageStats = false" in text
    assert 'toolbarMode = "minimal"' in text  # supported by the installed Streamlit


def test_no_white_boxes_in_app_code_or_chart_template() -> None:
    pattern = re.compile(r"#fff(?:fff)?|\"white\"|rgba\(\s*255,\s*255,\s*255,\s*0\.[3-9]",
                         re.IGNORECASE)
    for path in APP_DIR.rglob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name
    layout = pio.templates["olist_dark"].layout
    assert layout.paper_bgcolor == "rgba(0,0,0,0)" and layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert layout.legend.y < 0  # legend below the plot


# --------------------------------------------------------------------------
# Content, per page
# --------------------------------------------------------------------------


def test_home_is_title_description_page_list_and_synthetic_note(real_outputs: None) -> None:
    at = _run("Home")
    assert [t.value for t in at.title] == [
        "Which sellers should we back, and which are we overlooking?"
    ]
    text = _links_and_text(at)
    for page in ("Intelligence", "Design", "Choice", "Discovery", "Implementation",
                 "DSS Architecture"):
        assert page in text
    assert "synthetic" in text
    assert not at.metric and not at.expander and not at.get("plotly_chart")


def test_money_and_id_formatting_helpers() -> None:
    assert money_short(13_220_249) == "R$ 13.2M"
    assert money_short(842_300) == "R$ 842K"
    assert money_short(950) == "R$ 950"
    assert short_id("0123456789abcdef") == "01234567…"
    assert short_id("abc") == "abc"


def test_intelligence_kpis_match_fact_table_and_synthetic_is_tagged(real_outputs: None) -> None:
    at = _run("Intelligence")
    fact = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "FactOrderItems.parquet",
        columns=["order_id", "seller_id", "price", "is_late", "review_score"],
    )
    metrics = _metrics(at)
    assert metrics["Delivered orders"] == f"{fact['order_id'].nunique():,}"
    assert metrics["Item revenue"] == money_short(float(fact["price"].sum()))
    assert metrics["Late-delivery rate"] == f"{fact['is_late'].mean() * 100:.2f}%"
    # One review per (seller, order), the definition behind SellerFacts.avg_review; NOT the mean
    # over item rows, which counts a review once for every item of its order.
    per_order = fact.groupby(["seller_id", "order_id"])["review_score"].first().mean()
    per_item = fact["review_score"].mean()
    assert round(per_order, 2) != round(per_item, 2)
    assert metrics["Average review score"] == f"{per_order:.2f} / 5"
    review_kpi = next(m for m in at.metric if m.label == "Average review score")
    distinct = fact.drop_duplicates("order_id")["review_score"].mean()
    assert f"{per_item:.2f}" in review_kpi.help and f"{distinct:.2f}" in review_kpi.help
    assert "one review per seller-order pair" in review_kpi.help
    text = _markdown(at)
    assert text.count('class="tag">Synthetic data') >= 2  # funnel and call sections
    assert any("no audio and no speech recognition" in c.value for c in at.caption)
    assert any("Of every 100 sessions that start" in c.value for c in at.caption)
    assert "How customers sound on calls" in text


def test_intelligence_funnel_uses_thousands_separators_and_a_negative_share_chart(
    real_outputs: None,
) -> None:
    at = _run("Intelligence")
    charts = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    funnel = charts[0]["data"][0]
    assert funnel["type"] == "funnel" and "%{value:,}" in funnel["texttemplate"]
    assert funnel["x"] == [70000, 15000, 9000, 6000] or len(funnel["x"]) == 4
    calls = pd.read_parquet(config.PROCESSED_DATA_DIR / "CallSentiment.parquet")
    bars = charts[1]["data"][0]
    assert bars["type"] == "bar" and len(bars["x"]) == calls["agent_id"].nunique()


def test_design_shows_the_largest_sensitivity_effects(real_outputs: None) -> None:
    at = _run("Design")
    assert "What if an input changes?" in _markdown(at)
    sens = pd.read_csv(config.REPORTS_DIR / "sensitivity.csv")
    rf = sens[(sens["model"] == "random_forest")
              & ~sens["feature"].isin(["order_month", "day_of_week"])]
    strongest = rf.loc[rf["mean_late_risk_change"].abs().idxmax()]
    bars = json.loads(at.get("plotly_chart")[1].proto.spec)["data"][0]
    assert bars["type"] == "bar" and len(bars["y"]) == 5
    assert not any("Month of purchase" in label or "Day of the week" in label
                   for label in bars["y"])  # calendar nudges are not realistic scenarios
    assert max(abs(v) for v in _decode(bars["x"])) == pytest.approx(
        abs(strongest["mean_late_risk_change"])
    )
    assert any("largest effects" in c.value for c in at.caption)


def _decode(values: object) -> list[float]:
    """Plotly may ship numpy arrays as {dtype, bdata}; return plain floats either way."""
    if isinstance(values, dict):
        import base64

        import numpy as np

        raw = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        return [float(v) for v in raw]
    return [float(v) for v in values]  # type: ignore[attr-defined]


def test_design_kpis_come_from_the_metrics_file(real_outputs: None) -> None:
    at = _run("Design")
    metrics = json.loads((config.REPORTS_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    rf = metrics["random_forest"]
    shown = _metrics(at)
    assert shown["Late-delivery rate"] == f"{rf['base_rate'] * 100:.2f}%"
    assert shown["Late-order ranking score"] == f"{rf['roc_auc'] * 100:.0f}%"
    assert "Ranking score" not in shown
    assert shown["Inputs used"] == str(len(config.ALLOWED_FEATURES))
    text = _markdown(at)
    assert "The random forest is the primary model" in text
    assert "accuracy" not in text.lower()
    helps = [m.help for m in at.metric if m.label == "Late-order ranking score"]
    assert helps and "coin flip" in helps[0]
    assert "randomly chosen late order is ranked above a randomly chosen on-time order" in helps[0]
    assert "not accuracy and not the probability" in helps[0]
    # Technical details still hold the comparison, and the sensitivity selector works.
    at.selectbox(key="design_sensitivity_model").select("logistic_regression").run()
    assert not at.exception


def test_choice_default_consistency_and_live_matrix_editing(real_outputs: None) -> None:
    at = _run("Choice")
    assert len(at.number_input) == 6
    _, _, _, configured_cr = compute_weights(config.AHP_PAIRWISE_MATRIX)
    assert _metrics(at)["Consistency ratio"] == f"{configured_cr:.3f}"
    assert any("The judgments are consistent" in c.value for c in at.caption)

    for index, value in enumerate([9.0, 1 / 9, 9.0, 1 / 9, 9.0, 1 / 9]):
        at.number_input[index].set_value(value)
    at.run()
    assert not at.exception
    assert any("The judgments contradict each other" in c.value for c in at.caption)

    at.button[0].click().run()  # Reset to the configured matrix
    assert _metrics(at)["Consistency ratio"] == f"{configured_cr:.3f}"
    assert any("The judgments are consistent" in c.value for c in at.caption)


def test_choice_top_table_uses_short_ids_and_keeps_the_discovery_pointer(
    real_outputs: None,
) -> None:
    at = _run("Choice")
    assert f"{config.AHP_MIN_ORDERS}+ items sold" in _markdown(at)
    assert "Discovery" in _markdown(at)
    top = next(df.value for df in at.dataframe if "Rank" in df.value.columns)
    assert len(top) == 10 and top["Seller"].str.len().max() <= 9
    assert f"AHP score ({config.AHP_MIN_ORDERS}+ items pool)" in top.columns


def test_discovery_caution_counts_line_table_and_download(real_outputs: None) -> None:
    at = _run("Discovery")
    n = config.DISCOVERY_CONFIDENCE_MIN_ORDERS
    assert any(
        c.value == (
            "This page surfaces sellers with strong quality signals but few items sold. "
            f"Sellers with fewer than {n} items sold are flagged low-confidence; a few good "
            "reviews on a handful of items sold is promising, not proof."
        )
        for c in at.caption
    )
    text = _markdown(at)
    assert "not food or nutrition" in text

    quadrants = pd.read_csv(config.REPORTS_DIR / "discovery_quadrants.csv")
    assert not at.metric  # the four counts are one line of text, not cards
    for label, count in quadrants["quadrant"].value_counts().items():
        assert re.search(rf"<b>{label}</b></span> {count:,}", text), label

    tables = [df.value for df in at.dataframe if "Average review (per order)" in df.value.columns
              and "Group" not in df.value.columns]
    assert len(tables) == 2  # Stronger evidence, then Early signal
    actionable, watchlist = tables
    for table in tables:
        assert list(table.columns) == [
            "Seller", "Items sold", f"Quality score ({config.DISCOVERY_MIN_ORDERS}+ items pool)",
            "Average review (per order)", "Confidence",
        ]
        assert table["Seller"].str.len().max() <= 9
    assert any("ranked by quality score" in c.value for c in at.caption)
    gems = quadrants[quadrants["quadrant"] == "Hidden Gem"]
    n_actionable = int((gems["confidence"] == "normal").sum())
    n_watch = int((gems["confidence"] == "low").sum())
    top = min(10, n_actionable), min(10, n_watch)
    assert f"Stronger evidence ({n}+ items sold): top {top[0]:,} of {n_actionable:,}" in text
    assert f"Early signal (under {n} items sold): top {top[1]:,} of {n_watch:,}" in text
    assert "the full list is in the download below and under Technical details" in text
    assert (actionable["Confidence"] == "normal").all() and (actionable["Items sold"] >= n).all()
    assert len(actionable) == min(10, n_actionable)
    assert watchlist["Confidence"].str.startswith("low").all()
    assert (watchlist["Items sold"] < n).all()
    assert "by definition, hidden gems have under" in text.lower()
    assert "items sold, so most picks" in text
    assert "short rather than padded" not in text  # 44 Actionable gems: no padding message
    assert at.get("download_button")
    assert len(at.get("plotly_chart")) == 1


def test_discovery_scatter_marks_low_confidence_with_hollow_markers(real_outputs: None) -> None:
    at = _run("Discovery")
    figure = json.loads(at.get("plotly_chart")[0].proto.spec)
    symbols = {t["marker"].get("symbol") for t in figure["data"] if "marker" in t}
    assert {"circle", "circle-open"} <= symbols
    assert figure["layout"]["legend"]["y"] < 0  # legend below the chart


def test_quadrant_thresholds_are_consistent_with_stored_labels(real_outputs: None) -> None:
    quadrants = pd.read_csv(config.REPORTS_DIR / "discovery_quadrants.csv")
    quality, popularity = quadrant_thresholds(quadrants)
    high_quality = quadrants["quadrant"].isin(["Star", "Hidden Gem"])
    high_popularity = quadrants["quadrant"].isin(["Star", "Overrated"])
    assert (quadrants.loc[high_quality, "ahp_score"] >= quality).all()
    assert (quadrants.loc[~high_quality, "ahp_score"] < quality).all()
    assert (quadrants.loc[high_popularity, "order_volume"] >= popularity).all()
    assert (quadrants.loc[~high_popularity, "order_volume"] < popularity).all()


def test_implementation_default_seller_is_not_the_default_rule(real_outputs: None) -> None:
    recommendations = pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv")
    at = _run("Implementation")
    row = recommendations.set_index("seller_id").loc[at.session_state["impl_seller"]]
    assert row["rule_id"] != "R12"
    assert _displayed_action(at) == row["action"]
    assert len(at.metric) == 4  # items sold, delivered orders, average review, late rate
    text = _markdown(at)
    assert "Why:" not in text  # the subtitle and the ticked conditions carry the reason
    assert "- ✓" in text or "- ✕" in text


def test_implementation_example_dropdown_replaces_the_buttons(real_outputs: None) -> None:
    recommendations = pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv")
    fact = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "FactOrderItems.parquet",
        columns=["order_id", "seller_id", "review_score"],
    )
    reviewed = (
        fact.groupby(["seller_id", "order_id"])["review_score"].first().groupby("seller_id").count()
    )
    at = _run("Implementation")
    assert not at.button
    for action in ("Warn", "Suspend", "Feature", "Promote", "Keep"):
        at.selectbox(key="impl_example").select(action).run()
        assert not at.exception
        pool = recommendations[recommendations["action"] == action]
        enough_reviews = pool["seller_id"].map(reviewed).fillna(0) >= config.DISCOVERY_MIN_ORDERS
        strong = pool[(pool["rule_id"] != "R12") & enough_reviews]
        # The example is the first seller (file order) whose rule fired for a real reason and who
        # has enough REVIEWED orders; only if there is none does it fall back to the first of the
        # action.
        expected = (strong if not strong.empty else pool)["seller_id"].iloc[0]
        assert at.session_state["impl_seller"] == expected
        assert _displayed_action(at) == action
        if not strong.empty:
            chosen = recommendations.set_index("seller_id").loc[at.session_state["impl_seller"]]
            assert chosen["rule_id"] != "R12"
            assert reviewed[at.session_state["impl_seller"]] >= config.DISCOVERY_MIN_ORDERS


def test_implementation_live_engine_matches_batch_for_many_sellers(real_outputs: None) -> None:
    recommendations = pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv")
    facts = pd.read_parquet(config.PROCESSED_DATA_DIR / "SellerFacts.parquet")

    sampled = recommendations["seller_id"].sample(15, random_state=config.SEED).tolist()
    sparse = facts.loc[facts["ahp_score"].isna(), "seller_id"].sample(3, random_state=1).tolist()
    one_per_action = recommendations.groupby("action")["seller_id"].first().tolist()
    sellers = list(dict.fromkeys(one_per_action + sparse + sampled))
    assert len(sellers) >= 20

    at = _run("Implementation")
    by_id = recommendations.set_index("seller_id")
    seen_actions = set()
    for seller_id in sellers:
        at.selectbox(key="impl_seller").select(seller_id).run()
        assert not at.exception, (seller_id, [e.value for e in at.exception])
        expected = by_id.loc[seller_id]
        assert _displayed_action(at) == expected["action"]
        assert f"({expected['action']}, rule {expected['rule_id']})" in _live_matches_batch(at)
        seen_actions.add(expected["action"])
    assert seen_actions == {"Keep", "Warn", "Suspend", "Feature", "Promote"}


def test_implementation_flags_low_confidence_hidden_gem(real_outputs: None) -> None:
    quadrants = pd.read_csv(config.REPORTS_DIR / "discovery_quadrants.csv")
    gem = quadrants[(quadrants["quadrant"] == "Hidden Gem") & (quadrants["confidence"] == "low")]
    at = _run("Implementation")
    at.selectbox(key="impl_seller").select(gem["seller_id"].iloc[0]).run()
    assert "Low confidence" in _markdown(at)


def test_implementation_keeps_near_miss_inside_technical_details(real_outputs: None) -> None:
    at = _run("Implementation")
    text = _markdown(at)
    assert "What would change this recommendation?" in text  # inside the expander
    assert "would need to be" in text


def test_dss_catalog_row_counts_are_read_live(real_outputs: None) -> None:
    at = _run("DSS_Architecture")
    catalog = next(t.value for t in at.table if "Rows" in t.value.columns)
    assert len(catalog) == len(DATA_CATALOG)
    for table, row in catalog.iterrows():
        entry = next(e for e in DATA_CATALOG if e.name == table)
        assert int(str(row["Rows"]).replace(",", "")) == entry.row_count()
    synthetic = set(catalog.index[catalog["Synthetic"] == "Synthetic data"])
    assert {"Clickstream", "CallTranscripts", "CallSentiment"} <= synthetic
    text = _markdown(at)
    assert text.count('class="flow-box"') == 4
    assert 'class="flow-note"' in text and "SQL Server" in text
    assert "height: 16rem" in text  # all four boxes share one fixed height


# --------------------------------------------------------------------------
# Honesty, theme and offline guarantees
# --------------------------------------------------------------------------


def test_app_never_derives_lift_or_gain_from_in_sample_predictions(real_outputs: None) -> None:
    for path in (APP_DIR / "pages").glob("*.py"):
        assert 'load_parquet("LatePredictions' not in path.read_text(encoding="utf-8")
    for name in CONTENT_PAGES:
        text = _markdown(_run(name)).lower()
        for banned in ("catches", "lift", "cumulative gain", "captures"):
            assert banned not in text, (name, banned)


def _lab(hex_color: str) -> tuple[float, float, float]:
    """sRGB hex -> CIE L*a*b* (D65), for a normal-vision colour-difference check."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    x = 0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = 0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x / 0.95047), f(y), f(z / 1.08883)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def test_theme_colours_meet_contrast_requirements_on_the_dark_surfaces() -> None:
    for group in (theme.QUADRANTS, theme.ACTIONS):
        for name, swatch in group.items():
            for surface in (theme.PAGE, theme.SURFACE):
                assert theme.contrast_ratio(swatch.color, surface) >= 3.0, (name, "graphic")
                assert theme.contrast_ratio(swatch.text, surface) >= 4.5, (name, "text")
    for ink in (theme.TEXT, theme.TEXT_MUTED, theme.ACCENT):
        for surface in (theme.PAGE, theme.SURFACE):
            assert theme.contrast_ratio(ink, surface) >= 4.5, ink
    for step in theme.FUNNEL_RAMP:  # dark funnel labels on the gold steps
        assert theme.contrast_ratio(theme.FUNNEL_TEXT, step) >= 4.5, step


def test_quadrant_colours_are_distinguishable_from_each_other() -> None:
    import itertools
    import math

    colours = {name: _lab(s.color) for name, s in theme.QUADRANTS.items()}
    for (a, ca), (b, cb) in itertools.combinations(colours.items(), 2):
        assert math.dist(ca, cb) >= 25, (a, b)  # CIE76; blue/rose and green/orange included


def test_semantic_colours_exist_only_for_actions_and_quadrants() -> None:
    assert set(theme.ACTIONS) == {"Suspend", "Warn", "Keep", "Feature", "Promote"}
    assert set(theme.QUADRANTS) == {"Star", "Hidden Gem", "Overrated", "Overlooked-Low-Quality"}
    assert len({s.color for s in theme.QUADRANTS.values()}) == 4


def test_app_loads_no_remote_resources() -> None:
    sources = [*APP_DIR.rglob("*.py"), *(config.PROJECT_ROOT / ".streamlit").glob("*")]
    pattern = re.compile(r"https?://|@import|<link\b|<script\b", re.IGNORECASE)
    for path in sources:
        assert not pattern.search(path.read_text(encoding="utf-8")), path
    # The only static file is the tab icon (a local PNG); Streamlit's static serving stays off.
    assert sorted(p.name for p in (APP_DIR / "static").glob("*")) == ["icon.png"]
    streamlit_config = config.PROJECT_ROOT / ".streamlit" / "config.toml"
    assert "enableStaticServing" not in streamlit_config.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Missing files: a plain message with the header and footer still shown
# --------------------------------------------------------------------------

_DIRS = {
    "processed": "PROCESSED_DATA_DIR",
    "reports": "REPORTS_DIR",
    "rules": "RULES_DIR",
}


def _break_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *which: str) -> None:
    for key in which:
        monkeypatch.setattr(config, _DIRS[key], tmp_path / f"empty_{key}")


@pytest.mark.parametrize(
    ("name", "broken"),
    [
        ("Intelligence", ("processed",)),
        ("Design", ("reports",)),
        ("Choice", ("reports",)),
        ("Discovery", ("reports",)),
        ("Implementation", ("processed",)),
        ("Implementation", ("reports",)),
        ("Implementation", ("rules",)),
        ("DSS_Architecture", ("rules",)),
    ],
)
def test_page_shows_friendly_message_when_outputs_missing(
    name: str, broken: tuple[str, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _break_dirs(monkeypatch, tmp_path, *broken)
    at = _run(name)
    assert not at.exception, [e.value for e in at.exception]
    text = _markdown(at)
    assert "A required pipeline output is missing" in text and "run_all.py" in text
    assert [t.value for t in at.title] == [PAGE_TITLES[name]]  # the header still renders
    assert "Syllabus topic:" in text  # and so does the footer


def test_dss_architecture_marks_missing_catalog_files_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _break_dirs(monkeypatch, tmp_path, "processed", "reports")
    at = _run("DSS_Architecture")
    assert not at.exception
    text = _markdown(at)
    assert f"{len(DATA_CATALOG)} of {len(DATA_CATALOG)} catalog files are missing" in text
    assert "run_all.py" in text
    catalog = next(t.value for t in at.table if "Rows" in t.value.columns)
    assert (catalog["Rows"] == "—").all()
    assert catalog["Status"].str.contains("missing").all()


# --------------------------------------------------------------------------
# Wording: "items sold" for seller-level counts, "delivered orders" for distinct orders
# --------------------------------------------------------------------------


def test_seller_level_counts_are_labelled_items_sold_not_orders(real_outputs: None) -> None:
    for name in ("Choice", "Discovery", "Implementation"):
        at = _run(name)
        labels = [m.label for m in at.metric]
        columns = [c for df in at.dataframe for c in df.value.columns]
        assert "Orders" not in labels and "Orders" not in columns, name
        assert "Minimum orders" not in labels, name
    assert "Items sold" in _metrics(_run("Implementation"))
    assert "Minimum items sold" in _metrics(_run("Choice"))
    assert "Delivered orders" in _metrics(_run("Intelligence"))  # distinct orders keep "orders"


def test_technical_details_explain_items_sold_versus_delivered_orders(
    real_outputs: None,
) -> None:
    note = "Items sold counts order lines"
    for name in ("Intelligence", "Choice", "Discovery", "Implementation"):
        assert note in _markdown(_run(name)), name


def test_choice_full_table_shows_raw_reviews_and_keeps_the_shrunk_columns(
    real_outputs: None,
) -> None:
    at = _run("Choice")
    full = next(df.value for df in at.dataframe if "Average review (shrunk)" in df.value.columns)
    wanted = {
        "Average review (per order)", "On-time rate (per order)", "Average review (shrunk)",
        "On-time rate (shrunk)", "Reviewed orders", "Delivered orders",
    }
    assert wanted <= set(full.columns)
    text = _markdown(at)
    assert "pulled toward the marketplace mean" in text
    # The sentence names exactly the column groups the table has (raw, orders, shrunk, shares).
    assert "headed shrunk" in text and "contributions" in text
    assert (full["Average review (per order)"].between(1, 5)).all()
    # Shrinkage never moves a value past the marketplace mean, so it stays between them.
    assert (full["Average review (shrunk)"] - full["Average review (per order)"]).abs().max() < 1.5
    assert (full["Reviewed orders"] <= full["Delivered orders"]).all()
    assert "scales every criterion" in text  # the Choice-versus-Discovery pool explanation
