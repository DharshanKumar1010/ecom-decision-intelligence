"""Tests for the second polish round: thin-evidence headline, Discovery headings, heading levels,
layout and font rules, link fallback, Choice wording, Makefile parity and the reason texts.

Page tests run the real pages under `streamlit.testing.v1.AppTest` against the real pipeline
outputs (skipped when they have not been generated).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
import yaml
from streamlit.testing.v1 import AppTest

from app import _components
from app._common import DATA_CATALOG, hidden_gems_download
from src import config, explain
from src.ahp import build_seller_criteria
from src.expert import KnowledgeBase, Rule

APP_DIR = config.PROJECT_ROOT / "app"
PAGES = {
    "Home": APP_DIR / "Home.py",
    "Intelligence": APP_DIR / "pages" / "1_Intelligence.py",
    "Design": APP_DIR / "pages" / "2_Design.py",
    "Choice": APP_DIR / "pages" / "3_Choice.py",
    "Discovery": APP_DIR / "pages" / "4_Discovery.py",
    "Implementation": APP_DIR / "pages" / "5_Implementation.py",
    "DSS Architecture": APP_DIR / "pages" / "6_DSS_Architecture.py",
}
THIN_SELLER = "010da0602d7774602cd1b3f5fb7b709e"
RULES_PATH = config.RULES_DIR / "seller_rules.yaml"


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture(scope="module")
def real_outputs() -> None:
    missing = [e.name for e in DATA_CATALOG if not e.path.exists()]
    if missing or not RULES_PATH.exists():
        pytest.skip(f"pipeline outputs not generated (missing: {missing}); run run_all.py")


def _run(name: str) -> AppTest:
    return AppTest.from_file(str(PAGES[name]), default_timeout=180).run()


def _markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _headline(at: AppTest) -> str:
    for md in at.markdown:
        match = re.search(r'class="action-word[^"]*" data-action="\w+"[^>]*>([^<]+)<', md.value)
        if match:
            return match.group(1)
    raise AssertionError("no headline rendered")


# --------------------------------------------------------------------------
# F1: evidence_status reads the gates from the rules, never from literals
# --------------------------------------------------------------------------


def _yaml_gates() -> dict[str, float]:
    """Independent reading of the YAML: the lowest `>=` gate on each evidence fact in a Warn or
    Suspend rule."""
    gates: dict[str, float] = {}
    for rule in yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")):
        if rule["then"] not in ("Warn", "Suspend"):
            continue
        for cond in rule["if"]:
            if cond["op"] == ">=" and cond["fact"] in ("order_volume", "n_orders", "n_reviewed"):
                gates[cond["fact"]] = min(float(cond["value"]), gates.get(cond["fact"], 1e18))
    return gates


def test_the_helper_reads_the_minimums_from_the_rule_file() -> None:
    rules = KnowledgeBase.from_yaml(RULES_PATH).rules
    minimums = explain.warn_suspend_minimums(rules)
    assert minimums == _yaml_gates()
    assert set(minimums) == {"order_volume", "n_orders", "n_reviewed"}


def _rule(rule_id: str, action: str, **gates: float) -> Rule:
    from src.expert import Condition

    conditions = tuple(Condition(fact=f, op=">=", value=v) for f, v in gates.items())
    return Rule(id=rule_id, priority=70, conditions=conditions, action=action, because="x")


def test_changing_a_gate_in_the_rules_changes_the_result() -> None:
    seller = {"order_volume": 12, "n_orders": 4, "n_reviewed": 4}
    loose = [_rule("RA", "Warn", order_volume=5, n_orders=3, n_reviewed=3)]
    strict = [_rule("RA", "Warn", order_volume=5, n_orders=6, n_reviewed=3)]
    assert not explain.evidence_status(seller, loose).below
    tightened = explain.evidence_status(seller, strict)
    assert tightened.below
    assert [(g.fact, g.have, g.minimum) for g in tightened.gaps] == [("n_orders", 4.0, 6.0)]
    # a gate on a Keep or Promote rule is not a minimum for warning or suspending
    keep_only = [_rule("RK", "Keep", n_orders=50)]
    assert explain.evidence_status(seller, keep_only).minimums == {}


def test_missing_counts_count_as_zero_and_the_sentence_gives_have_and_need() -> None:
    rules = [_rule("RA", "Warn", order_volume=5, n_orders=3, n_reviewed=3)]
    status = explain.evidence_status({"order_volume": 5, "n_orders": None}, rules)
    assert status.have == {"order_volume": 5.0, "n_orders": 0.0, "n_reviewed": 0.0}
    assert {g.fact for g in status.gaps} == {"n_orders", "n_reviewed"}
    sentence = explain.evidence_sentence(status)
    assert "5 items sold, 0 delivered orders, 0 reviewed orders" in sentence
    assert "at least 5 items sold, 3 delivered orders, 3 reviewed orders" in sentence
    one = explain.evidence_status({"order_volume": 1, "n_orders": 1, "n_reviewed": 1}, rules)
    assert "1 item sold, 1 delivered order, 1 reviewed order" in explain.evidence_sentence(one)


def test_the_thin_keep_headline_is_the_agreed_wording() -> None:
    assert explain.THIN_KEEP_HEADLINE == "Keep for now: too few orders to warn or suspend"


def test_a_thin_evidence_seller_gets_the_new_headline_and_never_no_change(
    real_outputs: None,
) -> None:
    at = _run("Implementation")
    at.selectbox(key="impl_seller").select(THIN_SELLER).run()
    assert not at.exception, [e.value for e in at.exception]
    assert _headline(at) == explain.THIN_KEEP_HEADLINE
    text = _markdown(at)
    assert "No change is needed for now" not in text
    facts = pd.read_parquet(config.PROCESSED_DATA_DIR / "SellerFacts.parquet").set_index(
        "seller_id"
    )
    row = facts.loc[THIN_SELLER]
    status = explain.evidence_status(
        {
            "order_volume": row["order_volume"],
            "n_orders": row["n_orders"],
            "n_reviewed": row["n_reviewed"],
        },
        KnowledgeBase.from_yaml(RULES_PATH).rules,
    )
    assert explain.evidence_sentence(status) in text
    assert "Thin evidence:" in text
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Delivered orders"] == str(int(row["n_orders"]))
    assert list(metrics) == [
        "Items sold",
        "Delivered orders",
        "Average review",
        "Late-delivery rate",
    ]


def test_sellers_with_enough_history_keep_the_no_change_wording(real_outputs: None) -> None:
    facts = pd.read_parquet(config.PROCESSED_DATA_DIR / "SellerFacts.parquet")
    recs = pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv")
    minimums = _yaml_gates()
    enough = facts[
        (facts["order_volume"] >= minimums["order_volume"])
        & (facts["n_orders"] >= minimums["n_orders"])
        & (facts["n_reviewed"] >= minimums["n_reviewed"])
    ]
    keep = recs[(recs["action"] == "Keep") & recs["seller_id"].isin(set(enough["seller_id"]))]
    assert len(keep) > 100
    at = _run("Implementation")
    for seller in (keep["seller_id"].iloc[0], keep["seller_id"].iloc[-1]):
        at.selectbox(key="impl_seller").select(seller).run()
        assert _headline(at) == "Keep"
        assert "No change is needed for now" in _markdown(at)
        assert "Thin evidence:" not in _markdown(at)


def test_a_non_keep_outcome_is_never_given_the_thin_headline(real_outputs: None) -> None:
    recs = pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv")
    at = _run("Implementation")
    for action in ("Warn", "Suspend", "Feature", "Promote"):
        seller = recs[recs["action"] == action]["seller_id"].iloc[0]
        at.selectbox(key="impl_seller").select(seller).run()
        assert _headline(at) == action


# --------------------------------------------------------------------------
# F2: Discovery headings, caption, one tier column
# --------------------------------------------------------------------------


def test_discovery_headings_say_top_n_of_the_total_from_the_files(real_outputs: None) -> None:
    quad = pd.read_csv(config.REPORTS_DIR / "discovery_quadrants.csv")
    gems = quad[quad["quadrant"] == "Hidden Gem"]
    normal, low = (
        int((gems["confidence"] == "normal").sum()),
        int((gems["confidence"] == "low").sum()),
    )
    n = config.DISCOVERY_CONFIDENCE_MIN_ORDERS
    text = _markdown(_run("Discovery"))
    assert f"Stronger evidence ({n}+ items sold): top {min(10, normal)} of {normal}" in text
    assert f"Early signal (under {n} items sold): top {min(10, low)} of {low}" in text
    assert "the full list is in the download below and under Technical details" in text


def test_discovery_technical_table_keeps_one_tier_column_and_the_download_is_unchanged(
    real_outputs: None,
) -> None:
    at = _run("Discovery")
    tables = [df.value for df in at.dataframe if "Evidence" in df.value.columns]
    assert tables, "the top-gems table with the tier column is missing"
    for table in tables:
        assert "Confidence" not in table.columns
        assert set(table["Evidence"]) <= {"Stronger evidence", "Early signal"}
    quad = pd.read_csv(config.REPORTS_DIR / "discovery_quadrants.csv")
    from src.discovery import top_hidden_gems

    csv = hidden_gems_download(top_hidden_gems(quad, n=len(quad), tiered=True)).decode("utf-8")
    header = csv.splitlines()[0].split(",")
    assert header == ["seller_id", "ahp_score", "order_volume", "quadrant", "confidence", "tier"]
    assert "Actionable" not in csv and "Watchlist" not in csv


# --------------------------------------------------------------------------
# F3 to F7: Design formatter, heading levels, tab icon, links, layout, fonts
# --------------------------------------------------------------------------


def test_design_and_intelligence_show_the_late_rate_with_the_same_formatter(
    real_outputs: None,
) -> None:
    design = {m.label: m.value for m in _run("Design").metric}
    intelligence = {m.label: m.value for m in _run("Intelligence").metric}
    assert design["Late-delivery rate"] == "%.2f%%" % (
        pd.read_parquet(config.PROCESSED_DATA_DIR / "FactOrderItems.parquet", columns=["is_late"])[
            "is_late"
        ].mean()
        * 100
    )
    assert re.fullmatch(r"\d+\.\d\d%", design["Late-delivery rate"])
    assert design["Late-delivery rate"] == intelligence["Late-delivery rate"]


@pytest.mark.parametrize("name", list(PAGES))
def test_heading_levels_start_at_one_and_never_skip(name: str, real_outputs: None) -> None:
    at = _run(name)
    levels = [1] * len(at.title)
    levels += [len(m.value.split(" ")[0]) for m in at.markdown if re.match(r"^#{1,6} ", m.value)]
    assert at.title, "each page has exactly one h1 title"
    assert len(at.title) == 1
    shown = sorted(set(levels))
    assert shown[0] == 1 and all(b - a <= 1 for a, b in zip(shown, shown[1:], strict=False)), shown
    assert not any(m.value.startswith("### ") for m in at.markdown), "section headings are h2"


def test_hover_link_icons_on_headings_are_hidden_by_css() -> None:
    assert '[data-testid="stHeaderActionElements"] {{ display: none; }}' in Path(
        _components.__file__
    ).read_text(encoding="utf-8")


def test_the_tab_icon_is_a_small_local_png_used_by_every_page(real_outputs: None) -> None:
    icon = _components.PAGE_ICON
    assert icon.exists() and icon.parent == APP_DIR / "static"
    header = icon.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    assert 16 <= width <= 128 and width == height
    assert "page_icon=str(PAGE_ICON)" in Path(_components.__file__).read_text(encoding="utf-8")
    assert not re.search(r"https?://", Path(_components.__file__).read_text(encoding="utf-8"))
    # no emoji in any heading text of the pages
    emoji = re.compile("[\U0001f300-\U0001faff☀-➿]")
    for name in PAGES:
        at = _run(name)
        headings = [t.value for t in at.title] + [
            m.value for m in at.markdown if re.match(r"^#{1,6} ", m.value)
        ]
        assert not any(emoji.search(h) for h in headings), name


def test_home_lists_every_page_as_a_page_link(real_outputs: None) -> None:
    at = _run("Home")
    links = [(link.proto.label, link.proto.page) for link in at.get("page_link")]
    assert links == [
        ("Intelligence", "Intelligence"),
        ("Design", "Design"),
        ("Choice", "Choice"),
        ("Discovery", "Discovery"),
        ("Implementation", "Implementation"),
        ("DSS Architecture", "DSS_Architecture"),
    ]


def test_without_a_page_registry_the_link_falls_back_to_a_plain_markdown_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.errors import StreamlitPageNotFoundError

    def no_registry(*args: object, **kwargs: object) -> None:
        raise StreamlitPageNotFoundError(page="pages/6_DSS_Architecture.py")

    shown: list[str] = []
    monkeypatch.setattr(st, "page_link", no_registry)
    monkeypatch.setattr(st, "markdown", lambda body, **kw: shown.append(body))
    _components.page_link_or_label("pages/6_DSS_Architecture.py", "DSS Architecture")
    _components.page_link_or_label("pages/3_Choice.py", "Choice")
    assert shown == ["[DSS Architecture](DSS_Architecture)", "[Choice](Choice)"]


def test_page_links_are_styled_so_they_look_like_links() -> None:
    css = Path(_components.__file__).read_text(encoding="utf-8")
    assert 'stPageLink-NavLink"] p' in css and "text-decoration: underline" in css


def _css_source() -> str:
    return _components._CSS


def test_no_css_min_width_above_600_pixels_and_no_wide_fixed_widths() -> None:
    sources = [_css_source(), *(p.read_text(encoding="utf-8") for p in APP_DIR.rglob("*.py"))]
    for text in sources:
        for match in re.finditer(r"min-width\s*:\s*(\d+(?:\.\d+)?)\s*(px|rem|em)", text):
            value = float(match.group(1)) * (16 if match.group(2) != "px" else 1)
            assert value <= 600, match.group(0)
        for match in re.finditer(r"(?<![-\w])width\s*:\s*(\d+)\s*px", text):
            assert int(match.group(1)) <= 400, match.group(0)
    assert "min-height: 16rem" in _css_source() and "height: 16rem" not in _css_source().replace(
        "min-height: 16rem", ""
    )


def test_the_smallest_font_is_at_least_13_pixels() -> None:
    sizes = [
        float(m.group(1)) * 16 for m in re.finditer(r"font-size:\s*([\d.]+)rem", _css_source())
    ]
    assert sizes and min(sizes) >= 13, sorted(sizes)[:3]
    from app import _theme

    template = _theme.pio.templates["olist_dark"].layout
    assert template.xaxis.tickfont.size >= 13 and template.legend.font.size >= 13
    assert template.font.size >= 13


# --------------------------------------------------------------------------
# V3: the shrunk values are the same on Choice and Discovery
# --------------------------------------------------------------------------


def test_shrunk_review_and_on_time_values_are_the_same_in_both_pools(real_outputs: None) -> None:
    fact = pd.read_parquet(config.PROCESSED_DATA_DIR / "FactOrderItems.parquet")
    dim = pd.read_parquet(config.PROCESSED_DATA_DIR / "DimSeller.parquet")
    choice_pool = build_seller_criteria(fact, dim, min_orders=config.AHP_MIN_ORDERS)
    discovery_pool = build_seller_criteria(fact, dim, min_orders=config.DISCOVERY_MIN_ORDERS)
    both = choice_pool.merge(discovery_pool, on="seller_id", suffixes=("_30", "_5"))
    assert len(both) == len(choice_pool) and len(discovery_pool) > len(choice_pool)
    for column in ("avg_review_score", "on_time_rate", "avg_review_raw", "on_time_rate_raw"):
        diff = (both[f"{column}_30"] - both[f"{column}_5"]).abs().max()
        assert diff == 0, f"{column} differs between the two pools by up to {diff}"
    ranking = pd.read_csv(config.REPORTS_DIR / "ahp_ranking.csv").set_index("seller_id")
    saved = ranking.loc[both["seller_id"], ["avg_review_score", "on_time_rate"]]
    for column in ("avg_review_score", "on_time_rate"):
        assert abs(saved[column].to_numpy() - both[f"{column}_5"].to_numpy()).max() < 1e-9


def test_choice_states_the_shared_shrunk_values_in_its_technical_details() -> None:
    source = (APP_DIR / "pages" / "3_Choice.py").read_text(encoding="utf-8")
    flat = re.sub(r'"\s*\n\s*"', "", source)  # join the adjacent string literals
    assert "The shrunk review and on-time " in flat
    assert "values themselves are the same on both" in flat


# --------------------------------------------------------------------------
# V2: every numeric claim in a because text is backed by the data (re-derived here)
# --------------------------------------------------------------------------


def _because(rule_id: str) -> str:
    return next(
        r["because"]
        for r in yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
        if r["id"] == rule_id
    )


def test_reason_texts_make_only_claims_the_data_supports(real_outputs: None) -> None:
    facts = pd.read_parquet(config.PROCESSED_DATA_DIR / "SellerFacts.parquet")
    quad = pd.read_csv(config.REPORTS_DIR / "discovery_quadrants.csv")
    rules = {
        r["id"]: {c["fact"]: c["value"] for c in r["if"]}
        for r in yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    }
    # R06: "top 10% of the sellers that have a quality score"
    scored = facts["ahp_score"].dropna()
    share = float((scored >= rules["R06"]["ahp_score"]).mean())
    assert 0.09 < share < 0.11, share
    assert "established" not in _because("R06") and "established" not in _because("R11")
    # R11: "bottom decile of the sellers that have one", "75th percentile of all sellers"
    assert 0.09 < float((scored < rules["R11"]["ahp_score"]).mean()) < 0.11
    assert 0.73 < float((facts["late_rate"] <= rules["R11"]["late_rate"]).mean()) < 0.78
    # R08: "below 0.3, which holds for about half of the hidden gems"
    gems = quad[quad["quadrant"] == "Hidden Gem"].merge(facts[["seller_id", "avg_late_risk"]])
    assert 0.45 < float((gems["avg_late_risk"] < rules["R08"]["avg_late_risk"]).mean()) < 0.55
    assert "CLAUDE.md" not in _because("R08")
    assert f"{rules['R08']['avg_late_risk']:g}" in _because("R08")
    # R10: "median of all sellers" for risk and for the review
    assert abs(float((facts["avg_late_risk"] <= rules["R10"]["avg_late_risk"]).mean()) - 0.5) < 0.01
    assert (
        abs(float((facts["avg_review"].dropna() < rules["R10"]["avg_review"]).mean()) - 0.5) < 0.01
    )
    # R05: "only the worst 6% of sellers with enough history"
    gate = facts[
        (facts["order_volume"] >= rules["R05"]["order_volume"])
        & (facts["n_orders"] >= rules["R05"]["n_orders"])
    ]
    worst = float((gate["late_rate"] > rules["R05"]["late_rate"]).mean())
    assert round(worst * 100) == int(re.search(r"worst (\d+)%", _because("R05")).group(1)), worst
    # R07: "90th percentile" of items sold across all sellers, "a little above the median" review
    assert abs(float((facts["order_volume"] < rules["R07"]["order_volume"]).mean()) - 0.9) < 0.01
    median = float(facts["avg_review"].median())
    assert median < rules["R07"]["avg_review"] < median + 0.1
    # no stale "recalibrated" remark in any text
    for rule_id in rules:
        assert "recalibrated" not in _because(rule_id), rule_id


# --------------------------------------------------------------------------
# F10: the Makefile mirrors tasks.ps1
# --------------------------------------------------------------------------


def test_every_makefile_target_mirrors_a_tasks_ps1_target_and_back() -> None:
    root = config.PROJECT_ROOT
    make = (root / "Makefile").read_text(encoding="utf-8")
    ps1 = (root / "tasks.ps1").read_text(encoding="utf-8")
    make_targets = set(re.findall(r"^([a-z][a-z-]*):", make, flags=re.M))
    declared = set(re.search(r"\.PHONY:\s*(.+)", make).group(1).split())
    ps_targets = set(
        re.search(r"ValidateSet\(([^)]*)\)", ps1)
        .group(1)
        .replace('"', "")
        .replace(" ", "")
        .split(",")
    )
    assert make_targets == declared == ps_targets, make_targets ^ ps_targets
    assert {"verify", "rebuild"} <= make_targets
    verify = make.split("\nverify:", 1)[1].split("\n\n", 1)[0]
    for step in (
        "ruff check",
        "mypy src",
        "pytest -q",
        "src.load_sql --verify-only",
        "make_results.py --check",
        "app_smoke.py",
    ):
        assert step in verify, step
    rebuild = make.split("\nrebuild:", 1)[1]
    assert "run_all.py --export" in rebuild and "src.load_sql" in rebuild and "verify" in rebuild
