"""Edge cases for the dashboard: thin or missing seller facts, missing single files, empty tiers.

Also compares the live rule engine the Implementation page builds (`app._common.working_memory`)
with the stored batch recommendation for EVERY seller. Runs the real pages under
`streamlit.testing.v1.AppTest` against the real pipeline outputs (skipped when they have not been
generated); file-level failures are simulated on temporary copies, never on the real files.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app._common import DATA_CATALOG, working_memory
from src import config
from src.expert import InferenceEngine, KnowledgeBase, _facts_from_row

APP_DIR = config.PROJECT_ROOT / "app"
PAGE_FILES = {
    "Intelligence": APP_DIR / "pages" / "1_Intelligence.py",
    "Design": APP_DIR / "pages" / "2_Design.py",
    "Choice": APP_DIR / "pages" / "3_Choice.py",
    "Discovery": APP_DIR / "pages" / "4_Discovery.py",
    "Implementation": APP_DIR / "pages" / "5_Implementation.py",
    "DSS Architecture": APP_DIR / "pages" / "6_DSS_Architecture.py",
}
_TIMEOUT = 180
_FACTS = config.PROCESSED_DATA_DIR / "SellerFacts.parquet"
_RECS = config.REPORTS_DIR / "seller_recommendations.csv"
_RULES = config.RULES_DIR / "seller_rules.yaml"


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture(scope="module")
def real_outputs() -> None:
    missing = [e.name for e in DATA_CATALOG if not e.path.exists()]
    if missing or not _RULES.exists():
        pytest.skip(f"pipeline outputs not generated (missing: {missing}); run run_all.py")


def _run(name: str) -> AppTest:
    return AppTest.from_file(str(PAGE_FILES[name]), default_timeout=_TIMEOUT).run()


def _text(at: AppTest) -> str:
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def _action_shown(at: AppTest) -> str:
    import re

    for md in at.markdown:
        match = re.search(r'data-action="(\w+)"', md.value)
        if match:
            return match.group(1)
    raise AssertionError("no recommendation rendered")


# --------------------------------------------------------------------------
# The live engine receives every fact the batch engine uses, for every seller
# --------------------------------------------------------------------------


def test_live_engine_facts_and_outcome_equal_the_batch_for_all_sellers(real_outputs: None) -> None:
    facts = pd.read_parquet(_FACTS)
    recs = pd.read_csv(_RECS).set_index("seller_id")
    engine = InferenceEngine(KnowledgeBase.from_yaml(_RULES))
    assert len(facts) == len(recs) and len(facts) > 2000
    wrong = []
    for _, row in facts.iterrows():
        live_facts = working_memory(str(row["seller_id"]), row).as_facts()
        batch_facts = _facts_from_row(row)
        assert set(live_facts) == set(batch_facts), (
            "a fact the batch engine uses is not passed live"
        )
        assert live_facts == batch_facts, row["seller_id"]
        fired = engine.run(live_facts).fired_rule
        expected = recs.loc[row["seller_id"]]
        if (fired.action, fired.id) != (expected["action"], expected["rule_id"]):
            wrong.append((row["seller_id"], fired.id, expected["rule_id"]))
    assert not wrong, f"{len(wrong)} sellers differ between the live and batch engine: {wrong[:5]}"


def test_a_seller_with_every_fact_missing_falls_to_the_default_rule() -> None:
    row = pd.Series(
        {
            "ahp_score": np.nan,
            "avg_late_risk": np.nan,
            "avg_review": np.nan,
            "order_volume": np.nan,
            "late_rate": np.nan,
            "is_hidden_gem": False,
            "n_orders": np.nan,
            "n_reviewed": np.nan,
        }
    )
    memory = working_memory("nobody", row)
    assert all(v is None for k, v in memory.as_facts().items() if k != "is_hidden_gem")
    fired = InferenceEngine(KnowledgeBase.from_yaml(_RULES)).run(memory.as_facts()).fired_rule
    assert (fired.id, fired.action) == ("R12", "Keep")


# --------------------------------------------------------------------------
# Implementation page: thin sellers, first and last, every example
# --------------------------------------------------------------------------


def test_sellers_with_no_reviewed_order_render_without_error(real_outputs: None) -> None:
    facts = pd.read_parquet(_FACTS)
    recs = pd.read_csv(_RECS).set_index("seller_id")
    none_reviewed = facts[facts["n_reviewed"] == 0]["seller_id"].tolist()
    assert len(none_reviewed) >= 1
    at = _run("Implementation")
    for seller_id in none_reviewed:
        at.selectbox(key="impl_seller").select(seller_id).run()
        assert not at.exception, (seller_id, [e.value for e in at.exception])
        assert _action_shown(at) == recs.loc[seller_id, "action"]
        assert {m.label: m.value for m in at.metric}["Average review"] == "n/a"


def test_one_item_seller_is_keep_by_default_and_says_so(real_outputs: None) -> None:
    facts = pd.read_parquet(_FACTS)
    recs = pd.read_csv(_RECS).set_index("seller_id")
    one = facts[facts["order_volume"] == 1]["seller_id"].tolist()
    assert one
    at = _run("Implementation")
    at.selectbox(key="impl_seller").select(one[0]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert recs.loc[one[0], "action"] == "Keep" and _action_shown(at) == "Keep"
    assert {m.label: m.value for m in at.metric}["Items sold"] == "1"


def test_first_and_last_seller_of_the_selector_render(real_outputs: None) -> None:
    facts = pd.read_parquet(_FACTS)
    ids = facts["seller_id"].tolist()
    at = _run("Implementation")
    options = list(at.selectbox(key="impl_seller").options)
    assert len(options) == len(ids)
    for seller_id in (ids[0], ids[-1]):
        at.selectbox(key="impl_seller").select(seller_id).run()
        assert not at.exception, (seller_id, [e.value for e in at.exception])


def test_the_longest_seller_id_can_be_found_with_the_search_box(real_outputs: None) -> None:
    dim = pd.read_parquet(config.PROCESSED_DATA_DIR / "DimSeller.parquet")
    facts = pd.read_parquet(_FACTS)
    longest = max(facts["seller_id"], key=len)
    assert longest in set(dim["seller_id"])
    at = _run("Implementation")
    at.text_input(key="impl_lookup").input(longest).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["impl_seller"] == longest


def test_every_try_an_example_outcome_lands_on_that_outcome(real_outputs: None) -> None:
    at = _run("Implementation")
    options = [
        o for o in at.selectbox(key="impl_example").options if not str(o).startswith("Choose")
    ]
    assert set(options) == {"Keep", "Warn", "Suspend", "Feature", "Promote"}
    for option in options:
        at.selectbox(key="impl_example").select(option).run()
        assert not at.exception, (option, [e.value for e in at.exception])
        assert _action_shown(at) == option


# --------------------------------------------------------------------------
# Missing single files and an empty Discovery tier (temporary copies)
# --------------------------------------------------------------------------

_DIR_ATTR = {"processed": "PROCESSED_DATA_DIR", "reports": "REPORTS_DIR", "rules": "RULES_DIR"}


def _copy_without(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, area: str, missing: str) -> None:
    """Point `config` at a temporary copy of one data area that lacks exactly one file."""
    target = tmp_path / area
    shutil.copytree(getattr(config, _DIR_ATTR[area]), target)
    if missing == "diagnostics":
        (target / "diagnostics" / "diagnostic_temporal.txt").unlink()
    else:
        (target / missing).unlink()
    monkeypatch.setattr(config, _DIR_ATTR[area], target)


_MISSING_ONE = [
    ("Intelligence", "processed", "FactOrderItems.parquet"),
    ("Intelligence", "processed", "Clickstream.parquet"),
    ("Intelligence", "processed", "CallSentiment.parquet"),
    ("Design", "reports", "model_metrics.json"),
    ("Design", "reports", "oof_metrics.json"),
    ("Design", "reports", "sensitivity.csv"),
    ("Design", "reports", "decision_tree.txt"),
    ("Design", "reports", "rf_feature_importance.csv"),
    ("Choice", "reports", "ahp_ranking.csv"),
    ("Choice", "reports", "ahp_prior_strength.csv"),
    ("Discovery", "reports", "discovery_quadrants.csv"),
    ("Implementation", "processed", "SellerFacts.parquet"),
    ("Implementation", "reports", "seller_recommendations.csv"),
    ("Implementation", "reports", "discovery_quadrants.csv"),
    ("Implementation", "rules", "seller_rules.yaml"),
    ("DSS Architecture", "rules", "seller_rules.yaml"),
]


@pytest.mark.parametrize(("page", "area", "missing"), _MISSING_ONE)
def test_a_single_missing_file_never_shows_a_traceback(
    page: str,
    area: str,
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    real_outputs: None,
) -> None:
    _copy_without(monkeypatch, tmp_path, area, missing)
    at = _run(page)
    assert not at.exception, (page, missing, [e.value for e in at.exception])
    assert at.title, "the page header should still render"


def test_the_temporal_report_missing_does_not_break_the_design_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_outputs: None
) -> None:
    _copy_without(monkeypatch, tmp_path, "reports", "diagnostics")
    at = _run("Design")
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("empty_tier", ["normal", "low"])
def test_discovery_renders_when_one_evidence_tier_is_empty(
    empty_tier: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_outputs: None
) -> None:
    target = tmp_path / "reports"
    shutil.copytree(config.REPORTS_DIR, target)
    quad = pd.read_csv(target / "discovery_quadrants.csv")
    other = "low" if empty_tier == "normal" else "normal"
    quad.loc[quad["quadrant"] == "Hidden Gem", "confidence"] = other  # the chosen tier has no gem
    quad.to_csv(target / "discovery_quadrants.csv", index=False)
    monkeypatch.setattr(config, "REPORTS_DIR", target)
    at = _run("Discovery")
    assert not at.exception, [e.value for e in at.exception]


def test_implementation_renders_a_seller_with_every_fact_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_outputs: None
) -> None:
    processed, reports = tmp_path / "processed", tmp_path / "reports"
    shutil.copytree(config.PROCESSED_DATA_DIR, processed)
    shutil.copytree(config.REPORTS_DIR, reports)
    facts = pd.read_parquet(processed / "SellerFacts.parquet")
    blank = facts.iloc[[0]].copy()
    blank["seller_id"] = "0" * 32
    for column in (
        "ahp_score",
        "avg_late_risk",
        "avg_review",
        "order_volume",
        "late_rate",
        "n_orders",
        "n_reviewed",
    ):
        blank[column] = np.nan
    blank["is_hidden_gem"] = False
    pd.concat([facts, blank], ignore_index=True).to_parquet(processed / "SellerFacts.parquet")
    recs = pd.read_csv(reports / "seller_recommendations.csv")
    row = {"seller_id": "0" * 32, "action": "Keep", "rule_id": "R12", "because": "default"}
    pd.concat([recs, pd.DataFrame([row])], ignore_index=True).to_csv(
        reports / "seller_recommendations.csv", index=False
    )
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    at = _run("Implementation")
    at.selectbox(key="impl_seller").select("0" * 32).run()
    assert not at.exception, [e.value for e in at.exception]
    assert _action_shown(at) == "Keep"
    assert at.session_state["impl_seller"] == "0" * 32
    labels = [str(o) for o in at.selectbox(key="impl_seller").options]
    assert any(label.endswith("items sold unknown") for label in labels)
    assert any(label.endswith("1 item sold") for label in labels)


def test_whole_number_format_has_a_dash_for_missing_values() -> None:
    from app._format import fmt_int

    assert fmt_int(float("nan")) == "–" and fmt_int(None) == "–"
    assert fmt_int(110189) == "110,189" and fmt_int(2.0) == "2"


# --------------------------------------------------------------------------
# First-load time
# --------------------------------------------------------------------------


@pytest.mark.parametrize("page", list(PAGE_FILES))
def test_every_page_loads_in_a_reasonable_time(page: str, real_outputs: None) -> None:
    started = time.perf_counter()
    at = _run(page)
    elapsed = time.perf_counter() - started
    assert not at.exception, [e.value for e in at.exception]
    assert elapsed < 20, f"{page} took {elapsed:.1f}s to render for the first time"


# --------------------------------------------------------------------------
# Every heading that shows synthetic data carries the tag, one by one
# --------------------------------------------------------------------------

_SYNTHETIC_HEADINGS = (
    "Where visitors drop out",
    "How customers sound on calls",
    "Funnel table",
    "Funnel by device",
    "Call sentiment by agent",
)


@pytest.mark.parametrize("heading", _SYNTHETIC_HEADINGS)
def test_each_synthetic_heading_on_intelligence_carries_the_tag(
    heading: str, real_outputs: None
) -> None:
    at = _run("Intelligence")
    lines = [m.value for m in at.markdown if m.value.startswith(f"### {heading}")]
    assert len(lines) == 1, f"heading {heading!r} not found exactly once"
    assert 'class="tag">Synthetic data' in lines[0]
