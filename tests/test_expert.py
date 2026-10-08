"""Tests for src.expert: KnowledgeBase validation, inference, explanation.

Action-outcome tests run against the REAL `rules/seller_rules.yaml` (not a
hand-built knowledge base) with hand-picked fact sets chosen to land on the
correct side of each rule's real, data-justified thresholds (see
docs/knowledge_engineering.md) — this both exercises the shipped rule file
and pins down each rule's intended behavior.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import PROCESSED_DATA_DIR, RULES_DIR
from src.expert import (
    Condition,
    ExplanationFacility,
    InferenceEngine,
    KnowledgeBase,
    Rule,
    WorkingMemory,
)

_REAL_RULES_PATH = RULES_DIR / "seller_rules.yaml"


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    return InferenceEngine(KnowledgeBase.from_yaml(_REAL_RULES_PATH))


def test_real_ruleset_has_at_least_twelve_rules_covering_all_actions(
    engine: InferenceEngine,
) -> None:
    assert len(engine.knowledge_base.rules) >= 12
    actions = {rule.action for rule in engine.knowledge_base.rules}
    assert actions == {"Keep", "Warn", "Suspend", "Feature", "Promote"}


def test_suspend_fires_for_high_risk_poorly_reviewed_seller(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        seller_id="bad", ahp_score=None, avg_late_risk=0.6, avg_review=2.5,
        order_volume=20, late_rate=0.2, is_hidden_gem=False,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Suspend"


def test_warn_fires_for_elevated_risk_seller(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        seller_id="borderline", ahp_score=0.65, avg_late_risk=0.56, avg_review=4.0,
        order_volume=20, late_rate=0.05, is_hidden_gem=False,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Warn"


def test_feature_fires_for_top_established_seller(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        seller_id="top", ahp_score=0.85, avg_late_risk=0.45, avg_review=4.9,
        order_volume=200, late_rate=0.02, is_hidden_gem=False,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Feature"


def test_promote_fires_via_r08_for_moderate_low_risk_hidden_gem(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        seller_id="gem", ahp_score=0.75, avg_late_risk=0.2, avg_review=4.5,
        order_volume=8, late_rate=0.0, is_hidden_gem=True,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Promote"
    # 0.2 is below R08's 0.3 but not below R09's tighter 0.05 -- only R08 matches.
    assert result.fired_rule.id == "R08"


def test_promote_fires_via_r09_for_very_low_risk_hidden_gem(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        # ahp_score kept below R06's 0.711 Feature threshold so this fixture
        # isolates the Promote tier rather than accidentally matching Feature.
        seller_id="best_gem", ahp_score=0.68, avg_late_risk=0.02, avg_review=4.8,
        order_volume=6, late_rate=0.0, is_hidden_gem=True,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Promote"
    # 0.02 is below BOTH thresholds; R09 (priority 80, the tighter/high-confidence
    # tier) outranks R08 (priority 78) when both match.
    assert result.fired_rule.id == "R09"


def test_keep_fires_via_default_rule_for_sparse_seller(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        # avg_late_risk kept below R03's 0.29 Warn threshold so this fixture
        # isolates the default rule rather than accidentally matching R03.
        seller_id="quiet", ahp_score=None, avg_late_risk=0.15, avg_review=None,
        order_volume=2, late_rate=0.0, is_hidden_gem=False,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Keep"
    assert result.fired_rule.id == "R12"


def test_explanation_facility_renders_rationale_and_trace(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        seller_id="bad", ahp_score=None, avg_late_risk=0.6, avg_review=2.5,
        order_volume=20, late_rate=0.2, is_hidden_gem=False,
    ).as_facts()
    result = engine.run(facts)

    explanation = ExplanationFacility.explain(result)
    assert "Suspend" in explanation
    assert result.fired_rule.because in explanation

    trace = ExplanationFacility.trace_summary(result)
    assert sum(row["fired"] for row in trace) == 1
    assert len(trace) == len(engine.knowledge_base.rules)


def test_priority_conflict_resolution_and_id_tie_break() -> None:
    rule_low = Rule(id="RA", priority=10, conditions=(), action="ActionA", because="a")
    rule_tie_1 = Rule(id="RB", priority=20, conditions=(), action="ActionB", because="b")
    rule_tie_2 = Rule(id="RC", priority=20, conditions=(), action="ActionC", because="c")
    engine = InferenceEngine(KnowledgeBase((rule_low, rule_tie_1, rule_tie_2)))

    result = engine.run({})
    # RB and RC tie at priority 20; RB wins because "RB" < "RC" ascending.
    assert result.fired_rule.id == "RB"


def test_no_matching_rule_raises_runtime_error() -> None:
    kb = KnowledgeBase(
        (Rule(id="R01", priority=1, conditions=(Condition("x", "==", 1),),
              action="Keep", because="only matches x==1"),)
    )
    engine = InferenceEngine(kb)
    with pytest.raises(RuntimeError):
        engine.run({"x": 2})


def test_knowledge_base_rejects_duplicate_ids(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "- id: R01\n  priority: 10\n  if: []\n  then: Keep\n  because: \"first\"\n"
        "- id: R01\n  priority: 5\n  if: []\n  then: Warn\n  because: \"dup\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        KnowledgeBase.from_yaml(bad_yaml)


def test_knowledge_base_rejects_unknown_field(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "- id: R01\n  priority: 10\n  if: []\n  then: Keep\n  because: \"ok\"\n"
        "  extra_field: \"not allowed\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown field"):
        KnowledgeBase.from_yaml(bad_yaml)


def test_knowledge_base_rejects_unknown_operator(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "- id: R01\n  priority: 10\n"
        "  if:\n    - {fact: avg_review, op: \"~=\", value: 3.0}\n"
        "  then: Keep\n  because: \"bad operator\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown operator"):
        KnowledgeBase.from_yaml(bad_yaml)


def test_promote_rules_remain_discriminating_among_real_hidden_gems(
    engine: InferenceEngine,
) -> None:
    """Regression guard against the exact drift this file was recalibrated for.

    R09 (and R08) are meant to split Hidden Gems into meaningfully different
    confidence tiers. If a future model change pushes `avg_late_risk`'s
    distribution around again, one of these thresholds could silently drift
    to firing on ~0% or ~100% of Hidden Gems (as R09's old 0.474 threshold
    did when RandomForest replaced LogisticRegression as primary — see
    BUILD_LOG.md). This checks the REAL current data, not a fixture, so
    that drift is caught automatically.
    """
    seller_facts_path = PROCESSED_DATA_DIR / "SellerFacts.parquet"
    if not seller_facts_path.exists():
        pytest.skip("data/processed/SellerFacts.parquet not available; run run_all.py first")

    seller_facts = pd.read_parquet(seller_facts_path)
    hidden_gems = seller_facts[seller_facts["is_hidden_gem"]]
    assert len(hidden_gems) > 0, "no Hidden Gems in the real data; can't test discrimination"

    rules_by_id = {rule.id: rule for rule in engine.knowledge_base.rules}
    for rule_id in ("R08", "R09"):
        rule = rules_by_id[rule_id]
        match_rate = hidden_gems.apply(
            lambda row, r=rule: r.matches(
                {"is_hidden_gem": bool(row["is_hidden_gem"]),
                 "avg_late_risk": row["avg_late_risk"]}
            ),
            axis=1,
        ).mean()
        assert 0.0 < match_rate < 1.0, (
            f"{rule_id} fires on {match_rate:.1%} of real Hidden Gems -- "
            "not discriminating (0% or 100%); its threshold likely needs "
            "recalibrating against the current model's late_risk distribution."
        )


def test_percentile_calibrated_rules_remain_discriminating_among_all_sellers(
    engine: InferenceEngine,
) -> None:
    """Same drift guard as the Hidden-Gem Promote check, for the other six
    rules whose thresholds were calibrated against a PERCENTILE of
    avg_late_risk or late_rate over ALL sellers (see
    docs/knowledge_engineering.md section 5b for which rules are
    percentile-intent vs. fixed-threshold-intent -- R02 is deliberately
    excluded here since its 40% late_rate threshold is a fixed severity
    threshold by design, not a percentile, and a near-0% fire rate is its
    correct, intended behavior, not drift).

    R01/R03/R06/R10 (avg_late_risk) drifted to the ~97th percentile when
    RandomForest replaced LogisticRegression as primary and were
    recalibrated; R05/R11 (late_rate, empirical and model-independent) were
    checked and found not to have drifted. This test guards all six against
    future drift the same way, using the REAL current data, not a fixture.
    """
    seller_facts_path = PROCESSED_DATA_DIR / "SellerFacts.parquet"
    if not seller_facts_path.exists():
        pytest.skip("data/processed/SellerFacts.parquet not available; run run_all.py first")

    seller_facts = pd.read_parquet(seller_facts_path)
    assert len(seller_facts) > 0

    rules_by_id = {rule.id: rule for rule in engine.knowledge_base.rules}
    for rule_id in ("R01", "R03", "R05", "R06", "R10", "R11"):
        rule = rules_by_id[rule_id]
        match_rate = seller_facts.apply(
            lambda row, r=rule: r.matches(
                {
                    "ahp_score": None if pd.isna(row["ahp_score"]) else row["ahp_score"],
                    "avg_late_risk": row["avg_late_risk"],
                    "avg_review": None if pd.isna(row["avg_review"]) else row["avg_review"],
                    "order_volume": row["order_volume"],
                    "late_rate": row["late_rate"],
                }
            ),
            axis=1,
        ).mean()
        assert 0.0 < match_rate < 1.0, (
            f"{rule_id} fires on {match_rate:.1%} of all real sellers -- "
            "not discriminating (0% or 100%); its threshold likely needs "
            "recalibrating against the current model's late_risk distribution."
        )


def test_knowledge_base_rejects_missing_required_field(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "- id: R01\n  priority: 10\n  if: []\n  then: Keep\n",  # missing 'because'
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        KnowledgeBase.from_yaml(bad_yaml)
