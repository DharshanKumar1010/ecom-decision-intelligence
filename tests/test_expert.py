"""Tests for src.expert: KnowledgeBase validation, inference, explanation.

Action-outcome tests run against the REAL `rules/seller_rules.yaml` (not a
hand-built knowledge base) with hand-picked fact sets chosen to land on the
correct side of each rule's real, data-justified thresholds (see
docs/knowledge_engineering.md) — this both exercises the shipped rule file
and pins down each rule's intended behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import RULES_DIR
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


def test_promote_fires_for_hidden_gem_with_low_late_risk(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        seller_id="gem", ahp_score=0.75, avg_late_risk=0.2, avg_review=4.5,
        order_volume=8, late_rate=0.0, is_hidden_gem=True,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Promote"
    assert result.fired_rule.id == "R08"  # the CLAUDE.md-literal (<0.3) rule outranks R09


def test_keep_fires_via_default_rule_for_sparse_seller(engine: InferenceEngine) -> None:
    facts = WorkingMemory(
        seller_id="quiet", ahp_score=None, avg_late_risk=0.50, avg_review=None,
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


def test_knowledge_base_rejects_missing_required_field(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "- id: R01\n  priority: 10\n  if: []\n  then: Keep\n",  # missing 'because'
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        KnowledgeBase.from_yaml(bad_yaml)
