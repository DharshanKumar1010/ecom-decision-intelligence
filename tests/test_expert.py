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

from src import explain
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, RULES_DIR
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


def _risk_cutoff(engine: InferenceEngine, rule_id: str) -> float:
    rule = next(r for r in engine.knowledge_base.rules if r.id == rule_id)
    return float(next(c.value for c in rule.conditions if c.fact == "avg_late_risk"))


def test_promote_fires_via_r08_for_moderate_low_risk_hidden_gem(engine: InferenceEngine) -> None:
    # Between R09's tighter cutoff and R08's looser one, read from the real rule file.
    risk = (_risk_cutoff(engine, "R09") + _risk_cutoff(engine, "R08")) / 2
    facts = WorkingMemory(
        seller_id="gem", ahp_score=0.65, avg_late_risk=risk, avg_review=4.5,
        order_volume=8, late_rate=0.0, is_hidden_gem=True,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Promote"
    # Below R08's cutoff but not below R09's tighter one -- only R08 matches.
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
        # A seller with 2 items sold has too little evidence for any rule, even with a risk
        # score far above every Warn/Suspend cutoff, so only the default rule applies.
        seller_id="quiet", ahp_score=None, avg_late_risk=0.95, avg_review=None,
        order_volume=2, late_rate=0.0, is_hidden_gem=False,
    ).as_facts()
    result = engine.run(facts)
    assert result.fired_rule.action == "Keep"
    assert result.fired_rule.id == "R12"


def _minimum_items_sold(rule: Rule) -> float | None:
    """The `order_volume >= N` gate of a rule, or None if it has none."""
    gates = [c.value for c in rule.conditions if c.fact == "order_volume" and c.op == ">="]
    return max(gates) if gates else None


def test_every_warn_and_suspend_rule_requires_minimum_evidence(engine: InferenceEngine) -> None:
    """A warning or suspension needs at least 5 items sold, in the rule file itself."""
    for rule in engine.knowledge_base.rules:
        if rule.action in ("Warn", "Suspend"):
            gate = _minimum_items_sold(rule)
            assert gate is not None and gate >= 5, f"{rule.id} ({rule.action}) has no gate >= 5"


def test_risk_only_rules_do_not_fire_for_tiny_sellers(engine: InferenceEngine) -> None:
    high_risk_tiny = WorkingMemory(
        seller_id="tiny", ahp_score=None, avg_late_risk=0.99, avg_review=1.0,
        order_volume=4, late_rate=1.0, is_hidden_gem=False,
    ).as_facts()
    result = engine.run(high_risk_tiny)
    assert result.fired_rule.id == "R12"
    just_enough = {**high_risk_tiny, "order_volume": 5}
    assert engine.run(just_enough).fired_rule.action in ("Suspend", "Warn")


def test_no_warn_or_suspend_recommendation_for_a_seller_below_five_items() -> None:
    facts_path = PROCESSED_DATA_DIR / "SellerFacts.parquet"
    recs_path = REPORTS_DIR / "seller_recommendations.csv"
    if not facts_path.exists() or not recs_path.exists():
        pytest.skip("pipeline outputs not available; run run_all.py first")
    facts = pd.read_parquet(facts_path)[["seller_id", "order_volume"]]
    recs = pd.read_csv(recs_path).merge(facts, on="seller_id", how="left", validate="1:1")
    assert recs["order_volume"].notna().all()
    tiny = recs[recs["order_volume"] < 5]
    assert len(tiny) > 0, "no seller below 5 items sold; the check would prove nothing"
    assert not tiny["action"].isin(["Warn", "Suspend"]).any(), tiny[
        tiny["action"].isin(["Warn", "Suspend"])
    ][["seller_id", "order_volume", "action", "rule_id"]].to_string()


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


def _seller_facts() -> pd.DataFrame:
    seller_facts_path = PROCESSED_DATA_DIR / "SellerFacts.parquet"
    if not seller_facts_path.exists():
        pytest.skip("data/processed/SellerFacts.parquet not available; run run_all.py first")
    seller_facts = pd.read_parquet(seller_facts_path)
    assert len(seller_facts) > 0
    return seller_facts


def _rule_threshold(rule: Rule, fact: str) -> float:
    return float(next(c.value for c in rule.conditions if c.fact == fact))


def _reference_population(seller_facts: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    """The sellers a rule can apply to: those meeting its own items-sold gate, else everyone.

    R01 and R03 (risk rules with a gate) are percentile-calibrated on this group, not on all
    sellers; R06 and R10 keep their all-seller median, R06 because it has no gate and R10 by
    decision (see docs/knowledge_engineering.md section 10).
    """
    gate = _minimum_items_sold(rule)
    if gate is None:
        return seller_facts
    return seller_facts[seller_facts["order_volume"] >= gate]


def test_percentile_calibrated_rules_remain_discriminating_in_their_reference_population(
    engine: InferenceEngine,
) -> None:
    """Drift guard for the rules calibrated against a PERCENTILE of avg_late_risk or late_rate
    (docs/knowledge_engineering.md sections 5b and 10 say which rules are percentile-intent;
    R02's 40 % late_rate is a fixed severity level, so a near-0 % fire rate is correct there).

    Each rule is measured over the sellers it can apply to (`_reference_population`): R01 and
    R03 over sellers with at least 5 items sold, the rest over all sellers (R05 and R10 over
    their own gate). A rate of 0 % or 100 % means the threshold drifted away from the
    distribution it was set against. Uses the REAL current data, not a fixture.
    """
    seller_facts = _seller_facts()
    rules_by_id = {rule.id: rule for rule in engine.knowledge_base.rules}
    for rule_id in ("R01", "R03", "R05", "R06", "R10", "R11"):
        rule = rules_by_id[rule_id]
        population = _reference_population(seller_facts, rule)
        assert len(population) > 0, rule_id
        match_rate = population.apply(
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
            f"{rule_id} fires on {match_rate:.1%} of its {len(population):,} reference sellers -- "
            "not discriminating (0% or 100%); its threshold likely needs "
            "recalibrating against the current model's late_risk distribution."
        )


@pytest.mark.parametrize(("rule_id", "intended_percentile"), [("R01", 0.75), ("R03", 0.90)])
def test_r01_and_r03_thresholds_sit_at_their_percentile_of_sellers_with_five_items(
    engine: InferenceEngine, rule_id: str, intended_percentile: float
) -> None:
    """R01 is 'top quartile' and R03 'top 10%' of the sellers they apply to (items sold >= 5)."""
    seller_facts = _seller_facts()
    rule = next(r for r in engine.knowledge_base.rules if r.id == rule_id)
    threshold = _rule_threshold(rule, "avg_late_risk")
    eligible = _reference_population(seller_facts, rule)["avg_late_risk"]
    share_at_or_below = float((eligible <= threshold).mean())
    assert abs(share_at_or_below - intended_percentile) < 0.01, (
        f"{rule_id} threshold {threshold} is the {share_at_or_below:.1%} percentile of "
        f"{len(eligible):,} sellers with >= 5 items sold, not {intended_percentile:.0%}"
    )


def test_r06_and_r10_keep_the_all_seller_median_risk(engine: InferenceEngine) -> None:
    """Documented decision: R06 and R10 stay at the median of avg_late_risk over ALL sellers."""
    seller_facts = _seller_facts()
    rules_by_id = {rule.id: rule for rule in engine.knowledge_base.rules}
    for rule_id in ("R06", "R10"):
        threshold = _rule_threshold(rules_by_id[rule_id], "avg_late_risk")
        share = float((seller_facts["avg_late_risk"] <= threshold).mean())
        assert abs(share - 0.5) < 0.01, f"{rule_id}: {threshold} is the {share:.1%} percentile"


def test_condition_checks_and_near_miss_agree_with_the_engine_for_every_seller_and_rule(
    engine: InferenceEngine,
) -> None:
    """All sellers x all rules: the page's tick list equals Rule.matches, and near-miss only
    lists unmatched, higher-priority rules with a different action (one per action)."""
    from src.expert import _facts_from_row

    seller_facts = _seller_facts()
    rules = engine.knowledge_base.rules
    pairs = 0
    for _, row in seller_facts.iterrows():
        facts = _facts_from_row(row)
        for rule in rules:
            ticks = explain.condition_checks(rule, facts)
            assert all(ok for ok, _ in ticks) == rule.matches(facts), (row["seller_id"], rule.id)
            pairs += 1
        fired = engine.run(facts).fired_rule
        near = explain.near_miss(facts, rules, limit=len(rules))
        by_id = {r.id: r for r in rules}
        for miss in near:
            assert not by_id[miss.rule_id].matches(facts)
            assert miss.priority > fired.priority and miss.action != fired.action
            assert miss.gaps
        expected_actions = {
            r.action for r in rules
            if r.priority > fired.priority and r.action != fired.action and not r.matches(facts)
        }
        assert {m.action for m in near} == expected_actions
    assert pairs == len(seller_facts) * len(rules)


def test_knowledge_base_rejects_missing_required_field(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "- id: R01\n  priority: 10\n  if: []\n  then: Keep\n",  # missing 'because'
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        KnowledgeBase.from_yaml(bad_yaml)
