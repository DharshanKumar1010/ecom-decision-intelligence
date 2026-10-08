"""Tests for src.explain: plain-language builders must reflect the live values passed in."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.expert import Condition, KnowledgeBase, Rule
from src.explain import (
    ACTION_MEANING,
    action_sentence,
    architecture_answer,
    auc_sentence,
    choice_answer,
    condition_checks,
    cr_sentence,
    cv_sentence,
    design_answer,
    discovery_answer,
    feature_label,
    funnel_sentence,
    intelligence_answer,
    near_miss,
    quadrant_definitions,
    quadrant_sentence,
    sentiment_range_sentence,
    top_drivers_sentence,
)


def _funnel(view: int, cart: int, checkout: int, purchase: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"sessions": [view, cart, checkout, purchase]},
        index=pd.Index(["page_view", "add_to_cart", "checkout", "purchase"], name="stage"),
    )


def _rules() -> tuple[Rule, ...]:
    return (
        Rule("P", 80, (Condition("is_hidden_gem", "==", True),
                       Condition("avg_late_risk", "<", 0.05)), "Promote", "tight gem"),
        Rule("W", 70, (Condition("avg_late_risk", ">", 0.29),), "Warn", "risky"),
        Rule("D", 1, (), "Keep", "default"),
    )


# --- labels ---------------------------------------------------------------


def test_feature_label_covers_every_model_feature_with_a_readable_name() -> None:
    for feature in config.ALLOWED_FEATURES:
        label = feature_label(feature)
        assert label != feature
        assert "_" not in label
    assert feature_label("order_month") == "Month of purchase"
    assert feature_label("seller_historical_late_rate") == "Seller's past late-delivery rate"
    assert feature_label("geo_distance") == "Seller-to-customer distance"


def test_feature_label_has_a_humanised_fallback_for_unknown_names() -> None:
    assert feature_label("some_new_feature") == "Some new feature"


# --- model-quality sentences ----------------------------------------------


def test_auc_sentence_states_the_live_value_and_the_coin_flip_baseline() -> None:
    sentence = auc_sentence(0.7841)
    assert "78%" in sentence
    assert "coin flip" in sentence and "50%" in sentence
    assert "late" in sentence and "on-time" in sentence
    assert "78%" not in auc_sentence(0.6347) and "63%" in auc_sentence(0.6347)


@pytest.mark.parametrize("auc", [0.2, 0.5, 0.63, 0.7841, 0.99])
def test_auc_sentence_never_calls_it_accuracy(auc: float) -> None:
    assert "accura" not in auc_sentence(auc).lower()


def test_auc_below_a_coin_flip_is_called_out() -> None:
    assert "worse than a coin flip" in auc_sentence(0.4)
    assert "worse than a coin flip" not in auc_sentence(0.6)


def test_cv_sentence_reports_live_mean_and_spread_and_stability() -> None:
    stable = cv_sentence(0.7928, 0.0051)
    assert "79%" in stable and "0.5 percentage points" in stable and "stable" in stable
    wobbly = cv_sentence(0.70, 0.06)
    assert "6.0 percentage points" in wobbly and "caution" in wobbly and "stable" not in wobbly
    assert "5 different slices" in cv_sentence(0.7, 0.01, folds=5)


def test_cr_sentence_distinguishes_consistent_from_contradictory() -> None:
    ok = cr_sentence(0.0189, 0.10)
    assert "consistent" in ok and "0.019" in ok and "0.10" in ok
    bad = cr_sentence(0.494, 0.10)
    assert "contradict" in bad and "0.494" in bad
    assert "consistent" in cr_sentence(0.10, 0.10)  # at the limit is still acceptable


def test_top_drivers_sentence_orders_by_importance_with_plain_labels() -> None:
    sentence = top_drivers_sentence(
        {"order_month": 0.16, "geo_distance": 0.05, "seller_historical_late_rate": 0.06,
         "price": 0.01}
    )
    assert sentence.index("month of purchase") < sentence.index("past late-delivery rate")
    assert sentence.index("past late-delivery rate") < sentence.index("seller-to-customer distance")
    assert "price" not in sentence.lower().replace("item price", "")  # only the top three
    assert "order_month" not in sentence


# --- marketplace / AHP sentences ------------------------------------------


def test_funnel_sentence_is_computed_from_the_frame() -> None:
    sentence = funnel_sentence(_funnel(70_000, 15_000, 9_000, 6_000))
    assert "Of every 100 sessions that start" in sentence
    assert "about 21 add something to the cart" in sentence
    assert "about 13 reach checkout" in sentence
    assert "about 9 complete a purchase" in sentence
    assert "about 25 reach checkout" in funnel_sentence(_funnel(100, 50, 25, 10))


def test_funnel_sentence_handles_an_empty_funnel() -> None:
    assert "No sessions" in funnel_sentence(_funnel(0, 0, 0, 0))


def test_quadrant_sentences_use_the_live_lines_and_counts() -> None:
    definitions = quadrant_definitions(0.716, 18.0)
    assert set(definitions) == {"Star", "Hidden Gem", "Overrated", "Overlooked-Low-Quality"}
    assert "0.716" in definitions["Hidden Gem"] and "18 orders" in definitions["Hidden Gem"]
    assert "0.9" in quadrant_definitions(0.9, 18.0)["Star"]
    sentence = quadrant_sentence("Hidden Gem", 1478, definitions["Hidden Gem"])
    assert sentence.startswith("1,478 sellers are in the Hidden Gem group")


def test_sentiment_range_sentence_names_the_extremes() -> None:
    sentence = sentiment_range_sentence({"AG01": 0.06, "AG02": 0.27, "AG03": 0.36})
    assert "+0.06 (AG01)" in sentence and "+0.36 (AG03)" in sentence


# --- expert-system sentences ----------------------------------------------


def test_action_sentence_combines_plain_meaning_and_the_rules_own_rationale() -> None:
    rule = _rules()[0]
    sentence = action_sentence("Promote", rule)
    assert ACTION_MEANING["Promote"] in sentence and rule.because in sentence
    other = Rule("X", 5, (), "Warn", "a different reason")
    assert "a different reason" in action_sentence("Warn", other)
    assert "Quarantine" in action_sentence("Quarantine", other)  # unknown action still works


def test_action_sentence_with_facts_builds_the_reason_from_conditions_and_live_values() -> None:
    rule = _rules()[0]  # Promote: hidden gem AND risk below 0.05
    facts = {"is_hidden_gem": True, "avg_late_risk": 0.03}
    sentence = action_sentence("Promote", rule, facts)
    assert "predicted late risk is below 0.05 (here: 0.03)" in sentence
    assert "the seller is a Hidden Gem" in sentence
    assert "tight gem" not in sentence  # the stored rationale is for Technical mode
    assert "(here: 0.01)" in action_sentence("Promote", rule, {**facts, "avg_late_risk": 0.01})
    default = action_sentence("Keep", _rules()[2], {})
    assert "default is to keep the seller" in default and ".." not in default
    assert "(here: unknown)" in action_sentence("Warn", _rules()[1], {"avg_late_risk": None})


def test_near_miss_reports_the_exact_gap_on_a_hand_built_fixture() -> None:
    facts = {"is_hidden_gem": True, "avg_late_risk": 0.08}
    result = near_miss(facts, _rules())
    assert [n.rule_id for n in result] == ["P", "W"]
    promote = result[0]
    assert promote.action == "Promote" and len(promote.gaps) == 1
    assert promote.gaps[0].gap == pytest.approx(0.03)
    assert promote.sentences == (
        "To be promoted, predicted late risk would need to be below 0.05; "
        "this seller is at 0.08.",
    )
    assert result[1].gaps[0].gap == pytest.approx(0.21)


def test_near_miss_describes_a_boolean_condition_without_a_numeric_gap() -> None:
    result = near_miss({"is_hidden_gem": False, "avg_late_risk": 0.02}, _rules())
    promote = next(n for n in result if n.rule_id == "P")
    assert len(promote.gaps) == 1 and promote.gaps[0].gap is None
    assert "be a Hidden Gem" in promote.sentences[0]


def test_near_miss_handles_missing_facts() -> None:
    result = near_miss({"is_hidden_gem": True, "avg_late_risk": None}, _rules())
    promote = next(n for n in result if n.rule_id == "P")
    assert "has no predicted late risk yet" in promote.sentences[0]
    assert promote.gaps[0].gap is None


def test_near_miss_ignores_same_action_and_lower_priority_rules() -> None:
    # Fires P (the top rule): nothing outranks it, so nothing can change the outcome.
    assert near_miss({"is_hidden_gem": True, "avg_late_risk": 0.01}, _rules()) == []
    # Fires W: the only higher rule is P (different action, so it is reported).
    fired_w = near_miss({"is_hidden_gem": False, "avg_late_risk": 0.5}, _rules())
    assert [n.rule_id for n in fired_w] == ["P"]


def test_near_miss_keeps_the_closest_rule_per_action_and_respects_the_limit() -> None:
    rules = (
        Rule("S1", 100, (Condition("avg_late_risk", ">", 0.5), Condition("avg_review", "<", 3)),
             "Suspend", "s1"),
        Rule("S2", 98, (Condition("late_rate", ">", 0.4),), "Suspend", "s2"),
        Rule("F", 90, (Condition("ahp_score", ">=", 0.8),), "Feature", "f"),
        Rule("D", 1, (), "Keep", "d"),
    )
    facts = {"avg_late_risk": 0.1, "avg_review": 4.5, "late_rate": 0.0, "ahp_score": 0.7}
    result = near_miss(facts, rules)
    assert [n.action for n in result] == ["Suspend", "Feature"]  # one per action
    assert result[0].rule_id == "S2"  # closest Suspend rule (1 unmet condition vs 2)
    assert len(near_miss(facts, rules, limit=1)) == 1


def test_near_miss_is_deterministic_and_empty_without_a_matching_rule() -> None:
    facts = {"is_hidden_gem": True, "avg_late_risk": 0.08}
    assert near_miss(facts, _rules()) == near_miss(facts, _rules())
    assert near_miss(facts, _rules()[:2]) == []  # no default rule and nothing matches


def test_near_miss_quotes_thresholds_from_the_real_rules_not_literals() -> None:
    rules = KnowledgeBase.from_yaml(config.RULES_DIR / "seller_rules.yaml").rules
    facts = {"ahp_score": 0.6, "avg_late_risk": 0.12, "avg_review": 4.0, "order_volume": 40,
             "late_rate": 0.05, "is_hidden_gem": True}
    result = near_miss(facts, rules)
    assert result
    by_id = {r.id: r for r in rules}
    for miss in result:
        for gap in miss.gaps:
            threshold = next(c.value for c in by_id[miss.rule_id].conditions if c.fact == gap.fact)
            if not isinstance(threshold, bool):
                assert f"{threshold:g}" in gap.sentence or f"{threshold:,}" in gap.sentence


def test_condition_checks_tick_each_condition_with_live_values() -> None:
    rule = _rules()[0]
    checks = condition_checks(rule, {"is_hidden_gem": True, "avg_late_risk": 0.08})
    assert checks == [
        (True, "the seller is a Hidden Gem"),
        (False, "predicted late risk is below 0.05 (here: 0.08)"),
    ]
    assert condition_checks(_rules()[2], {}) == []  # the default rule has no conditions


# --- page answers ----------------------------------------------------------


def test_page_answers_contain_the_live_values_passed_in() -> None:
    funnel = _funnel(70_000, 15_000, 9_000, 6_000)
    assert "13 reach checkout" in intelligence_answer(funnel, 0.0791)
    assert "7.9%" in intelligence_answer(funnel, 0.0791)
    assert "8.5%" in intelligence_answer(funnel, 0.0851)
    assert "78%" in design_answer(0.7841, 0.0791) and "7.9%" in design_answer(0.7841, 0.0791)
    choice = choice_answer(681, 30)
    assert "681 established" in choice and "at least 30 orders" in choice
    answer = discovery_answer(478, 400, 15)
    assert "478" in answer and "400" in answer and "fewer than 15 orders" in answer
    architecture = architecture_answer(16, 12)
    assert "16 data tables" in architecture and "12 rules" in architecture


# --- agreement with the real engine on the real data --------------------------


def test_near_miss_and_condition_checks_agree_with_the_engine_for_every_seller_and_rule() -> None:
    """For all sellers x all rules, "would this rule fire" must equal the engine's match."""
    from src.expert import ExplanationFacility, InferenceEngine, _facts_from_row

    path = config.PROCESSED_DATA_DIR / "SellerFacts.parquet"
    if not path.exists():
        pytest.skip("SellerFacts.parquet not generated; run run_all.py")
    sellers = pd.read_parquet(path)
    kb = KnowledgeBase.from_yaml(config.RULES_DIR / "seller_rules.yaml")
    engine = InferenceEngine(kb)
    by_id = {rule.id: rule for rule in kb.rules}
    checked = 0

    for _, row in sellers.iterrows():
        facts = _facts_from_row(row)
        result = engine.run(facts)
        matched = {t["rule_id"]: t["matched"] for t in ExplanationFacility.trace_summary(result)}
        checks = {rule.id: condition_checks(rule, facts) for rule in kb.rules}
        for rule in kb.rules:
            would_fire = all(ok for ok, _ in checks[rule.id])
            assert would_fire == matched[rule.id], (row["seller_id"], rule.id)
            checked += 1

        for miss in near_miss(facts, kb.rules, limit=len(kb.rules)):
            rule = by_id[miss.rule_id]
            failed = sum(1 for ok, _ in checks[miss.rule_id] if not ok)
            assert not matched[miss.rule_id], (row["seller_id"], miss.rule_id)
            assert len(miss.gaps) == failed > 0, (row["seller_id"], miss.rule_id)
            assert rule.priority > result.fired_rule.priority
            assert rule.action != result.fired_rule.action

    assert checked == len(sellers) * len(kb.rules) > 0
