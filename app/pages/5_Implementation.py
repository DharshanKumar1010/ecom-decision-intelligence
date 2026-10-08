"""Page 5 — Implementation: the expert system explains a seller recommendation (Unit 5).

Only the rule engine runs live here. A seller's predicted late risk is read from the
precomputed pipeline output (`SellerFacts`), which is largely scored on rows the model was
trained on, so it is shown as a pipeline output, never as a held-out measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import _theme as theme
from app._common import load_knowledge_base, load_parquet, load_report_csv
from app._components import (
    footer,
    kpis,
    note,
    page_header,
    run_page,
    set_page,
    short_id,
    subtitle,
    technical,
)
from src import config, explain
from src.expert import (
    ExplanationFacility,
    InferenceEngine,
    InferenceResult,
    KnowledgeBase,
    Rule,
    WorkingMemory,
)

set_page("Implementation")
_SUBTITLE = page_header("What should we do about this seller?")

_SELLER_KEY = "impl_seller"
_EXAMPLE_KEY = "impl_example"
_LOOKUP_KEY = "impl_lookup"
_SELLER_IDS = "_impl_seller_ids"
_NO_EXAMPLE = "Choose an outcome…"
_DEFAULT_RULE = "R12"


def _num(value: object) -> float | None:
    """NaN -> None (a missing fact), otherwise float."""
    return None if pd.isna(value) else float(value)


def _pick_example() -> None:
    """Jump to the first seller whose batch recommendation is the chosen action."""
    action = st.session_state.get(_EXAMPLE_KEY, _NO_EXAMPLE)
    batch = st.session_state.get("_impl_batch")
    if action != _NO_EXAMPLE and batch is not None:
        sellers = batch.loc[batch["action"] == action, "seller_id"]
        if not sellers.empty:
            st.session_state[_SELLER_KEY] = str(sellers.iloc[0])
    st.session_state[_EXAMPLE_KEY] = _NO_EXAMPLE


def _lookup() -> None:
    typed = str(st.session_state.get(_LOOKUP_KEY, "")).strip()
    if typed and typed in st.session_state.get(_SELLER_IDS, ()):
        st.session_state[_SELLER_KEY] = typed


def _default_seller(facts: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    """A first seller worth looking at: an actionable outcome from a real rule, not the default."""
    in_facts = recommendations["seller_id"].isin(set(facts["seller_id"]))
    informative = recommendations[(recommendations["rule_id"] != _DEFAULT_RULE) & in_facts]
    actionable = informative[informative["action"] != "Keep"]
    for pool in (actionable, informative, recommendations[in_facts]):
        if not pool.empty:
            return str(pool["seller_id"].iloc[0])
    return str(facts["seller_id"].iloc[0])


def _selector(facts: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    seller_ids = facts["seller_id"].tolist()
    st.session_state[_SELLER_IDS] = frozenset(seller_ids)
    st.session_state["_impl_batch"] = recommendations
    orders = dict(zip(facts["seller_id"], facts["order_volume"], strict=True))
    st.session_state.setdefault(_SELLER_KEY, _default_seller(facts, recommendations))
    st.session_state.setdefault(_EXAMPLE_KEY, _NO_EXAMPLE)

    left, right = st.columns([3, 2])
    with left:
        chosen = st.selectbox(
            f"Seller (all {len(seller_ids):,}; type to search)",
            seller_ids,
            format_func=lambda sid: f"{short_id(sid)} · {int(orders[sid])} orders",
            key=_SELLER_KEY,
        )
    with right:
        st.selectbox(
            "Try an example",
            [_NO_EXAMPLE, *theme.ACTIONS],
            key=_EXAMPLE_KEY,
            on_change=_pick_example,
        )
    return str(chosen)


def _result_block(result: InferenceResult) -> None:
    fired = result.fired_rule
    swatch = theme.ACTIONS.get(fired.action, theme.ACTIONS["Keep"])
    st.markdown(
        f'<div class="action-word" data-action="{fired.action}" '
        f'style="color:{swatch.text}">{fired.action}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Why:** {explain.action_sentence(fired.action, fired, result.facts)}")
    checks = explain.condition_checks(fired, result.facts)
    if checks:
        st.markdown("\n".join(f"- {'✓' if ok else '✕'} {text}" for ok, text in checks))
    else:
        note("No stronger rule matched this seller, so the default applies.")


def _confidence_line(quadrant_row: pd.DataFrame) -> None:
    if quadrant_row.empty:
        note(f"Fewer than {config.DISCOVERY_MIN_ORDERS} orders: no quality score and no group, "
             "so quality-based rules cannot apply to this seller.")
        return
    quadrant = quadrant_row.iloc[0]
    if quadrant["confidence"] == "low":
        note(f"Low confidence: this seller has {int(quadrant['order_volume'])} orders (under "
             f"{config.DISCOVERY_CONFIDENCE_MIN_ORDERS}) and sits in the {quadrant['quadrant']} "
             "group. Quality signals from so few orders are promising, not proof: treat any "
             "Promote or Feature outcome as directional.")


def _rules_table(result: InferenceResult, kb: KnowledgeBase) -> None:
    rules: dict[str, Rule] = {rule.id: rule for rule in kb.rules}
    trace = ExplanationFacility.trace_summary(result)

    def describe(rule: Rule) -> str:
        if not rule.conditions:
            return "Always matches (default)"
        return " AND ".join(f"{c.fact} {c.op} {c.value}" for c in rule.conditions)

    table = pd.DataFrame({
        "Rule": [t["rule_id"] for t in trace],
        "Priority": [t["priority"] for t in trace],
        "Action": [t["action"] for t in trace],
        "Conditions": [describe(rules[t["rule_id"]]) for t in trace],
        "Outcome": [
            "FIRED" if t["fired"]
            else ("Also matched, outranked" if t["matched"] else "No match")
            for t in trace
        ],
    }).sort_values(["Priority", "Rule"], ascending=[False, True])
    st.markdown(f"**All {len(table)} rules, in priority order**")
    st.table(table.set_index("Rule"))
    note("The engine fires only the single highest-priority matching rule (ties go to the "
         "lowest rule id); lower-priority matches are shown as outranked.")


def _agreement(result: InferenceResult, batch: pd.DataFrame) -> None:
    fired = result.fired_rule
    if batch.empty:
        note("This seller has no row in the batch file seller_recommendations.csv.")
        return
    stored = batch.iloc[0]
    if stored["action"] == fired.action and stored["rule_id"] == fired.id:
        st.markdown(f"Live result = batch output ({stored['action']}, rule {stored['rule_id']}).")
    else:
        st.markdown(
            f"**Live result differs from the stored batch output.** Live: {fired.action} "
            f"({fired.id}). Batch: {stored['action']} ({stored['rule_id']}). The batch file is "
            "probably stale: re-run `python run_all.py`."
        )


def _technical(result: InferenceResult, kb: KnowledgeBase, row: pd.Series,
               batch: pd.DataFrame) -> None:
    _agreement(result, batch)
    _rules_table(result, kb)

    misses = explain.near_miss(result.facts, kb.rules)
    st.markdown("**What would change this recommendation?**")
    if not misses:
        note("No higher-priority rule with a different outcome exists for this seller.")
    for miss in misses:
        st.markdown(f"{miss.action} (rule {miss.rule_id})")
        for sentence in miss.sentences or ("All of its conditions are already met.",):
            st.markdown(f"- {sentence}")

    st.markdown("**Other facts the rules use**")
    quality = "n/a" if pd.isna(row["ahp_score"]) else f"{row['ahp_score']:.3f}"
    st.markdown(
        f"- Predicted late risk: {row['avg_late_risk']:.3f} (a pipeline output, not a "
        "held-out measurement: the model has seen most of these items while learning)\n"
        f"- Quality (AHP) score: {quality} (needs at least {config.DISCOVERY_MIN_ORDERS} "
        "orders)\n"
        f"- Hidden Gem: {'yes' if bool(row['is_hidden_gem']) else 'no'}"
    )
    st.markdown("**Rule engine explanation**")
    st.code(ExplanationFacility.explain(result), language="text")
    st.text_input("Search by full seller id", key=_LOOKUP_KEY, on_change=_lookup)
    typed = str(st.session_state.get(_LOOKUP_KEY, "")).strip()
    if typed and typed not in st.session_state[_SELLER_IDS]:
        st.caption("No seller has that id.")
    st.caption(f"Selected seller id: `{row['seller_id']}`")


def render() -> None:
    facts = load_parquet("SellerFacts.parquet")
    recommendations = load_report_csv("seller_recommendations.csv")
    quadrants = load_report_csv("discovery_quadrants.csv")
    kb = load_knowledge_base()

    seller_id = _selector(facts, recommendations)
    row = facts.loc[facts["seller_id"] == seller_id].iloc[0]
    memory = WorkingMemory(
        seller_id=seller_id,
        ahp_score=_num(row["ahp_score"]),
        avg_late_risk=_num(row["avg_late_risk"]),
        avg_review=_num(row["avg_review"]),
        order_volume=_num(row["order_volume"]),
        late_rate=_num(row["late_rate"]),
        is_hidden_gem=bool(row["is_hidden_gem"]),
    )
    result = InferenceEngine(kb).run(memory.as_facts())
    action = result.fired_rule.action
    subtitle(_SUBTITLE, f"{action}: {explain.ACTION_MEANING.get(action, 'see the reason below')}")

    _result_block(result)
    review = "n/a" if pd.isna(row["avg_review"]) else f"{row['avg_review']:.2f}"
    kpis([
        ("Orders", f"{int(row['order_volume'])}", None),
        ("Average review", review, "Average review score (1 to 5)."),
        ("Late-delivery rate", f"{row['late_rate'] * 100:.1f}%",
         "Share of this seller's items that really were delivered late."),
    ])
    _confidence_line(quadrants.loc[quadrants["seller_id"] == seller_id])

    with technical():
        _technical(result, kb, row, recommendations.loc[recommendations["seller_id"] == seller_id])


run_page(render)
footer(
    "Expert systems: rules, inference and explanation (Unit 5)",
    "Prescriptive",
    "Operational",
)
