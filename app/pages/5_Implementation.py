"""Page 5 — Implementation: the expert system explains a seller recommendation (Unit 5).

Only the rule engine runs live here. A seller's predicted late risk is read from the
precomputed pipeline output (`SellerFacts`): the average of OUT-OF-FOLD scores, each made by a
model that never saw any item of that item's order. It is still a model estimate, not a
measured outcome, and is labelled as such.
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
    ITEMS_SOLD_NOTE,
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
from app._format import fmt_int, fmt_num, fmt_pct
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


def _reviewed_orders(facts: pd.DataFrame) -> pd.Series:
    """Reviewed orders per seller id (`n_reviewed` in SellerFacts; 0 if the seller has none)."""
    return facts.set_index("seller_id")["n_reviewed"]


def _example_order(facts: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    """Recommendations reordered so "Try an example" lands on a meaningful seller first.

    Sellers with enough reviewed orders (`config.DISCOVERY_MIN_ORDERS`, the project's minimum
    amount of evidence) and a rule other than the default come first, then well-evidenced
    default-rule sellers, then thin ones; file order is kept inside each group, so the choice
    is deterministic.
    """
    reviewed = recommendations["seller_id"].map(_reviewed_orders(facts)).fillna(0)
    thin = reviewed < config.DISCOVERY_MIN_ORDERS
    default_rule = recommendations["rule_id"] == _DEFAULT_RULE
    rank = thin.astype(int) * 2 + default_rule.astype(int)
    ranked = recommendations.assign(_rank=rank).sort_values("_rank", kind="stable")
    return ranked.drop(columns="_rank")


def _pick_example() -> None:
    """Jump to the first seller (in `_example_order`) whose recommendation is the chosen action."""
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
    """A first seller worth looking at: a real rule fired (not the default) for a seller with
    enough REVIEWED orders (`config.DISCOVERY_MIN_ORDERS`) for the outcome to mean something.

    Prefers an actionable outcome (anything but Keep); falls back step by step, so the page
    always opens on some seller.
    """
    reviewed = _reviewed_orders(facts)
    well_evidenced = set(reviewed[reviewed >= config.DISCOVERY_MIN_ORDERS].index)
    in_facts = recommendations["seller_id"].isin(set(facts["seller_id"]))
    informative = recommendations[(recommendations["rule_id"] != _DEFAULT_RULE) & in_facts]
    informative = informative[informative["seller_id"].isin(well_evidenced)]
    actionable = informative[informative["action"] != "Keep"]
    for pool in (actionable, informative, recommendations[in_facts]):
        if not pool.empty:
            return str(pool["seller_id"].iloc[0])
    return str(facts["seller_id"].iloc[0])


def _selector(facts: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    seller_ids = facts["seller_id"].tolist()
    st.session_state[_SELLER_IDS] = frozenset(seller_ids)
    st.session_state["_impl_batch"] = _example_order(facts, recommendations)
    orders = dict(zip(facts["seller_id"], facts["order_volume"], strict=True))
    st.session_state.setdefault(_SELLER_KEY, _default_seller(facts, recommendations))
    st.session_state.setdefault(_EXAMPLE_KEY, _NO_EXAMPLE)

    left, right = st.columns([3, 2])
    with left:
        chosen = st.selectbox(
            f"Seller (all {len(seller_ids):,}; type to search)",
            seller_ids,
            format_func=lambda sid: f"{short_id(sid)} · {int(orders[sid])} items sold",
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
    checks = explain.condition_checks(fired, result.facts)
    if checks:
        st.markdown("\n".join(f"- {'✓' if ok else '✕'} {text}" for ok, text in checks))
    else:
        note("No stronger rule matched this seller, so the default applies.")


def _confidence_line(quadrant_row: pd.DataFrame) -> None:
    if quadrant_row.empty:
        note(f"Fewer than {config.DISCOVERY_MIN_ORDERS} items sold: no quality score and no group, "
             "so quality-based rules cannot apply to this seller.")
        return
    quadrant = quadrant_row.iloc[0]
    if quadrant["confidence"] == "low":
        note(f"Low confidence: this seller has {int(quadrant['order_volume'])} items sold (under "
             f"{config.DISCOVERY_CONFIDENCE_MIN_ORDERS}) and sits in the {quadrant['quadrant']} "
             "group. Quality signals from so few items are promising, not proof: treat any "
             "Promote or Feature outcome as directional.")


def _plain(value: object) -> str:
    """A fact or threshold without float noise: 5.0 -> 5, 0.33900000000000002 -> 0.339."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None or pd.isna(value):  # type: ignore[call-overload]
        return "unknown"
    number = float(value)  # type: ignore[arg-type]
    return fmt_int(number) if number.is_integer() else fmt_num(number, 3).rstrip("0").rstrip(".")


def _explanation_text(result: InferenceResult) -> str:
    """The engine's explanation as short lines (nothing wide enough to be clipped)."""
    rule = result.fired_rule
    lines = [
        f"Recommendation: {rule.action} (rule {rule.id}, priority {rule.priority}).",
        f"Because: {rule.because}",
    ]
    if not rule.conditions:
        lines.append("Matched on: no conditions (default rule).")
        return "\n".join(lines)
    lines.append("Matched on:")
    for cond in rule.conditions:
        lines.append(
            f"  {explain.fact_label(cond.fact)} = {_plain(result.facts.get(cond.fact))} "
            f"{cond.op} {_plain(cond.value)}"
        )
    return "\n".join(lines)


def _rules_table(result: InferenceResult, kb: KnowledgeBase) -> None:
    rules: dict[str, Rule] = {rule.id: rule for rule in kb.rules}
    trace = ExplanationFacility.trace_summary(result)

    def describe(rule: Rule) -> str:
        if not rule.conditions:
            return "Always matches (default)"
        return " AND ".join(
            f"{explain.fact_label(c.fact)} {c.op} {_plain(c.value)}" for c in rule.conditions
        )

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
               batch: pd.DataFrame, orders: tuple[int, int]) -> None:
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
    quality = "n/a" if pd.isna(row["ahp_score"]) else fmt_num(row["ahp_score"], 3)
    delivered, reviewed = orders
    st.markdown(
        f"- Items sold: {fmt_int(row['order_volume'])} · Delivered orders: {fmt_int(delivered)} "
        f"· Reviewed orders: {fmt_int(reviewed)}\n"
        f"- Predicted late risk: {fmt_num(row['avg_late_risk'], 3)} (the average of out-of-fold "
        "scores: each item was scored by a model that never saw any item of its order. It is "
        "still a model estimate, not a measured outcome; the measured outcome is the actual "
        "late rate)\n"
        f"- Quality score ({config.DISCOVERY_MIN_ORDERS}+ items pool): {quality} (needs at "
        f"least {config.DISCOVERY_MIN_ORDERS} items sold)\n"
        f"- Hidden Gem: {'yes' if bool(row['is_hidden_gem']) else 'no'}"
    )
    note(ITEMS_SOLD_NOTE)
    st.markdown("**Rule engine explanation**")
    st.code(_explanation_text(result), language="text", wrap_lines=True)
    id_length = len(str(row["seller_id"]))
    st.text_input("Search by full seller id", key=_LOOKUP_KEY, on_change=_lookup,
                  placeholder=f"Paste a full {id_length}-character seller id")
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
        n_orders=_num(row["n_orders"]),
        n_reviewed=_num(row["n_reviewed"]),
    )
    result = InferenceEngine(kb).run(memory.as_facts())
    action = result.fired_rule.action
    subtitle(_SUBTITLE, explain.ACTION_MEANING.get(action, "See the reason below."))

    _result_block(result)
    review = "n/a" if pd.isna(row["avg_review"]) else fmt_num(row["avg_review"], 2)
    orders = (int(row["n_orders"]), int(row["n_reviewed"]))
    kpis([
        ("Items sold", fmt_int(row["order_volume"]), None),
        ("Average review", review, "Average review score (1 to 5), one review per order."),
        ("Late-delivery rate", fmt_pct(row["late_rate"], 1),
         "Share of this seller's items that really were delivered late."),
    ])
    _confidence_line(quadrants.loc[quadrants["seller_id"] == seller_id])

    with technical():
        _technical(result, kb, row,
                   recommendations.loc[recommendations["seller_id"] == seller_id], orders)


run_page(render)
footer(
    "Expert systems: rules, inference and explanation (Unit 5)",
    "Prescriptive",
    "Operational",
)
