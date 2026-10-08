"""Plain-language sentence builders for the dashboard (pure functions, no I/O).

Every function turns LIVE values passed in by the caller into a sentence, so
no metric is ever baked into prose: change the input and the sentence changes
(`tests/test_explain.py` checks this). Rule thresholds quoted by `near_miss`
come from the `Rule` objects loaded from `rules/seller_rules.yaml`, never from
literals here, so a later recalibration of the rules cannot leave stale copy.

Wording rules: `auc_sentence` explains ROC-AUC as a ranking probability and
never calls it "accuracy" (it is not accuracy); `cv_sentence` only calls a
result stable when the fold-to-fold spread is small.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import AHP_CR_THRESHOLD
from src.expert import Condition, Rule

# Fold-to-fold std (in AUC units) at or below which a CV result is called stable.
_STABLE_CV_STD = 0.02

_FEATURE_LABELS: dict[str, str] = {
    "price": "Item price",
    "freight_value": "Shipping cost",
    "product_weight_g": "Product weight",
    "same_state": "Seller and customer in the same state",
    "product_category_freq": "How common the product category is",
    "order_month": "Month of purchase",
    "day_of_week": "Day of the week",
    "n_items_in_order": "Number of items in the order",
    "payment_installments": "Payment installments",
    "seller_historical_late_rate": "Seller's past late-delivery rate",
    "geo_distance": "Seller-to-customer distance",
    "geo_distance_missing": "Distance unknown (location not found)",
}

# Facts the expert system reasons over, in words (no leading article, so they
# slot into "To be promoted, <label> would need to be ...").
# One plain sentence per predictive model, keyed like reports/model_metrics.json.
MODEL_BLURB: dict[str, str] = {
    "random_forest": "Hundreds of small decision trees vote. Ranks late orders best, "
    "but cannot be read as one rule.",
    "decision_tree": "One flowchart of yes/no questions. Easy to read, ranks less well.",
    "logistic_regression": "One weight per input, added up into a risk. Easy to read, "
    "ranks least well.",
}

_FACT_LABELS: dict[str, str] = {
    "ahp_score": "AHP quality score",
    "avg_late_risk": "predicted late risk",
    "avg_review": "average review score",
    "order_volume": "number of orders",
    "late_rate": "actual late-delivery rate",
    "is_hidden_gem": "Hidden Gem status",
}
_BOOLEAN_NEEDS: dict[str, tuple[str, str]] = {
    "is_hidden_gem": ("be a Hidden Gem", "not be a Hidden Gem"),
}

ACTION_MEANING: dict[str, str] = {
    "Keep": "No change is needed for now.",
    "Warn": "Warn this seller and watch them closely.",
    "Suspend": "Pause this seller until the problems are fixed.",
    "Feature": "Give this proven seller extra visibility.",
    "Promote": "Give this under-exposed seller a visibility boost.",
}
_ACTION_PHRASE: dict[str, str] = {
    "Keep": "kept as is",
    "Warn": "warned",
    "Suspend": "suspended",
    "Feature": "featured",
    "Promote": "promoted",
}
_OP_WORDS: dict[str, str] = {
    "<": "below",
    "<=": "at most",
    ">": "above",
    ">=": "at least",
    "==": "equal to",
    "!=": "different from",
}


def _humanise(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _fmt(value: float) -> str:
    """Compact number: integers without decimals, otherwise up to 3 decimals."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def feature_label(name: str) -> str:
    """Readable label for a model feature (humanised fallback for unknown names)."""
    return _FEATURE_LABELS.get(name, _humanise(name))


# --------------------------------------------------------------------------
# Model-quality sentences
# --------------------------------------------------------------------------


def auc_sentence(auc: float) -> str:
    """What a ROC-AUC means, as a random-pair ranking probability.

    Deliberately never uses the word for plain correctness: AUC measures how
    well the model ranks late orders above on-time ones, not how often it is
    right.
    """
    sentence = (
        "If you pick one late order and one on-time order at random, the model gives "
        f"the late one the higher risk score about {auc * 100:.0f}% of the time. "
        "A coin flip would be 50%; a perfect ranking would be 100%."
    )
    if auc < 0.5:
        sentence += " That is worse than a coin flip."
    return sentence


def cv_sentence(mean: float, std: float, folds: int | None = None) -> str:
    """Cross-validation result in words; 'stable' only if the spread is small."""
    slices = f"{folds} different slices" if folds else "different slices"
    lead = f"Re-tested on {slices} of the data, the score averaged {mean * 100:.0f}%"
    spread = f"with a typical swing of {std * 100:.1f} percentage points"
    if std <= _STABLE_CV_STD:
        return f"{lead}, {spread}, so the result is stable rather than a lucky split."
    return (
        f"{lead}, {spread}; that is a wide swing, so treat the headline number with caution."
    )


def top_drivers_sentence(importances: Mapping[str, float], n: int = 3) -> str:
    """Name the biggest drivers of a model's predictions, using plain labels."""
    ranked = sorted(importances.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    if not ranked:
        return "No feature importances are available."
    names = [feature_label(name).lower() for name, _ in ranked]
    first = feature_label(ranked[0][0])
    if len(names) == 1:
        return f"The biggest driver of the predictions is {first.lower()}."
    rest = ", ".join(names[1:-1])
    tail = names[-1] if len(names) > 2 else names[1]
    followed = f"{rest}, and {tail}" if rest else tail
    return f"The biggest driver of the predictions is {first.lower()}, followed by {followed}."


# --------------------------------------------------------------------------
# AHP / marketplace sentences
# --------------------------------------------------------------------------


def cr_sentence(cr: float, threshold: float = AHP_CR_THRESHOLD) -> str:
    """Consistency of the pairwise judgments, in words."""
    if cr <= threshold:
        return (
            "The judgments are consistent: they do not contradict each other "
            f"(consistency ratio {cr:.3f}, limit {threshold:.2f})."
        )
    return (
        f"The judgments contradict each other (consistency ratio {cr:.3f} is above the "
        f"{threshold:.2f} limit), so these weights should not be trusted until the "
        "comparisons are revisited."
    )


def funnel_sentence(funnel: pd.DataFrame) -> str:
    """'Of every 100 sessions that start, about N reach checkout' from a funnel frame.

    `funnel` is the output of `clickstream.funnel_conversion` (indexed by stage,
    with a `sessions` column).
    """
    sessions = funnel["sessions"]
    start = float(sessions.iloc[0])
    if start <= 0:
        return "No sessions were recorded, so there is no funnel to describe."

    def per_hundred(stage: str) -> int:
        return round(100 * float(sessions.loc[stage]) / start)

    return (
        f"Of every 100 sessions that start, about {per_hundred('add_to_cart')} add something "
        f"to the cart, about {per_hundred('checkout')} reach checkout and about "
        f"{per_hundred('purchase')} complete a purchase."
    )


def sentiment_range_sentence(means: Mapping[str, float]) -> str:
    """Spread of mean call sentiment across agents."""
    if not means:
        return "No call sentiment is available."
    low = min(means.items(), key=lambda kv: (kv[1], kv[0]))
    high = max(means.items(), key=lambda kv: (kv[1], kv[0]))
    return (
        f"Mean sentiment runs from {low[1]:+.2f} ({low[0]}) to {high[1]:+.2f} ({high[0]}); "
        "higher means friendlier wording."
    )


def quadrant_definitions(quality_line: float, popularity_line: float) -> dict[str, str]:
    """One-line definition of each quadrant using the live split lines."""
    q, p = f"{quality_line:.3f}", f"{popularity_line:,.0f}"
    return {
        "Star": f"quality score at least {q} and at least {p} orders: good and well known",
        "Hidden Gem": f"quality score at least {q} but under {p} orders: good but overlooked",
        "Overrated": f"at least {p} orders but quality score under {q}: popular, not good",
        "Overlooked-Low-Quality": (
            f"under {p} orders and quality score under {q}: neither good nor popular"
        ),
    }


def quadrant_sentence(name: str, count: int, definition: str) -> str:
    """Count and meaning of one quadrant."""
    return f"{count:,} sellers are in the {name} group: {definition}."


# --------------------------------------------------------------------------
# Expert-system sentences
# --------------------------------------------------------------------------


def _condition_phrase(condition: Condition, facts: Mapping[str, Any]) -> str:
    """One condition in words, with the seller's actual value where it is numeric."""
    value = facts.get(condition.fact)
    label = _FACT_LABELS.get(condition.fact, _humanise(condition.fact))
    if isinstance(condition.value, bool):
        need = _BOOLEAN_NEEDS.get(condition.fact, (f"has {label}", f"lacks {label}"))
        return "the seller " + need[0 if condition.value else 1].replace("be ", "is ", 1)
    op_word = _OP_WORDS.get(condition.op, condition.op)
    shown = "unknown" if value is None or _is_missing(value) else _fmt(float(value))
    return f"{label} is {op_word} {_fmt(float(condition.value))} (here: {shown})"


def condition_checks(rule: Rule, facts: Mapping[str, Any]) -> list[tuple[bool, str]]:
    """Each condition of `rule` as `(satisfied, plain sentence)` for a tick/cross list."""
    fact_dict = dict(facts)
    return [(c.evaluate(fact_dict), _condition_phrase(c, fact_dict)) for c in rule.conditions]


def _plain_reason(rule: Rule, facts: Mapping[str, Any]) -> str:
    """Why a rule fired, in words, from its own conditions and the seller's real values."""
    if not rule.conditions:
        return (
            "no rule found a strong signal, so the default is to keep the seller "
            "and wait for more data"
        )
    return ", and ".join(_condition_phrase(c, facts) for c in rule.conditions)


def action_sentence(action: str, fired_rule: Rule, facts: Mapping[str, Any] | None = None) -> str:
    """Plain meaning of a recommendation plus why.

    With `facts`, the reason is built from the fired rule's own conditions and
    the seller's actual values (so it is always live and free of project
    jargon). Without `facts`, the rule's recorded rationale is used instead.
    """
    meaning = ACTION_MEANING.get(action, f"Recommended action: {action}.")
    reason = _plain_reason(fired_rule, facts) if facts is not None else fired_rule.because
    return f"{meaning} Why: {reason.rstrip('.')}."


@dataclass(frozen=True)
class ConditionGap:
    """One unmet condition of a rule, with how far the seller is from it."""

    fact: str
    op: str
    threshold: Any
    value: Any
    gap: float | None
    sentence: str


@dataclass(frozen=True)
class NearMiss:
    """A higher-priority rule that did not fire, and what it would take."""

    rule_id: str
    action: str
    priority: int
    gaps: tuple[ConditionGap, ...]

    @property
    def sentences(self) -> tuple[str, ...]:
        """The plain-words gap for each unmet condition."""
        return tuple(g.sentence for g in self.gaps)


def _condition_gap(
    condition: Condition, facts: Mapping[str, Any], action: str
) -> ConditionGap | None:
    """Describe how a seller misses one condition; `None` if it is satisfied."""
    fact_dict = dict(facts)
    if condition.evaluate(fact_dict):
        return None

    value = fact_dict.get(condition.fact)
    phrase = _ACTION_PHRASE.get(action, f"treated as {action}")
    label = _FACT_LABELS.get(condition.fact, _humanise(condition.fact))
    lead = f"To be {phrase},"

    if isinstance(condition.value, bool):
        needs_yes, needs_no = _BOOLEAN_NEEDS.get(
            condition.fact, (f"have {label}", f"not have {label}")
        )
        need = needs_yes if condition.value else needs_no
        if _is_missing(value):
            status = "this seller has no value for it"
        else:
            status = "for this seller that is currently " + ("true" if value else "not true")
        sentence = f"{lead} the seller would need to {need}; {status}."
        return ConditionGap(condition.fact, condition.op, condition.value, value, None, sentence)

    op_word = _OP_WORDS.get(condition.op, condition.op)
    threshold = _fmt(float(condition.value))
    if value is None or _is_missing(value):
        sentence = (
            f"{lead} {label} would need to be {op_word} {threshold}; "
            f"this seller has no {label} yet."
        )
        return ConditionGap(condition.fact, condition.op, condition.value, value, None, sentence)

    sentence = (
        f"{lead} {label} would need to be {op_word} {threshold}; "
        f"this seller is at {_fmt(float(value))}."
    )
    gap = abs(float(value) - float(condition.value))
    return ConditionGap(condition.fact, condition.op, condition.value, value, gap, sentence)


def near_miss(facts: Mapping[str, Any], rules: Sequence[Rule], limit: int = 3) -> list[NearMiss]:
    """Closest higher-priority rules that did NOT fire, with the gap per failed condition.

    The fired rule is the highest-priority matching rule (ties by rule id), the
    same resolution as `InferenceEngine`. Candidates are rules with a higher
    priority and a DIFFERENT action (a rule that would give the same
    recommendation would not change anything). Ordered by (fewest unmet
    conditions, higher priority, rule id); only the closest rule per action is
    kept so the list is short and varied. Pure and deterministic.
    """
    fact_dict = dict(facts)
    matched = [r for r in rules if r.matches(fact_dict)]
    if not matched:
        return []
    fired = sorted(matched, key=lambda r: (-r.priority, r.id))[0]

    candidates: list[NearMiss] = []
    for rule in rules:
        if rule.priority <= fired.priority or rule.action == fired.action:
            continue
        gaps = tuple(
            gap
            for condition in rule.conditions
            if (gap := _condition_gap(condition, fact_dict, rule.action)) is not None
        )
        candidates.append(NearMiss(rule.id, rule.action, rule.priority, gaps))

    candidates.sort(key=lambda n: (len(n.gaps), -n.priority, n.rule_id))
    closest_per_action: dict[str, NearMiss] = {}
    for candidate in candidates:
        closest_per_action.setdefault(candidate.action, candidate)
    ordered = sorted(
        closest_per_action.values(), key=lambda n: (len(n.gaps), -n.priority, n.rule_id)
    )
    return ordered[:limit]


# --------------------------------------------------------------------------
# One-sentence "Answer" for each page header
# --------------------------------------------------------------------------


def intelligence_answer(funnel: pd.DataFrame, late_rate: float) -> str:
    """Headline for the Intelligence page."""
    return (
        f"{funnel_sentence(funnel)} Meanwhile {late_rate * 100:.1f}% of delivered items "
        "arrive late."
    )


def design_answer(auc: float, base_rate: float) -> str:
    """Headline for the Design page."""
    return (
        f"Only about {base_rate * 100:.1f}% of items arrive late, and the best model ranks a late "
        f"order above an on-time one about {auc * 100:.0f}% of the time (50% is a coin flip)."
    )


def choice_answer(n_sellers: int, min_orders: int) -> str:
    """Headline for the Choice page."""
    return (
        f"{n_sellers:,} established sellers (each with at least {min_orders} orders) are ranked "
        "on reviews, on-time delivery, price and volume; you can change how much each matters."
    )


def discovery_answer(n_gems: int, n_low_confidence: int, min_orders: int) -> str:
    """Headline for the Discovery page."""
    return (
        f"{n_gems:,} sellers look good but are overlooked; {n_low_confidence:,} of them have "
        f"fewer than {min_orders} orders, so treat those as leads, not proof."
    )


def architecture_answer(n_tables: int, n_rules: int) -> str:
    """Headline for the DSS Architecture page."""
    return (
        f"The system turns {n_tables} data tables into a risk score, a seller ranking and one of "
        f"five recommendations, using {n_rules} rules a person can read."
    )
