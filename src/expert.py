"""Forward-chaining rule-based expert system for seller recommendations
(CLAUDE.md section 7.5).

Four classes:
- `KnowledgeBase` loads and validates `rules/seller_rules.yaml` (rejects
  duplicate rule ids, unknown fields, and unknown operators at load time).
- `WorkingMemory` holds one seller's facts: `ahp_score`, `avg_late_risk`,
  `avg_review`, `order_volume`, `late_rate`, `is_hidden_gem` — the exact
  contract produced by `src.discovery.build_seller_facts`.
- `InferenceEngine` evaluates every rule against a fact set (match), and
  fires the single highest-`priority` match, tie-breaking by ascending
  rule `id` (resolve). A knowledge base without a matching default rule
  raises, since every seller must get a recommendation.
- `ExplanationFacility` renders a fired rule's rationale and the specific
  fact values that satisfied it, in plain English.

`ahp_score` (and, for a handful of very-low-volume sellers, other facts)
may be `None`/NaN: a seller with fewer than `config.DISCOVERY_MIN_ORDERS`
orders has no AHP score (see `src.discovery`). Any condition on a missing
fact simply fails to match rather than raising, so the low-priority default
rule still catches those sellers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, RULES_DIR, get_logger

logger = get_logger(__name__)

_VALID_OPERATORS: frozenset[str] = frozenset({"<", "<=", ">", ">=", "==", "!="})
_REQUIRED_RULE_FIELDS: frozenset[str] = frozenset({"id", "priority", "if", "then", "because"})
_VALID_CONDITION_FIELDS: frozenset[str] = frozenset({"fact", "op", "value"})

_OPERATORS: dict[str, Any] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _is_missing(value: Any) -> bool:
    """Return True if `value` is None or a float NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


@dataclass(frozen=True)
class Condition:
    """One `if` clause: `fact <op> value`."""

    fact: str
    op: str
    value: Any

    def evaluate(self, facts: dict[str, Any]) -> bool:
        """Evaluate this condition against a fact set.

        Args:
            facts: mapping of fact name to value (may be missing or None).

        Returns:
            False if the fact is absent or missing (None/NaN); otherwise
            the result of applying `op` to the fact's value and `value`.
        """
        fact_value = facts.get(self.fact)
        if _is_missing(fact_value):
            return False
        return bool(_OPERATORS[self.op](fact_value, self.value))


@dataclass(frozen=True)
class Rule:
    """One knowledge-base rule: conditions, action, priority, and rationale."""

    id: str
    priority: int
    conditions: tuple[Condition, ...]
    action: str
    because: str

    def matches(self, facts: dict[str, Any]) -> bool:
        """Return True iff every condition in this rule is satisfied by `facts`."""
        return all(condition.evaluate(facts) for condition in self.conditions)


class KnowledgeBase:
    """Loads and validates a set of rules."""

    def __init__(self, rules: tuple[Rule, ...]) -> None:
        self.rules = rules

    @classmethod
    def from_yaml(cls, path: Path) -> KnowledgeBase:
        """Load and validate rules from a YAML file.

        Args:
            path: path to a YAML file containing a list of rule mappings.

        Returns:
            A `KnowledgeBase` holding the parsed, validated rules.

        Raises:
            ValueError: on a duplicate rule id, an unknown or missing
                field, a malformed condition, or an unknown operator.
        """
        with path.open("r", encoding="utf-8") as fh:
            raw_rules = yaml.safe_load(fh)
        if not isinstance(raw_rules, list):
            raise ValueError(f"{path}: expected a YAML list of rules, got {type(raw_rules)}")

        seen_ids: set[str] = set()
        rules: list[Rule] = []
        for entry in raw_rules:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: each rule must be a mapping, got {entry!r}")

            missing_fields = _REQUIRED_RULE_FIELDS - set(entry)
            if missing_fields:
                raise ValueError(f"{path}: rule {entry} missing required field(s) {missing_fields}")
            unknown_fields = set(entry) - _REQUIRED_RULE_FIELDS
            if unknown_fields:
                raise ValueError(
                    f"{path}: rule {entry['id']} has unknown field(s) {unknown_fields}"
                )

            rule_id = str(entry["id"])
            if rule_id in seen_ids:
                raise ValueError(f"{path}: duplicate rule id {rule_id!r}")
            seen_ids.add(rule_id)

            if not isinstance(entry["if"], list):
                raise ValueError(f"{path}: rule {rule_id}'s 'if' must be a list")

            conditions = []
            for cond in entry["if"]:
                if not isinstance(cond, dict) or set(cond) != _VALID_CONDITION_FIELDS:
                    raise ValueError(f"{path}: rule {rule_id} has a malformed condition {cond!r}")
                if cond["op"] not in _VALID_OPERATORS:
                    raise ValueError(
                        f"{path}: rule {rule_id} has unknown operator {cond['op']!r}"
                    )
                conditions.append(Condition(fact=cond["fact"], op=cond["op"], value=cond["value"]))

            rules.append(
                Rule(
                    id=rule_id,
                    priority=int(entry["priority"]),
                    conditions=tuple(conditions),
                    action=str(entry["then"]),
                    because=str(entry["because"]),
                )
            )

        return cls(tuple(rules))


@dataclass
class WorkingMemory:
    """Facts for one seller, as consumed by the inference engine."""

    seller_id: str
    ahp_score: float | None
    avg_late_risk: float | None
    avg_review: float | None
    order_volume: float | None
    late_rate: float | None
    is_hidden_gem: bool

    def as_facts(self) -> dict[str, Any]:
        """Return this seller's facts as a plain dict, keyed by fact name."""
        return {
            "ahp_score": self.ahp_score,
            "avg_late_risk": self.avg_late_risk,
            "avg_review": self.avg_review,
            "order_volume": self.order_volume,
            "late_rate": self.late_rate,
            "is_hidden_gem": self.is_hidden_gem,
        }


@dataclass(frozen=True)
class InferenceResult:
    """The outcome of running the inference engine against one fact set."""

    facts: dict[str, Any]
    fired_rule: Rule
    trace: tuple[tuple[Rule, bool], ...]


class InferenceEngine:
    """Match-resolve-act over a `KnowledgeBase`."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    def run(self, facts: dict[str, Any]) -> InferenceResult:
        """Evaluate every rule and fire the highest-priority match.

        Args:
            facts: fact set to evaluate (e.g. `WorkingMemory.as_facts()`).

        Returns:
            An `InferenceResult` holding the fired rule and the full trace
            of every rule evaluated (rule, matched).

        Raises:
            RuntimeError: if no rule matches, meaning the knowledge base is
                missing a default (always-true) rule.
        """
        trace = tuple((rule, rule.matches(facts)) for rule in self.knowledge_base.rules)
        matched = [rule for rule, ok in trace if ok]
        if not matched:
            raise RuntimeError(
                "No rule matched these facts; the knowledge base must include a "
                "default rule with no conditions so every seller gets a recommendation."
            )
        best_priority = max(rule.priority for rule in matched)
        top_candidates = sorted(
            (rule for rule in matched if rule.priority == best_priority), key=lambda r: r.id
        )
        return InferenceResult(facts=facts, fired_rule=top_candidates[0], trace=trace)


class ExplanationFacility:
    """Renders an `InferenceResult` as a plain-English explanation."""

    @staticmethod
    def explain(result: InferenceResult) -> str:
        """Return a plain-English explanation of the fired rule.

        Args:
            result: an `InferenceResult` from `InferenceEngine.run`.

        Returns:
            A multi-line string: the recommendation, the rule's rationale,
            and the fact values that satisfied each condition.
        """
        rule = result.fired_rule
        if rule.conditions:
            condition_lines = [
                f"{cond.fact} = {result.facts.get(cond.fact)!r} {cond.op} {cond.value!r}"
                for cond in rule.conditions
            ]
            matched_on = "; ".join(condition_lines)
        else:
            matched_on = "no conditions (default rule)"
        return (
            f"Recommendation: {rule.action} (rule {rule.id}, priority {rule.priority}).\n"
            f"Because: {rule.because}\n"
            f"Matched on: {matched_on}"
        )

    @staticmethod
    def trace_summary(result: InferenceResult) -> list[dict[str, Any]]:
        """Return the full rule-evaluation trace as a list of plain dicts.

        Args:
            result: an `InferenceResult` from `InferenceEngine.run`.

        Returns:
            One dict per rule in evaluation order: `rule_id`, `priority`,
            `matched` (bool), `action`, and whether it `fired`.
        """
        return [
            {
                "rule_id": rule.id,
                "priority": rule.priority,
                "matched": matched,
                "action": rule.action,
                "fired": rule.id == result.fired_rule.id,
            }
            for rule, matched in result.trace
        ]


def _facts_from_row(row: pd.Series) -> dict[str, Any]:
    """Convert one SellerFacts row into a facts dict, mapping NaN to None."""

    def _or_none(value: Any) -> Any:
        return None if pd.isna(value) else float(value)

    return {
        "ahp_score": _or_none(row["ahp_score"]),
        "avg_late_risk": _or_none(row["avg_late_risk"]),
        "avg_review": _or_none(row["avg_review"]),
        "order_volume": _or_none(row["order_volume"]),
        "late_rate": _or_none(row["late_rate"]),
        "is_hidden_gem": bool(row["is_hidden_gem"]),
    }


def main() -> None:
    """Run every seller's facts through the expert system and write recommendations."""
    seller_facts = pd.read_parquet(PROCESSED_DATA_DIR / "SellerFacts.parquet")
    knowledge_base = KnowledgeBase.from_yaml(RULES_DIR / "seller_rules.yaml")
    engine = InferenceEngine(knowledge_base)

    rows = []
    for _, row in seller_facts.iterrows():
        result = engine.run(_facts_from_row(row))
        rows.append(
            {
                "seller_id": row["seller_id"],
                "action": result.fired_rule.action,
                "rule_id": result.fired_rule.id,
                "because": result.fired_rule.because,
            }
        )
    recommendations = pd.DataFrame(rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(REPORTS_DIR / "seller_recommendations.csv", index=False)

    action_counts = recommendations["action"].value_counts().to_dict()
    logger.info("Wrote seller_recommendations.csv (%d sellers): %s", len(recommendations),
                action_counts)


if __name__ == "__main__":
    main()
