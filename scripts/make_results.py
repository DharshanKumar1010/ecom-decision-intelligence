"""Generate `reports/RESULTS.md`, the single place the project's numbers are quoted from.

Every number is read from a file the pipeline wrote (reports/, data/export/, data/processed/,
rules/seller_rules.yaml, config) or from the committed diagnostic reports; nothing is typed in
here. The output has no timestamps, so it is identical on every run for the same inputs and a
test can check that the committed file is current.

    python scripts/make_results.py            # write reports/RESULTS.md
    python scripts/make_results.py --check    # exit 1 if the committed file is out of date

No database is touched. The live comparison of SQL Server with these CSVs is
`python -m src.load_sql --verify-only` (also run by `.\\tasks.ps1 verify`).
"""

# ruff: noqa: E501  (the report prose is written as long literal lines)

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ahp, config, expert  # noqa: E402
from src.load_sql import CSV_FILES, TABLES, parse_docs_counts  # noqa: E402

RESULTS_PATH = config.REPORTS_DIR / "RESULTS.md"
DIAG = config.REPORTS_DIR / "diagnostics"
_MODEL_ORDER = ("random_forest", "decision_tree", "logistic_regression")
_BANDS = (("fewer than 5", 0, 4), ("5 to 14", 5, 14), ("15 or more", 15, 10**9))


class ResultsError(RuntimeError):
    """A source file is missing or no longer has the format this script reads."""


def _json(path: Path) -> dict:
    if not path.exists():
        raise ResultsError(f"{path} not found; run the pipeline first")
    return json.loads(path.read_text(encoding="utf-8"))


def _text(path: Path) -> str:
    if not path.exists():
        raise ResultsError(f"{path} not found")
    return path.read_text(encoding="utf-8")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def _csv_rows(table: str) -> int:
    path = config.EXPORT_DATA_DIR / CSV_FILES[table]
    if not path.exists():
        raise ResultsError(f"{path} not found; run python -m src.export_sql")
    return len(pd.read_csv(path, usecols=[0], dtype=str, keep_default_na=False))


def _marketplace_section() -> list[str]:
    fact = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "FactOrderItems.parquet",
        columns=["order_id", "seller_id", "price", "is_late", "review_score"],
    )
    per_pair = float(ahp.seller_order_table(fact)["review"].mean())
    per_item = float(fact["review_score"].mean())
    per_order = float(fact.drop_duplicates("order_id")["review_score"].mean())
    return [
        "## Marketplace (real Olist data)",
        "",
        f"{fact['order_id'].nunique():,} delivered orders, {len(fact):,} items sold, item revenue "
        f"R$ {fact['price'].sum():,.2f} (excluding freight), late-delivery rate "
        f"{fact['is_late'].mean() * 100:.2f}% of items. Average review per seller-order pair: "
        f"**{per_pair:.2f}** (one review per pair, an order with items from two sellers counts "
        f"once for each seller; the definition behind every seller's average review). Averaged "
        f"over distinct reviewed orders it is {per_order:.2f}; counting a review once per item "
        f"would give {per_item:.2f}. Only the first is used. Source `FactOrderItems.parquet`.",
        "",
    ]


def _models_section() -> list[str]:
    metrics = _json(config.REPORTS_DIR / "model_metrics.json")
    oof = _json(config.REPORTS_DIR / "oof_metrics.json")
    registry = _json(config.MODELS_DIR / "registry.json")
    primary = registry["primary_model"]
    split = oof["split"]
    lines = [
        "## Late-delivery risk models",
        "",
        f"Evaluation is order-grouped: {oof['n_folds']} folds over `order_id`, so no order is on both "
        f"sides. Fold 0 is the train/test split ({split['train_items']:,} / {split['test_items']:,} "
        f"items; {split['train_orders']:,} / {split['test_orders']:,} orders); the five folds together "
        "give the out-of-fold scores stored in `LatePredictions`. Base rate of late items "
        f"{metrics[primary]['base_rate'] * 100:.2f}% on the test fold. Source: "
        "`reports/model_metrics.json`, `reports/oof_metrics.json`, `models/registry.json`.",
        "",
    ]
    rows = []
    for name in _MODEL_ORDER:
        m = oof["models"][name]
        role = "primary" if name == primary else "kept for interpretability"
        rows.append(
            [
                f"`{name}`",
                role,
                f"{m['grouped_test_auc']:.4f}",
                f"{m['oof_auc']:.4f}",
                f"{m['oof_fold_auc_mean']:.4f} ± {m['oof_fold_auc_std']:.4f}",
                f"{m['in_sample_auc_train_rows']:.4f}",
            ]
        )
    lines += _table(
        [
            "model",
            "role",
            "test AUC (fold 0)",
            "out-of-fold AUC",
            "fold mean ± std",
            "AUC on its own training rows",
        ],
        rows,
    )
    rf = metrics[primary]
    lines += [
        f"At the default cut-off the primary model flags {rf['recall']['1'] * 100:.0f}% of late items "
        f"(recall {rf['recall']['1']:.2f}) with precision {rf['precision']['1']:.2f}. "
        f"`late_risk` is a ranking score, not a probability: with balanced class weights its median "
        f"({_late_risk_median():.2f}) is far above the {rf['base_rate'] * 100:.1f}% late rate.",
        "",
    ]
    return lines + _superseded_section() + _correlation_section()


def _late_risk_median() -> float:
    late = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "LatePredictions.parquet", columns=["late_risk"]
    )
    return float(late["late_risk"].median())


def _superseded_section() -> list[str]:
    leak = _text(DIAG / "diagnostic_group_leak.txt")
    cv = _text(DIAG / "diagnostic_rf_cv.txt")
    rows = []
    for name in _MODEL_ORDER:
        m = re.search(rf"^{name}\s+(\d\.\d+)\s+(\d\.\d+)", leak, re.MULTILINE)
        c = re.search(
            rf"^\s*{name.replace('_', '')}\s+(\d\.\d+) \+/- (\d\.\d+)",
            cv,
            re.MULTILINE | re.IGNORECASE,
        )
        if not m or not c:
            raise ResultsError(f"cannot read the historical numbers for {name}")
        rows.append([f"`{name}`", m.group(1), m.group(2), f"{c.group(1)} ± {c.group(2)}"])
    return [
        "### Superseded numbers (do not quote)",
        "",
        "Earlier stages split item rows at random and used a fully grown forest. Items of one order "
        "share the same outcome, so siblings on both sides of the split inflated the scores, most of "
        "all for the forest. Kept for the record; sources `reports/diagnostics/diagnostic_group_leak.txt` "
        "and `diagnostic_rf_cv.txt`.",
        "",
        *_table(
            [
                "model",
                "random item split (superseded)",
                "grouped, forest leaf 1 (superseded)",
                "random-fold 5-fold CV (superseded)",
            ],
            rows,
        ),
    ]


def _correlation_section() -> list[str]:
    facts = pd.read_parquet(config.PROCESSED_DATA_DIR / "SellerFacts.parquet")
    everyone = facts["avg_late_risk"].corr(facts["late_rate"])
    enough = facts[facts["order_volume"] >= config.DISCOVERY_MIN_ORDERS]
    established = enough["avg_late_risk"].corr(enough["late_rate"])
    return [
        "### Seller-level agreement with what really happened",
        "",
        f"Correlation between a seller's average out-of-fold risk and its actual late rate: "
        f"{everyone:.2f} over all {len(facts):,} sellers, {established:.2f} over the "
        f"{len(enough):,} sellers with at least {config.DISCOVERY_MIN_ORDERS} items sold "
        "(source `SellerFacts.parquet`).",
        "",
    ]


def _temporal_section() -> list[str]:
    text = _text(DIAG / "diagnostic_temporal.txt")
    split = re.findall(r"^(train|test)\s+= .*?late rate ([\d.]+)%", text, re.MULTILINE)
    rows = []
    for label, pattern in (
        ("all 12 features", r"^all 12 features"),
        ("without `order_month` and `day_of_week`", r"^without order_month"),
    ):
        m = re.search(pattern + r".*?(\d\.\d{4})\s+(\d\.\d{4})\s+(\d\.\d{4})", text, re.MULTILINE)
        if not m:
            raise ResultsError("cannot read diagnostic_temporal.txt")
        rows.append([label, m.group(1), m.group(2), m.group(3)])
    rates = dict(split)
    return [
        "## Forward-looking check (train on the earliest 80% of orders, test on the latest 20%)",
        "",
        f"Late rate {rates.get('train', '?')}% in the training period against {rates.get('test', '?')}% in "
        "the test period. An AUC of 0.5 is chance. Source `reports/diagnostics/diagnostic_temporal.txt`.",
        "",
        *_table(["inputs", "logistic regression", "decision tree", "random forest"], rows),
        "No model has useful forward-looking skill; the risk score ranks orders within the period "
        "it was built on and does not forecast future deliveries.",
        "",
    ]


def _leaf_section() -> list[str]:
    text = _text(DIAG / "diagnostic_rf_leaf.txt")
    rows = []
    for m in re.finditer(
        r"^\s*(\d+)\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)\s+[+\-]\d\.\d+\s+([\d.]+)\s+[\d.]+%",
        text,
        re.MULTILINE,
    ):
        rows.append([m.group(1), m.group(2), f"{m.group(3)} ± {m.group(4)}", f"{m.group(5)} MB"])
    if not rows:
        raise ResultsError("cannot read diagnostic_rf_leaf.txt")
    saved = config.MODELS_DIR / "random_forest.joblib"
    size = f"{saved.stat().st_size / 1e6:.1f} MB" if saved.exists() else "not present"
    return [
        "## Forest leaf size (`min_samples_leaf`)",
        "",
        f"Adopted: {config.PREDICT_RF_MIN_SAMPLES_LEAF} (`config.PREDICT_RF_MIN_SAMPLES_LEAF`). "
        f"Saved forest on disk: {size}. Source `reports/diagnostics/diagnostic_rf_leaf.txt`.",
        "",
        *_table(["leaf", "pooled out-of-fold AUC", "fold mean ± std", "fitted size"], rows),
    ]


def _tables_section() -> list[str]:
    docs = parse_docs_counts()
    rows, total = [], 0
    for table in TABLES:
        n = _csv_rows(table)
        total += n
        rows.append(
            [
                f"`{table}`",
                f"`{CSV_FILES[table]}`",
                f"{n:,}",
                f"{docs.get(table, 0):,}",
                "yes" if docs.get(table) == n else "NO",
            ]
        )
    rows.append(
        ["**total**", "", f"**{total:,}**", f"**{sum(docs.get(t, 0) for t in TABLES):,}**", ""]
    )
    return [
        "## Tables exported for SQL Server and Power BI",
        "",
        "Rows in each `data/export/*.csv` against the count recorded in `docs/powerbi_setup.md`. "
        "`python -m src.load_sql` refuses to load while they disagree and checks the same counts in "
        "SQL Server; `python -m src.load_sql --verify-only` repeats that check without changing anything.",
        "",
        *_table(["SQL table", "CSV", "rows", "rows in docs", "match"], rows),
    ]


def _gem_outcomes_line(gems: pd.DataFrame) -> str:
    """What the expert system finally recommends for the hidden gems, and why most do not Promote."""
    facts = pd.read_parquet(config.PROCESSED_DATA_DIR / "SellerFacts.parquet")
    recs = pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv")
    kb = expert.KnowledgeBase.from_yaml(config.RULES_DIR / "seller_rules.yaml")
    r08 = next(r for r in kb.rules if r.id == "R08")
    cutoff = float(next(c.value for c in r08.conditions if c.fact == "avg_late_risk"))
    merged = gems[["seller_id", "confidence"]].merge(
        recs[["seller_id", "action"]], on="seller_id"
    ).merge(facts[["seller_id", "avg_late_risk", "order_volume", "n_orders"]], on="seller_id")
    acts = merged["action"].value_counts()
    promote = merged[merged["action"] == "Promote"]
    return (
        f"Final recommendation of the {len(merged):,} hidden gems: "
        + ", ".join(f"{a} {int(acts.get(a, 0)):,}" for a in ("Promote", "Keep", "Feature", "Warn"))
        + f". Of the {len(promote):,} Promote gems {int((promote['confidence'] == 'normal').sum()):,} "
        f"have stronger evidence and {int((promote['confidence'] == 'low').sum()):,} an early signal; "
        f"{int((promote['n_orders'] < 3).sum()):,} have fewer than 3 delivered orders (Promote is not "
        f"gated on orders). {int((merged['avg_late_risk'] >= cutoff).sum()):,} gems have a risk score at "
        f"or above R08's cut-off ({cutoff:g}). Median items sold of a hidden gem: "
        f"{merged['order_volume'].median():g}."
    )


def _sellers_section() -> list[str]:
    quad = pd.read_csv(config.REPORTS_DIR / "discovery_quadrants.csv")
    ranking = pd.read_csv(config.REPORTS_DIR / "ahp_ranking.csv")
    prior = pd.read_csv(config.REPORTS_DIR / "ahp_prior_strength.csv")
    _, _, _, cr = ahp.compute_weights(config.AHP_PAIRWISE_MATRIX)
    counts = quad["quadrant"].value_counts()
    gems = quad[quad["quadrant"] == "Hidden Gem"]
    tiers = gems["confidence"].value_counts()
    lines = [
        "## Sellers",
        "",
        f"AHP ranking: {len(ranking):,} established sellers (at least {config.AHP_MIN_ORDERS} items "
        f"sold), consistency ratio {cr:.4f} (limit {config.AHP_CR_THRESHOLD}). "
        f"Quadrants cover {len(quad):,} sellers with at least {config.DISCOVERY_MIN_ORDERS} items sold.",
        "",
        *_table(
            ["quadrant", "sellers"],
            [
                [q, f"{int(counts.get(q, 0)):,}"]
                for q in ("Star", "Hidden Gem", "Overrated", "Overlooked-Low-Quality")
            ],
        ),
        f"Hidden gems by evidence: {int(tiers.get('normal', 0)):,} with at least "
        f"{config.DISCOVERY_CONFIDENCE_MIN_ORDERS} items sold (stronger evidence), "
        f"{int(tiers.get('low', 0)):,} with fewer (early signal). Hidden gems are directional, not proven.",
        "",
        _gem_outcomes_line(gems),
        "",
        "### Empirical-Bayes shrinkage of the quality score",
        "",
        *_table(
            [
                "criterion",
                "sellers",
                "within variance",
                "between variance",
                "k estimated",
                "k configured",
            ],
            [
                [
                    r.criterion,
                    f"{int(r.sellers):,}",
                    f"{r.within_variance:.4f}",
                    f"{r.between_variance:.4f}",
                    f"{r.k_estimate:.2f}",
                    str(int(r.k_configured)),
                ]
                for r in prior.itertuples()
            ],
        ),
    ]
    return lines


def rule_condition_text(rule: expert.Rule) -> str:
    """A rule's conditions as one canonical string, e.g. `avg_review < 3.5 and order_volume >= 5`."""
    return " and ".join(f"{c.fact} {c.op} {c.value}" for c in rule.conditions) or "(always)"


_EVIDENCE_COLUMNS = (
    ("order_volume", "items sold"),
    ("n_orders", "delivered orders"),
    ("n_reviewed", "reviewed orders"),
)


def _minimum_evidence_line(facts: pd.DataFrame, recs: pd.DataFrame) -> str:
    """The sentence on sellers with too little evidence for Warn or Suspend (items and orders)."""
    merged = recs.merge(facts[["seller_id", "order_volume", "n_orders", "n_reviewed"]], on="seller_id")
    thin_items = merged[merged["order_volume"] < 5]
    thin = merged[(merged["order_volume"] < 5) | (merged["n_orders"] < 3)]
    acts = ", ".join(f"{k} {v:,}" for k, v in thin["action"].value_counts().items())
    flagged = merged["action"].isin(["Warn", "Suspend"])
    return (
        f"Minimum evidence: {len(thin_items):,} sellers have fewer than 5 items sold and "
        f"{len(thin):,} have fewer than 5 items sold or fewer than 3 delivered orders; the "
        f"actions of the latter are {acts}, and {int((thin['rule_id'] == 'R12').sum()):,} of them "
        "sit in the default Keep rule. Warn or Suspend for those sellers: "
        f"{int(flagged[thin.index].sum())}. Warn or Suspend with fewer than 3 delivered orders: "
        f"{int((flagged & (merged['n_orders'] < 3)).sum())}; rules R01 or R04 firing with fewer "
        f"than 3 reviewed orders: "
        f"{int((merged['rule_id'].isin(['R01', 'R04']) & (merged['n_reviewed'] < 3)).sum())}."
    )


def _rules_section() -> list[str]:
    kb = expert.KnowledgeBase.from_yaml(config.RULES_DIR / "seller_rules.yaml")
    facts = pd.read_parquet(config.PROCESSED_DATA_DIR / "SellerFacts.parquet")
    recs = pd.read_csv(config.REPORTS_DIR / "seller_recommendations.csv")
    fact_dicts = [expert._facts_from_row(r) for _, r in facts.iterrows()]
    fired = recs["rule_id"].value_counts()
    actions = recs["action"].value_counts()
    rows = []
    for rule in sorted(kb.rules, key=lambda r: -r.priority):
        eligible = facts
        parts = []
        for column, name in _EVIDENCE_COLUMNS:
            gate = [c.value for c in rule.conditions if c.fact == column and c.op == ">="]
            if gate:
                eligible = eligible[eligible[column] >= max(gate)]
                parts.append(f"{name} ≥ {int(max(gate))}")
        population = len(eligible)
        label = ", ".join(parts) if parts else "all sellers"
        matches = sum(1 for f in fact_dicts if rule.matches(f))
        cond = rule_condition_text(rule)
        rows.append(
            [
                rule.id,
                rule.action,
                str(rule.priority),
                f"`{cond}`",
                f"{matches:,}",
                f"{population:,} ({label})",
                f"{int(fired.get(rule.id, 0)):,}",
            ]
        )
    return [
        "## Recommendations (expert system)",
        "",
        f"{len(recs):,} sellers. Actions: "
        + ", ".join(
            f"{a} {int(actions.get(a, 0)):,}"
            for a in ("Keep", "Warn", "Suspend", "Feature", "Promote")
        )
        + ".",
        "",
        "Rules in priority order. *matches* counts sellers whose conditions are all true; *fires* "
        "counts sellers whose final recommendation is that rule (a higher-priority rule can take "
        "over). Thresholds are read from `rules/seller_rules.yaml`, never typed elsewhere.",
        "",
        *_table(
            [
                "rule",
                "action",
                "priority",
                "conditions",
                "matches",
                "reference population",
                "fires",
            ],
            rows,
        ),
        _minimum_evidence_line(facts, recs),
        "",
    ]


def build_results() -> str:
    """The full text of RESULTS.md for the current files (deterministic)."""
    parts = [
        "# Results",
        "",
        "Generated by `scripts/make_results.py` from the files named under each heading. Do not edit "
        "by hand; regenerate with `python scripts/make_results.py` (a test fails if this file is "
        "out of date). Other documents point here instead of repeating these numbers.",
        "",
        *_marketplace_section(),
        *_models_section(),
        *_temporal_section(),
        *_leaf_section(),
        *_tables_section(),
        *_sellers_section(),
        *_rules_section(),
    ]
    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write (or with --check, verify) reports/RESULTS.md."""
    parser = argparse.ArgumentParser(description="Generate reports/RESULTS.md")
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is out of date")
    args = parser.parse_args(argv)
    text = build_results()
    if args.check:
        current = (
            RESULTS_PATH.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if RESULTS_PATH.exists()
            else ""
        )
        if current != text:
            sys.stderr.write(
                "reports/RESULTS.md is out of date; run python scripts/make_results.py\n"
            )
            return 1
        return 0
    RESULTS_PATH.write_bytes(text.encode("utf-8"))
    sys.stdout.write(f"wrote {RESULTS_PATH} ({len(text.splitlines())} lines)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
