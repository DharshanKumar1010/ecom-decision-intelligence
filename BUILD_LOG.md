# Build log

Short summary of how the project got here. The complete stage-by-stage log (661 lines, with the
real output of every run) is `docs/archive/BUILD_LOG_full.md`. Numbers are not repeated here:
every figure is in `reports/RESULTS.md`, generated from the pipeline outputs. What each stage
decided, and why, is below; the reasoning behind each threshold is in `docs/knowledge_engineering.md`.

## Environment notes

- Windows 11, Python 3.11, PowerShell as the main shell; `tasks.ps1` is the task runner (`Makefile` mirrors most targets).
- Windows Smart App Control (reputation-based) blocked freshly installed compiled packages at import:
  pandas 3.0.6 and mypy 2.3.1 were replaced by pandas 2.2.3 and mypy 1.13.0; scikit-learn's `.pyd` files were
  blocked for a while and the block cleared on its own. Nothing was bypassed (no `Unblock-File`, no policy change).
  Details in the archive, section "scikit-learn Application Control block".
- SQL: SQL Server Express (`localhost\SQLEXPRESS`), database `EcomDecisionIntelligence`, Microsoft ODBC Driver 17 and 18
  installed, `pyodbc` 5.3.0 imported without trouble. The loader uses Windows authentication and
  `TrustServerCertificate=yes` (a client-side setting for the local self-signed certificate; not for remote servers).
- A clean rebuild (`python run_all.py --export` in a copy of the tree) takes about 9 to 10 minutes, almost all in the model
  fits, and is byte-identical from run to run apart from timestamps in `models/*_metadata.json`.

## Stages

**Stage 1, data layer.** Raw Olist tables to a delivered-orders fact table, dimensions and a data-quality report; seeded
synthetic clickstream and call transcripts (Olist has neither, so every use is labelled synthetic); funnel metrics.

**Stage 2, models and rules.** VADER call sentiment, a first late-delivery model on four features (weak, and said so), AHP
seller scoring with a consistency check, hidden-gem quadrants with an evidence flag, and a 12-rule YAML expert system with
explanations.

**Stage 3 and 3.5, features, export, forest.** Features widened to 12 (no date or delivery-derived column; a leakage guardrail
on AUC), a CSV export per SQL table, and RandomForest adopted as the primary model after a diagnostic and a cross-validation
check (logistic regression and decision tree kept for interpretability). Recalibrating the rules that read the risk score
followed. The cross-validation behind the adoption later turned out to be inflated (see Stage 6).

**Stages 4, 4b, 4d, dashboard.** Six Streamlit pages named after the decision phases, rewritten in plain language, simplified to one
template (dark, colour-blind-safe palette) with one "Technical details" expander per page; hidden gems split into
Stronger evidence and Early signal; a demo script with the questions a reviewer is likely to ask.

**Stages 5 and 5b, small-sample honesty.** Empirical-Bayes shrinkage of review and on-time rate toward the marketplace mean, first
per item, then per distinct order (a review belongs to an order); strengths measured from the data, not chosen. One definition of
"average review" (one review per distinct order) across facts, rules, Choice and Implementation. Wording changed to "items sold".

**Stage 6, order-grouped evaluation.** Two problems found and measured: items of one order sit on both sides of a random split
(sibling leakage), and the stored risk scores came from models that had trained on those very rows. Fixed with order-grouped
folds and out-of-fold scores; forest leaf size tuned under that protocol (fully grown trees were larger and worse); every
risk-based rule threshold re-derived at the same percentile; a temporal check showed no forward-looking skill. The model
family did not change.

**Stage 7, SQL loader.** The SSMS Import Flat File wizard would have rejected or truncated several columns (blank weights, an
em dash, long text). Owner-authorised exception to the "no SQL" rule for exactly three files: `sql/schema.sql` (typed tables, keys,
indexes, 8 foreign keys that all had zero orphans, `dbo.vw_SellerScorecard`), `src/load_sql.py` (typed read, allow-listed
statements, one transaction per table, count and checksum verification, `--dry-run`, `--verify-only`, `--only`) and its tests.
The old hand-imported `DimCustomer` and `DimProduct` were replaced.

**Stage 8, minimum evidence, verification and cleanup (this stage).**
- *Verified first, nothing changed:* a clean-room rebuild twice (identical), ruff, mypy, the whole test suite, data
  reconciliation from raw orders to SQL, model integrity (no order in train and test, out-of-fold scores reproduce from the saved
  model, AUCs under the guardrail), rule integrity (2,970 sellers x 12 rules parity between the engine and the explanations),
  and that no action changes when values are rounded the way SQL stores them.
- *One deliberate rule change.* R01 (Suspend) and R03 (Warn) had no volume gate and fired mostly on sellers with one to four
  items sold (half of R03's Warn sellers had never delivered late). Both now need `order_volume >= 5`; their risk thresholds were
  re-derived at the 75th and 90th percentile of the sellers they now apply to (0.416 to 0.412, 0.501 to 0.476). Suspend 143 to 76,
  Warn 325 to 241, Keep 2,118 to 2,269; 232 sellers changed action; all 1,106 sellers under 5 items sold are now Keep with the
  default rule. Nothing else changed: models, risk scores, AHP, quadrants and the 437 gems are byte-identical. SQL Server was
  reloaded and re-verified. Reasoning and sensitivity: `docs/knowledge_engineering.md` section 10.
- *Cleanup.* `reports/RESULTS.md` (generated) is now the only place numbers are quoted, with tests that fail when a document
  drifts; `CLAUDE.md`, `README.md` and this log were rewritten shorter; stale thresholds and counts removed; dead review-flag and
  CV-parser helpers deleted; the Implementation page opens on a seller with at least 5 items sold; the forest is now shown in
  `run_all`'s summary; diagnostics moved to `scripts/diagnostics/` and `reports/diagnostics/` and marked current or historical;
  `docs/SYLLABUS_MAP.md`, `docs/MANUAL_CHECKS.md` (browser checks and ten SSMS queries) and a docs index added;
  `tasks.ps1 verify` and `rebuild` added. Full account: `docs/archive/CLEANUP_REPORT.md`.
- *Follow-up.* The reason texts of R02, R04, R05 and R10 now say "items sold" instead of "orders" (only the `because` column of 273 recommendations changed; actions and rules identical; SQL reloaded and verified); the one test that failed without `data/raw` now skips; the Makefile has `load-sql`; "Try an example" opens on a seller with enough items sold. Details: `docs/archive/CLEANUP_REPORT.md`.
- *Final polish before Power BI.* Every Technical-details table now has plain-language headers, one number format and one
  risk colour pair; the average review on Intelligence is per order; Design says what the strongest input is and that nothing
  is forward-looking; Implementation opens on a seller with 5+ reviewed orders and shows reviewed and delivered orders;
  tier names read Stronger evidence / Early signal everywhere; `docs/powerbi_measures.md` added. No output, model or table
  changed (hash-compared). Details: `docs/archive/FINAL_POLISH_REPORT.md`.
- *Stage 9, evidence in orders.* Items of one order are not independent, so every Warn and Suspend rule now also needs 3
  delivered orders (R01, R02, R03, R05) and/or 3 reviewed orders (R01, R04), read from two new SellerFacts columns
  (`n_orders`, `n_reviewed`); R01's threshold re-derived (0.412 to 0.413), R03's unchanged. 8 sellers changed action; models,
  quadrants, AHP and `sql/schema.sql` byte-identical; SQL reloaded. See `docs/knowledge_engineering.md` section 11.

## Corrections that mattered

1. **Sibling leakage:** random item-level splits inflated the forest most; evaluation is order-grouped.
2. **In-sample risk scores:** seller-level risk was an echo of the seller's own late rate; scores are now out-of-fold.
3. **One review per order:** a review repeats on each item of its order; averages count distinct orders.
4. **Small-sample shrinkage:** a few perfect reviews no longer beat a long record; the hidden-gem list is tiered by evidence.
5. **No forward-looking skill:** the risk score ranks historical orders; it does not forecast and is not a probability.
6. **Small-seller warnings:** no Warn or Suspend on fewer than 5 items sold.
7. **A normalisation bug** that overwrote raw AHP criteria (found in Stage 2) is guarded by keeping `norm_*` columns separate.

## Open items

- Power BI report (T4-T15): built by hand from the loaded tables; it lives in `powerbi/` (git-ignored). After any reload press Refresh.
- Decisions left to the owner are listed in `docs/archive/CLEANUP_REPORT.md` (for example the rule `because` texts that still say
  "orders" for items sold, a `load-sql` target in the Makefile, and an optional GitHub Actions workflow for the tests that need
  neither `data/raw` nor SQL Server).
- Real SQL Server integration is tested read-only; the full reload is run by hand (`.\tasks.ps1 load-sql`).
