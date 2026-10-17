# Stage 8 cleanup report (2026-10-17)

Verification, one deliberate rule change, documentation cleanup, SQL reload and re-verification, done before the
Power BI stage. Nothing was committed. `powerbi/EcomDSS.pbix` was never opened, copied or changed (its SHA-256 was
`eaa0c60e...8a45` before and after). `reports/RESULTS.md` holds the current numbers.

Baseline for this report: HEAD `271687f` (SQL loader commit), manifests `M0` (before anything), `M1` (after the
rule change and SQL reload) and `M3` (after cleanup) in `%TEMP%` (`ecom_manifest_M*.json`), a backup of `reports/`, `models/`,
`data/processed`, `data/export` and `rules/` in `%TEMP%\ecom_backup_final`.

## A. Phase 1: the current state was verified before anything changed

| Check | Result |
|---|---|
| Clean-room rebuild, twice (copy of the tree without `.git`, `.venv`, models, processed data, exports, `powerbi/`) | `run_all.py --export` exit 0 both times (597 s and 544 s, each including its own pytest, ruff and mypy gates, all PASSED). Outputs vs the working repo: 43 byte-identical, 4 differ only in a timestamp field (`models/*_metadata.json`, `models/registry.json`), 0 other differences. Every report file was regenerated (none missing). Run 1 vs run 2: identical apart from the same timestamps. |
| ruff, mypy, pytest | clean; mypy 13 source files; **273 passed, 0 skipped** (188 s, run beside a rebuild). Slowest: `test_predict.py` fixtures (36 s and 24 s). |
| Data reconciliation | 99,441 raw orders, 96,478 delivered in raw, 96,470 orders and 110,189 item rows in the fact table (8 delivered orders do not reach the fact table); 97,811 seller-order pairs; `item_key` unique; only the two review fields have nulls (0.7505%); late rate 7.9082%; dimension counts 3,095 / 32,951 / 99,441; **0 orphan keys** in every relationship; every seller in `SellerFacts`, `ahp_ranking`, `discovery_quadrants`, `seller_recommendations` exists in `DimSeller`. |
| Model integrity | saved features equal `config.ALLOWED_FEATURES` (12, no date or delivery-derived name; `predict.py` never reads those columns); all three persisted models load and predict 1,000 rows in [0,1]; five folds partition the rows and each order is held out in exactly one; **the stored out-of-fold score equals the persisted model's prediction on fold-0 test rows and differs on fold-0 training rows**, so scores are out-of-fold; `late_risk`, `late_risk_lr`, `late_risk_dt` in [0,1] with no NaN; every AUC below the 0.95 guardrail (largest: the forest on its own training rows, 0.9346); no file in `app/` imports `joblib` or `pickle`. One false alarm of my own check: the Design page names `random_forest.joblib` only to show its file size. |
| Rules integrity | 12 unique rule ids; every rule matches between 0% and 100% of all sellers except R12, the default, which matches everyone by design; every rule fires on a fixture built from its own conditions; action counts sum to 2,970 and equal the per-rule counts; the engine reproduces every row of `seller_recommendations.csv` and is deterministic; `condition_checks` agrees with `Rule.matches` for all 2,970 x 12 = 35,640 pairs and `near_miss` lists only unmatched, higher-priority rules with a different action. |
| Rounding safety for SQL | recomputing every seller's facts from `late_risk` and `ahp_score` rounded as SQL stores them (6 decimals): largest difference 5e-7, **0 sellers change rule or action**. |
| Defect found and not fixed (outside this task) | `tests/test_data_integrity.py::test_verify_raw_files_passes_on_real_raw_dir` fails instead of skipping when `data/raw` is absent (seen in a data-less copy, see F). |

### Language and stale-number audit

Each row is a finding at audit time (line numbers are from before the cleanup) and what became of it.

| File : line | Text | Issue | Outcome |
|---|---|---|---|
| `src/predict.py:22-26` | "5-fold CV mean AUC 0.7928 vs. DT's 0.6870 ... stable low-variance advantage" | stale evidence (random-fold CV was inflated) | rewritten to point to `reports/oof_metrics.json` and `RESULTS.md` |
| `src/predict.py:509, 522` | "predicted late probability", "one probability column per builder" | the score is a ranking score | reworded ("risk score") |
| `src/config.py:259-277` | `PROMOTE_LATE_RISK_THRESHOLD_STRICT = 0.05`, "recalibrated to ~0.05" | unused constants; the rule file says 0.252 | removed; thresholds live only in the YAML |
| `src/config.py:191, 251` and 3 test/doc references | `reports/diagnostic_*.txt`, `scripts/diagnostic_*.py` | paths moved | updated |
| `CLAUDE.md:402`, `BUILD_LOG.md:185`, `docs/knowledge_engineering.md:83` | "(materially) better-calibrated" | the score is not calibrated | CLAUDE.md rewritten, doc reworded; the archived log keeps its historical wording |
| `CLAUDE.md` (many) | AUCs 0.7841 / 0.6738 / 0.6347, 423 MB, quadrants 454/478/480/452, old thresholds and fire counts, "61 passed", "10 source files" | stale | file rewritten, numbers removed; a test blocks them in current documents |
| `docs/knowledge_engineering.md` section 6 | thresholds R01 0.53, R03 0.55, R06/R10 0.503, R05 "105 sellers", old percentiles | stale and mislabelled "current" | table regenerated from the YAML and the real counts; test compares it with the YAML |
| `docs/knowledge_engineering.md:619-620`, `docs/DEMO_SCRIPT.md:178-179` | temporal forest 0.5210 / 0.4402 (0.44) | from a fully grown forest | `diagnostic_temporal` regenerated; now 0.5178 and 0.4198 (0.52 and 0.42) |
| `docs/DEMO_SCRIPT.md:109-110` | "precision 0.20" | actual 0.1946, i.e. 0.19 | corrected |
| `docs/DEMO_SCRIPT.md:137` | "typical value (about 0.34)" | item-level median is 0.33 (0.34 is the seller-level median) | corrected to 0.33 |
| `README.md` | "stub ... not yet implemented", "make models/run/app will fail" | wrong | rewritten |
| `run_all.py:177-203` | summary lists LR and DT only; "STAGE 1 + 2 + 3"; ">= 30 orders" | forest missing, stale title, items sold | all three models with the primary labelled; title and wording fixed |
| `rules/seller_rules.yaml` `because` of R02, R04, R05, R10 | "orders" where items sold is meant | wording | **not changed** (would alter `seller_recommendations.csv` text and the SQL table); decision E1 |
| `src/discovery.py:8, 47, 66, 97`, `src/ahp.py` | "orders" for items sold in comments and names (`AHP_MIN_ORDERS`) | wording | not changed (comments and constant names in files outside this task's list); decision E2 |
| `src/explain.py:9, 122` | "ranking probability" | the standard reading of ROC-AUC (probability a random late order outranks a random on-time one), not a claim about `late_risk` | left; decision E3 |
| `tests/test_explain.py`, `tests/test_app.py` | 0.7841, 0.7928, 0.6347 as inputs to pure sentence functions | arbitrary fixture numbers, not claims | left |
| `reports/diagnostics/*`, `BUILD_LOG.md` (archive) | old AUCs, leaf-1 numbers, "not committed" headings | historical | annotated (reports) or archived unchanged (log) |
| Everywhere else | "forecast", "predicts the future" | all occurrences say the model does **not** forecast | none needed |

## B. Phase 2: the small-seller rule fix (the only phase that changed outputs)

**Problem (measured before the change).** R01 (Suspend) and R03 (Warn) read `avg_late_risk` with no volume gate. Of R01's 137 final
recommendations, 70 had fewer than 5 items sold (37 had 5-14, 30 had 15 or more; median 4 items; 34 never delivered late). Of R03's 219,
120 had fewer than 5 (50 with 5-14, 49 with 15 or more; median 3 items) and **104 had never delivered late**.

**Change.** `rules/seller_rules.yaml`: `order_volume >= 5` added to R01 and R03, `because` texts say "at least 5 items sold". Nothing else
in any rule. Thresholds re-derived on the sellers the rules now apply to (1,864 with at least 5 items sold):

| | before | adopted (eligible population) | sensitivity (all-seller thresholds with the gate) |
|---|---|---|---|
| R01 `avg_late_risk >` | 0.416 | **0.412** (75th percentile; share at or below 0.750) | 0.416 (67 fire) |
| R03 `avg_late_risk >` | 0.501 | **0.476** (90th percentile; 0.901) | 0.501 (99 fire) |

All-seller percentiles for reference: 75th 0.416381, 90th 0.501273; eligible: 0.411893 and 0.475514. R06 and R10 keep 0.339 (all-seller
median); the documents state each rule's reference population.

| Rule | R01 | R02 | R03 | R04 | R05 | R06 | R07 | R08 | R09 | R10 | R11 | R12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fires before | 137 | 6 | 219 | 69 | 37 | 137 | 37 | 106 | 104 | 170 | 713 | 1,235 |
| fires after | 70 | 6 | 144 | 66 | 31 | 137 | 37 | 106 | 104 | 170 | 701 | 1,398 |

| Action | Keep | Warn | Suspend | Feature | Promote | Sum |
|---|---|---|---|---|---|---|
| before | 2,118 | 325 | 143 | 174 | 210 | 2,970 |
| after | 2,269 | 241 | 76 | 174 | 210 | 2,970 |
| sensitivity variant | 2,308 | 205 | 73 | 174 | 210 | 2,970 |

**Sellers whose action changed: 232** (rule changed: 238; rows whose rule or reason text changed: 404).

| from \ to | Keep | Warn | Suspend | Feature | Promote |
|---|---|---|---|---|---|
| Keep | 2,079 | 39 | 0 | 0 | 0 |
| Warn | 120 | 202 | 3 | 0 | 0 |
| Suspend | 70 | 0 | 73 | 0 | 0 |
| Feature | 0 | 0 | 0 | 174 | 0 |
| Promote | 0 | 0 | 0 | 0 | 210 |

- **Tiny sellers.** All 1,106 sellers with fewer than 5 items sold are now Keep through R12 ("pending more data"); 190 of them were previously on R01 or R03
  (R12: 1,235 to 1,398).
- **R03 Warn sellers with zero late deliveries:** 104 to 10 (R01: 34 to 3). Warn or Suspend below 5 items sold: 0.
- **Side effect of adopting the eligible-population thresholds** (the owner's decision): lowering R03 from 0.501 to 0.476 moves **39** sellers
  with at least 5 items sold from Keep to Warn, and lowering R01 from 0.416 to 0.412 moves 3 from Warn to Suspend. The all-seller thresholds
  would not have.
- **Invariants held** (file by file against M0): `SellerFacts.parquet`, `ahp_ranking.csv`, `discovery_quadrants.csv`, `discovery_summary.md`,
  `LatePredictions` (Parquet and CSV), `FactOrderItems`, every model file, every other export and every other report are byte-identical. Quadrants
  495 / 437 / 439 / 493 and the same 437 hidden-gem ids. The only changed outputs: `rules/seller_rules.yaml`, `reports/seller_recommendations.csv`,
  `data/export/seller_recommendations.csv` (plus tests and documents).
- **SQL reload:** `python -m src.load_sql` 21.6 s in the loader (25 s wall), all ten tables and the view OK (`SellerRecommendations` 2,970), then
  `--verify-only` OK in 0.5 s.
- **Tests added** (`tests/test_expert.py`): every Warn and Suspend rule has an `order_volume >= 5` gate; a seller with 4 items stays on R12 and one with 5 does not;
  no real Warn or Suspend seller below 5 items; the drift test uses each rule's own reference population; R01 and R03 sit at the 75th and 90th percentile
  of sellers with 5+ items; R06 and R10 at the all-seller median; all 2,970 x 12 engine and near-miss parity.

## C. Phase 4: re-verification against M1

| Check | Result |
|---|---|
| Sandbox rebuild x2 on the cleaned tree | both `run_all.py --export` exit 0, **530 s** each (predict 342 s, export 4.5 s); gates PASSED. Run 1 = run 2 byte for byte except the four timestamp files. Sandbox vs working repo: 44 byte-identical, 4 timestamp-only, 0 other. |
| `.\tasks.ps1 verify` (working repo) | ruff PASS, mypy PASS (13 files), **pytest 289 passed in 175 s** (the 273 above plus the new tests), `load_sql --verify-only` PASS (all ten tables), `RESULTS.md` fresh PASS, app smoke PASS; exit 0, 184 s total. |
| Database | `--verify-only`: DimSeller 3,095; DimProduct 32,951; DimCustomer 99,441; FactOrderItems 110,189; LatePredictions 110,189; CallSentiment 300; Clickstream 100,000; AhpRanking 681; DiscoveryQuadrants 1,864; SellerRecommendations 2,970; all OK. |
| Headless app | all 7 pages run without an exception under `AppTest`; first-load times 0.24 s (Home), 1.5 s (Intelligence), 0.13 (Design), 0.11 (Choice), 0.20 (Discovery), 0.15 (Implementation), 0.15 (DSS Architecture) (page caches are shared within the process, so only the first page pays for data loading). Real `streamlit run app/Home.py --server.headless true` on port 8611: `/_stcore/health` returned 200 `ok`, `/` returned 200; server stopped afterwards. |
| M3 vs M1 | differences are only: documents, moved scripts and reports (6 + 6), new files (`RESULTS.md`, `make_results.py`, `app_smoke.py`, `MANUAL_CHECKS.md`, `SYLLABUS_MAP.md`, `docs/README.md`, `test_results.py`, archive files), tests, `tasks.ps1`, `Makefile`, `run_all.py` summary, dead-code removals in `app/` and `src/config.py`, the Implementation default-seller change, wording in `src/predict.py`, the regenerated `diagnostic_temporal` report. **No difference in `data/processed`, `data/export`, `models`, `LatePredictions`, `SellerFacts`, `ahp_ranking`, `discovery_quadrants`, `seller_recommendations`, `sql/schema.sql` or the database.** |

Determinism: confirmed twice (this phase and Phase 1). The new Implementation default seller has 5 items sold, an average review of 1.00, no late delivery, and is Warn through R04.

## D. Line counts

| File | before | after |
|---|---|---|
| `BUILD_LOG.md` | 661 (full copy kept at `docs/archive/BUILD_LOG_full.md`, sha-identical) | 89 |
| `CLAUDE.md` | 408 | 229 |
| `README.md` | 35 | 98 |

## Audit findings (23) and what was done

| # | Finding | Action |
|---|---|---|
| 1 | `product_weight_g` blanks break the wizard | solved by `sql/schema.sql` + loader (column `int NULL`, blanks loaded as NULL; the load and the weight checksum pass; a direct read-back during the loader stage showed the 2 NULLs) |
| 2 | em dash in `because` | solved (`nvarchar` loaded through pyodbc; the loader's `SUM(LEN(because))` checksum equals the CSV's UTF-16 length sum after this stage's reload, which would not hold if the dash were mangled; the earlier direct read-back found every dash row intact) |
| 3 | text columns truncated by the wizard | solved (columns sized from the profile; the loader refuses any value that does not fit and SQL Server would raise on truncation; row counts and checksums pass) |
| 4 | `powerbi_setup.md` total 361,860 | verified: it says 461,680; `RESULTS.md` shows rows match |
| 5 | `CLAUDE.md` stale numbers | fixed (rewritten; test blocks known-stale numbers) |
| 6 | `knowledge_engineering.md` section 6 stale | fixed (regenerated, tested against the YAML) |
| 7 | README stub | fixed |
| 8 | CLAUDE.md layout, missing `SYLLABUS_MAP.md`, status | fixed (file created, tested) |
| 9 | temporal figures not reproducible | fixed (regenerated; quotes updated; tested) |
| 10 | `models` target omits `src.expert` | fixed in `tasks.ps1` and `Makefile` |
| 11 | diagnostics do not regenerate their reports | marked: three HISTORICAL, two CURRENT (they reproduce exactly), one regenerated |
| 12 | no final test count in the log | the log now records the stage result; count is in section C |
| 13 | `run_all` summary omits the forest | fixed |
| 14 | "probability" / "better-calibrated" | fixed in `predict.py`, docs, CLAUDE.md; `explain.py` AUC wording left (E3) |
| 15 | "(not committed)" headings | gone from the new log (archive unchanged) |
| 16 | Implementation opens on a one-item seller | fixed |
| 17 | R01/R03 fire on tiny sellers | fixed by Phase 2 |
| 18 | Stage 3 AUCs not marked inflated | new log and `RESULTS.md` mark them superseded |
| 19 | four `.gitkeep` | deleted |
| 20 | diagnostics clutter | moved into `scripts/diagnostics/`, `reports/diagnostics/` |
| 21 | dead helpers | removed with their tests, plus two stale unused constants in `config.py` |
| 22 | long log without summary | 89-line log |
| 23 | duplicated narrative | CLAUDE.md rewritten without it; one home per topic |

## E. Needs your decision (nothing below was changed)

1. **`because` texts of R02, R04, R05, R10 say "orders"** where items sold is meant. Fixing them changes `seller_recommendations.csv` text (and the SQL table), so it needs a reload.
2. **"orders" in `src/discovery.py` comments and in constant names** (`AHP_MIN_ORDERS`, `DISCOVERY_MIN_ORDERS`, `order_volume`): renaming is wide and outside this task.
3. `src/explain.py` calls ROC-AUC a "ranking probability" (the usual definition). Keep, or rephrase to avoid the word near the risk score?
4. **R01/R03 thresholds:** adopted the eligible-population values (0.412, 0.476); it moves 39 Keep sellers to Warn and 3 Warn sellers to Suspend. The all-seller values would move none (sensitivity in section B and `knowledge_engineering.md` section 10).
5. **R06 and R10 reference population:** both keep the all-seller median (0.339); R10 is gated at 5 items, so its median could be re-derived on those sellers. Left as instructed.
6. **Diagnostic reports:** you asked for the "HISTORICAL ... superseded" prefix on all five others; two of them (`rf_leaf`, `shrinkage`) reproduce exactly and are still the evidence for current settings, so they are marked CURRENT instead.
7. **CLAUDE.md constraints rewritten rather than dropped:** "commit per stage" became "do not commit unless asked" (what you have been asking for); "constants only in `config.py`" now has two stated exceptions (the loader's few constants, rule thresholds in the YAML); the model-family rule is unchanged (forest primary; anything new needs evidence and your confirmation); the app is described as the internal check view with Power BI as the presented deliverable. Say if any of these should differ.
8. **`Makefile`** has no `load-sql`, `verify` or `rebuild` target (only `tasks.ps1` does; `make` is not installed here, so the Makefile can only be checked by reading it).
9. **Test fails instead of skipping without raw data:** `test_verify_raw_files_passes_on_real_raw_dir` (needed for the proposed CI job).
10. **"Try an example" on Implementation** jumps to the first seller in file order with the chosen outcome; for Keep that can be a one-item default-rule seller.
11. The SQL integration test compares the live database with the CSVs, so a pipeline re-run before `load-sql` makes it fail until the reload (intended, but easy to trip over).
12. **Commit hygiene:** `.gitignore` (adds `*.pbix`) has been modified and uncommitted since before this task; `docs/archive/AUDIT_REPORT.md` is untracked (it was untracked on purpose); the new BUILD_LOG no longer says "(not committed)" anywhere.

### Follow-up to section E (same day, after the owner's decisions)

| # | Item | Status |
|---|---|---|
| E1 | reason texts say "orders" | **Done.** R02, R04, R05, R10 now say "items sold" (and "items" for the late-rate phrases, since the late rate is taken over item rows). Only the `because` column of 273 recommendations changed; action and rule_id are identical for all 2,970 sellers; action and per-rule counts unchanged; every other output byte-identical; SQL reloaded and verified. |
| E2 | "orders" in `src/discovery.py` | **Comments, docstrings and one log message fixed.** Renaming `AHP_MIN_ORDERS`, `DISCOVERY_MIN_ORDERS`, `order_volume` and the `min_orders` parameters is wide and still open. |
| E3 | "ranking probability" in `src/explain.py` | Open: it is the standard definition of ROC-AUC; your call whether to rephrase. |
| E4 | R01/R03 thresholds from the eligible population | Your decision, unchanged (sensitivity in `docs/knowledge_engineering.md` section 10). |
| E5 | R06/R10 reference population | Your decision, unchanged. |
| E6 | `rf_leaf` and `shrinkage` marked CURRENT, not HISTORICAL | Open: I kept the accurate label; say if you want HISTORICAL anyway. |
| E7 | CLAUDE.md constraints rewritten | Open for your confirmation (commit policy, constants location, model-family rule, app role). |
| E8 | Makefile lacks `load-sql` | **Done** (`load-sql` target and `.PHONY`). `verify` and `rebuild` are still `tasks.ps1` only. `make` is not installed here, so the Makefile was checked by reading it. |
| E9 | a test failed instead of skipping without raw data | **Done.** It is `tests/test_config.py::test_verify_raw_files_passes_on_real_raw_dir` (not in `test_data_integrity.py`); it now skips with a reason when `data/raw` holds no CSV, and still fails if the directory is incomplete. |
| E10 | "Try an example" could open a one-item default-rule seller | **Done.** Examples are taken from sellers with at least 5 items sold and a rule other than R12 first, then the old order; one test. |
| E11 | integration test depends on the database matching the CSVs | Open by design (it is how a stale database is noticed). |
| E12 | commit hygiene | Open: see the single commit in the reply. |

Suite run in three sandbox copies (not the working repo): without `data/raw` **268 passed, 21 skipped, 0 failed**; without `data/`, `models/` and
anything generated **181 passed, 108 skipped, 0 failed**; with SQL Server unreachable (the loader pointed at a non-existent instance in the copy)
**288 passed, 1 skipped (the integration test), 0 failed**.

## F. Recommended commits (not run)

Check each with `git status --short` and `git diff --cached --stat` before committing.

```
# 1. The rule change and its outputs
git add rules/seller_rules.yaml reports/seller_recommendations.csv tests/test_expert.py
git diff --cached --stat        # expect 3 files
git commit -m "fix: minimum evidence (5 items sold) for Warn and Suspend; R01/R03 recalibrated on eligible sellers"

# 2. Code and wording fixes, dead code, Makefile
git add src/predict.py src/config.py run_all.py app/_common.py app/_components.py app/pages/2_Design.py app/pages/5_Implementation.py tests/test_app.py Makefile .gitignore scripts/app_smoke.py
git add -A -- app/.gitkeep app/pages/.gitkeep docs/.gitkeep rules/.gitkeep
git diff --cached --stat        # expect 15 files (4 are .gitkeep deletions); no data, models or powerbi/
git commit -m "chore: wording (risk score is not a probability), remove dead review-flag helpers, default seller needs 5 items, app smoke script"

# 3. Moves, new documentation, tooling that depends on them
git add -A -- scripts/diagnostic_group_leak.py scripts/diagnostic_rf_ceiling.py scripts/diagnostic_rf_cv.py scripts/diagnostic_rf_leaf.py scripts/diagnostic_shrinkage.py scripts/diagnostic_temporal.py
git add -A -- reports/diagnostic_group_leak.txt reports/diagnostic_rf_ceiling.txt reports/diagnostic_rf_cv.txt reports/diagnostic_rf_leaf.txt reports/diagnostic_shrinkage.txt reports/diagnostic_temporal.txt
git add scripts/diagnostics reports/diagnostics scripts/make_results.py reports/RESULTS.md tests/test_results.py tests/test_predict.py tasks.ps1
git add README.md CLAUDE.md BUILD_LOG.md docs/README.md docs/SYLLABUS_MAP.md docs/MANUAL_CHECKS.md docs/DEMO_SCRIPT.md docs/knowledge_engineering.md docs/powerbi_setup.md docs/archive
git diff --cached --stat        # expect about 41 paths (12 of them moves, git shows renames); still no data, models or powerbi/
git commit -m "docs: single source of numbers (RESULTS.md), shorter CLAUDE/README/BUILD_LOG, diagnostics folders, verify/rebuild tasks"
```

Commits 2 and 3 are split by theme, so between them a few comments refer to `reports/diagnostics/` before it exists in history, and `tasks.ps1 verify` (commit 3) is the first commit where every file it calls exists.

### Proposed GitHub Actions workflow (not created)

```yaml
name: checks
on: [push, pull_request]
jobs:
  checks:
    runs-on: windows-latest        # requirements.txt was frozen on Windows
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: python -m pip install -r requirements.txt
      - run: python -m ruff check .
      - run: python -m mypy src
      - run: python -m pytest -q -rs
```

Measured by running the suite in a copy without `data/`, `models/` and generated reports: **177 passed, 110 skipped, 1 real failure** (item 9 above; a second failure in that copy was only my removal of tracked report files). Skipped there: `test_app.py` 23, `test_predict.py` 15, `test_load_sql.py` 11, `test_data_integrity.py` 6, `test_export_sql.py` 5, `test_discovery_outputs.py` 4, `test_seller_facts_review.py` 3, `test_results.py` 3, `test_expert.py` 3, `test_ahp_shrinkage.py` 2, `test_explain.py` 1. The SQL integration test and every test that needs `data/raw` skip. A real checkout also has the tracked `reports/` files.

## G. Surprises

1. For sellers with 5 or more items sold the volume gate changes nothing; the only change is the lower R03 and R01 thresholds (0.501 to 0.476, 0.416 to 0.412), which turns 39 Keep sellers into Warn and 3 Warn sellers into Suspend.
2. The documents carried wrong figures a quote-checking test found immediately: the demo script said precision 0.20 (it is 0.19), a typical risk of 0.34 (0.33 at item level), and 0.44 for the forest without calendar inputs (0.42).
3. `src/config.py` held two unused Promote constants, one of them stale (0.05 against the file's 0.252); nothing read them.
4. The number of reasons containing an em dash fell from 1,298 to 1,149. That is not a data loss: R01 and R03 now have new `because` texts and fewer firings.
5. `tasks.ps1 verify` first died at the app-smoke step: with redirected output PowerShell turns a native program's stderr text (a Streamlit warning) into a terminating error under `$ErrorActionPreference = "Stop"`. Fixed inside the task (and the warning is silenced in the smoke script).
6. The data-less test run shows 110 tests silently depend on pipeline outputs; only one fails instead of skipping.
7. `HEAD` had moved on from the audit (the loader was committed meanwhile); the Stage 7 notes in the old log were already committed.
