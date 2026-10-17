> **Archived.** Written against commit 9647f81 before the SQL loader, the Stage 8 rule fix and the documentation cleanup; see `CLEANUP_REPORT.md` for what was done about each finding. **The DDL in section B.5 is superseded: use `sql/schema.sql`, which is the source of truth for the tables, and do not copy that block.**

# Audit report — E-Commerce Decision Support System

Audit of commit `9647f81` (Stage 6, working tree clean at the start). Audit only: the sandbox clone
lived in `%TEMP%\ecom_e2e`, all helper scripts were kept outside the repo, and this file is the
only change to the working repo.

**Verdict, one line:** the Python pipeline is sound and fully reproducible (byte-identical rebuild,
198 tests, ruff and mypy clean, every export equals its source, every rule/action count reconciles).
What is not sound is the paperwork around it and the SQL import: `CLAUDE.md`, `README.md`,
`docs/knowledge_engineering.md` §6 and `docs/powerbi_setup.md` still quote numbers from earlier
stages, and three export columns will very likely trip the SSMS Import Flat File wizard. Details and
fixes are in the findings table at the end.

Contents: [A. Clean-room rebuild](#a-clean-room-rebuild) · [B. Consistency audit](#b-consistency-audit)
· [C. Repo hygiene and app check](#c-repo-hygiene-and-app-check) · [D. Findings by severity](#d-findings-by-severity)
· [Verdict](#overall-verdict)

---

## A. Clean-room rebuild

### A.1 Procedure and result

`git clone` of the working repo (HEAD only) into `%TEMP%\ecom_e2e`, `data\raw` copied in, then, from
inside the clone and with the project `.venv` python (no new venv, nothing installed):
`python run_all.py --export`, `ruff check .`, `mypy src`, full `pytest`.

| Step | Result | Time |
|---|---|---|
| `python run_all.py --export` | exit 0 | **527 s** total |
| &nbsp;&nbsp;`build_tables` | ok | 1.28 s |
| &nbsp;&nbsp;`synthetic` | ok | 0.43 s |
| &nbsp;&nbsp;`clickstream_checks` | ok | 0.73 s |
| &nbsp;&nbsp;`sentiment` | ok | 0.03 s |
| &nbsp;&nbsp;`predict` | ok | **347.44 s** (five-fold forest fits; ~66 % of the run) |
| &nbsp;&nbsp;`ahp` | ok | 0.32 s |
| &nbsp;&nbsp;`discovery` | ok | 0.33 s |
| &nbsp;&nbsp;`expert` | ok | 0.14 s |
| &nbsp;&nbsp;`export_sql` | ok | 4.31 s |
| `run_all`'s own gates (pytest, ruff, mypy) | all PASSED | within the 527 s |
| `ruff check .` (separate run) | passed | 2 s |
| `mypy src` (separate run) | passed, 12 source files | 0 s (cached) |
| full `pytest` (separate run, includes `tests/test_predict.py`) | **198 passed** | 162.68 s |

**No failure, and no failure caused by a file missing in the clone.** The pipeline has no silent
dependency on an untracked file; the only untracked inputs are `data/raw` (copied, by design) and
what the pipeline itself generates.

Defect found in the run (severity 3): the closing summary printed by `run_all.py` (lines ~177-201,
"Late-delivery models (test-set ROC-AUC)") lists only logistic regression and decision tree. The
primary model, `random_forest`, is not printed.

### A.2 Output comparison, clone vs working repo

48 files compared by SHA-256 across `reports/`, `data/processed/`, `data/export/`, `models/`.

- **42 byte-identical**, including every CSV, Parquet, text report and all three `.joblib` models.
- **4 differ, timestamp fields only** (acceptable): `models/logistic_regression_metadata.json`,
  `models/decision_tree_metadata.json`, `models/random_forest_metadata.json`, `models/registry.json`.
  The first differing field is `timestamp` (or `updated` in `registry.json`) in each.
- **2 present only in the working repo**: `data/processed/.gitkeep` and `models/.gitkeep`
  (the directories are gitignored, so the clone has no `.gitkeep`). Harmless.
- Nothing else differs (this comparison was made before the diagnostic re-runs in A.3 overwrote the clone's own copies), so the committed `reports/` are reproducible from a clean checkout.

### A.3 Do the committed one-off diagnostics reproduce? (extra check)

I re-ran the five fast diagnostic scripts in the clone and diffed the output against the committed
`reports/diagnostic_*.txt` (`diagnostic_rf_cv.py` was not re-run: slow, and it is superseded; judged
by reading it).

| Script | Re-run vs committed report |
|---|---|
| `diagnostic_shrinkage` | identical |
| `diagnostic_group_leak` | **differs.** The forest section reads 0.7841 / 0.7274 in the report (random vs grouped split, fully grown trees) and 0.7628 / 0.7410 on re-run, because the forest default is now `min_samples_leaf=20`. Part (d) now reads the out-of-fold `LatePredictions`: AUCs 0.7367 / 0.7224 / 0.7402 and seller-level correlations 0.339 / 0.428 and 0.517 / 0.583, against 0.906 / 0.775 and 0.925 / 0.885 in the report |
| `diagnostic_temporal` | **forest differs; LR and DT identical.** All 12 inputs 0.5178 (report 0.5210); without `order_month`, `day_of_week` 0.4198 (report 0.4402) |
| `diagnostic_rf_ceiling` | **differs.** Forest 0.7274 on re-run against 0.7841 in the report (the script now uses the order-grouped split); the permutation order also changes (`seller_historical_late_rate` 0.0593 → 0.0373, `geo_distance` becomes second). The LR 0.6347 and DT 0.6738 lines are hard-coded constants (`scripts/diagnostic_rf_ceiling.py:43-44`) from the old random split, so the report now compares a grouped forest with random-split baselines |
| `diagnostic_rf_leaf` | **identical in every number**: pooled OOF AUC 0.7233 / 0.7311 / 0.7367 for leaf 1 / 5 / 20, fitted size 421.2 / 193.3 / 67.7 MB (only the minutes column differs, as expected) |

Two reproduce exactly (`shrinkage`, `rf_leaf`); the three that do not (`group_leak`, `temporal`, `rf_ceiling`) share one cause: those reports were produced with a fully grown forest (leaf 1) or the old random split, and the scripts import
`predict`'s current defaults, so they no longer regenerate their own committed reports.
`DEMO_SCRIPT.md` and `docs/knowledge_engineering.md` quote the leaf-1 temporal figures
(see B.3).

---

## B. Consistency audit

### B.3 Every quoted number against the current files

"Actual" values come from the current repo files (`reports/`, `data/processed/`, `rules/`, `src/config.py`)
and from the clean-room rebuild, which is byte-identical. Truth values used: test AUC LR 0.6259 /
DT 0.6888 / RF 0.7410; out-of-fold AUC LR 0.6241 / DT 0.6856 / RF 0.7367; quadrants Star 495,
Hidden Gem 437, Overrated 439, Overlooked-Low-Quality 493; hidden gems low-confidence 393, normal 44;
actions Keep 2,118, Warn 325, Promote 210, Feature 174, Suspend 143 (= 2,970); rule fire counts
R01 137, R02 6, R03 219, R04 69, R05 37, R06 137, R07 37, R08 106, R09 104, R10 170, R11 713, R12 1,235;
YAML thresholds R01 0.416, R03 0.501, R06 0.339, R08 0.3, R09 0.252, R10 0.339; `random_forest.joblib`
67,748,322 bytes; 198 tests; mypy 12 files.

Entries in `BUILD_LOG.md` that describe an earlier stage and say so are marked *historical* (not a
defect). The same figure in a current-state document is judged strictly.

| File | Line | Claim | Actual | Status |
|---|---|---|---|---|
| README.md | 7, 12-14 | "This README is a stub"; "Stage 1 only … models, AHP, expert system, Streamlit app are not yet implemented" | Stages 1-6 are built and committed | **wrong** |
| README.md | 32 | "`make models`, `make run`, `make app` will fail until later stages" | all targets run | **wrong** |
| README.md | 35 | nine raw CSVs go in `data/raw/` | correct | ok |
| CLAUDE.md | 13-18 | Status list ends at Stage 3.5 | Stage 4, 4b, 4d, 5, 5b, 6 are done and absent | stale |
| CLAUDE.md | 24 | RF justified by 5-fold CV 0.7928 vs 0.6870 vs 0.6247 | those were random-fold, inflated; grouped OOF is 0.7367 / 0.6856 / 0.6241 | stale |
| CLAUDE.md | 117 | `docs/SYLLABUS_MAP.md` in the layout | file does not exist | **wrong** |
| CLAUDE.md | 138 | "mypy clean on 10 source files" | 12 | stale |
| CLAUDE.md | 193 | "Stratified 80/20 split", RF without `min_samples_leaf` | order-grouped split; `min_samples_leaf=20` | stale |
| CLAUDE.md | 194 | test AUC 0.7841 / 0.6738 / 0.6347 (RF / DT / LR) | 0.7410 / 0.6888 / 0.6259 | **stale** |
| CLAUDE.md | 198, 408 | `random_forest.joblib` 423 MB | 67.7 MB (67,748,322 bytes) | stale |
| CLAUDE.md | 212-216 | thresholds R01 0.178, R03 0.29, R06/R10 0.106, R09 0.05 | 0.416, 0.501, 0.339, 0.252 | **stale** |
| CLAUDE.md | 215-216 | counts R09 119/478, R08 472/478; R01 165, R03 180, R06 164, R10 104; R02 2, R05 5, R11 607, R04 85, R07 69, R12 1,259 | R09 104, R08 106; R01 137, R03 219, R06 137, R10 170; R02 6, R05 37, R11 713, R04 69, R07 37, R12 1,235 | **stale** |
| CLAUDE.md | 225 | quadrants Star 454, Hidden Gem 478, Overrated 480, Overlooked 452 | 495, 437, 439, 493 | **stale** |
| CLAUDE.md | 241 | "61 passed (20 Stage 1 + 41 Stage 2)" | 198 passed | stale |
| CLAUDE.md | 249 | "Still pending: tests for `src/export_sql.py`" | `tests/test_export_sql.py` exists | stale |
| CLAUDE.md | 269 | `models` target includes expert; "Stage 3 will add an export step" | Makefile and tasks.ps1 `models` omit `src.expert`; export exists | stale |
| CLAUDE.md | 288 | "Definition of done (Stage 3, next up)", all boxes open | Stage 3 finished long ago | stale |
| CLAUDE.md | 337-342 | Stage 2 snapshot (AUC 0.5744 / 0.5692) | labelled as a historical snapshot | ok |
| CLAUDE.md | 397-398 | diagnostics 0.7841 / 0.7928 as the evidence trail | historical; produced on random splits | ok (historical) |
| CLAUDE.md | 402 | R08 472/478, R09 478/478 after adoption | historical, written as a present consequence | stale |
| CLAUDE.md | 6.2, 6.3 | 99,441 / 96,470 / 110,189 / 3,095 / 32,951 / 99,441; 7.9082 %; funnel 70,000 → 6,000; 300 calls | all verified against `dq_report.md` and files | ok |
| CLAUDE.md | 7.4 | 681 sellers, CR 0.0189 | 681, 0.0189 | ok |
| CLAUDE.md | 7.6 | 818 low-confidence / 1,046 normal | matches `discovery_quadrants.csv` | ok |
| docs/powerbi_setup.md | 27 | "10 tables, 361,860 rows exported" | **461,680** | **wrong** |
| docs/powerbi_setup.md | table | ten per-table row counts | all ten correct | ok |
| docs/knowledge_engineering.md | 263-266 | "All thresholds below are current as of the Stage 3.5 recalibration" | the table below still shows R01 0.53, R03 0.55, R06/R10 0.503 | **wrong** |
| docs/knowledge_engineering.md | 273 | `avg_review` percentiles 4.27 (median), 4.70 (75th) | per-order definition: 4.28, 4.67 | stale |
| docs/knowledge_engineering.md | 283-290 | R01 `> 0.53`, R03 `> 0.55`, R06 `< 0.503`, R10 `<= 0.503`; R05 "105 sellers" | YAML 0.416, 0.501, 0.339, 0.339; R05 37 | **stale** |
| docs/knowledge_engineering.md | 563-571, 592 | random vs grouped AUCs, 0.7841, 0.7928 | historical, labelled as the Stage 6 diagnostic | ok (historical) |
| docs/knowledge_engineering.md | 619-620 | temporal forest 0.5210; without calendar 0.4402 | re-run of the committed script: 0.5178; 0.4198 | stale |
| docs/DEMO_SCRIPT.md | 68, 115 | 437 gems, 44 normal / 393 low-confidence | 437, 44, 393 | ok |
| docs/DEMO_SCRIPT.md | 96-97, 170 | forest tuned to 0.74; model 424 MB → 68 MB | 0.7410; 67.7 MB | ok |
| docs/DEMO_SCRIPT.md | 109 | recall 0.52 | matches `oof_metrics.json` | ok |
| docs/DEMO_SCRIPT.md | 136 | median score about 0.34; correlation 0.34 (0.52 for 5+ items) | matches | ok |
| docs/DEMO_SCRIPT.md | 128 | 12 of 437 gems have reviews below the marketplace mean | matches | ok |
| docs/DEMO_SCRIPT.md | 178-179 | temporal: forest 0.52, tree 0.50, linear 0.40; without calendar 0.44, 0.42, 0.40 | forest 0.5178 → 0.52 ok; **without calendar the forest is 0.4198 → 0.42, not 0.44** | stale |
| BUILD_LOG.md | 192, 219, 294, 342, 414, 456, 520 | headings say "(not committed)" | everything is committed (`9647f81`) | **wrong** wording |
| BUILD_LOG.md | 29-30, 43-59 | Stage 2/3 AUCs 0.5744, 0.5692, 0.6347, 0.6738 | historical; Stage 3 values are random-split numbers, not annotated as inflated at that spot | ok (historical) |
| BUILD_LOG.md | 119-143, 197, 505-508 | 0.7841, 0.7928, 423 MB | historical, tied to their stage | ok (historical) |
| BUILD_LOG.md | 515 | "192 passed" | 198 now; the log never records the final post-Stage 6 count | stale |

### B.4 Integrity checks

- **Every export equals its source.** Row counts match for all ten files:
  `FactOrderItems` 110,189, `DimSeller` 3,095, `DimProduct` 32,951, `DimCustomer` 99,441,
  `LatePredictions` 110,189, `CallSentiment` 300, `Clickstream` 100,000, `ahp_ranking` 681,
  `discovery_quadrants` 1,864, `seller_recommendations` 2,970. Value-level comparison (six-decimal
  tolerance) of `LatePredictions.csv` against the Parquet and `seller_recommendations.csv` against
  `reports/`: no differing cell. `ahp_ranking` and `discovery_quadrants` also match.
- **Action counts sum to 2,970** (Keep 2,118 + Warn 325 + Promote 210 + Feature 174 + Suspend 143),
  equal to the number of sellers in `SellerFacts`; no duplicate seller ids.
- **Every rule's fire count equals the actions it produces.** Mapping the YAML action onto the fire
  counts (R01 137, R02 6, R03 219, R04 69, R05 37, R06 137, R07 37, R08 106, R09 104, R10 170,
  R11 713, R12 1,235) reproduces the action table exactly; zero rows have an action that differs from
  their rule's action.
- **`docs/powerbi_setup.md`:** all ten per-table counts are correct; the stated total
  "361,860" is wrong, the real total is **461,680**. No table is missing from its table.
- **Wording: forecast or probability.** No text in the app, `DEMO_SCRIPT.md`, README or docs claims the
  model forecasts future deliveries; `DEMO_SCRIPT.md:137, 175`, `knowledge_engineering.md:625, 632` and
  `BUILD_LOG.md:583` say the opposite ("not a probability", "does not forecast the future"). Hits
  that are still misleading:
  - `src/predict.py:509` "Out-of-fold predicted late **probability**", `:522` "one **probability** column per builder";
  - `src/explain.py:9, 122` explains ROC-AUC as a "ranking probability" (this one is a correct use of the
    term, for a random-pair statistic, but sits next to the same word elsewhere);
  - `CLAUDE.md:402` calls the forest's `late_risk` "materially better-calibrated" (and `BUILD_LOG.md:185`,
    `knowledge_engineering.md:83` use the same phrase). With balanced class weights the score's
    median is 0.34 against a 7.9 % late rate, so it is a ranking score, not a calibrated probability.

### B.5 SSMS Import Flat File readiness

All ten files: **UTF-8 without BOM, CRLF line endings, comma delimiter, `"` quote character, header row,
minimal quoting (a field is quoted only if it contains a comma or quote), ISO datetimes
`YYYY-MM-DD HH:MM:SS`** (19 characters, no time zone). Inference below assumes the wizard samples the
leading rows (about 200) to guess types, which is how it behaves in my experience; **I did not run SSMS**, so the
"wizard risk" column is an inference from the data, not an observed failure.

Full per-column table (max length, nulls, inferred type, contains comma/quote/newline, first-200 inference,
risk, recommended type) is generated from the real files; the recommended `CREATE TABLE` statements are in the
DDL block at the end of this section. Summary of what is risky:

| File . column | Finding | Why the wizard gets it wrong | Fix |
|---|---|---|---|
| DimProduct.csv . `product_weight_g` | **2 blank values, the first at row 8,578** (none in the first 200 rows) | Sample shows no blanks, so the column is guessed NOT NULL; the import then fails or drops those rows | Create the table first with the column `NULL`, or tick *Allow nulls* in the wizard |
| seller_recommendations.csv . `because` | **1,298 rows contain an em dash (U+2014)**, 12 distinct texts, up to 227 characters, with commas | File has no BOM; the wizard may read it as ANSI and show mojibake; a short guessed `varchar` truncates | Set code page 65001 (UTF-8), column `nvarchar(1000)`; or export with a BOM / replace the dash with `-` |
| CallSentiment.csv . `transcript` | up to 107 characters, commas (quoted), 65 with apostrophes, no double quotes | Short guessed length; embedded commas rely on correct quote handling | `nvarchar(1000)`; keep the text qualifier `"` |
| DimProduct.csv . `product_category_name` | max 46 characters vs 34 in the first 200 rows | Length guessed from the sample, truncation | `nvarchar(128)` |
| DimProduct.csv . `product_category_name_english` | max 39 vs 31 | same | `nvarchar(128)` |
| discovery_quadrants.csv . `quadrant` | max 22 ("Overlooked-Low-Quality") vs 10 in the first 200 rows | same, and this is the column Power BI filters on | `nvarchar(64)` |
| FactOrderItems.csv and LatePredictions.csv . `item_key` | max 35 vs 34 in the first 200 rows | off-by-one truncation of the join key breaks the relationship silently | `varchar(64)` on both tables (the same type on both sides) |
| FactOrderItems.csv . `review_score`, `review_creation_date` | 827 blanks (first at row 32, so nullable is inferred correctly); `review_score` is written as `5.000000` | the six-decimal text makes the wizard pick `float` or `decimal(7,6)` | `decimal(3,1)` NULL |
| FactOrderItems.csv . `price`, `freight_value` | written with six decimals but only two are significant | guessed as `float`, then rounding noise in DAX sums | `decimal(12,2)` |
| FactOrderItems.csv . `order_item_id` | small integers | none (low) | `int` |
| LatePredictions.csv, ahp_ranking.csv, discovery_quadrants.csv numeric columns | six decimals | `float` guess is acceptable, but ranges are 0-1 | `decimal(9,6)` |
| Clickstream.csv, CallSentiment.csv | synthetic data | n/a for types; label as synthetic in the Power BI report | n/a |

No file contains an embedded newline, a leading-zero id, scientific notation, or an unquoted
delimiter inside a field.

**Recommended approach for the import:** run the DDL below first, then use the wizard's *import into
existing table* path (or `BULK INSERT … WITH (CODEPAGE='65001', FORMAT='CSV', FIELDQUOTE='"', FIRSTROW=2)`),
so no type is guessed at all.

```sql
-- ahp_ranking.csv  (681 rows)
CREATE TABLE dbo.AhpRanking (
    seller_id                        nvarchar(64)     NOT NULL,
    seller_state                     nvarchar(64)     NOT NULL,
    avg_review_score                 decimal(9,6)     NOT NULL,
    on_time_rate                     decimal(9,6)     NOT NULL,
    avg_price                        decimal(14,6)    NOT NULL,
    order_volume                     int              NOT NULL,
    avg_review_raw                   decimal(9,6)     NOT NULL,
    on_time_rate_raw                 decimal(9,6)     NOT NULL,
    n_reviewed                       int              NOT NULL,
    n_orders                         int              NOT NULL,
    norm_avg_review_score            decimal(9,6)     NOT NULL,
    norm_on_time_rate                decimal(9,6)     NOT NULL,
    norm_avg_price                   decimal(9,6)     NOT NULL,
    norm_order_volume                decimal(9,6)     NOT NULL,
    contribution_avg_review_score    decimal(9,6)     NOT NULL,
    contribution_on_time_rate        decimal(9,6)     NOT NULL,
    contribution_avg_price           decimal(9,6)     NOT NULL,
    contribution_order_volume        decimal(9,6)     NOT NULL,
    ahp_score                        decimal(9,6)     NOT NULL
);

-- CallSentiment.csv  (300 rows)
CREATE TABLE dbo.CallSentiment (
    call_id                          nvarchar(64)     NOT NULL,
    agent_id                         nvarchar(64)     NOT NULL,
    transcript                       nvarchar(1000)   NOT NULL,
    hold_seconds                     decimal(9,1)     NOT NULL,
    silence_pct                      decimal(7,4)     NOT NULL,
    compound                         decimal(7,4)     NOT NULL,
    sentiment_label                  nvarchar(64)     NOT NULL
);

-- Clickstream.csv  (100,000 rows)
CREATE TABLE dbo.Clickstream (
    session_id                       nvarchar(64)     NOT NULL,
    event_type                       nvarchar(64)     NOT NULL,
    product_id                       nvarchar(64)     NOT NULL,
    device                           nvarchar(64)     NOT NULL,
    source                           nvarchar(64)     NOT NULL,
    event_time                       datetime2(0)     NOT NULL
);

-- DimCustomer.csv  (99,441 rows)
CREATE TABLE dbo.DimCustomer (
    customer_id                      nvarchar(64)     NOT NULL,
    customer_state                   nvarchar(64)     NOT NULL
);

-- DimProduct.csv  (32,951 rows)
CREATE TABLE dbo.DimProduct (
    product_id                       nvarchar(64)     NOT NULL,
    product_category_name            nvarchar(128)    NULL,
    product_weight_g                 decimal(12,1)    NULL,
    product_category_name_english    nvarchar(128)    NOT NULL
);

-- DimSeller.csv  (3,095 rows)
CREATE TABLE dbo.DimSeller (
    seller_id                        nvarchar(64)     NOT NULL,
    seller_state                     nvarchar(64)     NOT NULL
);

-- discovery_quadrants.csv  (1,864 rows)
CREATE TABLE dbo.DiscoveryQuadrants (
    seller_id                        nvarchar(64)     NOT NULL,
    ahp_score                        decimal(9,6)     NOT NULL,
    order_volume                     int              NOT NULL,
    quadrant                         nvarchar(64)     NOT NULL,
    confidence                       nvarchar(64)     NOT NULL
);

-- FactOrderItems.csv  (110,189 rows)
CREATE TABLE dbo.FactOrderItems (
    order_id                         nvarchar(64)     NOT NULL,
    order_item_id                    int              NOT NULL,
    product_id                       nvarchar(64)     NOT NULL,
    seller_id                        nvarchar(64)     NOT NULL,
    price                            decimal(12,2)    NOT NULL,
    freight_value                    decimal(12,2)    NOT NULL,
    customer_id                      nvarchar(64)     NOT NULL,
    order_status                     nvarchar(64)     NOT NULL,
    order_purchase_timestamp         datetime2(0)     NOT NULL,
    order_delivered_customer_date    datetime2(0)     NOT NULL,
    order_estimated_delivery_date    datetime2(0)     NOT NULL,
    order_date                       datetime2(0)     NOT NULL,
    delivery_days                    int              NOT NULL,
    is_late                          tinyint          NOT NULL,
    review_score                     decimal(3,1)     NULL,
    review_creation_date             datetime2(0)     NULL,
    item_key                         nvarchar(64)     NOT NULL
);

-- LatePredictions.csv  (110,189 rows)
CREATE TABLE dbo.LatePredictions (
    item_key                         nvarchar(64)     NOT NULL,
    late_risk                        decimal(9,6)     NOT NULL,
    late_risk_lr                     decimal(9,6)     NOT NULL,
    late_risk_dt                     decimal(9,6)     NOT NULL
);

-- seller_recommendations.csv  (2,970 rows)
CREATE TABLE dbo.SellerRecommendations (
    seller_id                        nvarchar(64)     NOT NULL,
    action                           nvarchar(64)     NOT NULL,
    rule_id                          nvarchar(64)     NOT NULL,
    because                          nvarchar(1000)   NOT NULL
);
```

---

## C. Repo hygiene and app check

### C.6 Hygiene

- **Tracked files over 1 MB: none.** 84 tracked files; the largest is `reports/seller_recommendations.csv`
  (620 KB). The 67.7 MB model and all data are gitignored.
- **`pip freeze` vs `requirements.txt`:** identical (57 packages, no version differences, nothing
  installed but unpinned, nothing pinned but missing).
- **Tasks:** every `python -m src.*` module referenced by `Makefile` and `tasks.ps1` resolves and has a
  `__main__` guard; `run_all.py`, `streamlit run app/Home.py`, `ruff`, `mypy` and `pytest` targets are
  well formed. I did not execute pipeline modules in the working repo (they would rewrite outputs).
  **One defect:** the `models` target in both files omits `src.expert`, so running it leaves
  `seller_recommendations.csv` stale after a model change (the documented `models` target in
  `CLAUDE.md` §9 includes it).
- **Files referenced nowhere:** four `.gitkeep` files (`app/`, `app/pages/`, `docs/`, `rules/`; all four directories
  hold real files already) and the test modules, which are referenced only by pytest discovery.
- **One-off diagnostics:** six scripts in `scripts/` (`diagnostic_group_leak`, `diagnostic_rf_ceiling`,
  `diagnostic_rf_cv`, `diagnostic_rf_leaf`, `diagnostic_shrinkage`, `diagnostic_temporal`) and six matching
  `reports/diagnostic_*.txt`. They are cited as evidence in `CLAUDE.md` and `BUILD_LOG.md`, but see A.3: three
  do not regenerate their committed report. `diagnostic_rf_ceiling.py` calls `_split_and_encode(table)`, which is now the
  grouped split; `diagnostic_rf_cv.py` still uses a random `StratifiedKFold` (its numbers are the ones
  that were later shown to be inflated).
- **Dead code ruff does not flag:**
  - `app/_common.py::parse_cv_summary` is used only by tests since Design now reads `oof_metrics.json`;
  - the `model_review_flag` / `REVIEW_TOKEN` / `review_line` machinery serves a flag that no longer
    exists in any output;
  - unused `mtime` arguments are cache keys on purpose (not dead).
  Nothing else substantive; no unused imports beyond ruff's.
- **Stale or contradictory `CLAUDE.md` sections:** see the B.3 table. The pattern: §1, §7.3-7.6, §8, §9, §11,
  §18 still describe the Stage 3.5 state; §4 layout lacks `src/explain.py`, the `app/` helper modules,
  `scripts/`, `.streamlit/`, `Makefile` and `docs/DEMO_SCRIPT.md`, and lists a file that does not exist
  (`docs/SYLLABUS_MAP.md`); §7.4 does not mention the shrinkage (k = 18 / 26); §7.3 still says the split is stratified 80/20.
- **Overlapping docs:** `CLAUDE.md` §7.5 and §18 repeat `docs/knowledge_engineering.md` §5-6 and
  `BUILD_LOG.md`'s Stage 3.5 section three times, which is why the same stale number appears in
  three places. `README.md` duplicates nothing and says nothing true about the current state.
- **`BUILD_LOG.md` size and structure:** 583 lines, 12 level-2 sections: Environment notes (8),
  Stage 1 (18), Stage 2 (26), Stage 3 Part A (41), Stage 3.5 (107), Stage 4 (192), Stage 4b (219),
  Stage 4d (294), Stage 5 (342), Stage 5b (414), scikit-learn block (456), Stage 6 (520). No table of
  contents, no current-state summary at the top, and seven headings say "not committed".

  Proposed 1-page summary to put at the top (about 40 lines) and keep the rest as an appendix:
  1. **Current state** (5 lines): commit, 198 tests, ruff/mypy clean, rebuild time (527 s).
  2. **Headline results** (8 lines): model table (test AUC and out-of-fold AUC), AHP 681 / CR 0.0189, quadrants, gem tiers, action counts.
  3. **What changed and why**, one line per stage (Stage 1 … Stage 6), each with a link to its section.
  4. **Honest caveats** (6 lines): score is a ranking not a probability; no forward-looking skill; synthetic data; tiny sellers dominate R01/R03; random-split numbers superseded.
  5. **Environment issues** (3 lines): pandas 2.2.3 pin, mypy 1.13.0, Application Control and sklearn.
  6. **How to rebuild** (3 lines): the commands and expected outputs.

### C.7 App check, headless AppTest in the clone

All seven pages run under Streamlit `AppTest` with **no exception**. Every metric shown, with its source:

| Page | Metric | Value shown | Source file | Matches source? |
|---|---|---|---|---|
| Home | none | | | |
| Intelligence | Delivered orders | 96,470 | `data/processed/FactOrderItems.parquet` | yes |
| Intelligence | Item revenue | R$ 13.2M | `FactOrderItems.parquet` | yes |
| Intelligence | Late-delivery rate | 7.91 % | `FactOrderItems.parquet` | yes |
| Intelligence | Average review score | 4.08 / 5 | `FactOrderItems.parquet` | yes |
| Intelligence | charts | funnel, device, sentiment | `Clickstream.parquet`, `CallSentiment.parquet` (synthetic) | n/a |
| Design | Late-delivery rate | 7.9 % | `reports/model_metrics.json` | yes |
| Design | Ranking score | 74 % | `model_metrics.json` (RF test AUC 0.7410) | yes |
| Design | Inputs used | 12 | `config.ALLOWED_FEATURES` | yes |
| Design | tables/charts | | `coefficients.csv`, `decision_tree.txt`, `oof_metrics.json`, `rf_feature_importance.csv`, `sensitivity.csv` | n/a |
| Choice | Established sellers ranked | 681 | `reports/ahp_ranking.csv` | yes |
| Choice | Minimum items sold | 30 | `config.AHP_MIN_ORDERS` | yes |
| Choice | Consistency ratio | 0.019 | `ahp_ranking.csv` / config matrix | yes (0.0189) |
| Discovery | quadrant counts line | Star 495, Hidden Gem 437, Overrated 439, Overlooked-Low-Quality 493 | `reports/discovery_quadrants.csv` | yes |
| Implementation | Items sold / Average review / Late-delivery rate | 1 / 5.00 / 0.0 % (default seller `003554e2dc`) | `SellerFacts.parquet` | yes |
| Implementation | action, rule | Warn, R03, predicted risk 0.567 | `seller_recommendations.csv`, `rules/seller_rules.yaml` | yes |
| DSS Architecture | Data tables | 16 | data catalog (row sum 465,119) | yes |
| DSS Architecture | Expert rules | 12 | `rules/seller_rules.yaml` | yes |

Two things the smoke surfaced:

1. **The Implementation page opens on a one-item, five-star seller that the batch run labels "Warn"
   (rule R03, predicted risk 0.567)** — a seller with a single order, a perfect review and no late delivery.
   `DEMO_SCRIPT.md` says to start from "a seller where a real rule fired". It is a true result of the rule,
   but a poor first impression.
2. **R01 and R03 fire largely on very small sellers.** 70 of R01's 137 and 120 of R03's 219 have fewer than
   5 items sold (median 4 and 3), and 104 of the 219 R03 "Warn" sellers have never delivered late; 37.2 % of all sellers
   have fewer than 5 items. R06 and R10 apply an items-sold floor, R01 and R03 do not. This is a rule-design
   question, not a bug.

---

## D. Findings by severity

Severity: **1** breaks something or will break the SQL import · **2** a wrong or stale number someone
could quote · **3** misleading wording · **4** clutter.

| # | Sev | Finding | File | Suggested fix |
|---|---|---|---|---|
| 1 | 1 | `product_weight_g` has 2 blanks, the first at row 8,578; the wizard will guess NOT NULL from the first rows and the import fails or drops rows | `data/export/DimProduct.csv` (from `src/export_sql.py`) | Create the tables from the DDL in B.5 with the column NULL; or document *Allow nulls* in `docs/powerbi_setup.md` |
| 2 | 1 | `seller_recommendations.because` has 1,298 rows with an em dash and no BOM; likely mojibake or truncation | `data/export/seller_recommendations.csv` | Import with code page 65001 into `nvarchar(1000)`; or write the CSV with a BOM, or use `-` in `rules/seller_rules.yaml` |
| 3 | 1 | Columns whose length grows after the sampled rows (`product_category_name` 46 vs 34, `…_english` 39 vs 31, `quadrant` 22 vs 10, `item_key` 35 vs 34) will be truncated, and a truncated `item_key` silently breaks the `FactOrderItems`↔`LatePredictions` relationship | `data/export/*.csv`, `docs/powerbi_setup.md` | Pre-create tables from the DDL; add the DDL and "import into existing table" step to `docs/powerbi_setup.md` |
| 4 | 2 | Total "361,860 rows" is wrong; real total 461,680 | `docs/powerbi_setup.md:27` | Replace with 461,680 |
| 5 | 2 | `CLAUDE.md` quotes superseded results as current: AUCs 0.7841/0.6738/0.6347, 423 MB, quadrants 454/478/480/452, rule thresholds and fire counts (§7.3, 7.5, 7.6, 18), "61 passed", "10 source files" | `CLAUDE.md` lines 24, 138, 193-198, 212-216, 225, 241, 402-408 | Update to the values in B.3, or replace the numbers with a pointer to `reports/` and `BUILD_LOG.md` so they cannot drift again |
| 6 | 2 | `docs/knowledge_engineering.md` §6 says "current" but the threshold table is stale (R01 0.53, R03 0.55, R06/R10 0.503, R05 105 sellers, avg_review percentiles) | `docs/knowledge_engineering.md:263-290` | Regenerate the §6 table from `rules/seller_rules.yaml` and the fire counts |
| 7 | 2 | `README.md` is a Stage 1 stub: says models, app, AHP are not implemented and that `make models/run/app` fail | `README.md` | Rewrite: purpose, quick start (`tasks.ps1 setup/data/run/app/check`), outputs, synthetic-data note, pointer to `CLAUDE.md` |
| 8 | 2 | `CLAUDE.md` layout lists a file that does not exist (`docs/SYLLABUS_MAP.md`) and omits real ones; status list stops at Stage 3.5; §11 "Stage 3, next up" is finished | `CLAUDE.md:13-18, 117, 288` | Create the file or delete the line; update status and layout |
| 9 | 2 | `DEMO_SCRIPT.md` and `knowledge_engineering.md` quote temporal figures from a fully grown forest that the committed script no longer reproduces (without calendar: 0.44 vs 0.42 on re-run; all-12: 0.5210 vs 0.5178) | `docs/DEMO_SCRIPT.md:178-179`, `docs/knowledge_engineering.md:619-620`, `reports/diagnostic_temporal.txt` | Re-run `scripts/diagnostic_temporal.py`, commit the new report, and update both quotes (or state "fully grown forest" next to them) |
| 10 | 2 | `make models` / `tasks.ps1 models` omit `src.expert`, so `seller_recommendations.csv` is not regenerated after a model change | `Makefile:17`, `tasks.ps1` | Append `python -m src.expert` |
| 11 | 2 | `diagnostic_group_leak`, `diagnostic_temporal` (forest) and `diagnostic_rf_ceiling` (see A.3) do not regenerate their committed reports because `predict` defaults changed | `scripts/diagnostic_*.py`, `reports/diagnostic_*.txt` | Pin the diagnostic's own forest settings inside each script, or mark the reports "historical, produced with leaf 1" |
| 12 | 2 | `BUILD_LOG.md` never records the final test count after Stage 6 (last "192 passed"; real 198) | `BUILD_LOG.md:515, 520-583` | Add one gate line to the Stage 6 section |
| 13 | 3 | `run_all.py`'s final summary prints LR and DT AUC but not the primary random forest | `run_all.py:177-201` | Add `random_forest`, and label it primary |
| 14 | 3 | "probability" and "better-calibrated" used for a score that is a ranking (median 0.34 vs 7.9 % late rate) | `src/predict.py:509, 522`; `CLAUDE.md:402`; `BUILD_LOG.md:185`; `docs/knowledge_engineering.md:83` | Say "out-of-fold risk score" and "ranking score, not calibrated" |
| 15 | 3 | Seven `BUILD_LOG.md` headings say "(not committed)" though everything is committed | `BUILD_LOG.md:192, 219, 294, 342, 414, 456, 520` | Remove the suffix |
| 16 | 3 | Implementation page opens on a one-item five-star seller flagged Warn (R03) | `app/pages/5_Implementation.py`, `docs/DEMO_SCRIPT.md` | Default to a seller with at least 5 items sold, or the first Keep/Suspend with a fired rule |
| 17 | 3 | R01 and R03 fire mostly on sellers with fewer than 5 items sold (104 of 219 R03 Warn sellers never late) | `rules/seller_rules.yaml` (R01, R03) | Add `order_volume >= 5` like R04, R05, R06, R10, then re-run the drift test; or document it |
| 18 | 3 | Stage 3 AUCs 0.6347 / 0.6738 sit in the log without a note that the random split inflated them | `BUILD_LOG.md:43-59` | One-line "superseded by Stage 6" note |
| 19 | 4 | Four `.gitkeep` files in directories that already hold files | `app/`, `app/pages/`, `docs/`, `rules/` | Delete them |
| 20 | 4 | Six one-off diagnostic scripts and six reports in the main tree | `scripts/`, `reports/diagnostic_*.txt` | Move to `scripts/archive/` and `reports/archive/` once item 11 is settled |
| 21 | 4 | Dead helpers: `parse_cv_summary`, the `model_review_flag` / `REVIEW_TOKEN` / `review_line` machinery | `app/_common.py`, `src/explain.py` and callers | Remove along with their tests |
| 22 | 4 | `BUILD_LOG.md` is 583 lines with no summary at the top | `BUILD_LOG.md` | Add the one-page summary proposed in C.6 |
| 23 | 4 | `CLAUDE.md` §7.5, §18 duplicate `knowledge_engineering.md` §5-6 and `BUILD_LOG.md` Stage 3.5 | those three files | Keep the narrative in one place, link from the others |

## Overall verdict

**Code and data: pass.** A clean checkout rebuilds byte-for-byte in about nine minutes, all 198 tests
pass, ruff and mypy are clean, nothing depends on an untracked file, every export equals its source, and the
action, rule and quadrant counts reconcile with each other.

**Documentation: fail until items 4-9 are fixed.** Someone quoting the AUC, the file size, the quadrant
counts, the rule thresholds or the row total from `CLAUDE.md`, `README.md`, `docs/knowledge_engineering.md`
§6 or `docs/powerbi_setup.md` would be quoting the wrong number.

**SQL Server import: do not run the wizard blind.** Items 1-3 are expected to fail or truncate. Create
the tables from the DDL above and import into them with UTF-8 (code page 65001); then the import should be
clean. I have not run SSMS, so that expectation is inferred from the data, not tested.

No tracked file was changed by this audit.
