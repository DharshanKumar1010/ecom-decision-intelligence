# Full project audit (2026-10-17 to 2026-10-18)

Written for the project owner. Starting point: HEAD `dfab042` (equal to `origin/main`), working tree clean, 338 tests.
Nothing was committed or staged. `powerbi/EcomDSS.pbix` SHA-256 before and after:
`eaa0c60e6df9732abf6c2781f4fe8431e4b15f4f48a7f3dbb48b6f59768e8a45` (unchanged). Scratch work lives in
`%TEMP%\ecom_audit\` (independent scripts, sandboxes, the SQL read-only script and its query log).

## A. Independent recomputation (phases 1 and 2)

Phase 1 was done by a fresh sub-agent given only `data/raw` and the definitions; I compared its output with the
pipeline. Phase 2 was done by me without importing `src/` (the pairwise matrix and the YAML were read as data).

| Target | Independent | Pipeline | Status |
|---|---|---|---|
| Delivered orders (status delivered and a delivered date) | 96,470 (96,478 delivered, 8 without a date) | 96,470 | equal |
| Item rows / seller-order pairs | 110,189 / 97,811 | 110,189 / 97,811 | equal |
| Revenue (sum of price) | 13,220,248.93 | 13,220,248.93 | equal |
| Late rate over items / over orders | 0.0790823040 / 0.0811236654 | 0.0790823040 / 0.0811236654 | equal |
| Average review per seller-order pair / distinct order / item | 4.137600 / 4.155796 / 4.081738 | 4.137621 / 4.155848 / 4.081966 | equal to 2 decimals (4.14 / 4.16 / 4.08); fifth-decimal difference from finding F1 |
| Share without a review (orders / items) | 0.6696% / 0.7505% | 0.6696% / 0.7505% | equal |
| Sellers with items / >= 5 / >= 30 | 2,970 / 1,864 / 681 | 2,970 / 1,864 / 681 | equal |
| Per seller, all 2,970: items sold, `n_orders`, `n_reviewed`, late rate | | | **0 differences** |
| Per seller, all 2,970: average review per order | | | 13 sellers differ (max 0.19), all caused by F1 |
| 20 named sellers (`n_orders`, `n_reviewed`, items, late rate; 5 with no reviewed order) | | | equal in files, pages and SQL |
| AHP weights / lambda max / CI / CR | 47.29, 28.44, 7.29, 16.99% / 4.05111 / 0.01704 / 0.01893 | same (ahp_ranking contributions match to 1e-15) | equal |
| Shrinkage strengths (method of moments, >= 5 items, order level) | review 17.80, on-time 25.79 to 25.91 | 17.82 and 25.91 | both round to 18 and 26 |
| Median items sold in the scored pool; quadrants | 18; Star 495, Hidden Gem 437, Overrated 439, Overlooked-Low-Quality 493 | same | equal counts; 2 sellers' labels differ (F1) |
| Gem tiers / confidence flag at 15 items | 44 normal, 393 low / 0 differences | same | equal |
| Old (item-level, unshrunk) gem definition | 478 gems; 420 stay, 58 leave, 17 enter | DEMO_SCRIPT question 4 | reproduced exactly |
| Independent rule evaluator on 2,970 sellers (YAML read as data, gates included) | Keep 2,277, Warn 234, Suspend 75, Feature 174, Promote 210 | same; 0 action and 0 rule differences, per-rule fire counts equal | equal |
| Same evaluator on the independent review values | 0 differences (R10 raw matches 484 instead of 485: one seller crosses 4.28) | | no action effect |
| Rounding as SQL stores it (6 decimals) | 0 action changes | | safe |
| Percentile drift of the thresholds | R01 0.7508 (n=1,846), R03 0.9009 (n=1,847), R06/R10 0.4997, R09 0.2540, R11 0.0998, R05 0.918, R11 late rate 0.758 | intent 0.75, 0.90, 0.50, 0.25, 0.10, about 0.90, about 0.75 | no drift beyond the documented rounding |
| Clickstream funnel (synthetic; cannot be verified against raw) | 70,000 / 15,000 / 9,000 / 6,000, causal within each session | same | consistent |

Definitions that differ from the owner's: only the review tie-break (F1). Delivered/late/items/pairs/`n_orders`/`n_reviewed`
follow the stated definitions exactly. `late` per order is the maximum over a seller-order pair's items (identical for all items of an order).

## B. Findings

Severity: 1 breaks something or contradicts evidence, 2 wrong or stale number someone could quote, 3 misleading wording or missing edge
case, 4 clutter.

| ID | Sev | Where | Issue | Fix | Tier | Status |
|---|---|---|---|---|---|---|
| F1 | 3 | `src/build_tables.py` `_dedupe_latest_reviews` | When two reviews of one order share the latest creation date (154 delivered orders; 17 with different scores) the pipeline keeps whichever the sort leaves first. Effect measured in a sandbox: 26 item rows, 13 sellers' average review, quality score shift <= 0.02, 2 quadrant label swaps; no recommendation, gem or tier changes | none; documented as a known limitation (`docs/knowledge_engineering.md`, end of section 8b) | 2 | declined by owner |
| F2 | 3 | `sql/schema.sql` view `vw_SellerScorecard` | `items_sold` was empty for 1,231 of 3,095 sellers (1,106 of them sold 1 to 4 items) | view now counts items from `FactOrderItems`, 0 for 125 sellers; `VIEW_SOURCE_TABLES` in `src/load_sql.py` lists the fact table; new tests; SQL reloaded and verified | 2 | applied (approved) |
| F3 | 2 | `docs/MANUAL_CHECKS.md` query 3 | expected em-dash count 1,149 was stale (1,155 in CSV and SQL) | corrected | 1 | fixed |
| F4 | 2 | `docs/DEMO_SCRIPT.md` questions 1, 10 | "424 MB" before tuning; RESULTS.md says 421.2 MB | corrected to 421 | 1 | fixed |
| F5 | 2 | `docs/DEMO_SCRIPT.md` question 8 | Warn and Suspend gate described as 5 items only | now includes the order gates | 1 | fixed |
| F6 | 3 | `docs/powerbi_setup.md` | first blank weight "row 8,578" (it is data row 8,579) | corrected | 1 | fixed |
| F7 | 3 | `CLAUDE.md` sections 1 and 11 | "no date column as a feature" contradicted the allowed purchase month and weekday features | reworded; owner-approved order-gate text added in 1, 7.4, 11 | 1 | fixed |
| F8 | 3 | Intelligence KPI tooltip, `RESULTS.md` | "per reviewed order" ambiguous (4.14 vs 4.16); 4.16 not in RESULTS.md; the 0.75% share was typed | tooltip names the seller-order definition and gives 4.16 and 4.08; RESULTS.md shows all three; share computed | 1 | fixed |
| F9 | 3 | Implementation page | crashed when a seller's counts were missing; "1 items sold" | missing counts show a dash; "1 item sold"; live-engine facts moved to a tested helper | 1 | fixed |
| F10 | 3 | `README.md` | Makefile target list lacked `load-sql`; no data attribution; run time "about 9 minutes" | list fixed; attribution and synthetic-data paragraph added; run time measured 25 min under load (earlier unloaded runs 9 to 10 min) so left as is | 1 | fixed |
| F11 | 3 | tests | two of twelve deliberate breaks survived (late-flag sign; one "Synthetic data" tag removed) | `tests/test_build_tables.py` (5 tests), per-heading tag tests | 1 | fixed |
| F12 | 4 | `src/load_sql.py` `assert_statement_allowed` | a `SELECT ... INTO #temp FROM dbo.<loader table>` would pass the allow-list (no such statement exists; all SELECTs are fixed strings) | not changed (hardening: add `INTO` to the forbidden words) | 2 | needs your decision |
| F13 | 4 | `docs/SYLLABUS_MAP.md` | no statement of topics not implemented | section added (neural networks, SVM, nearest neighbour; other out-of-scope topics) | 1 | fixed |
| F14 | 4 | `docs/DEMO_SCRIPT.md` | no answers to the five hard questions; Promote limitation not stated | questions 13 to 17 added with a test tying their figures to RESULTS.md | 1 | fixed |

Not defects (owner notes): 4.14 vs 4.16; Promote not gated on orders (now stated in question 16 and CLAUDE.md 7.4).

## C. Tier 2 proposals and your answers

1. **F1 review tie-break** (patch: `_REVIEWS_COLS` gains `review_answer_timestamp` and `review_id`; sort ascending by creation date,
   answer time, id with a stable sort and keep the last row; return the three original columns). Sandbox impact: `FactOrderItems`
   26 rows / 17 orders; `SellerFacts.avg_review` 13 sellers; `ahp_ranking.csv` ahp_score changes on 679 of 681 rows (max 0.0203); quadrant
   file 2 sellers swap labels, counts 495/437/439/493 and the 437 gems unchanged; **`seller_recommendations.csv` byte-identical**
   (Keep 2,277, Warn 234, Suspend 75, Feature 174, Promote 210; all fire counts equal); k estimate 17.819 to 17.804. **Answer: no, document it.**
2. **F2 view `items_sold`** (patch in `sql/schema.sql`: `COALESCE(i.items, 0) AS items_sold` with a left join on per-seller item counts).
   Impact: view only; 1,231 NULLs become counts (125 zeros), the other 1,864 values identical, sum 110,189; tables, CSVs, models,
   rules and quadrants unchanged. **Answer: yes. Applied.** Before/after hashes: `sql/schema.sql` changed, `src/load_sql.py`
   changed (one constant), every file under `reports/`, `data/` and `models/` identical (section G).

## D. Test-suite quality

Twelve deliberate breaks applied one at a time in a sandbox copy (`%TEMP%\ecom_audit\mut`):

| # | Break | Killed by the 338 original tests | Killed by the current suite |
|---|---|---|---|
| 1 | flip a comparison in R04 | yes | yes |
| 2 | R03 threshold x 1.1 | yes | yes |
| 3 | swap two actions in the YAML | yes | yes |
| 4 | `order_volume >= 5` to `> 5` in R04 | yes | yes |
| 5 | shrinkage k = 0 | yes | yes |
| 6 | average review per item instead of per order | yes | yes |
| 7 | late-flag sign reversed | **no** | yes (`tests/test_build_tables.py`) |
| 8 | 100 fact rows dropped before the export | yes | yes |
| 9 | "Synthetic data" tag removed from one heading | **no** | yes (per-heading tests) |
| 10 | two quadrant colours made equal | yes | yes |
| 11 | R03 `n_orders >= 3` to `>= 2` | yes | yes |
| 12 | `n_reviewed` not passed to the live engine | yes | yes |

Kill rate: **10 of 12 before, 12 of 12 after.** Flaky tests and slow tests: see section G.

## E. SQL cross-check (read-only)

75 SELECT statements (log: `%TEMP%\ecom_audit\sql_queries.log`; every statement starts with SELECT, enforced in the script) against
`localhost\SQLEXPRESS` / `EcomDecisionIntelligence`, Windows authentication; the only other database access was
`python -m src.load_sql` (full reload under the approved Tier 2) and `--verify-only`.

- Row counts of the ten tables equal the CSVs and `docs/powerbi_setup.md` (461,680 rows in total); the view has 3,095 rows, 3,095 distinct sellers.
- Primary keys unique (Clickstream has none by design; its natural key (session, event type) is unique); 8 foreign keys declared, trusted, 0 orphans.
- 70 columns: type, length, precision, scale, nullability and position equal `sql/schema.sql`; no extra columns.
- NOT NULL columns have 0 nulls; nullable columns: product category 610, weight 2, review score 827, review date 827, equal to the CSVs.
- **Every value of all ten tables equals the CSV** (full-table comparison, 0 differing columns); sums of price, freight and the three risk columns equal.
- `because`: 1,155 rows contain an em dash (equal to the CSV), longest text 371, no replacement characters, no truncation.
- After the reload the view's `items_sold` has 0 NULLs, 125 zeros, sums to 110,189, type int.

Seven Power BI measures computed in SQL (written to `docs/powerbi_measures.md`, "Expected values (verified in SQL)"):
Delivered Orders **96,470**; Items Sold **110,189**; Revenue **R$ 13,220,248.93**; Late Rate **7.91%**; Avg Late Risk Score **0.361**;
Hidden Gems **437**; Sellers Ranked **681**. No differences.

## F. Security, secrets, licence, dependencies

- `src/load_sql.py`: Windows authentication only, database and table allow-lists, values parameterised (`?`), statements checked against an
  allow-list before execution, one transaction per table with rollback on any error or count/checksum mismatch, no credential anywhere;
  `TrustServerCertificate=yes` is documented as local-only (README, docstring). Gap: F12.
- Secrets: working tree and all 24 commits scanned for passwords, tokens, keys, credentialed connection strings and e-mail addresses:
  only the statements "no password exists" and a test fixture. No file over 1 MB, `.pbix`, `.joblib`, parquet or raw CSV was ever committed.
- Dependencies: 58 exact pins, `pip freeze` equals `requirements.txt`, every import is pinned, every unused pin is a transitive or tool
  dependency. Licences: all permissive (MIT, BSD, Apache-2.0, PSF, MIT-CMU, dual BSD/Apache); two are MPL-2.0 (certifi, pathspec), a
  file-level copyleft that does not affect this project. pyodbc needs the Microsoft ODBC Driver 17 or 18; the README says so.
- Olist licence: **to verify**. The Kaggle page could not be read by the fetch tool (no licence text returned); the README now says
  "see the dataset page for licence terms". Check the page before publishing the repository.
- Personal data: none in the repository, exports or database: customers and sellers appear only as pseudonymous Olist ids and a
  two-letter state; no names, e-mail addresses, cities, zip codes or phone numbers (transcripts are synthetic and contain no long digit runs).
- Robustness: `run_all.py`, `src.build_tables` (missing raw file) and the other entry points exit non-zero on failure (1), `load_sql` exits 2 for a refusal;
  pipeline steps and tests run from a path with spaces; paths come from `config.py`; seed 42.
- Hygiene scans: no TODO/FIXME, no bare `except`, no mutable default arguments, no commented-out code, `print` only in the diagnostic scripts and `run_all.py`.

## G. Hashes, tests, reproducibility

- **Tests:** 338 at the start (`.	asks.ps1 verify`, exit 0) and **385 passed** at the end, twice in a row (a plain `pytest` run, then the
  pytest step of `.	asks.ps1 verify`): no flaky test. 47 tests were added (`tests/test_app_edge.py`, `tests/test_build_tables.py`, view,
  document and tooltip tests). Only one test takes over 30 s: the one-off setup of `tests/test_predict.py::test_late_risk_in_unit_interval`
  (38.8 s); next are `test_seed_reproducibility` (25.9 s) and `test_clickstream_causal_consistency` (15.6 s).
- **`.	asks.ps1 verify` at the end:** ruff PASS, mypy PASS (13 files), pytest PASS, SQL verify-only PASS, RESULTS.md fresh PASS, app smoke PASS.
- **Manifest Z0 (start) against Z1 (end), 126 files:** the 48 generated files that are checked (`reports/`, `data/processed/`,
  `data/export/`, `models/`, `sql/`) are byte-identical except `reports/RESULTS.md` (gem-outcome line and the three review averages) and
  `sql/schema.sql` (the approved view change). Models, parquet files, exports, `seller_recommendations.csv`, `ahp_ranking.csv`,
  `discovery_quadrants.csv` and `LatePredictions` did not change. Other changed files, all in the allowed set: README, CLAUDE.md, BUILD_LOG,
  `docs/`, `app/` (common helper, number format, two pages), `scripts/make_results.py`, `src/load_sql.py` (one constant, Tier 2 approved)
  and tests. New files: `docs/archive/CLAIM_REGISTER.md`, this report, `tests/test_app_edge.py`, `tests/test_build_tables.py`.
- **Clean-room rebuild** (clone of HEAD plus `data/raw`, `python run_all.py --export` twice; each run passed its own pytest, ruff and mypy gates):
  both runs exited 0 (1,578 s and 1,493 s on a busy machine) and produced identical outputs apart from four timestamp fields in
  `models/*_metadata.json` and `registry.json`; against the working repository only those timestamps, `RESULTS.md` (documentation edits made
  in this audit) and `sql/schema.sql` (line endings only: the clone has CRLF) differed.
- **`powerbi/EcomDSS.pbix` SHA-256:** `eaa0c60e6df9732abf6c2781f4fe8431e4b15f4f48a7f3dbb48b6f59768e8a45` before and after. Nothing is staged;
  `.pbix` and `powerbi/` are not tracked; `HEAD` equals `origin/main` (nothing unpushed before this audit's changes).

## H. Needs your decision

1. Whether to harden `assert_statement_allowed` against `SELECT ... INTO` (F12). It is a change to `src/load_sql.py`.
2. The Olist licence text: confirm on Kaggle (I could not read it).
3. Optional: apply the review tie-break fix (F1) after the Power BI report is finished; the evidence is in section C.
4. `CLAUDE.md` constraints that still match your decisions: SQL exception (three files; `src/load_sql.py` and `sql/schema.sql` were
   edited under the Tier 2 approval for F2), no neural networks/SVM/KNN (now mirrored in `docs/SYLLABUS_MAP.md`), numbers live in
   `reports/RESULTS.md`. None no longer matches.

## I. Still to check by hand

- The Power BI report (`powerbi/EcomDSS.pbix`): press Refresh (the view `items_sold` changed), then check the seven cards against section E.
- The PowerPoint, if any, against `docs/DEMO_SCRIPT.md` (questions 13 to 17 are new).
- How every page looks: the tick table in `docs/MANUAL_CHECKS.md`.
- The GitHub author e-mail shown by `git log -1 --format="%an <%ae>"` after the commit.
- Colour-blind separation is the weakest visual point (section K).

## J. Commit commands (PowerShell; one commit; not run; no attribution line)

```powershell
git add -A
git --no-pager diff --cached --stat
git diff --cached --name-only | Select-String -Pattern '\.pbix$|^powerbi/|^data/|^models/|^\.venv/'   # must print nothing
git commit -m "audit: independent verification, SQL view items_sold fix, edge-case and mutation tests, claim register"
git log -1 --format="%an <%ae>"
git push
```

## K. What surprised me

- The 17 tied reviews: the only place where the pipeline and an independent reading of the definitions disagree on data, and it moves the
  quality score of nearly every seller a little (the min-max scaling depends on a few extreme sellers) while changing no decision.
- Everything else matched to the last digit, including 2,970 rule outcomes from an independent evaluator and every value in SQL.
- The view's `items_sold` was empty for a third of the sellers; no document mentioned it.
- Under red-green colour blindness the smallest colour differences (OKLab x 100) are 7.0 between two quadrant colours (Hidden Gem and
  Overlooked-Low-Quality), 6.4 between two action colours (Keep and Promote) and 6.6 (Suspend and Promote under protanopia): inside the
  "needs a second cue" band the theme file already admits; each is also named in words on the pages. Contrast on the dark surfaces is
  fine (marks at least 4.8:1, text at least 7.0:1), with no white fills, one sans-serif font and no remote URLs.
- A full rebuild took about 25 minutes while the machine was busy; the documented 9 to 10 minutes is the idle figure.
