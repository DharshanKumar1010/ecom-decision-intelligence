# E-Commerce Decision Support System (BIA 21CSE421T course project)

A Python + Streamlit decision support system on the Olist Brazilian e-commerce dataset, presented to end users through a SQL Server + Power BI layer. It answers two business questions:

1. **Which sellers should the marketplace keep, warn, suspend, or feature, and why?**
2. **Which sellers and products are genuinely good but under-exposed — "hidden gems" that deserve a visibility boost rather than being buried under high-volume, high-marketing sellers?**

It does this by combining clickstream analytics, call-transcript sentiment, late-delivery risk prediction, multi-criteria (AHP) seller scoring, a quality-vs-popularity quadrant analysis, and a rule-based expert system with explanations. The Python side produces verified, versioned outputs; SQL Server and Power BI consume those outputs to deliver the BIA course's T1-T15 tutorial deliverables (star schema, DAX, what-if parameters, paginated report, row-level security).

This file is the source of truth for every Claude Code session. Read it fully before doing anything. If a request conflicts with this file, say so and ask before proceeding.

**Status:**
- Stage 1 (scaffolding, data build, synthetic data, clickstream) — complete and verified. See section 13.
- Stage 2 (sentiment, prediction, AHP, discovery, expert system) — complete and verified. See section 14.
- Stage 3 (CSV export for SQL Server import) — not started. See section 15.
- SQL Server + Power BI layer (T1-T15) — manual GUI work outside Claude Code's scope, done directly in SSMS/Power BI Desktop. See section 16 for what it consumes and how it maps to the course tutorials.

---

## 1. Hard constraints (never violate)

- **No SQL Server, no SSMS, no Power BI dependency inside `src/`, `app/`, or any Python code Claude Code writes.** All storage in the Python layer is CSV/Parquet under `data/`. SQL Server and Power BI are an approved **downstream, manual presentation layer** that reads the Stage 3 CSV exports — see section 16. This does not reopen scope for Claude Code to write SQL, connect to a database, or generate `.pbix` files; that layer is built by hand in SSMS and Power BI Desktop, not by Claude Code.
- **No neural networks, no SVM, no KNN**, anywhere: code, dependencies, comments, docs. Predictive models are limited to `LogisticRegression` and `DecisionTreeClassifier` from scikit-learn.
- **Only features Claude Code can build and verify from the terminal.** No GUI-only steps, no manual clicking, no paid APIs, no network calls at runtime (after `pip install`).
- **Scope is seven syllabus-anchored topics in the Python layer, not all of them.** Do not add extra topics unless asked. See section 2.
- **`data/raw/` is read-only.** Never modify, rename, or delete files there. Never commit it.
- **Never fabricate results.** Every metric, row count, AUC, or quadrant count quoted in docs or UI must come from a real run of the code. If something cannot be measured, say so.
- **Never claim statistical confidence a small sample doesn't support.** The hidden-gems feature (section 7.7) surfaces low-order-count sellers/products on purpose — every such output must carry an explicit low-confidence flag, never presented with the same certainty as the main AHP ranking.
- **Do not silently expand the predictive feature set.** Section 14's AUC numbers (0.57) are a known, disclosed limitation of a deliberately small 4-feature model, not a bug. Adding features (candidates listed in section 17) is an optional, explicitly-requested enhancement only — never do this without being asked.
- **Ask before adding any new dependency**, changing the model family, or changing the scope.

---

## 2. Scope: syllabus topics implemented (Python layer)

| Unit | Syllabus topic | Where it lives | Dashboard page |
|---|---|---|---|
| 1 | Clickstream analysis (metrics) | `src/clickstream.py` | Intelligence |
| 2 | Phases of decision making, DSS components | `app/` structure + `app/pages/6_DSS_Architecture.py` | All pages, DSS Architecture |
| 3 | Predictive modeling + sensitivity analysis | `src/predict.py` | Design |
| 3 | Sentiment analysis process + speech analytics | `src/sentiment.py` | Intelligence |
| 4 | Multi-criteria decision making, pairwise comparison (AHP) | `src/ahp.py` | Choice |
| 5 | Expert systems (rules, inference, explanation) | `src/expert.py`, `rules/seller_rules.yaml` | Implementation |
| 4 + 5 (extension) | Quality-vs-popularity long-tail analysis ("hidden gems") | `src/discovery.py` | Choice, Discovery |

The **SQL Server + Power BI layer** (section 16) separately covers the BIA course's own tutorial sequence (T1-T15: SSMS setup, data prep, loading, star-schema modeling, SQL-backed data model, DAX, report design, what-if parameters, paginated reports, row-level security) — this is course-required tooling practice, distinct from the seven syllabus *topics* above, which live in the Python layer.

Explicitly **out of scope for the Python layer**: competitive intelligence, uncertainty/payoff methods, linear programming, EV decision trees, maps, PDF reports, Docker, neural networks, SVM, KNN, literal product nutrition/health data (Olist has none; "healthier" in this project means business quality, not food health — see section 7.7.1).

Decision phase mapping (Simon): Intelligence (find the problem) -> Design (model it) -> Choice (evaluate and select) -> Implementation (act and explain). The main dashboard pages are named after these phases; Discovery is a satellite of Choice specifically for the hidden-gems view.

DSS subsystem mapping (used on the Architecture page and in docs):
- Data management: `src/build_tables.py`, `src/synthetic.py`, `data/processed/`, plus SQL Server as the downstream data store
- Model management: `src/predict.py`, `src/sentiment.py`, `src/ahp.py`, `src/discovery.py`
- Knowledge base: `src/expert.py`, `rules/seller_rules.yaml`
- User interface: `app/` (Streamlit, internal/dev view) and Power BI (the graded deliverable view)

---

## 3. Tech stack

- Python 3.11+ (verified working: 3.11/3.12 with the pinned versions below)
- pandas **2.2.3** (pinned — 3.0.6 was blocked by Windows Application Control / Smart App Control on this machine; do not upgrade without checking that first), numpy, pyarrow (Parquet I/O), scikit-learn, scipy, vaderSentiment
- streamlit, plotly
- pyyaml
- pytest, ruff, mypy **1.13.0** (pinned — 2.3.1 hit the same Application Control block)
- Pin exact versions in `requirements.txt` (generate with `pip freeze` after a clean install). Use a virtual environment at `.venv/`.
- No runtime downloads (for example no `nltk.download`). `vaderSentiment` bundles its lexicon.
- **Known environment issue:** on this Windows machine, freshly-installed compiled binaries for some packages (pandas 3.0.6, mypy 2.3.1 confirmed) get blocked at import by an Application Control policy with the error "An Application Control policy has blocked this file." numpy and pyarrow were unaffected. If a new package hits this, pin to an older, long-established release rather than fighting the OS policy, and note it in `BUILD_LOG.md`.
- **Downstream, outside Claude Code's toolchain:** SQL Server Express + SSMS (data store and import), Power BI Desktop (modeling, DAX, dashboard, paginated report, RLS). Neither is scripted or automated by Claude Code — see section 16.

---

## 4. Repository layout

```
.
├── CLAUDE.md
├── BUILD_LOG.md               # stage-by-stage log: what ran, real output, environment notes
├── README.md
├── tasks.ps1                  # PowerShell task runner (Windows-first; see section 9)
├── requirements.txt
├── pyproject.toml             # ruff + mypy + pytest config
├── run_all.py                 # runs the full Python pipeline in order
├── data/
│   ├── raw/                   # 9 Olist CSVs (gitignored, read-only)
│   ├── processed/             # generated parquet/csv (gitignored)
│   └── export/                # Stage 3 — flat CSVs for SQL Server import (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── build_tables.py
│   ├── synthetic.py
│   ├── clickstream.py
│   ├── sentiment.py
│   ├── predict.py
│   ├── ahp.py
│   ├── discovery.py
│   ├── expert.py
│   └── export_sql.py          # Stage 3 — Parquet -> CSV for SQL Server (no DB connection)
├── rules/
│   └── seller_rules.yaml
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Intelligence.py
│       ├── 2_Design.py
│       ├── 3_Choice.py
│       ├── 4_Discovery.py
│       ├── 5_Implementation.py
│       └── 6_DSS_Architecture.py
├── models/                    # joblib artifacts + metadata JSON (gitignored)
├── reports/                   # metrics, tree text, sensitivity, dq report, discovery report
├── docs/
│   ├── knowledge_engineering.md
│   ├── SYLLABUS_MAP.md
│   └── powerbi_setup.md       # manual steps: SSMS import, Power BI model, DAX, RLS
└── tests/
```

`.gitignore` must cover: `data/raw/`, `data/processed/`, `data/export/`, `models/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`. Power BI `.pbix` files are built and kept locally / submitted separately — not part of this repo's scope unless you decide otherwise later.

---

## 5. Engineering standards

- Type hints on every function signature. Docstrings (one-line summary plus params/returns for non-trivial functions).
- Use `logging` with a module-level logger configured in `config.py`. **No bare `print`** except in `run_all.py`'s final summary.
- All paths, constants, the seed, the AHP matrix, and the discovery thresholds live in `src/config.py`. No magic numbers or hard-coded paths elsewhere.
- **Seed = 42** for every random operation (`numpy.random.default_rng(SEED)`, `random_state=SEED`).
- Functions in `src/` are **pure where possible**: take DataFrames in, return DataFrames out. File I/O happens only in clearly named `load_*`/`save_*`/`main()` functions.
- **UI has no business logic.** Streamlit pages only call functions from `src/` and render results.
- Scripts are **idempotent**: rerunning overwrites outputs with identical content (given the same inputs and seed) — confirmed for Stage 1 and Stage 2.
- Every pipeline stage ends with **assertions** on schema, uniqueness, and row-count ranges, failing loudly with a clear message.
- Use vectorized pandas/numpy. No row-wise Python loops over the fact table (110,189 rows).
- Keep functions under ~50 lines where practical. Prefer small composable functions.
- Format and lint with `ruff` (line length 100). Type-check `src/` with `mypy --strict`; currently clean on 10 source files (Stage 2 added `sklearn.*`/`joblib.*`/`vaderSentiment.*`/`yaml.*` to the `pyproject.toml` ignore-missing-imports overrides, same pattern as the pre-existing pandas/pyarrow entries) — keep it clean as files are added.
- Windows-first tooling: prefer `tasks.ps1` / `python -m` invocations over `make` unless the user confirms `make` is available.

---

## 6. Data specification

### 6.1 Raw inputs (Olist, in `data/raw/`) — verified present, Stage 1

```
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_order_reviews_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
olist_customers_dataset.csv
olist_order_payments_dataset.csv
olist_geolocation_dataset.csv
product_category_name_translation.csv
```

### 6.2 Cleaning rules (`src/build_tables.py`) — implemented, Stage 1

Confirmed real output from the last run (quote these, do not recompute unless the pipeline changes):
- Raw orders: 99,441 -> delivered-filtered: 96,470 -> `FactOrderItems`: 110,189 rows
- `DimSeller`: 3,095 rows; `DimProduct`: 32,951 rows
- **`DimCustomer`: 99,441 rows.** (Correction: earlier versions of this file incorrectly quoted 96,470 for `DimCustomer`, copying the delivered-orders count. `DimCustomer` is built from all raw customers, not filtered to delivered orders — 99,441 is the correct, verified number, confirmed again in the Stage 2 run. This was a documentation error, not a code regression; `build_tables.py` was not modified.)
- Null rate: `review_score`/`review_creation_date` 0.7505% (orders with no review), all other fields 0%
- Late-delivery rate: 7.9082%
- Assertions passing: `item_key` unique, fact rows in 100,000-115,000, no null `seller_id`, `is_late` in {0,1}
- `reports/dq_report.md` written

### 6.3 Synthetic data (`src/synthetic.py`) — implemented, Stage 1

Confirmed real output:
- `Clickstream`: 100,000 rows. Funnel: page_view 70,000 (100%), add_to_cart 15,000 (21.43% stage, 21.43% overall), checkout 9,000 (60.00% stage, 12.86% overall), purchase 6,000 (66.67% stage, 8.57% overall). `bounce_rate` = 0.7857, `cart_abandonment_rate` = 0.600.
- `CallTranscripts`: 300 rows, agents AG01-AG10.
- Both are **synthetic and must be labelled as such everywhere they appear** (dashboard captions, README, report, and the Power BI report — see section 16) since Olist has no clickstream or call data.

---

## 7. Module specifications

### 7.1 `src/clickstream.py` (Unit 1) — implemented, Stage 1

`funnel_conversion`, `bounce_rate`, `cart_abandonment_rate`, `funnel_by_device` are all built and tested.

### 7.2 `src/sentiment.py` (Unit 3) — implemented, Stage 2

Four separate, individually testable stages: `preprocess`, `extract_features`, `classify`, `score`. `classify()` scores the **original**, not the aggressively-stripped preprocessed text, since VADER needs punctuation and capitalization cues — documented in the module docstring. Outputs: `CallSentiment.parquet` (300 rows, per-call labels) plus a per-agent aggregate table (mean sentiment, mean hold seconds, mean silence %, negative-call share). Speech analytics here starts from transcripts only — no audio, no ASR — stated plainly in docs and on the dashboard.

### 7.3 `src/predict.py` (Unit 3) — implemented, Stage 2

- **Target:** `is_late`. **Features (exactly these four, enforced by a test against `config.ALLOWED_FEATURES`):** `price`, `freight_value`, `product_weight_g`, `same_state`.
- **Leakage rule (critical, verified):** no date-derived column anywhere in the feature set. `grep` of `src/predict.py` confirms `delivery_days`/date references appear only in the exclusion docstring, never in code.
- Stratified 80/20 split, `random_state=42`. `Pipeline([StandardScaler, LogisticRegression(class_weight="balanced", max_iter=2000)])` and `DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)`. Missing features imputed with the train-set median only.
- **Real test-set results:** `logistic_regression` ROC-AUC = **0.5744**, `decision_tree` ROC-AUC = **0.5692**. Base rate of `is_late` ≈ 7.9%, matching the dq report.
- **This AUC is a known, disclosed limitation of the deliberately small 4-feature set — not a bug, and not something to silently "fix" by adding features.** It must be reported honestly in the README, the report, and on the Design dashboard page, with a one-line explanation of why (small, interpretable feature set was a deliberate design choice — see section 17 for what a legitimate expansion would look like, only if asked for later).
- Because predicted `late_risk` sits in a narrow band (~0.41-0.57) as a direct consequence of this weak signal, expert-system rules referencing `late_risk` needed calibrating against the real distribution — see section 7.5's R08/R09 note.
- Outputs: `reports/model_metrics.json`, `reports/decision_tree.txt`, `reports/coefficients.csv`, `reports/feature_importance.csv`, `reports/sensitivity.csv` (perturbation -25/-10/+10/+25%), `LatePredictions.parquet` (110,189 rows), fitted models in `models/` with metadata JSON.

### 7.4 `src/ahp.py` (Unit 4) — implemented, Stage 2

- Criteria (sellers with **>= 30 orders**, the "established sellers" ranking — `AHP_MIN_ORDERS=30`, justified as the 77.9th percentile of seller order counts, i.e. the top ~23% "established" sellers): average review score, on-time rate, average price, order volume.
- Pairwise matrix (Saaty 1-9) in `config.py`, reciprocity-validated at load. Weights via principal right eigenvector (`numpy.linalg.eig`), normalized to sum 1. Consistency ratio via the standard Random Index table; flagged if `CR > 0.10`.
- **Real result:** 681 established sellers scored, **CR = 0.0189** (well within the consistent range).
- Criteria are min-max normalized into **`norm_*` columns added alongside the raw columns** (not overwriting them — see the bug note directly below) before weighting.
- `compute_weights(matrix)` is a standalone function, callable independently (used later by `discovery.py` at a different order-count threshold, and by the Power BI what-if layer conceptually, though Power BI does not call Python at runtime — see section 16).
- **Bug found and fixed during Stage 2:** an earlier version of `normalize_criteria` overwrote the raw criterion columns (including `order_volume`) with their 0-1 normalized values. This silently broke `discovery.py`'s popularity axis, since normalized values are always below the confidence threshold — every seller was mis-flagged as "low confidence." Fixed by adding `norm_*` columns instead of overwriting the raw ones. This also makes `ahp_ranking.csv` more readable for a human reviewer. Do not reintroduce the overwrite pattern.
- Output: `ahp_ranking.csv`, 681 rows.

### 7.5 `src/expert.py` + `rules/seller_rules.yaml` (Unit 5) — implemented, Stage 2

`KnowledgeBase`, `WorkingMemory` (facts: `ahp_score, avg_late_risk, avg_review, order_volume, late_rate, is_hidden_gem`), `InferenceEngine` (priority-ordered, tie-break by rule id), `ExplanationFacility` (fired-rule chain with rationale). 12 rules (R01-R12) covering `Keep`, `Warn`, `Suspend`, `Feature`, `Promote`, plus a low-priority default. Output: `seller_recommendations.csv`, 2,970 rows.

**R08 vs. R09 — both are kept intentionally, and both must be explained in the report, not just in `docs/knowledge_engineering.md`:**
- **R08** (`Promote` if `is_hidden_gem` AND `avg_late_risk < 0.3`) is `CLAUDE.md`'s original literal specification threshold. Given the real observed `late_risk` range (~0.41-0.57, per section 7.3), this threshold **currently fires on 0 real sellers** — it is exercised only by a constructed test fixture, never by live data. It is kept, not deleted, as the as-specified rule.
- **R09** (`Promote` if `is_hidden_gem` AND `avg_late_risk < 0.474`, the 25th percentile of observed risk) is the threshold **actually calibrated against the real data distribution**, and it is the one that fires in practice: 132 of 478 Hidden Gems are promoted.
- **Why keep both rather than just fixing R08's number:** this is a legitimate, disclosable point about expert systems — a threshold written against a specification before seeing the model's real output range often needs recalibration once real data is in hand. Presenting both, with R09 explicitly labeled "recalibrated against observed data," is a stronger and more honest artifact for a BIA report than quietly rewriting R08's number and pretending it was always 0.474.
- All twelve rules' thresholds, with the empirical percentile behind each, are in `docs/knowledge_engineering.md` (R01 suspend on ~75th-pct risk + bottom-quintile review, 94 sellers; R02 suspend on severe empirical late-rate>0.4 with a volume gate, 7 sellers; R03 warn on ~90th-pct risk alone, 226; R04 warn on the same review band as R01 with a volume gate, 177; R05 warn on 90th-pct empirical lateness, 105; R06 feature on top-decile quality + at/below-median risk, 107; R07 feature on review≥4.3 (chosen specifically because high-volume sellers average only 4.08 review in this data, so 4.3 selects quality-at-scale) + volume≥83, 69; R10/R11 keep on median-or-better risk+review or top-90%-quality with late_rate≤75th pct, 446/1,264; R12 default catches 1,228 sellers, 41%, mostly those without enough orders to clear any other rule's gates).

### 7.6 `src/discovery.py` (Unit 4 extension, "hidden gems") — implemented, Stage 2

- Quality axis: AHP score recomputed at **`DISCOVERY_MIN_ORDERS=5`** (41.9th percentile — makes 63% of sellers AHP-eligible vs. 23% at the 30-order threshold), reusing `ahp.compute_weights`, not reimplemented.
- Popularity axis: total order volume per seller.
- `compute_quadrants`: median split on both axes by default (no override currently set). **Real result: Star 454, Hidden Gem 478, Overrated 480, Overlooked-Low-Quality 452** — a near-even split, confirming the classification isn't degenerate.
- `confidence_flag`: **`DISCOVERY_CONFIDENCE_MIN_ORDERS=15`** (sits just below the median of 18 within the AHP-eligible pool). **Real result: 818 low-confidence / 1,046 normal.**
- `top_hidden_gems`: verified to never return a non-Hidden-Gem seller.
- `is_hidden_gem` fact wired into `expert.py`'s `WorkingMemory` — the data contract between the two modules was reconciled explicitly during Stage 2, not left as two independently-guessed shapes.
- Output: `discovery_quadrants.csv`, 1,864 rows.

### 7.7 Quality-vs-popularity analysis — definition, read alongside 7.6

"Healthier" in this project means **sound business quality** (good reviews, reliable delivery), **not literal product nutrition or food health.** Olist is general e-commerce and has no nutrition data; state this explicitly on the Discovery dashboard page and in the report. This directly targets popularity bias / long-tail neglect in ranking systems.

### 7.8 Dashboard (`app/`, Streamlit) — internal/dev view, distinct from the Power BI deliverable

Six pages (Intelligence, Design, Choice, Discovery, Implementation, DSS Architecture) per the original spec — `st.cache_data`, no business logic in the UI, every page captioned with its syllabus topic, analytics type, and management level. This remains the fast internal way to sanity-check the Python outputs; **the Power BI dashboard (section 16) is the graded, presented deliverable** for the BIA course's own tutorial requirements. Building the Streamlit pages is still worthwhile for verification and for the portfolio, but is not itself a T1-T15 tutorial substitute.

---

## 8. Testing (`tests/`, pytest) — 61 passed (20 Stage 1 + 41 Stage 2)

- **Sentiment:** known positive/negative/neutral sentences classify correctly; per-agent aggregate matches a hand-computed example.
- **Predict:** `late_risk` in [0,1]; feature list equals the exact allowed set (`config.ALLOWED_FEATURES`); train/test disjoint; AUC below the 0.95 leak guardrail; seed reproducibility.
- **AHP:** a known consistent matrix gives CR ~0 with correct weights; a known inconsistent matrix gives CR > 0.10; reciprocity validator rejects a bad matrix.
- **Discovery:** `compute_quadrants` assigns all four labels correctly on a hand-built fixture; `confidence_flag` boundary test (14 vs. 15 orders); `top_hidden_gems` never returns a non-Hidden-Gem seller; thresholds respect `config.py` overrides.
- **Expert system:** hand-built fact sets for each of Keep/Warn/Suspend/Feature/Promote (including the R08 fixture that only fires under constructed data) produce the expected action and correct rule trace; priority-based conflict resolution verified; malformed YAML fails loudly.
- **Reproducibility:** full pipeline reruns with identical row counts and output hashes.
- Still pending (Stage 3, section 15): tests for `src/export_sql.py`.

---

## 9. Commands

`tasks.ps1` (PowerShell, primary on this machine) and an equivalent `Makefile` (secondary) both expose:

```
setup     # create .venv, install pinned requirements
data      # python -m src.build_tables ; python -m src.synthetic
models    # python -m src.sentiment ; python -m src.predict ; python -m src.ahp ; python -m src.discovery ; python -m src.expert
export    # python -m src.export_sql   (Stage 3 — writes data/export/*.csv for SQL Server import)
run       # python run_all.py
app       # streamlit run app/Home.py
test      # pytest -q
lint      # ruff check . ; mypy src
check     # lint + test (must pass before declaring any task done)
```

`run_all.py` runs `build_tables -> synthetic -> clickstream_checks -> sentiment -> predict -> ahp -> discovery -> expert`, logs each stage's duration, stops on first failure, and prints the real summary table (confirmed working — see section 14). Stage 3 will add an `export` step at the end, writing flat CSVs for SQL Server without opening any database connection from Python.

---

## 10. Working agreement for Claude Code

1. **Plan first.** For any task touching more than one file, outline the approach and wait for confirmation before editing.
2. **Small, verifiable steps.** After each stage, run it and show real output before moving on.
3. **Run `check` before saying a task is done.** Quote the real test results.
4. **Commit per stage** with a clear message. Do not commit data, models, or virtualenvs.
5. **Do not silently change behavior.** If a spec here is impossible or wrong, stop, explain, and propose a fix instead of working around it quietly.
6. **Preserve existing style.** When editing existing code, keep its formatting and naming conventions.
7. **When debugging, lead with the fix.** Show the corrected code first, then a short cause line.
8. **Report honestly.** Weak model performance (section 7.3), recalibrated rule thresholds (section 7.5), small effect sizes, and synthetic-data caveats belong in the docs, not hidden.
9. **Log environment issues in `BUILD_LOG.md`** as they occur.
10. **Never write SQL, connect to a database, or generate Power BI files.** Stage 3's `export_sql.py` writes plain CSVs only; everything past that (SSMS import, Power BI modeling, DAX, RLS) is done manually — see section 16.

---

## 11. Definition of done (Stage 3, next up)

- [ ] `src/export_sql.py` converts every Stage 2 Parquet output to a clean, flat CSV in `data/export/`, with column names and types suitable for SSMS's Import Flat File wizard (no nested/complex types, explicit datetime formatting).
- [ ] A short section in `docs/powerbi_setup.md` documenting exactly which CSV maps to which SQL Server table name, ready to hand to the manual SSMS import step.
- [ ] `check` passes with the new export step included.
- [ ] `python run_all.py` optionally includes the export step at the end (ask before wiring it in, since it changes the pipeline's final behavior).

---

## 12. Known pitfalls (avoid)

- Using `delivery_days` or date columns as model features (target leakage) — verified absent, keep it that way.
- Fitting scalers or imputers on the full dataset before the split.
- Reporting metrics on training data.
- Treating accuracy as the headline metric on an imbalanced target; lead with ROC-AUC and per-class recall.
- Scoring VADER on aggressively stripped text.
- Leaving expert-system thresholds unjustified, or rules that can never fire without disclosing why (R08 is the disclosed exception, not a hidden one).
- Hard-coding paths, the seed, or the discovery thresholds outside `config.py`.
- **Presenting a low-order-count "hidden gem" with the same confidence as a well-established top AHP seller.**
- Overwriting raw criterion columns during normalization instead of adding `norm_*` columns (the exact bug already found and fixed — don't reintroduce it).
- Implying Olist has food or nutrition data anywhere in the docs or UI.
- **Silently expanding the predictive feature set to chase a better AUC** — any such change is an explicit, separate, asked-for enhancement (section 17), never a quiet "fix."
- Writing SQL, opening a database connection, or generating `.pbix`/DAX files from Claude Code — that layer is manual (section 16).

---

## 13. Stage 1 — completed and verified (reference; do not redo)

| File | Rows |
|---|---|
| `data/processed/FactOrderItems.parquet` | 110,189 |
| `data/processed/DimSeller.parquet` | 3,095 |
| `data/processed/DimProduct.parquet` | 32,951 |
| `data/processed/DimCustomer.parquet` | **99,441** (corrected — see section 6.2) |
| `data/processed/Clickstream.parquet` | 100,000 (synthetic) |
| `data/processed/CallTranscripts.parquet` | 300 (synthetic) |

Data quality: raw orders 99,441 -> delivered-filtered 96,470 -> `FactOrderItems` 110,189 rows; null rate 0.7505% (review fields only); late-delivery rate 7.9082%. Clickstream funnel and quality gates as in section 6.3 and 6.2. Environment note: pandas 3.0.6 / mypy 2.3.1 blocked by Windows Application Control; pinned to 2.2.3 / 1.13.0.

---

## 14. Stage 2 — completed and verified (reference; do not redo)

Stage timings: `build_tables` 1.52s, `synthetic` 0.57s, `clickstream_checks` 1.16s, `sentiment` 0.03s, `predict` 2.75s, `ahp` 0.32s, `discovery` 0.44s, `expert` 0.29s.

| File | Rows |
|---|---|
| `CallSentiment.parquet` | 300 |
| `LatePredictions.parquet` | 110,189 |
| `SellerFacts` | 2,970 |
| `ahp_ranking.csv` | 681 |
| `discovery_quadrants.csv` | 1,864 |
| `seller_recommendations.csv` | 2,970 |

Models: logistic regression AUC 0.5744, decision tree AUC 0.5692 (see section 7.3 for the honest discussion of why, and section 17 for a possible future enhancement). AHP: 681 established sellers, CR 0.0189. Discovery quadrants: Star 454, Hidden Gem 478, Overrated 480, Overlooked-Low-Quality 452. Quality gates: 61 tests passed, ruff clean, mypy clean on 10 files.

---

## 15. Stage 3 — CSV export for SQL Server (next, Claude Code's last stage)

`src/export_sql.py` converts the Stage 2 Parquet/CSV outputs into flat CSVs in `data/export/`, one file per intended SQL Server table, with SSMS-import-friendly types (explicit ISO datetime strings, no nested columns). No database connection, no SQL, no ORM — this script's entire job is producing clean, importable flat files. This is the last piece Claude Code builds; everything downstream (section 16) is manual.

---

## 16. SQL Server + Power BI presentation layer (manual, outside Claude Code)

This section is documentation for the human-driven part of the project — Claude Code does not execute any of this, but should be aware of it so `export_sql.py` produces exactly what this layer needs.

**Flow:** `data/export/*.csv` (Stage 3) → SSMS **Import Flat File** wizard, one table per CSV → Power BI Desktop **Get Data → SQL Server** → star-schema model → DAX measures → dashboard pages → paginated report → row-level security.

**Table mapping** (CSV -> SQL Server table -> star-schema role):

| CSV (from `data/export/`) | SQL Server table | Role |
|---|---|---|
| `FactOrderItems` | `FactOrderItems` | Fact table |
| `DimSeller` | `DimSeller` | Dimension |
| `DimProduct` | `DimProduct` | Dimension |
| `DimCustomer` | `DimCustomer` | Dimension |
| `LatePredictions` | `LatePredictions` | Fact extension (joins to `FactOrderItems` on `item_key`) |
| `CallSentiment` | `CallSentiment` | Supporting fact (per-call/per-agent) |
| `Clickstream` | `Clickstream` | Supporting fact (synthetic — label it as such in the Power BI report) |
| `ahp_ranking` | `AhpRanking` | Dimension extension (joins to `DimSeller` on `seller_id`) |
| `discovery_quadrants` | `DiscoveryQuadrants` | Dimension extension (joins to `DimSeller` on `seller_id`) |
| `seller_recommendations` | `SellerRecommendations` | Dimension extension (joins to `DimSeller` on `seller_id`) |

**Relationships in Power BI Model view:** `FactOrderItems` <-> `DimSeller`/`DimProduct`/`DimCustomer` on their respective ids; `FactOrderItems` <-> `LatePredictions` on `item_key`; `DimSeller` <-> `AhpRanking`/`DiscoveryQuadrants`/`SellerRecommendations` on `seller_id`.

**Course tutorial mapping (T1-T15):**
- T1-T3: SSMS install, Power BI install, data prep — done directly in the GUI, no Python involvement.
- T4-T6: load the exported CSVs via SSMS, build the relationships above in Power BI's Model view.
- T7-T9: DAX measures built on top of the already-computed Python outputs — surfacing and slicing, not recomputing the ML (e.g., `Average Late Risk = AVERAGE(LatePredictions[late_risk])`, `Hidden Gem Count = CALCULATE(COUNTROWS(DiscoveryQuadrants), DiscoveryQuadrants[quadrant] = "Hidden Gem")`, `Seller AHP Score = AVERAGE(DiscoveryQuadrants[ahp_score])`).
- T10-T12: dashboard pages, including a quadrant scatter (popularity x, AHP score y, colored by quadrant — the Power BI equivalent of the Streamlit Discovery page) and a Power BI What-If Parameter on the late-risk threshold.
- T13-T15: a paginated "Seller Recommendation Report" (one page per seller: AHP score, late-risk, sentiment, recommended action and reason from `seller_recommendations.csv`), and row-level security (e.g., a Regional Manager role filtered to `seller_state`).

**Labelling requirement carried into Power BI:** every page or visual touching `Clickstream` or `CallSentiment`/`CallTranscripts` must note that the underlying data is synthetic, matching the Python dashboard's disclosure.

`docs/powerbi_setup.md` should hold the working notes for this section as the manual build proceeds (screenshots, gotchas, any deviation from the plan above) — Claude Code can help draft/update this file even though it doesn't perform the Power BI steps itself.

---

## 17. Optional future enhancement — expanding the predictive feature set (not started, ask before doing)

If asked to improve on the 0.57 AUC later, legitimate **non-leaky** candidate features (none depend on the delivery outcome):
- `product_category` (frequency-encoded)
- `order_month` / `day_of_week` of purchase (seasonality)
- `seller_historical_late_rate` — must be computed from strictly-prior orders only, with a proper train-only cutoff, or it reintroduces leakage
- `n_items_in_order`
- `payment_installments`
- a distance proxy from `olist_geolocation` (state-level or zip-prefix centroid distance between seller and customer)

This is optional and explicitly opt-in — do not add any of these without being asked, and treat it as a new, separately-tested addition to `src/predict.py`, not a silent edit.
