# E-Commerce Decision Support System (BIA 21CSE421T course project)

A Python + Streamlit decision support system on the Olist Brazilian e-commerce dataset. It answers one business question: **which sellers should the marketplace keep, warn, suspend, or feature, and why?** It does this by combining clickstream analytics, call-transcript sentiment, late-delivery risk prediction, multi-criteria (AHP) seller scoring, and a rule-based expert system with explanations.

This file is the source of truth for every Claude Code session. Read it fully before doing anything. If a request conflicts with this file, say so and ask before proceeding.

---

## 1. Hard constraints (never violate)

- **No SQL Server, no SSMS, no Power BI dependency.** All storage is CSV/Parquet under `data/`.
- **No neural networks, no SVM, no KNN**, anywhere: code, dependencies, comments, docs. Predictive models are limited to `LogisticRegression` and `DecisionTreeClassifier` from scikit-learn.
- **Only features Claude Code can build and verify from the terminal.** No GUI-only steps, no manual clicking, no paid APIs, no network calls at runtime (after `pip install`).
- **Scope is six syllabus topics, not all of them.** Do not add extra topics unless asked. See section 2.
- **`data/raw/` is read-only.** Never modify, rename, or delete files there. Never commit it.
- **Never fabricate results.** Every metric, row count, and AUC quoted in docs or UI must come from a real run of the code. If something cannot be measured, say so.
- **Ask before adding any new dependency**, changing the model family, or changing the scope.

---

## 2. Scope: syllabus topics implemented

| Unit | Syllabus topic | Where it lives | Dashboard page |
|---|---|---|---|
| 1 | Clickstream analysis (metrics) | `src/clickstream.py` | Intelligence |
| 2 | Phases of decision making, DSS components | `app/` structure + `app/pages/5_DSS_Architecture.py` | All pages, DSS Architecture |
| 3 | Predictive modeling + sensitivity analysis | `src/predict.py` | Design |
| 3 | Sentiment analysis process + speech analytics | `src/sentiment.py` | Intelligence |
| 4 | Multi-criteria decision making, pairwise comparison (AHP) | `src/ahp.py` | Choice |
| 5 | Expert systems (rules, inference, explanation) | `src/expert.py`, `rules/seller_rules.yaml` | Implementation |

Explicitly **out of scope**: competitive intelligence, uncertainty/payoff methods, linear programming, EV decision trees, maps, PDF reports, row-level security, Docker, Power BI, SQL Server, neural networks, SVM, KNN.

Decision phase mapping (Simon): Intelligence (find the problem) -> Design (model it) -> Choice (evaluate and select) -> Implementation (act and explain). The four main dashboard pages are named after these phases.

DSS subsystem mapping (used on the Architecture page and in docs):
- Data management: `src/build_tables.py`, `src/synthetic.py`, `data/processed/`
- Model management: `src/predict.py`, `src/sentiment.py`, `src/ahp.py`
- Knowledge base: `src/expert.py`, `rules/seller_rules.yaml`
- User interface: `app/`

---

## 3. Tech stack

- Python 3.11+
- pandas, numpy, pyarrow (Parquet I/O), scikit-learn, scipy, vaderSentiment
- streamlit, plotly
- pyyaml
- pytest, ruff, mypy
- Pin exact versions in `requirements.txt` (generate with `pip freeze` after a clean install). Use a virtual environment at `.venv/`.
- No runtime downloads (for example no `nltk.download`). `vaderSentiment` bundles its lexicon.

---

## 4. Repository layout

```
.
├── CLAUDE.md
├── README.md
├── Makefile
├── requirements.txt
├── pyproject.toml            # ruff + mypy + pytest config
├── run_all.py                # runs the full pipeline in order
├── data/
│   ├── raw/                  # 9 Olist CSVs (gitignored, read-only)
│   └── processed/            # generated parquet/csv (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── build_tables.py
│   ├── synthetic.py
│   ├── clickstream.py
│   ├── sentiment.py
│   ├── predict.py
│   ├── ahp.py
│   └── expert.py
├── rules/
│   └── seller_rules.yaml
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Intelligence.py
│       ├── 2_Design.py
│       ├── 3_Choice.py
│       ├── 4_Implementation.py
│       └── 5_DSS_Architecture.py
├── models/                   # joblib artifacts + metadata JSON (gitignored)
├── reports/                  # metrics, tree text, sensitivity, dq report
└── tests/
```

`.gitignore` must cover: `data/raw/`, `data/processed/`, `models/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`.

---

## 5. Engineering standards

- Type hints on every function signature. Docstrings (one-line summary plus params/returns for non-trivial functions).
- Use `logging` with a module-level logger configured in `config.py`. **No bare `print`** except in `run_all.py` final summary.
- All paths, constants, the seed, and the AHP matrix live in `src/config.py`. No magic numbers or hard-coded paths elsewhere.
- **Seed = 42** for every random operation (`numpy.random.default_rng(SEED)`, `random_state=SEED`).
- Functions in `src/` are **pure where possible**: take DataFrames in, return DataFrames out. File I/O happens only in clearly named `load_*`/`save_*`/`main()` functions.
- **UI has no business logic.** Streamlit pages only call functions from `src/` and render results.
- Scripts are **idempotent**: rerunning overwrites outputs with identical content (given the same inputs and seed).
- Every pipeline stage ends with **assertions** on schema, uniqueness, and row-count ranges, failing loudly with a clear message.
- Use vectorized pandas/numpy. No row-wise Python loops over the 100k-row fact table.
- Keep functions under ~50 lines where practical. Prefer small composable functions.
- Format and lint with `ruff` (line length 100). Type-check `src/` with `mypy --strict` where feasible; if a third-party stub is missing, use a scoped override in `pyproject.toml` rather than blanket ignores.

---

## 6. Data specification

### 6.1 Raw inputs (Olist, in `data/raw/`)

Expected files (fail with a clear message listing any missing):

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

Key columns used: orders (`order_id, customer_id, order_status, order_purchase_timestamp, order_delivered_customer_date, order_estimated_delivery_date`), items (`order_id, order_item_id, product_id, seller_id, price, freight_value`), reviews (`order_id, review_score, review_creation_date`), products (`product_id, product_category_name, product_weight_g`), sellers (`seller_id, seller_state`), customers (`customer_id, customer_state`), translation (`product_category_name, product_category_name_english`). Verify actual column names on load; note Olist has known typos in some product column names, so select only the columns above.

### 6.2 Cleaning rules (`src/build_tables.py`)

- Keep only `order_status == "delivered"` with a non-null `order_delivered_customer_date`.
- Parse timestamps with `pd.to_datetime(..., errors="coerce")` and log how many became NaT.
- Derive: `order_date` (normalized purchase date), `delivery_days`, `is_late` (1 if delivered date > estimated date, else 0), `item_key = f"{order_id}_{order_item_id}"`.
- Reviews: several orders have multiple reviews. Keep the **latest** per `order_id` by `review_creation_date`.
- Join category translation; fill missing English category with `"unknown"`.
- Outputs (Parquet): `FactOrderItems`, `DimSeller`, `DimProduct`, `DimCustomer`.
- Assertions: `item_key` unique, fact rows within 100,000 to 115,000, no null `seller_id`, `is_late` in {0,1}.
- Write a short data-quality summary to `reports/dq_report.md` (row counts before/after each filter, null rates, late rate).

### 6.3 Synthetic data (`src/synthetic.py`)

Olist has no clickstream or call transcripts, so these are **synthetic and must be labelled as such everywhere they appear** (dashboard captions, README, report).

**Clickstream** (100,000 rows): `session_id, event_type, product_id, device, source, event_time`.
- Event types: `page_view, add_to_cart, checkout, purchase`.
- Sessions must be **causally consistent**: a `purchase` implies earlier `checkout` and `add_to_cart` in the same session; timestamps strictly increase within a session.
- Overall funnel should be realistic (roughly 70% views, 15% carts, 9% checkouts, 6% purchases at the event level). Devices: mobile-heavy (about 60%). Sources: organic, ads, email, social.
- Sample `product_id` values from the real fact table.

**Call transcripts** (300 calls): `call_id, agent_id (AG01..AG10), transcript, hold_seconds, silence_pct`.
- At least 10 distinct templates per sentiment class (negative, neutral, positive), with light random variation so VADER scores are not identical.
- Negative calls should have higher hold time and silence percentage on average (state the generating relationship in the docstring).

Both generators use `numpy.random.default_rng(SEED)` and are deterministic.

---

## 7. Module specifications

### 7.1 `src/clickstream.py` (Unit 1)

Pure functions returning DataFrames:
- `funnel_conversion(events)`: sessions reaching each stage, with stage-to-stage and overall conversion rates.
- `bounce_rate(events)`: share of sessions containing exactly one event and it is a `page_view`.
- `cart_abandonment_rate(events)`: sessions with `add_to_cart` but no `purchase`, over sessions with `add_to_cart`.
- `funnel_by_device(events)` (and optionally by source).
- Invariant to test: each funnel stage count is less than or equal to the previous stage.

### 7.2 `src/sentiment.py` (Unit 3)

Implement the sentiment **process** as separate, individually testable stages:
1. `preprocess(text)`: lowercase, strip punctuation, tokenize, remove stop words (small built-in list), simple rule-based lemmatization (no external downloads).
2. `extract_features(tokens)`: token counts / negation flags (document what is used).
3. `classify(text)`: VADER compound score on the cleaned-but-not-over-stripped text (VADER needs punctuation and casing cues, so score the original text and use preprocessing for the feature/analysis stage; document this decision).
4. `score(compound)`: label `Negative` if <= -0.05, `Positive` if >= 0.05, else `Neutral`.
- Outputs: per-call labels and per-agent aggregates (mean sentiment, mean hold seconds, mean silence %, negative-call share).
- Note in docs: speech analytics here starts from **transcripts**. No audio and no ASR is implemented. Say this plainly.

### 7.3 `src/predict.py` (Unit 3)

- **Target:** `is_late`.
- **Features (only these unless asked):** `price`, `freight_value`, `product_weight_g`, `same_state` (`seller_state == customer_state`).
- **Leakage rule (critical):** never use `delivery_days`, delivered/estimated dates, or anything derived from them as a feature. They define the target.
- Drop rows with missing features (log the count) or impute with the train-set median inside the pipeline; do not impute using the full dataset.
- Stratified 80/20 split, `random_state=SEED`.
- Models: `Pipeline([StandardScaler, LogisticRegression(class_weight="balanced", max_iter=2000)])` and `DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=SEED)`.
- Metrics for both models on the **test set only**: ROC-AUC, precision/recall/F1 per class, confusion matrix, plus the base rate of `is_late`. Save to `reports/model_metrics.json`.
- Interpretability outputs: `reports/decision_tree.txt` (`export_text`), `reports/coefficients.csv` (LR coefficients, standardized features), `reports/feature_importance.csv` (permutation importance on the test set).
- **Sensitivity analysis:** for each feature, perturb by -25%, -10%, +10%, +25% (binary `same_state`: flip 0/1), hold others fixed, and record the mean change in predicted `late_risk`. Save `reports/sensitivity.csv`. Explain in docs that this is what "illuminating the black box with sensitivity" looks like for these models.
- Write `LatePredictions.parquet` (`item_key, late_risk`) for all fact rows using the logistic regression model. Persist the fitted pipeline with `joblib` in `models/` with a metadata JSON (features, metrics, timestamp, seed).
- Sanity guardrail: if test ROC-AUC exceeds 0.95, treat it as a probable leak, stop, and investigate before reporting it. If AUC is modest (say 0.55 to 0.70), report it honestly and explain why (few features).

### 7.4 `src/ahp.py` (Unit 4)

- Criteria (per seller, sellers with at least 30 orders): average review score (benefit), on-time rate (benefit), average price (define direction in config and justify), order volume (benefit).
- The pairwise comparison matrix uses Saaty's 1 to 9 scale and lives in `config.py` as a reciprocal matrix. Validate reciprocity (`a[j][i] == 1/a[i][j]`) at load time.
- Weights: principal right eigenvector via `numpy.linalg.eig`, normalized to sum 1.
- Consistency: `CI = (lambda_max - n) / (n - 1)`, `CR = CI / RI` with Random Index `{1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}`. Flag if `CR > 0.10`.
- Normalize each criterion (min-max, inverted for cost criteria) before weighting. Output a ranked table with per-criterion contribution.
- Expose `compute_weights(matrix)` separately so the UI can recompute live from an edited matrix.

### 7.5 `src/expert.py` + `rules/seller_rules.yaml` (Unit 5)

A forward-chaining rule-based system with distinct classes:
- `KnowledgeBase`: loads and validates rules from YAML (fail on duplicate ids or unknown fields).
- `WorkingMemory`: holds facts for one seller (`ahp_score, avg_late_risk, avg_review, order_volume, late_rate`).
- `InferenceEngine`: match rules against facts, resolve conflicts by **priority** (higher first, tie-break by rule id), fire, and record the trace.
- `ExplanationFacility`: returns the chain of fired rules, the facts that satisfied each, and each rule's rationale, in plain English.

Rule schema:

```yaml
- id: R01
  priority: 100
  if:
    - {fact: avg_late_risk, op: ">", value: 0.6}
    - {fact: avg_review, op: "<", value: 3.0}
  then: Suspend
  because: "High predicted late-delivery risk combined with poor customer reviews."
```

- At least 10 rules with actions `Keep`, `Warn`, `Suspend`, `Feature`, plus a low-priority default rule so every seller gets a recommendation.
- Thresholds must be justified in `docs/knowledge_engineering.md` (source: the data distributions, for example percentiles, not arbitrary numbers).
- Supported operators: `<, <=, >, >=, ==, !=`. Validate at load time.

### 7.6 Dashboard (`app/`)

- Multipage Streamlit, run with `streamlit run app/Home.py`.
- `st.cache_data` around every data load. No heavy computation on rerun.
- **Every page shows a caption naming the syllabus topic** it demonstrates, plus a badge for analytics type (Descriptive / Diagnostic / Predictive / Prescriptive) and management level (Operational / Tactical / Strategic).
- Pages:
  1. **Intelligence:** KPI cards (orders, revenue, late rate, average review), clickstream funnel, bounce and abandonment, funnel by device, call sentiment by agent (with hold vs silence scatter). Label clickstream and calls as synthetic.
  2. **Design:** both models' metrics side by side, decision tree text, coefficient bars, sensitivity chart, plain-English interpretation under each chart.
  3. **Choice:** AHP ranking, editable pairwise matrix with live weights and CR warning, and a what-if block (delivery time and price sliders showing projected change in late risk).
  4. **Implementation:** select a seller, run the expert system, show the recommendation and the full explanation chain.
  5. **DSS Architecture:** subsystem-to-module mapping, phase-to-page mapping, data catalog.
- Every chart has a title, axis labels, and units. No page may raise an exception when data files are missing: show a friendly instruction to run `python run_all.py` instead.

---

## 8. Testing (`tests/`, pytest)

Use small deterministic fixtures, not the full dataset, wherever possible. Required tests:
- **Data:** `item_key` unique, no orphan `seller_id`/`product_id`, `is_late` in {0,1}, row count in range (integration test, skipped if raw data absent).
- **Clickstream:** funnel monotonic, bounce and abandonment in [0,1], causal-consistency of synthetic sessions.
- **Sentiment:** known positive, negative, and neutral sentences classify correctly; per-agent aggregates match a hand-computed example.
- **Predict:** `late_risk` in [0,1], no forbidden feature columns (assert the feature list equals the allowed list), train/test disjoint, AUC below the leakage guardrail, fixed-seed reproducibility.
- **AHP:** a known consistent matrix gives CR near 0 and correct weights; a known inconsistent matrix gives CR above 0.10; reciprocity validator rejects a bad matrix.
- **Expert system:** three hand-built seller fact sets produce the expected action and the expected rule trace; conflict resolution respects priority; invalid YAML fails loudly.
- **App:** `streamlit.testing.v1.AppTest` smoke test that every page runs without exception.
- **Reproducibility:** running the pipeline twice yields identical output hashes.

---

## 9. Commands

`Makefile` targets (create them):

```
make setup     # create .venv, install requirements
make data      # python -m src.build_tables && python -m src.synthetic
make models    # python -m src.sentiment && python -m src.predict
make run       # python run_all.py
make app       # streamlit run app/Home.py
make test      # pytest -q
make lint      # ruff check . && mypy src
make check     # lint + test (must pass before declaring any task done)
```

`run_all.py` runs the stages in order (`build_tables -> synthetic -> clickstream checks -> sentiment -> predict -> ahp`), logs the duration of each stage, stops on the first failure, and prints a final summary table (files written with row counts, both model AUCs, test status).

**Automation (optional, only when asked):** a GitHub Actions workflow running `make check` on push, and instructions in the README for scheduling `python run_all.py` nightly (Windows Task Scheduler or cron).

---

## 10. Working agreement for Claude Code

1. **Plan first.** For any task touching more than one file, outline the approach and wait for confirmation before editing.
2. **Small, verifiable steps.** After each stage, run it and show real output before moving on.
3. **Run `make check` before saying a task is done.** Quote the real test results.
4. **Commit per stage** with a clear message (`feat: clickstream metrics`, `test: ahp consistency`). Do not commit data, models, or virtualenvs.
5. **Do not silently change behavior.** If a spec here is impossible or wrong (for example a column name differs from the Olist release), stop, explain, and propose a fix instead of working around it quietly.
6. **Preserve existing style.** When editing existing code, keep its formatting and naming conventions.
7. **When debugging, lead with the fix.** Show the corrected code first, then a short cause line.
8. **Report honestly.** Weak model performance, small effect sizes, and synthetic-data caveats belong in the docs, not hidden.

---

## 11. Definition of done

- [ ] `python run_all.py` runs end to end from a clean clone (with raw data supplied) and prints the summary table.
- [ ] `make check` passes: ruff clean, mypy clean, all tests green.
- [ ] `streamlit run app/Home.py` opens; all 5 pages render with no exceptions and captions naming their syllabus topic.
- [ ] `reports/` contains `model_metrics.json`, `decision_tree.txt`, `coefficients.csv`, `feature_importance.csv`, `sensitivity.csv`, `dq_report.md`.
- [ ] `docs/` contains `knowledge_engineering.md` (rule thresholds justified) and `SYLLABUS_MAP.md` (each implemented topic mapped to file, function, and page; a test asserts every listed path exists).
- [ ] README has: problem statement, architecture diagram (text), setup, how to run, results table, limitations.
- [ ] Limitations are stated in the README: clickstream and call data are synthetic; speech analytics uses transcripts only (no ASR); late-delivery model uses a small feature set; AHP weights are judgment-based.

---

## 12. Known pitfalls (avoid)

- Using `delivery_days` or date columns as model features (target leakage).
- Fitting scalers or imputers on the full dataset before the split.
- Reporting metrics on training data.
- Treating accuracy as the headline metric on an imbalanced target; lead with ROC-AUC and per-class recall.
- Scoring VADER on aggressively stripped text (it loses punctuation and capitalization cues).
- Leaving expert-system thresholds unjustified or rules that can never fire (write a test that every rule fires for at least one constructed fact set).
- Hard-coding paths or the seed outside `config.py`.
