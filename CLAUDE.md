# E-Commerce Decision Support System (BIA 21CSE421T course project)

A Python + Streamlit decision support system on the Olist Brazilian e-commerce dataset, presented
to end users through a SQL Server + Power BI layer. It answers two business questions:

1. **Which sellers should the marketplace keep, warn, suspend or feature, and why?**
2. **Which sellers and products are genuinely good but under-exposed ("hidden gems") and deserve
   a visibility boost rather than being buried under high-volume sellers?**

It combines clickstream analytics, call-transcript sentiment, late-delivery risk prediction,
multi-criteria (AHP) seller scoring, a quality-versus-popularity quadrant analysis and a
rule-based expert system with explanations. Python produces verified, versioned outputs; SQL Server and
Power BI consume them for the course's T1-T15 tutorial deliverables.

This file is the source of truth for every Claude Code session. Read it fully first. If a request
conflicts with it, say so and ask before proceeding. **Numbers live in `reports/RESULTS.md`
(generated); do not copy them into this file or other documents by hand.**

**Status.** Stages 1-3.5 (data, models, rules, export, RandomForest primary), 4-4d (dashboard),
5-5b (order-level shrinkage, one review definition), 6 (order-grouped evaluation and out-of-fold
risk), 7 (SQL Server loader) and 8 (minimum-evidence rule for Warn/Suspend, documentation cleanup,
single source of numbers) are complete and verified. History: `BUILD_LOG.md`, full detail in
`docs/archive/BUILD_LOG_full.md`. The Power BI report (`powerbi/`, git-ignored) is the next stage and is built by hand.

---

## 1. Hard constraints

- **No SQL Server or Power BI dependency in the Python layer, with one owner-authorised
  exception.** The only files that may contain SQL or open a database connection are
  `sql/schema.sql`, `src/load_sql.py` and `tests/test_load_sql.py`. Their limits, enforced in code and
  by tests: Windows authentication only (no password anywhere) to `localhost\SQLEXPRESS`
  (configurable); only the database `EcomDecisionIntelligence`; only the 10 export tables and the view
  `dbo.vw_SellerScorecard` may be dropped, created or written; never create or drop a database, never
  change a server setting, login or security option. Everything else in `src/`, `app/` and `tests/` stays
  database-free (CSV/Parquet under `data/`). Claude Code never generates `.pbix` files or DAX; the Power BI
  report is built by hand in Power BI Desktop and kept in `powerbi/` (never modified, moved or committed).
- **Models.** `RandomForestClassifier` is the primary model; `LogisticRegression` and
  `DecisionTreeClassifier` are kept for interpretability (readable coefficients, a printable rule path).
  **No neural networks, SVM or KNN anywhere** (code, dependencies, comments, docs). Any other model family
  needs the evidence-then-ask-then-confirm process that adopted the forest.
- **Do not silently expand the feature set or swap the model family.** The 12 features are exactly
  `config.ALLOWED_FEATURES`; no column derived from the delivered or estimated dates, `delivery_days`, the reviews or the
  target may enter (target leakage). The purchase month and weekday are allowed: they are known when the order is placed.
- **Only features Claude Code can build and verify from the terminal.** No GUI-only steps, paid APIs
  or network calls at run time (after `pip install`).
- **Scope: seven syllabus-anchored topics** (section 2). No extra topics unless asked. Out of scope:
  competitive intelligence, linear programming, EV decision trees, maps, PDF reports, Docker, neural
  networks, SVM, KNN, food or nutrition data (Olist has none; "healthier" means business quality).
- **`data/raw/` is read-only.** Never modify, rename or delete files there; never commit it.
- **Never fabricate results.** Every metric, row count, AUC or rule count quoted in docs or UI comes from a
  real run. Do not type a number that a file already holds; read it (or point to `reports/RESULTS.md`).
- **Never claim confidence a small sample does not support.** Hidden gems have few items sold, so each carries
  an evidence tier ("Stronger evidence" or "Early signal"); they are directional, never presented with the
  certainty of the main AHP ranking. Warn and Suspend rules need at least 5 items sold and independent orders (3 delivered, and 3 reviewed
  where they read the average review; section 7.4).
- **Wording about the risk score.** `late_risk` is an out-of-fold **ranking score, not a probability** and not
  "calibrated" (balanced class weights put its median far above the real late rate). The model has **no
  forward-looking skill** (temporal check at or below chance): never say it forecasts or predicts future
  deliveries. Say **items sold** for a seller's volume (order lines) and **delivered orders** for the
  marketplace total.
- **Rule thresholds live only in `rules/seller_rules.yaml`.** Code, tests and docs read them from the
  file, never retype them.
- **Ask before adding any dependency**, changing the model family or changing the scope.

## 2. Scope: syllabus topics (Python layer)

| Unit | Topic | Where | Page |
|---|---|---|---|
| 1 | Clickstream analysis | `src/clickstream.py` | Intelligence |
| 2 | Phases of decision making, DSS components | `app/` structure, `app/pages/6_DSS_Architecture.py` | all, DSS Architecture |
| 3 | Predictive modeling, sensitivity analysis | `src/predict.py` | Design |
| 3 | Sentiment analysis, speech analytics | `src/sentiment.py` | Intelligence |
| 4 | Multi-criteria decisions, AHP | `src/ahp.py` | Choice |
| 5 | Expert systems | `src/expert.py`, `rules/seller_rules.yaml` | Implementation |
| 4+5 | Quality-vs-popularity ("hidden gems") | `src/discovery.py` | Choice, Discovery |

Exact files, functions and pages: `docs/SYLLABUS_MAP.md` (a test checks it), which also maps the DSS subsystems.
Pages follow Simon's phases (Intelligence, Design, Choice, Implementation); Discovery is a satellite of Choice.

## 3. Tech stack

Python 3.11/3.12, Windows-first. Exact versions are pinned in `requirements.txt` (`pip freeze` equals it):
pandas **2.2.3**, numpy, pyarrow, scikit-learn, scipy, vaderSentiment, streamlit, plotly, pyyaml,
**pyodbc 5.3.0** (loader only; needs the Microsoft ODBC Driver 17 or 18 for SQL Server installed on the machine),
pytest, ruff, mypy **1.13.0**. Virtual environment at `.venv/`. No runtime downloads (VADER bundles its lexicon).

**Known environment issue.** Windows Smart App Control / Application Control blocks some freshly installed
compiled packages at import (seen with pandas 3.0.6, mypy 2.3.1, and for a time scikit-learn's `.pyd` files).
Pin an older, long-established release instead of fighting the policy. Never change Smart App Control,
Defender, the registry, execution policy or file ACLs, and never use `Unblock-File`. Log new cases in `BUILD_LOG.md`.

## 4. Repository layout

```
CLAUDE.md  README.md  BUILD_LOG.md  tasks.ps1  Makefile  run_all.py  requirements.txt  pyproject.toml
src/        config.py build_tables.py synthetic.py clickstream.py sentiment.py predict.py ahp.py
            discovery.py expert.py explain.py export_sql.py load_sql.py
app/        Home.py pages/1..6_*.py _common.py _components.py _format.py _labels.py _theme.py
            (.streamlit/config.toml: dark theme)
rules/      seller_rules.yaml
sql/        schema.sql                      (tables, keys, indexes, dbo.vw_SellerScorecard)
scripts/    make_results.py app_smoke.py diagnostics/   (one-off experiments; not part of the pipeline)
tests/      pytest suite
docs/       README.md (index) knowledge_engineering.md DEMO_SCRIPT.md SYLLABUS_MAP.md MANUAL_CHECKS.md
            powerbi_setup.md powerbi_measures.md archive/
reports/    generated outputs; RESULTS.md; diagnostics/
data/       raw/ (input, ignored)  processed/  export/ (ignored)        models/ (ignored)    powerbi/ (ignored)
```

`.gitignore` covers `data/raw`, `data/processed`, `data/export`, `models`, `.venv`, caches and `*.pbix`.

## 5. Engineering standards

- Type hints on every signature; docstrings (one-line summary plus parameters for non-trivial functions).
- `logging` with a module logger from `config.get_logger`. No bare `print` except `run_all.py`'s closing summary
  and the loader/generator scripts that write their result tables to stdout.
- Paths, the seed, the AHP matrix and the discovery and shrinkage constants live in `src/config.py`. The loader keeps its own
  few constants in `src/load_sql.py` so the SQL exception stays confined to three files. Rule thresholds live in the YAML.
- **Seed 42** for every random operation. Scripts are idempotent: rerunning gives byte-identical outputs
  (verified by a clean-room rebuild; only timestamp fields in `models/*_metadata.json` change).
- Functions in `src/` are pure where possible (DataFrames in and out); file I/O only in `load_*`, `save_*`, `main()`.
- **The UI has no business logic.** Pages call `src/` and render. `src/explain.py` is the only `src` module that
  exists to serve the UI (plain-language sentences, near-miss, tick lists); it calls the rule engine rather than
  duplicating it.
- Vectorised pandas/numpy; no row-wise loops over the fact table. Keep functions under about 50 lines.
- `ruff` (line length 100) and `mypy --strict` on `src/` must be clean. Every pipeline stage ends with assertions on schema,
  uniqueness and row-count ranges.
- Prefer `.\tasks.ps1` / `python -m` over `make`.

## 6. Data

Cleaning rules: `src/build_tables.py`, `reports/dq_report.md`; row counts: `reports/RESULTS.md`. `DimCustomer` comes
from all raw customers. The clickstream and call transcripts are **synthetic** (`src/synthetic.py`); label them as such
everywhere, including the Power BI report.

## 7. Modules

7.1 **`clickstream.py`**: funnel, bounce rate, cart abandonment, by device. **`sentiment.py`**: `preprocess`,
`extract_features`, `classify`, `score` plus a per-agent aggregate; VADER scores the original text; transcripts only, no audio.

7.2 **`predict.py`**: target `is_late`, exactly 12 features. **Order-grouped evaluation:** five
`StratifiedGroupKFold` folds over `order_id` (items of one order share their outcome, so item-level random
splits leak siblings); fold 0 is the train/test split for every metric, importance, sensitivity run and saved model;
all folds together give the **out-of-fold** scores in `LatePredictions` (columns `late_risk` from the forest, plus
`late_risk_lr` and `late_risk_dt`). The forest uses `min_samples_leaf = config.PREDICT_RF_MIN_SAMPLES_LEAF`
(chosen under this protocol; fully grown trees were larger and worse). Imputers and encoders are fitted on training
rows only. A test ROC-AUC above `config.PREDICT_AUC_LEAK_GUARDRAIL` raises. Metrics, importance, sensitivity and
models are written to `reports/` and `models/` (ignored); the current numbers are in `reports/RESULTS.md`.

7.3 **`ahp.py`**: established sellers (`AHP_MIN_ORDERS` items sold) scored on average review, on-time rate, average price
and items sold; weights from the principal eigenvector of the Saaty matrix in `config.py`, consistency ratio flagged above
0.10. **Review and on-time rate are shrunk toward the marketplace mean (empirical Bayes, per distinct order,
`config.AHP_SHRINKAGE_K_*` measured by `scripts/diagnostics/diagnostic_shrinkage.py`)**, so a few perfect reviews cannot beat a long
record. Criteria are normalised into `norm_*` columns **added alongside the raw ones; never overwrite the raw columns**
(an earlier bug). `compute_weights` is reused by `discovery.py`.

7.4 **`expert.py` + `rules/seller_rules.yaml`**: knowledge base, working memory (`ahp_score`, `avg_late_risk`,
`avg_review` as one review per distinct order, `order_volume` as items sold, `late_rate`, `is_hidden_gem`, `n_orders`
(distinct delivered orders) and `n_reviewed` (distinct reviewed orders)), inference
(highest priority wins, ties by rule id) and explanation. Twelve rules R01-R12 with actions Keep, Warn, Suspend, Feature,
Promote and a default (R12). **Minimum evidence: every Warn and Suspend rule requires `order_volume >= 5` (items sold) and independent orders**
(`n_orders >= 3` for R01, R02, R03 and R05; `n_reviewed >= 3` for R01 and R04), because items of one order are not
independent evidence. A seller with fewer than 5 items sold or fewer than 3 delivered orders can only be Keep
("pending more data"). Promote is deliberately not gated on orders (a directional suggestion that carries its evidence tier). Percentile-intent thresholds were set on the
population each rule applies to (R01 and R03 on sellers with at least 5 items sold). The reasoning per rule and per
recalibration is in `docs/knowledge_engineering.md`; current thresholds and fire counts are in `reports/RESULTS.md`. A drift test
fails if a threshold stops discriminating or R01/R03 leave their percentile.

7.5 **`discovery.py`**: quadrants by median split of the AHP quality score (recomputed at `DISCOVERY_MIN_ORDERS`) and items
sold; `confidence_flag` at `DISCOVERY_CONFIDENCE_MIN_ORDERS` gives the evidence tier; `top_hidden_gems` never returns a
non-gem; `build_seller_facts` builds the contract the rules read.

7.6 **`explain.py`**: plain-language sentences for the UI, the tick list and `near_miss` ("what would have to change");
it quotes rule thresholds from the rule objects, never from literals.

7.7 **Dashboard (`app/`)**: Streamlit, six pages named after the decision phases, one dark template (`.streamlit/config.toml`,
`app/_theme.py`, colour-blind-safe palette), plain-language headline per page, key numbers, one main chart and a single
"Technical details" expander; every page notes its syllabus topic. It is the internal way to sanity-check outputs; the
Power BI report is the presented deliverable. The app never loads the forest; it reads precomputed files.
Numbers are formatted by `app/_format.py` (thousands separators, fixed decimals, no "-0.000"), table headers come from
`app/_labels.py` (raw column names appear only in help text and CSV downloads), Technical-details tables go through
`app/_components.table()`, and the evidence tiers are shown as "Stronger evidence" and "Early signal".

7.8 **SQL Server layer.** `python -m src.export_sql` writes ten flat CSVs to `data/export/`; `python -m src.load_sql` loads them
into SQL Server using `sql/schema.sql` (drop and recreate in dependency order, one transaction per table, row count against
the CSV and `docs/powerbi_setup.md`, plus one checksum per table; flags `--dry-run`, `--verify-only`, `--only <table>`).
Power BI modelling, DAX, pages, paginated report and row-level security (T4-T15) are manual; see `docs/powerbi_setup.md`.

## 8. Testing

`pytest` covers each module, the app (headless `AppTest` per page), the export and loader (fake connection for rollback;
one read-only integration test that skips without a database) and the documents (`tests/test_results.py`: `RESULTS.md`
current, figures quoted in `docs/DEMO_SCRIPT.md` match it, the rule table equals the YAML, no known-stale numbers,
syllabus-map paths exist). Run everything with `.\tasks.ps1 verify`.

## 9. Commands

```
.\tasks.ps1 setup | data | models | export | load-sql | run | app | test | lint | check | verify | rebuild
python run_all.py [--export]        # full pipeline (+ CSV export), then pytest, ruff, mypy
python -m src.load_sql [--dry-run | --verify-only | --only <table>]
python scripts/make_results.py [--check]       # regenerate / check reports/RESULTS.md
```

`verify` = ruff, mypy, pytest, `load_sql --verify-only`, RESULTS freshness, headless app smoke. `rebuild` = `run_all --export`,
`load-sql`, `verify`. After changing outputs, regenerate `reports/RESULTS.md` and reload SQL Server, then refresh Power BI.

## 10. Working agreement for Claude Code

1. **Plan first** for any task touching more than one file; wait for confirmation before editing.
2. **Small, verifiable steps**; run each stage and show real output.
3. **Run `verify` (or at least ruff, mypy, pytest) before saying a task is done**, and quote the real results.
4. **Do not commit unless asked.** Never commit data, models, `powerbi/` or virtual environments.
5. **Do not silently change behaviour.** If a specification is impossible or wrong, stop, explain, propose a fix.
6. Preserve existing style and line endings. When debugging, lead with the fix, then a short cause line.
7. **Report honestly:** weak model performance, recalibrated thresholds, small effects and synthetic data belong in the docs.
8. Log environment issues and stage results in `BUILD_LOG.md` (short; numbers by reference to `reports/RESULTS.md`).
9. SQL and database access only inside the exception in section 1. Phases that must not change outputs are checked with a
   SHA-256 manifest before and after.
10. Never touch the Power BI report in `powerbi/`.

## 11. Known pitfalls

- Delivered or estimated dates, `delivery_days`, review fields or the target as features (leakage); fitting scalers or imputers on all data; metrics on training data;
  accuracy as the headline on an imbalanced target (lead with ROC-AUC); item-level random splits.
- Scoring VADER on stripped text; unjustified or never-firing rules; hard-coded paths, seeds, thresholds or numbers.
- Presenting a low-item-count hidden gem with the certainty of an established seller; warning or suspending on fewer than 5 items sold or fewer than 3 delivered orders.
- Calling the risk score a probability, "calibrated", or a forecast; saying "orders" for items sold.
- Overwriting raw criterion columns during normalisation; implying Olist has food or nutrition data.
- Quoting a number from memory or an old document instead of `reports/RESULTS.md`.

## 12. Superseded designs (history only; `docs/archive/BUILD_LOG_full.md`)

Logistic regression and decision tree as the only models (forest primary since Stage 3.5; both kept); a four-feature baseline
(12 features since Stage 3); random item-level splits and in-sample risk scores (order-grouped, out-of-fold since Stage 6);
unshrunk item-level quality averages (order-level shrinkage, Stages 5-5b); Warn/Suspend with no volume gate (5-item minimum,
Stage 8); the manual SSMS Import Flat File wizard (`src.load_sql`, Stage 7).
