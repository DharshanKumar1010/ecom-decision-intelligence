# Build Log

Stage-by-stage record of what actually ran, real output, and environment
issues. Numbers here are quoted from real runs, not recomputed by hand — if
the pipeline changes, rerun and update this file rather than editing numbers
in place.

## Environment notes

- **Windows Application Control blocked pandas 3.0.6 and mypy 2.3.1** at
  import, with the error "An Application Control policy has blocked this
  file." `numpy` and `pyarrow` were unaffected. Pinned down to `pandas==2.2.3`
  and `mypy==1.13.0`, both verified working. If a new package hits this same
  block, pin to an older, long-established release rather than fighting the
  OS policy.
- **Application Control now also blocks scikit-learn's compiled extensions (observed 2026-10-07).** After the OS build changed from 10.0.26200 to 10.0.26300 mid-project, `import sklearn` still works but `sklearn.metrics`, `sklearn.ensemble` etc. fail with `DLL load failed ... An Application Control policy has blocked this file` (a different `.pyd` each attempt: `_cython_blas`, `_radius_neighbors`, `_partition_nodes`). numpy, pandas, pyarrow, scipy and streamlit are unaffected. Effect: `tests/test_predict.py` cannot be collected and `python -m src.predict` cannot run on this machine until the policy is resolved. Nothing in `src/` was changed; the same tests passed earlier in the project. Not worked around (no policy bypass, no unrequested dependency change); needs a decision from the project owner.

## Stage 1 — data layer (reference)

`build_tables.py`/`synthetic.py`/`clickstream.py`: `FactOrderItems` 110,189
rows, `DimSeller` 3,095, `DimProduct` 32,951, `DimCustomer` 99,441 (see
CLAUDE.md section 6.2 for the DimCustomer row-count correction — a
documentation error, not a code regression). Synthetic `Clickstream` 100,000
rows, `CallTranscripts` 300 rows. 20 tests passing, ruff/mypy clean.

## Stage 2 — sentiment, prediction, AHP, discovery, expert system (reference)

Baseline predictive model (4 features: `price`, `freight_value`,
`product_weight_g`, `same_state`): logistic regression ROC-AUC **0.5744**,
decision tree ROC-AUC **0.5692**. AHP: 681 established sellers, CR 0.0189.
Discovery: Star 454 / Hidden Gem 478 / Overrated 480 /
Overlooked-Low-Quality 452. 61 tests passing, ruff/mypy clean.

One real bug found and fixed during this stage: `ahp.normalize_criteria` was
overwriting raw criterion columns (including `order_volume`) with their 0-1
normalized values, which silently broke `discovery.py`'s popularity axis
(every seller read as "low confidence" since normalized values are always
below the raw order-count threshold). Fixed by adding `norm_*` columns
instead of overwriting the raw ones.

## Stage 3, Part A — expanding the predictive feature set

Baseline (start of Part A): LR **0.5744**, DT **0.5692**. Six candidate
features added ONE AT A TIME, each retrained, evaluated, leak-guardrail
checked (`AUC > 0.95` would trigger a stop), tested, and committed
separately before moving to the next. None triggered the guardrail; none
were backed out.

| # | Feature | LR AUC (before -> after) | DT AUC (before -> after) | Verdict |
|---|---|---|---|---|
| 1 | `product_category_freq` (frequency-encoded, train-fold-only fit) | 0.5744 -> 0.5740 | 0.5692 -> 0.5667 | Kept — essentially flat/negligibly negative, within noise of a single split; not harmful, conceptually sound |
| 2 | `order_month`, `day_of_week` (from `order_purchase_timestamp`) | 0.5740 -> 0.5853 | 0.5667 -> 0.6692 | Kept — real, substantial improvement. The decision tree splits primarily on `order_month`; consistent with Olist's documented 2017 seasonal delivery disruptions (e.g. November 2017 Black Friday congestion), not a leak |
| 3 | `n_items_in_order` | 0.5853 -> 0.5889 | 0.6692 -> 0.6692 | Kept — small LR improvement; DT unchanged (its depth-4 budget was already spent on `order_month`/`freight_value`/`price`/`same_state`) |
| 4 | `payment_installments` (raw payments CSV, MAX per order) | 0.5889 -> 0.5891 | 0.6692 -> 0.6692 | Kept — negligible effect either way; not harmful |
| 5 | `seller_historical_late_rate` (expanding, strictly-prior-orders-only) | 0.5891 -> **0.6246** | 0.6692 -> 0.6724 | Kept — real, meaningful improvement. Highest leakage risk of the six; built and tested most carefully (see below) |
| 6 | `geo_distance` + `geo_distance_missing` (haversine, zip-prefix centroids) | 0.6246 -> **0.6347** | 0.6724 -> **0.6738** | Kept — real, meaningful improvement |

**Final model: LR ROC-AUC 0.6347, DT ROC-AUC 0.6738** — up from the Stage 2
baseline of 0.5744/0.5692. Both remain well under the 0.95 leak guardrail at
every step.

### `seller_historical_late_rate` — the leakage-risk feature, in detail

Two real ties in the data required actual handling, not just documentation:

1. **Same-order ties.** Items sharing one `order_id` and `seller_id` share
   `order_purchase_timestamp` and `is_late` by construction (an order's
   delivery outcome is one fact, duplicated per item row). Computing history
   at the item level would let a sibling item of the SAME order leak into
   its own "prior" feature. Fixed by collapsing to one row per
   `(seller_id, order_id)` before computing history.
2. **Same-timestamp, different-order ties.** 112 of 97,697 real
   `(seller_id, order_id)` pairs share a seller and an exact-second
   timestamp across two or three DIFFERENT orders (verified by direct
   query, not assumed) — these are simultaneous, not strictly ordered
   relative to each other. Fixed by further collapsing to one row per
   `(seller_id, order_purchase_timestamp)` before the expanding
   computation, so simultaneous orders never see each other either.

The actual computation uses `groupby("seller_id")[...].cumsum()` followed by
a **separate** `groupby("seller_id")[...].shift(1)` call — both standalone
groupby operations, which correctly reset at every seller boundary. This was
a deliberate choice over `groupby(...).expanding().shift()`, which does NOT
reset at group boundaries and would silently leak the last row of one seller
into the first row of the next.

Five dedicated tests were added (`tests/test_predict.py`), including the
explicitly requested one: mutate the *latest* order's outcome and confirm
every earlier order's computed feature value is bit-for-bit unchanged.

### R08 vs. R09 after the expanded model (real numbers, not assumed)

Re-ran `ahp`/`discovery`/`expert` (no code changes) against the new
`LatePredictions.parquet`. Seller-level `avg_late_risk` now ranges
**0.3096 - 0.8837** (was 0.4141 - 0.8371 under the Stage 2 4-feature model) —
meaningfully wider, and the minimum moved much closer to R08's literal
`< 0.3` threshold, but **R08 still fires on 0 real sellers** — the honest
answer, not the hoped-for one. R09 (`avg_late_risk < 0.474`, the
Stage-2-recalibrated threshold) now fires on **388 of 478 Hidden Gems**
(was 132/478), since the wider risk distribution spreads sellers out more.
Full expert-system action counts also shifted: Keep 2231 -> 1881, Warn
382 -> 461, Feature 176 -> 250, Suspend 101 -> 131, Promote 80 -> 247.

No feature was backed out during Part A — this section exists per CLAUDE.md
section 17's instruction to log the process either way, successes included.

## Stage 3.5 — RandomForestClassifier adopted as primary model

**This is a formal, confirmed override of CLAUDE.md's original hard
constraint** ("Predictive models are limited to LogisticRegression and
DecisionTreeClassifier"). The user was told explicitly this conflicted with
the constraint and confirmed proceeding, with CLAUDE.md itself updated to
match (see section 1 and 7.3). LR and DT are retained in the pipeline for
interpretability, not removed.

### Evidence trail (two diagnostics, run before this adoption)

1. `reports/diagnostic_rf_ceiling.txt` — single 80/20 split, same features:
   RF 0.7841 vs. DT 0.6738 vs. LR 0.6347. Top feature by permutation
   importance: `order_month`.
2. `reports/diagnostic_rf_cv.txt` — 5-fold stratified CV, same config:
   RF **0.7928 ± 0.0051** vs. DT **0.6870 ± 0.0083** vs. LR **0.6247 ± 0.0083**.
   `order_month` was the top feature in **5/5 folds**. No unusually
   high/low fold beyond normal variance. This confirmed the single-split
   result was real, generalizable signal, not a lucky split.

### What changed in the committed pipeline

- `src/predict.py`: `RandomForestClassifier(n_estimators=300, max_depth=None,
  class_weight="balanced", random_state=42)` added as a third model,
  trained on the identical 12-feature set and identical 80/20 split as LR/DT.
- **Real single-split test AUCs (this run):** LR 0.6347, DT 0.6738,
  **RF 0.7841** — matches the earlier ceiling diagnostic exactly (same
  split, same seed).
- `LatePredictions.parquet`: `late_risk` (primary column) now comes from
  RandomForest; `late_risk_lr`/`late_risk_dt` added alongside, not
  discarded.
- A real gap was caught and fixed while doing this: `reports/sensitivity.csv`
  was previously **LR-only** (the request's premise that it already covered
  "LR/DT's existing" results was not quite accurate — DT sensitivity had
  never been computed). Fixed by adding a `model` column and running the
  same perturbation methodology for LR, DT, and RF all three.
- `models/`: added `random_forest.joblib` (**423 MB** — worth flagging: an
  unlimited-depth, 300-tree forest on ~88K rows is a real storage cost
  compared to LR's/DT's few-KB files; not a blocker for this project's
  architecture since Python models aren't served through the SQL
  Server/Power BI layer, but notable). Replaced the single combined
  `model_metadata.json` with per-model metadata files, and added
  `models/registry.json` marking `random_forest` as `primary` and the
  other two as `retained_for_interpretability`.

### R08 vs. R09 — the picture inverts under RandomForest (real numbers)

RF's per-item `late_risk` spans the FULL **0.0 - 1.0** range (mean 0.150,
median 0.106), a dramatically wider and better-discriminating distribution
than LR's narrow 0.31 - 0.88 band. Seller-level `avg_late_risk` now ranges
**0.0 - 1.0** (was 0.3096 - 0.8837 under LR). Consequence:

- **R08** (`avg_late_risk < 0.3`, CLAUDE.md's original literal threshold)
  now fires on **472 of 478 Hidden Gems (98.7%)** — up from 0. It is now
  the more discriminating of the two Promote rules.
- **R09** (`avg_late_risk < 0.474`, the value recalibrated against LR's
  narrower range) now fires on **478 of 478 (100%)** — every Hidden Gem
  clears it, so R09 is now essentially non-discriminating given RF's wider
  spread. This is worth revisiting in a future rules pass (not done here —
  out of scope for this diagnostic-to-adoption change), since a
  non-discriminating rule sitting at higher priority than a now-meaningful
  R08 changes which rule actually determines the outcome in practice.

### Full expert-system action-count shift (real, before vs. after)

Quadrant counts (Star/Hidden Gem/Overrated/Overlooked-Low-Quality) are
**unchanged** — confirmed by rerunning `discovery.py`, not assumed;
`compute_quadrants` only ever depended on `ahp_score`/`order_volume`,
neither of which involves `late_risk`.

| Action | Before (LR primary) | After (RF primary) |
|---|---|---|
| Keep | 1,881 | 2,091 |
| Warn | 461 | 242 |
| Feature | 250 | 256 |
| Suspend | 131 | 60 |
| Promote | 247 | 321 |

Suspend and Warn both dropped substantially (RF's tighter, better-calibrated
risk estimates flag fewer sellers as high-risk in aggregate), while Promote
rose (consistent with R08 now firing broadly). This is a real, meaningful
shift in the expert system's real-world output driven purely by swapping
which model produces `late_risk` — underscores why the diagnostic-then-CV
evidence process mattered before making this change.

## Stage 4 — Streamlit dashboard (built, awaiting review; not committed)

`app/Home.py` plus six pages (`1_Intelligence` ... `6_DSS_Architecture`) and
`app/_common.py` (cached loaders, page chrome, friendly missing-file handling,
data catalog). UI only: pages call existing `src/` functions and read existing
pipeline outputs; the 423 MB RandomForest is never loaded. 32 new tests in
`tests/test_app.py` (smoke, content against the real files, missing-file per
page). Friction worth knowing:

- `st.page_link` raises `StreamlitPageNotFoundError` under `AppTest` (no page
  registry). Kept real clickable links; fall back to a plain label on exactly
  that exception (`_common.page_link_or_label`).
- Each page puts the project root on `sys.path` before importing `src`/`app`,
  which trips ruff E402; added a narrow `per-file-ignores` for `app/**` in
  `pyproject.toml` rather than scattering `noqa`.
- CLAUDE.md 7.7.3 (the Discovery caution caption) no longer exists as a
  sub-section after later consolidation; the caption wording was restored from
  the original spec, with the order threshold read from `config`, not hardcoded.
- `config.DISCOVERY_*_THRESHOLD` are `None` (median default), so the scatter's
  quadrant lines use the axis medians via `_common.quadrant_thresholds`, which is
  tested for consistency against the stored quadrant labels.
- A 4-class scatter fails the dataviz all-pairs floors with the default slot-4
  colour; the quadrant palette (blue/orange/violet/aqua) was validated instead.
  Aqua is 2.74:1 on white, so direct quadrant labels and a table view carry it.
- Streamlit follows the system theme but chart colours were validated for the
  light surface only; dark mode is not validated.

## Stage 4b — plain-language dashboard, simplified (built, awaiting review; not committed)

Goal: a non-technical viewer gets each page's point in about three seconds, while a
technical examiner can still reach every number. UI and explanation layer only.

**Current design.**
- One template on every content page: the plain question as the title, one grey live
  subtitle, at most four numbers, one main chart with a single caption, one
  **Technical details** expander at the bottom (tables, matrices, sensitivity, tree text,
  rules, near-miss, catalog), and a small grey footer (syllabus topic · analytics type ·
  management level). Home is a title, a short description, the six pages and one note.
  Intelligence ("How customers sound on calls") and Design ("What if an input changes?")
  each carry one small second section so that the call-sentiment and sensitivity syllabus
  topics are visible without opening Technical details.
- Dark theme (`.streamlit/config.toml`: page #0C1015, surface #141A22, text #F2EFE9, no
  white boxes anywhere, transparent chart backgrounds), one gold accent (#D9B26F, 9.6:1
  on the page, so it stays readable for bars and text), and one sans-serif family for
  everything. Semantic colours exist only for the five actions and the four
  seller groups, are muted, and are identical on every page. Synthetic data gets a small
  grey "Synthetic data" tag beside its section title, nothing louder.
- Files: `app/_theme.py` (tokens and the Plotly template), `app/_components.py` (template
  helpers), `app/_common.py` (cached loaders, catalog, `model_review_flag`), `app/Home.py`
  and `app/pages/1_Intelligence.py` ... `6_DSS_Architecture.py`, `.streamlit/config.toml`,
  `docs/DEMO_SCRIPT.md`, and `src/explain.py` (pure, `mypy --strict`: plain-language
  sentences and `near_miss`, which now only feeds the Technical details of Implementation).

**What was tried, and why it was removed.** The first Stage 4 dashboard (above) was correct
but read like an engineer's report. The first Stage 4b redesign answered that with a dark
theme, a Plain English / Technical reading mode, a Presentation mode, a sidebar glossary
with inline tooltips, a guided tour with step indicators, eyebrow labels, "ANSWER" bars,
"how to read this chart" expanders, status pills, callout cards, tabs, a gauge and
KPI cards. Review feedback was that it was too busy and confusing: every one of those
added a second thing to learn before the data. It also rendered numbers and some sentences
in a serif fallback, which looked inconsistent. All of it was removed rather than tuned:
the content that Plain English mode showed was kept, and everything technical moved into
the one expander. The simplification pass first moved to a light theme, then the dark theme
was restored on request and the semantic colours were re-derived and re-validated on the
dark surface (all pairs, with the dataviz validator). Blue and violet were too close under
red-green colour blindness (delta E about 2), so Overlooked-Low-Quality became a rose and
Overrated an orange; green vs orange and blue vs rose are well separated, and rose vs
green under deuteranopia is the one pair that relies on the second cue (corner labels and
legend).
Tests for the removed features (modes, glossary, tour navigation, header pills, callouts,
gauge, dark-theme config, glyph uniqueness, Home pipeline-status warning) were removed
with them; `tests/test_explain.py` was kept whole.

**Guardrails, verified.** No existing `src/` file changed (`git diff --stat` empty for
`src/`, `rules/`, `data/`, `reports/`, `models/`, requirements and task runners). A
SHA-256 manifest of 71 files taken before the work and 72 after differs only by the new
`src/explain.py`, and was re-checked after the simplification. No new dependency; no
network at runtime (a test scans `app/` and `.streamlit/` for remote URLs; a test fails if
any serif font name appears).

**Honesty notes.**
- `LatePredictions.parquet` is largely in-sample (models are fitted on ~80% of those
  rows), so no lift or "catches X%" is derived from it; quality claims use only the
  held-out `model_metrics.json` and the cross-validation report. The Implementation page
  describes a seller's predicted risk as a pipeline output.
- The "model under review" line is wired to an anchored `FLAG:`/`STATUS:` line naming the
  token `REVIEW_TOKEN` in `app/_common.py`, in this file or under `reports/`. No such line
  exists today, so nothing is shown; a test covers both the shown and absent cases.
- The YAML `because` strings contain project jargon and `rules/` was out of scope, so the
  plain-language reason is built from each rule's conditions and the seller's live values.
- `near_miss` and `condition_checks` call the engine's own `Condition.evaluate`, not a
  second copy of the operator logic. A test checks, for all 2,970 sellers and all 12
  rules, that "this rule would fire" equals the real `InferenceEngine` match.
- Supersedes the Stage 4 note about validating colours for the light surface only (the
  palette is now validated on the dark surface the app uses).

**Gates.** `ruff check .`, `mypy src` (12 files) and `pytest --ignore=tests/test_predict.py`
all pass (counts in the final report of the change); `test_predict.py` still cannot run
here because scikit-learn is blocked by Application Control (see the environment notes).
A headless boot returned HTTP 200 on the health endpoint and the server was stopped.
Visual quality has not been checked in a browser.
