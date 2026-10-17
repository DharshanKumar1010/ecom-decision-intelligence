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
- **Application Control blocked scikit-learn's compiled extensions on 2026-10-07 and 2026-10-08; it no longer reproduces as of 2026-10-13.** After the OS build changed from 10.0.26200 to 10.0.26300, `import sklearn.metrics` etc. failed with `DLL load failed ... An Application Control policy has blocked this file` (a different `.pyd` each attempt). Details, the diagnosis and the re-run of `src.predict` are in the "scikit-learn Application Control block: diagnosis and resolution" entry at the end of this file. The sklearn binaries are still unsigned, so the verdict is Smart App Control's reputation decision and could change again; nothing was done to the policy.

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

## Stage 4d — hidden gems by confidence tier, shrinkage diagnostic, small app fixes (not committed)

**Problem.** The Discovery page's "Top 10 hidden gems" was 9/10 low-confidence and every
one averaged exactly 5.00. A Hidden Gem has fewer orders than the popularity median (18) and
low confidence is under 15, so only 15 to 17 order sellers can ever be normal-confidence.

**Measured before any change** (`discovery_quadrants.csv` + `SellerFacts`): of the 478
gems, 434 are low and 44 normal confidence. Order counts: 5:76, 6:54, 7:54, 8:44, 9:40,
10:42, 11:41, 12:33, 13:27, 14:23, 15:16, 16:16, 17:12. 54 gems (11%) average exactly 5.0
(2 of them normal confidence). Inside the gem quadrant the Spearman correlation between
orders and AHP score is -0.19 (Pearson -0.20): more orders goes with a lower score, so the
top of a plain AHP ranking is a small-sample artefact. Note that `order_volume` counts
order items (`item_key`), not distinct orders; the UI wording was not changed.

**Change (presentation only).** `src/discovery.py` gains `TIER_ACTIONABLE`/`TIER_WATCHLIST`,
`hidden_gem_tier`, and `top_hidden_gems(..., tiered=False)`; with `tiered=True` it adds a
`tier` column and lists Actionable (normal confidence) then Watchlist, each by AHP score
with ties broken by order count, `n` rows per tier. The default call is unchanged and it
still returns gems only. `main()` uses the tiered output for `reports/discovery_summary.md`.
Quadrant membership, `is_hidden_gem`, rules and thresholds are untouched: re-running
`python -m src.discovery` left `SellerFacts.parquet`, `discovery_quadrants.csv` and
`seller_recommendations.csv` byte-identical (quadrants still 454/478/480/452); only the
summary file changed. The Discovery page now shows two stacked tables, Actionable
(15+ orders, 44 sellers) and Watchlist (under 15 orders, 434 sellers), with orders, average
review and confidence, a plain sentence that gems by definition have under 18 orders, and a
download with a `tier` column.

**Diagnostic, nothing adopted** (`scripts/diagnostic_shrinkage.py`,
`reports/diagnostic_shrinkage.txt`). Shrinking average review and on-time rate toward the
global mean (prior strength k): k=5 keeps 464 gems (2 of the top 10 normal-confidence,
Spearman 0.99 with the current scores), k=10 keeps 450 (4 of 10), k=15 keeps 441 (4 of 10),
k=30 keeps 428 and adds 1 (5 of 10, Spearman 0.92); the 43 or 44 normal-confidence gems
survive at every k. Shrinkage only removes sellers whose quality was a small-sample luck
and reorders the top, but even k=30 leaves half of the top 10 low-confidence. Orders per
active month as the popularity axis changes the definition itself: 444 gems, 117 of them
new and 151 dropped, with sellers of up to 70 orders now counted as "low popularity", and
0 of the top 10 normal-confidence. Recommendation: keep the tiers (done); consider
shrinkage around k=10 to 15 as a separate, asked-for change, because it moves gem
membership (about 30 to 40 fewer gems) and therefore `is_hidden_gem`, the Promote rules and
their calibration; do not adopt the rate-based popularity measure.

**App fixes.** (a) Design "What if an input changes?" leaves out month of purchase and day
of the week (multiplicative nudges to calendar numbers are not realistic); they stay in the
Technical details heatmap with its caveat, and the chart still shows five bars. (b)
Implementation drops the separate "Why" sentence; the subtitle and the ticked conditions
carry the reason. (c) `.streamlit/config.toml` sets `client.toolbarMode = "minimal"`
(supported by the installed Streamlit 1.64).

## Stage 5 — empirical-Bayes shrinkage adopted in the AHP quality score (not committed)

**Why.** Stage 4d showed that inside the Hidden Gem quadrant items sold and AHP score
correlated at -0.19: small sellers with a few perfect reviews were over-ranked (54 of 478
gems averaged exactly 5.0). Shrinkage was adopted as an explicit, separately tested change
to the model, not a presentation tweak. It was built twice, and the history matters:

1. *Item-level version (superseded).* n counted items, k = 11 (review) and 14 (on-time),
   and `n_reviewed` counted reviewed items. Reviews and lateness belong to an ORDER and repeat
   on every item of it, so counting items overstated the evidence behind a seller.
2. *Order-level version (current).* n counts distinct orders per seller: `n_reviewed` =
   distinct reviewed orders, `n_orders` = distinct delivered orders; the raw columns
   `avg_review_raw` / `on_time_rate_raw` are per order (one review per order, an order is late
   if delivered late). Eligibility is unchanged and still defined by items sold (>= 5 for
   Discovery, >= 30 for Choice). Verified: in all 97,811 (seller, order) pairs the items agree
   on review and lateness.

**What changed.**
- `src/ahp.py`: `seller_order_table`, `estimate_prior_strengths` (order level, with the
  intermediate variances), `select_prior_strength` (rounded estimate if inside the tested 5 to
  30 range, else 10), `shrink_toward_mean` (`(n*x + k*mean)/(n + k)`), and
  `build_seller_criteria(..., shrink=True)`, the one shared scoring path for Choice and
  Discovery. Price, items sold, `compute_weights` and the pairwise matrix are untouched;
  `python -m src.ahp` also writes `reports/ahp_prior_strength.csv`. `src/discovery.py`
  needed no change.
- `src/config.py`: `AHP_SHRINKAGE_K_REVIEW = 18`, `AHP_SHRINKAGE_K_ON_TIME = 26`, range
  (5, 30), fallback 10. A test checks them against the order-level estimate from the data.
- `rules/seller_rules.yaml`: only the two rules that read `ahp_score`, both percentile
  intent, re-derived at the same percentile of the new distribution: R06 0.789 to 0.711
  (90th), R11 0.604 to 0.549 (10th). Late-risk thresholds untouched (to be recalibrated
  separately).
- Wording: seller-level counts are "items sold" in the UI, docs and `src/explain.py`
  strings ("Delivered orders" keeps its name on Intelligence); Discovery tiers are shown as
  "Stronger evidence (15+ items sold)" and "Early signal (under 15 items sold)"; a sentence in
  Technical details and the demo script explains items sold versus delivered orders.
  `explain._fmt` shows thresholds to 4 decimals. `scripts/diagnostic_shrinkage.py` now builds
  its own frozen item-level criteria so its historical report is reproducible (byte-identical).

**Real numbers (order level).** k estimates on the pool of 1,864 sellers with >= 5 items sold:
review within variance 1.6332, observed 0.23422, noise 0.14256, between 0.09165, k = 17.82
(used 18); on-time within 0.07376, observed 0.009380, noise 0.006534, between 0.002847,
k = 25.91 (used 26). The expected "about 22 and 28" came from an earlier sensitivity on
sellers with >= 5 distinct orders (1,766 sellers: 21.6 and 27.8); on the stated items-sold
pool the estimates are lower. Choice-pool sensitivity (>= 30 items, 681 sellers): 25.65 and
41.82 (the on-time value is outside the range, so it would fall back to 10); one k per
criterion is used for both pages.

Three-way comparison (no shrinkage, item level, order level): Hidden Gems 478 / 450 / 437;
Star 454 / 482 / 495; Overrated 480 / 452 / 439; Overlooked-Low-Quality 452 / 480 / 493;
gems vs no shrinkage: item level 450 stay, 28 leave, 0 enter; order level 420 stay, 58
leave, 17 enter. Normal-confidence gems in the top 10: 1 / 4 / 4. Top-10 gems with a raw 5.0
average: 10 / 6 / 1. Spearman of items sold vs score inside the gem quadrant: -0.19 / +0.17 /
+0.23. Choice top-10 overlap with the no-shrinkage ranking: 7 (item) and 5 (order) of 10.
Actions Keep / Promote / Feature / Warn / Suspend: 1,970 / 330 / 233 / 270 / 167, then
1,946 / 397 / 190 / 270 / 167, then 1,949 / 412 / 173 / 269 / 167 (256 sellers changed
action no-shrinkage to order level). Full per-rule tables: `docs/knowledge_engineering.md`
section 8.

**Honest limits.** Shrinkage compresses scores, so the median quality line that defines the
quadrants fell (0.716 to 0.644); the 17 sellers that entered the gem quadrant all came from
Overlooked-Low-Quality (they average 4.24 per reviewed order, 5 of 17 below the marketplace
mean of 4.14, 2.2% late; an earlier version of this entry said 3.81, which was the per-item
mean and is superseded by Stage 5b). When this stage was first written `SellerFacts.avg_review`
was still the per-item mean while Choice showed the per-order mean; Stage 5b below closes that.
The k estimate depends on the pool, and a single k is a compromise between small and large
sellers.

**Environment.** scikit-learn is still blocked by Application Control, so only `src.ahp`,
`src.discovery`, `src.expert` and `src.export_sql` were re-run, individually; `src.predict`
and `run_all.py` were not run. `LatePredictions`, the models and the model reports are
byte-identical.

## Stage 5b — one definition of "average review" for Choice, Discovery, Implementation and the rules (not committed)

**Problem.** `SellerFacts.avg_review` (the fact behind R01, R04, R07, R10 and the
Implementation page) was the per-item mean, while Choice's `avg_review_raw` was the per-order
mean (one review per order); 749 sellers differed by more than 0.05, because one unhappy review
on a multi-item order counted once per item.

**Change.** `src/discovery.py::build_seller_facts` now computes `avg_review` as the mean of one
review per distinct order (`src.ahp.seller_order_table`) for ALL 2,970 sellers; the 5 sellers
without a reviewed order keep NaN. Fact names are unchanged; items sold, late rate and
predicted late risk are unchanged. Re-run individually: `discovery`, `expert`, `export_sql`
(sklearn is still blocked; `predict` and `run_all.py` were not run). `discovery_quadrants.csv`
and `ahp_ranking.csv` came out byte-identical (437 gems, same quadrants); `SellerFacts`, the
recommendations and their exports changed.

**Rules reading `avg_review`** (intent from each rule's `because` text and the docs):
R01 and R04 (`< 3.5`) are fixed-level (the cutoff is the stated 3.5; its percentile moved
12.85th to 11.60th, still inside the documented 10th to 25th band) and were left; R07
(`>= 4.3`) is fixed-level in practice (the 51st percentile, 51.5th to 50.7th; its text says
"top quartile", a pre-existing mismatch that is flagged, not changed) and was left; R10
(`>= 4.27`) is percentile-intent (median) and was recalibrated to the new median, **4.28**.
Late-risk and `ahp_score` thresholds untouched. Fired counts before / after the fact change /
after R10: R01 165 / 152 / 152, R02 2 / 5 / 5, R03 181 / 188 / 188, R04 83 / 58 / 58, R05
5 / 5 / 5, R06 130 / 130 / 130, R07 43 / 47 / 47, R08 306 / 306 / 306, R09 106 / 106 / 106,
R10 63 / 67 / 63, R11 601 / 617 / 621, R12 1,285 / 1,289 / 1,289. Actions (Keep / Promote /
Feature / Warn / Suspend) 1,949 / 412 / 173 / 269 / 167 to 1,973 / 412 / 177 / 251 / 157; 77
sellers changed action (Warn to Keep 32, Suspend to Keep 9, Suspend to Warn 9, Keep to Feature
8, Keep to Warn 7, Keep to Suspend 6, Feature to Keep 4, Warn to Suspend 2).

**Read-only gem diagnostic** (gem definition untouched). Per-order review vs the marketplace
mean 4.14: 12 of 437 gems (2.7%) are below it, none below 4.0 (minimum 4.07), mean 4.59, late
rate 1.4%. The 17 entrants average 4.24 (5 below the mean) with 2.2% late; the 58 that left
average 4.43 with 9.9% late. Below the 5% bar, so no extra caveat sentence was added to the
Discovery page or the demo script. Full table: `docs/knowledge_engineering.md` section 8b.

**Correction.** Stage 5 reported that the 17 entrants "average only 3.81 raw review". That
figure used the per-item mean; per reviewed order it is 4.24. Stage 5's docs and the demo
script were corrected.

**Gates.** See the final report of this change for ruff, mypy and pytest counts and the
hash comparison.

## scikit-learn Application Control block: diagnosis and resolution (2026-10-13, not committed)

**Constraint.** No system security setting was changed (no Smart App Control, Defender,
registry, WDAC policy or ACL change; no Unblock-File; no Mark-of-the-Web removal). Everything
below was read-only except re-running `src.predict`.

**Diagnosis (read-only).**
- Python 3.11.9 64-bit, Windows 10.0.26300. Project `.venv`: scikit-learn 1.9.1, numpy 2.4.6,
  scipy 1.17.1, pandas 2.2.3, joblib 1.6.0, threadpoolctl 3.7.0 (identical to `requirements.txt`).
- Smart App Control is in enforcement mode (`VerifiedAndReputablePolicyState = 1`). The
  CodeIntegrity/Operational log held 575 events of ID 3033/3077/3034 since 2026-10-04; nearly
  all are Chrome's `libLiteRtWebGpuAccelerator.dll` (unrelated).
- The only Python blocks (3033 "did not meet the Enterprise signing level requirements" with a
  matching 3077 "or violated code integrity policy", Policy ID
  {0283ac0f-fff1-49ae-ada1-8a933130cad6}, process `...\Python311\python.exe`):
  `.venv\Lib\site-packages\sklearn\utils\_cython_blas.cp311-win_amd64.pyd` (2026-10-07 13:52
  and 2026-10-08 19:05), `...\sklearn\metrics\_pairwise_distances_reduction\_radius_neighbors.cp311-win_amd64.pyd`
  (2026-10-07 13:53) and `...\sklearn\neighbors\_partition_nodes.cp311-win_amd64.pyd`
  (2026-10-07 13:53). No Python block was logged after 2026-10-08.
- Import check (`import_check.py`, scratchpad): **23 of 23 pass** on 2026-10-13: numpy, scipy
  (sparse, linalg, special, stats), joblib, threadpoolctl, sklearn (base, utils, impute,
  preprocessing, pipeline, linear_model, tree, ensemble, metrics, model_selection, inspection,
  and the three formerly blocked modules `utils._cython_blas`,
  `metrics._pairwise_distances_reduction`, `neighbors`). A tiny in-memory fit of
  RandomForest/DecisionTree/LogisticRegression pipelines, `roc_auc_score`,
  `permutation_importance` and `StratifiedKFold` also ran.
- The sklearn `.pyd` files are still `NotSigned` (69 compiled files), so the change is not a
  signature; Smart App Control decides unsigned code by reputation, and that verdict changed.
  It can change again.

**Fix attempts.** Fresh-venv attempts with scikit-learn 1.5.2, 1.4.2 and 1.3.2 (and an older
scipy) were **not run**: the project's own combination (1.9.1 / 2.4.6 / 1.17.1) imported
cleanly, so there was nothing to fix and older versions would only force older numpy/scipy for
no gain. The first combination that imports cleanly is therefore the installed one: nothing
was installed, upgraded or force-reinstalled, and `requirements.txt` already pins these
versions (checked against `pip freeze`; unchanged). If the block returns, the plan is: record
the exact blocked paths from the event log, try 1.5.2, 1.4.2, 1.3.2 in that order each in a
fresh venv under `%TEMP%`, and, if none imports, use WSL2 Ubuntu (install, clone, copy
`data/raw`, create the venv, run the tests) rather than any bypass.

| Combination | Result |
|---|---|
| scikit-learn 1.9.1, numpy 2.4.6, scipy 1.17.1, pandas 2.2.3 (project `.venv`) | 23/23 imports pass; `src.predict` and the full test suite run |
| scikit-learn 1.5.2 / 1.4.2 / 1.3.2, fresh venvs | Not attempted (not needed) |

**Re-running `src.predict`.** `reports/`, `models/` and `data/processed/LatePredictions.parquet`
were first copied (with SHA-256 hashes) to `%TEMP%\ecom_backup_predict_20261013`, outside the
repo. `python -m src.predict` then ran to completion (exit 0; about 7 minutes, almost all
RandomForest training and permutation importance). Test ROC-AUC: logistic regression
**0.6347**, decision tree **0.6738**, random forest **0.7841**, identical to the committed
values (tolerance was 0.003). Comparison with the backup, file by file: 21 files, namely the reports,
`LatePredictions.parquet` and the model pickles, are **byte-identical** (including the three
`.joblib` models: random forest 423,821,922 bytes, decision tree 4,739 bytes, logistic
regression 2,561 bytes). The only differences are the `timestamp`/`updated` fields in the four
untracked `models/*_metadata.json` and `models/registry.json` files (rewritten with today's
date). `git status` shows no change to any tracked file. The backup was therefore not needed
for a restore.

**Gates.** `ruff check .` clean; `mypy src` clean (12 files); the FULL `pytest` including
`tests/test_predict.py` (no `--ignore`): **192 passed** (the previous 171 plus the 21
`test_predict.py` tests that could not be collected while sklearn was blocked). A hash manifest
of src, rules, data, reports and models taken before the run differs afterwards only in the
untracked `models/*_metadata.json` and `models/registry.json` timestamps.

## Stage 6 — order-grouped evaluation and out-of-fold late risk (not committed)

**Two suspected problems, measured first** (`scripts/diagnostic_group_leak.py`,
`reports/diagnostic_group_leak.txt`; nothing committed was changed by it).
1. Sibling leakage: items of one order share `is_late` and most features, but the split put
   item rows at random into train and test; 18.34% of test items (16.75% of test orders) had a
   sibling in train. Test AUC, random item split vs grouped by order (same seed and sizes):
   logistic regression 0.6347 / 0.6259, decision tree 0.6738 / 0.6888, random forest (fully
   grown) 0.7841 / 0.7274. The forest alone lost 0.057; its lead over the tree fell from 0.110
   to 0.039.
2. In-sample late risk: the committed `late_risk` scored every row with a model that had trained
   on ~80% of them: AUC 1.0000 on its own training rows, 0.7841 on unseen rows; seller-level
   `avg_late_risk` correlated 0.906 (0.925 for sellers with >= 5 items sold) with the actual
   late rate, i.e. mostly an echo of it, then fed to the rules.

**Changes.**
- `src/predict.py`: `grouped_stratified_folds` (5 `StratifiedGroupKFold` folds over `order_id`,
  shuffle, seed 42, `config.PREDICT_N_FOLDS`); fold 0 is THE train/test split (metrics,
  permutation importance, sensitivity, persisted models); `out_of_fold_predictions` writes
  out-of-fold `late_risk` (forest), `late_risk_lr`, `late_risk_dt` to `LatePredictions.parquet`
  (fresh model per fold and model, category encoding refitted per fold, nothing persisted);
  `reports/oof_metrics.json` holds grouped-test, out-of-fold and in-sample AUCs. The feature
  table also carries `order_id` as a grouping key; the model features are unchanged.
- Forest leaf size: `reports/diagnostic_rf_leaf.txt` (same grouped OOF protocol) gave pooled OOF
  AUC 0.7233 / 0.7311 / 0.7367 and fitted size 421 / 193 / 68 MB for `min_samples_leaf`
  1 / 5 / 20. Adopted 20 (`config.PREDICT_RF_MIN_SAMPLES_LEAF`), reading the "within 0.005"
  rule as non-inferiority, since larger leaves were better, not just close; chosen on the
  folds it is reported on, so slightly optimistic; end of the grid, larger values untried.
- New numbers: grouped test AUC logistic regression 0.6259, decision tree 0.6888, random forest
  0.7410; out-of-fold 0.6241 / 0.6856 / 0.7367 (forest fold std 0.0041); forest AUC on its own
  training rows 0.9346 (1.0000 with fully grown trees). Persisted forest 68 MB (was 424 MB).
  Seller-level correlation of `avg_late_risk` with the actual late rate: 0.339 (0.517 for
  sellers with >= 5 items sold).
- Rules: all `avg_late_risk` cutoffs were percentile-intent and were re-derived at the same
  percentile of the new distribution: R01 0.178 to 0.416 (75th), R03 0.29 to 0.501 (90th), R06
  and R10 0.106 to 0.339 (median), R09 0.05 to 0.252 (25th among gems). R08 keeps its literal
  0.3 (now fires on 50.8% of gems, not 0% or 100%). R07's `because` text corrected (threshold
  unchanged). Fired counts for all 12 rules and the action table are in
  `docs/knowledge_engineering.md` section 9: Keep / Promote / Feature / Warn / Suspend went from
  1,973 / 412 / 177 / 251 / 157 to 2,118 / 210 / 174 / 325 / 143 (521 sellers changed action;
  Promote roughly halves because the in-sample risk of a gem was near zero).
- App and docs: the Implementation fact note now says predicted risk is out-of-fold and a model
  estimate (not a measured outcome); the Design page reads its cross-validation figures from
  `reports/oof_metrics.json` (the old random-fold `diagnostic_rf_cv.txt` is superseded) and its
  trade-off sentence is computed from the AUCs in the files; DEMO_SCRIPT questions 1, 3, 5, 6, 10
  and 11 updated. No "model under review" flag existed, so none was removed.

**Gates that held.** Quadrants unchanged (Star 495, Hidden Gem 437, Overrated 439,
Overlooked-Low-Quality 493), the same 437 gems, and `ahp_ranking.csv`,
`discovery_quadrants.csv` and `discovery_summary.md` are byte-identical.

**Temporal check** (`scripts/diagnostic_temporal.py`, `reports/diagnostic_temporal.txt`; train on
the earliest 80% of orders, test on the latest 20%, fully grown forest): all 12 features,
logistic regression 0.3975, decision tree 0.5049, random forest 0.5210; without `order_month` and
`day_of_week`, 0.4007 / 0.4188 / 0.4402. The forest stays nominally ahead, but no model has
useful forward-looking skill (test period late rate 5.2% against 8.6% in training; possible
censoring of long-delayed orders not verified). The random-forest adoption of Stage 3.5 was
justified with random-fold CV that is now known to have been inflated; it remains the best model
on unseen orders from the same period, with a smaller margin. The model family was NOT changed.

**Surprises.** (1) The forward-looking AUCs are at or below 0.5. (2) The forest's memorised
training rows did most of the work behind the seller-level risk the rules used. (3) With balanced
class weights the score's median is 0.34 against a 7.9% late rate: it ranks, it is not a
probability. (4) Fully grown trees were the worst leaf size by AUC as well as by size.

## Stage 7 — SQL Server loader (built, not committed)

**Why.** The audit found the SSMS Import Flat File wizard would mis-type or reject several export
columns: `DimProduct.product_weight_g` has two blanks (first at row 8,578, so the column is guessed
NOT NULL), `seller_recommendations.because` has 1,298 rows with an em dash and the file has no BOM,
and several text columns grow past the sampled rows (`item_key` 35 vs 34, category names 46 vs
34, `quadrant` 22 vs 10).

**Owner exception.** CLAUDE.md sections 1 and 10 forbid writing SQL or connecting to a database.
The owner lifted that for exactly `sql/schema.sql`, `src/load_sql.py` and `tests/test_load_sql.py`.
CLAUDE.md was not edited; a proposed diff was handed to the owner. `pyodbc==5.3.0` is the one new
dependency (`requirements.txt`, `pyproject.toml` mypy override and a `sql_server` pytest marker).

**Driver check.** ODBC drivers installed: SQL Server, ODBC Driver 17, ODBC Driver 18 (32 and 64
bit). `MSSQL$SQLEXPRESS` running, `SQLBrowser` stopped (not needed for a local connection).
`pyodbc` 5.3.0 imported under Smart App Control on the first try, in a throwaway venv outside the
repo; no older version had to be tried and no policy was touched. Connection to
`localhost\SQLEXPRESS` with Windows Authentication works through Driver 18 with
`TrustServerCertificate=yes` (a client-side option for the local self-signed certificate). The
database `EcomDecisionIntelligence` already existed and held two older hand-imported tables,
`DimCustomer` and `DimProduct` (nvarchar(50), `product_weight_g` as float); the loader replaced
them.

**Schema (`sql/schema.sql`, hand written from a profile of the real files).** Text is nvarchar at
about twice the longest value (ids 32 to `nvarchar(64)`, `item_key` 35 to 80, `because` 227 to 512,
`transcript` 107 to 256); no `nvarchar(max)`, nothing needs it. No float anywhere: money
`decimal(12,2)`, scores and risks `decimal(9,6)`, `avg_price` `decimal(14,6)`, `hold_seconds`
`decimal(9,1)`, `silence_pct` and `compound` `decimal(7,4)`; whole numbers use `tinyint`
(`is_late`, `review_score`), `smallint` (`order_item_id`) or `int` (`product_weight_g`, which is whole
grams); datetimes `datetime2(0)`. NOT NULL everywhere except `product_category_name` (610 blanks),
`product_weight_g` (2), `review_score` and `review_creation_date` (827). Primary key on each
table's natural key (`Clickstream`: clustered index on `session_id`, which repeats), 19 indexes
including the Power BI join paths and `seller_state` for row-level security. Foreign keys: all
nine candidates were counted on the real data and had 0 orphans (the `item_key` link was checked
both ways); eight are declared (the reverse `FactOrderItems` to `LatePredictions` one is not, a
cycle). View `dbo.vw_SellerScorecard`: `DimSeller` left joined to the three seller-level tables,
3,095 rows, 2,414 of them with no AHP score (below the 30-item threshold).

**Loader (`src/load_sql.py`).** Reads every CSV as text, converts to the DDL type and raises on
anything that would need rounding, truncating or a NULL in a NOT NULL column, all before the
database is touched; checks each CSV's row count against `docs/powerbi_setup.md`; drops the selected
tables in one transaction, then creates, loads (`fast_executemany`, 10,000-row batches) and
verifies each table in its own transaction. Verification: SQL count = CSV count = docs count, plus
one exact checksum (price x 100, late risk x 1e6, AHP score x 1e6, weight, hold seconds x 10,
seconds since 2000 of `event_time`, `LEN(because)`; `DimSeller` and `DimCustomer` have no numeric
column, so theirs is `SUM(LEN(state))`, a weak check). A mismatch rolls the table back and exits 1;
the first failure stops the run. Safety: exact database name, exact table names, an allow-list on
every statement (no `DROP DATABASE`, `ALTER`, `EXEC`, logins, `sys.*`, other tables), `--dry-run`
opens no connection, `--verify-only` only reads, `--only` is refused for a table that others
reference. Constants live in the module rather than `config.py` to keep the exception to three files.

**Real run.** Dry run 11.4 s (reads and type-checks all CSVs, no connection). Full load 42 s wall,
33.2 s inside the loader (FactOrderItems 16.7 s, LatePredictions 6.3 s, Clickstream 5.4 s,
DimCustomer 2.9 s, the rest under 1.2 s each); a second full load through `.\tasks.ps1 load-sql`
took 41 s and gave the same result, so drop-and-recreate over existing foreign keys works.
`--verify-only` 0.8 s. All 10 tables and the view: rows loaded = expected, status OK. Read back
from SQL independently: 1,298 `because` rows contain the em dash, 2 NULL weights, 610 NULL
categories, 827 NULL review scores, longest `transcript` 107, longest `quadrant` 22, longest
`item_key` 35, average `is_late` 0.079082, sum of price 13,220,248.93, 8 foreign keys all trusted.

**Tests.** `tests/test_load_sql.py`: 75 tests, no live database needed for 74 of them. The one
integration test (`sql_server` marker, read-only) runs when the database is reachable and its
tables are loaded, and skips itself otherwise (it skipped before the first load and passed after).
They cover schema parsing, CSV/DDL column and type fit, the safety rules, dry run opening no
connection, and a fake connection proving that a count or checksum mismatch rolls back and exits
non-zero. Gates after the load: `ruff check .` clean, `mypy src` clean on 13 files, full
`pytest`: **273 passed** in 300 s (198 existing + 75 new).

**Fixed while building.** The CSV has `item_key` as its last column and the DDL has it first, so
columns are matched by name. My first `SELECT` guard let `SELECT ... FROM sys.databases` through;
it now requires `FROM dbo.<loader object>` and rejects `sys.`, `INFORMATION_SCHEMA`, other
databases and `@@`.

**Not done / caveats.** The Makefile has no `load-sql` target (only `tasks.ps1` was asked for).
`TrustServerCertificate=yes` skips certificate validation, acceptable for a local instance but
not for a remote one. The pre-existing `DimCustomer` and `DimProduct` were dropped without a
backup; their row counts (99,441 and 32,951) already matched the CSVs.
