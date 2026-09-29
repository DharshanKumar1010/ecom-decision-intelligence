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
