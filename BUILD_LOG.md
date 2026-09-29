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
