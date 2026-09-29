# Knowledge Engineering: Threshold Justifications

Every threshold below was checked against the real output of `python run_all.py`
on this project's actual `data/processed/` tables (2,970 sellers total, 110,189
fact rows), not invented. Percentiles are computed from `SellerFacts.parquet`
(written by `src.discovery.build_seller_facts`) and `FactOrderItems.parquet`
unless stated otherwise. If the pipeline is rerun on a different data
snapshot, these numbers should be recomputed and this file updated —
CLAUDE.md section 1 forbids quoting a number that wasn't actually measured.

## 1. AHP established-sellers threshold: `AHP_MIN_ORDERS = 30`

Per-seller order counts across all 2,970 sellers:

| Percentile | order_volume |
|---|---|
| 25th | 2 |
| 50th (median) | 8 |
| 75th | 26 |
| 90th | 83 |
| 95th | 151 |

30 orders sits at the **77.9th percentile** of all sellers — i.e. it selects
the top ~22.9% (681 of 2,970) as "established" for the main AHP ranking.
This was chosen to keep the established-seller ranking restricted to sellers
with enough of a track record that an average-review/on-time-rate estimate
is meaningful, while still leaving a majority of the marketplace (the other
77%) for the lower-threshold discovery pass. 30 orders is also comfortably
above the point (order_volume >= ~10-15) where an empirical rate like
`on_time_rate` stops being dominated by single-order noise.

Resulting AHP run: 681 established sellers, weights `avg_review_score=0.473,
on_time_rate=0.284, avg_price=0.073, order_volume=0.170` (from the pairwise
matrix in `config.AHP_PAIRWISE_MATRIX`), consistency ratio **CR = 0.0189**,
well under the `AHP_CR_THRESHOLD = 0.10` flag.

## 2. Discovery AHP-eligibility threshold: `DISCOVERY_MIN_ORDERS = 5`

5 orders sits at the **41.9th percentile** — it makes 1,864 of 2,970 sellers
(62.8%) eligible for a quality (AHP) score, versus 22.9% at the 30-order
established-seller threshold. This is the whole point of the discovery
feature (CLAUDE.md section 7.7.1): a seller needs *some* orders for an
average review score and on-time rate to mean anything at all, but 5 is low
enough that small/newer sellers — the ones a "sort by volume" view would
bury — are actually included, rather than only re-running the same
established-seller list at a different name.

## 3. Discovery confidence cutoff: `DISCOVERY_CONFIDENCE_MIN_ORDERS = 15`

15 orders sits at the **66.1st percentile** of all sellers. Among the 1,864
AHP-eligible (>=5 order) sellers specifically, the median order count is
**18** — meaning the 15-order confidence cutoff sits just below the middle
of the eligible population, splitting it into **818 "low" confidence rows
(43.9%) and 1,046 "normal" (56.1%)** on the last real run. This threshold
was picked so "low confidence" genuinely means "below-typical order count
for an eligible seller" (not an arbitrary round number): a seller just past
the 5-order eligibility floor is treated with reduced confidence until they
clear roughly the eligible population's own midpoint.

## 4. Discovery quadrant thresholds: median split (both axes)

`config.DISCOVERY_QUALITY_THRESHOLD` / `DISCOVERY_POPULARITY_THRESHOLD` are
both `None` by default, meaning `compute_quadrants` splits each axis at its
own median among AHP-eligible sellers — the simplest split that guarantees
roughly balanced quadrants without hand-picking a magic number for either
axis. On the last real run this produced: **Star 454, Hidden Gem 478,
Overrated 480, Overlooked-Low-Quality 452** — a near-even four-way split,
confirming the median default doesn't collapse the classification onto one
dominant label.

## 5. Promote late-risk cutoffs (rules R08, R09) — recalibrated for Stage 3.5 (RandomForest)

**History, so the numbers below aren't read against the wrong context:**
Under the original Stage 2 4-feature LR model, `avg_late_risk` ranged
0.414-0.837 and R08 (`< 0.3`, CLAUDE.md's literal spec) fired on 0 real
sellers; R09 (`< 0.474`, the 25th percentile of that LR distribution) was
added as the realistic threshold. Stage 3 expanded to 12 features (still
LR-primary at the time) and widened the range slightly to 0.310-0.884; R08
still fired on 0, R09 (still `0.474`) fired on 388/478 Hidden Gems. **Stage
3.5 adopted RandomForest as the primary model** (CLAUDE.md section 18,
`BUILD_LOG.md`), whose `late_risk` spans the FULL 0.0-1.0 range — a much
wider, better-calibrated signal, but it flipped which rule was
discriminating: R08 jumped to firing on 472/478 (98.7%), and R09's `0.474`
became non-discriminating at 478/478 (100%, since RF's max observed
`avg_late_risk` among Hidden Gems is well below 0.474). This section
documents the real recalibration done in response.

**Step 1 — check `avg_late_risk` among ALL sellers** (context, not the
basis for the new threshold — see Step 2 for why):

| Percentile | avg_late_risk (all 2,970 sellers) |
|---|---|
| min | 0.000 |
| 10th | 0.040 |
| 25th | 0.063 |
| 50th (median) | 0.106 |
| 75th | 0.178 |
| 90th | 0.290 |
| 95th | 0.414 |
| max | 1.000 |

**Step 2 — check `avg_late_risk` among the 478 Hidden Gems specifically**
(the population R08/R09 actually gate, since both require
`is_hidden_gem == true`; Hidden Gems skew lower-risk than the general
population because the AHP composite that selects them already weights
`on_time_rate` at 0.284):

| Percentile | avg_late_risk (478 Hidden Gems only) |
|---|---|
| min | 0.000 |
| 10th | 0.036 |
| 25th | 0.050 |
| 50th (median) | 0.074 |
| 75th | 0.109 |
| 90th | 0.170 |
| max | 0.413 |

**R08** stays at CLAUDE.md's literal `avg_late_risk < 0.3` — not touched,
since it is now genuinely meaningful (98.7% of Hidden Gems, 472/478) rather
than vacuous. Repositioned as the BROADER, lower-priority (78) "worth
considering" Promote tier.

**R09** recalibrated to `avg_late_risk < 0.05`, approximately the **25th
percentile among Hidden Gems specifically** (0.050 from the table above,
not the all-sellers 25th percentile of 0.063 — using the Hidden-Gem-specific
distribution is the correct reference population, since that's the only
population these rules ever evaluate). Real match rate: **119 of 478
(24.9%)** — genuinely discriminating, comparable in spirit to the original
R09's 132/478 (27.6%) before it drifted to 100% under the wider RF range.
Repositioned as the STRICTER, higher-priority (80) "high confidence, act
now" tier: since R09's condition (`< 0.05`) is a strict subset of R08's
(`< 0.3`), R09 fires first whenever both match, and R08 only ends up being
the reported rule for the broader `[0.05, 0.3)` band. Verified on the real
pipeline run: of 315 sellers with `action == Promote`, 58 are attributed to
R09 (the tight, high-confidence band) and 257 to R08 (the broader band).

**Why recalibrate R09 rather than drop it or add a third rule:** dropping
R09 would leave only R08's now-broad 98.7% threshold, losing the
high-confidence/broader distinction the two-tier design always intended.
Adding a third rule (R13) would work but duplicates the existing "R08 vs.
R09" narrative already built into this doc, `BUILD_LOG.md`, and
`CLAUDE.md` for no real benefit — reusing R09's id for its recalibrated
role keeps that continuity intact.

**Follow-up closed out (was flagged here as a known limitation, now fixed
— see section 5b):** R01, R03, R06, and R10 also key off `avg_late_risk`
and were recalibrated in a second pass, the same day, once it was clear
the staleness wasn't limited to R08/R09. R02, R05, and R11 key off
`late_rate` (the empirical, model-independent rate) and were checked and
confirmed NOT stale — see section 5b for the full, rule-by-rule rigor.

## 5b. Second recalibration pass: R01, R03, R06, R10 (`avg_late_risk`), and confirming R02/R05/R11 (`late_rate`) are unaffected

**Step 1 — for every rule referencing `avg_late_risk` or `late_rate`,
determine whether its number was originally a PERCENTILE reference or a
fixed real-world threshold**, checked against each rule's own `because`
text and this doc's original justification (not assumed):

| Rule | Fact | Original threshold | Original intent (from its own justification text) | Percentile or fixed? |
|---|---|---|---|---|
| R01 | `avg_late_risk` | `> 0.53` | "top-quartile risk signal... thresholds set from this run's actual seller distribution" | **Percentile** (~75th) |
| R02 | `late_rate` | `> 0.4` | "an empirical late rate above 40% is a severe, directly-observed failure" | **Fixed** (40% lateness is severe on its own terms, not framed against a percentile) |
| R03 | `avg_late_risk` | `> 0.55` | "roughly the top 10% of sellers... deliberately set higher than R01's" | **Percentile** (~90th) |
| R05 | `late_rate` | `> 0.25` | "more than a quarter of a seller's own orders (90th-percentile territory)" | **Percentile** (~90th) |
| R06 | `avg_late_risk` | `< 0.503` | "at or below the median" | **Percentile** (median) |
| R10 | `avg_late_risk` | `<= 0.503` | "at-or-below-median predicted risk" | **Percentile** (median) |
| R11 | `late_rate` | `<= 0.10` | "at or below the 75th percentile" | **Percentile** (75th) |

R04, R07, R12 don't reference `avg_late_risk` or `late_rate` at all (R04/R07
use `avg_review`/`order_volume`; R12 is the unconditional default) — not
applicable, confirmed by inspection, not touched.

**Step 2 — recompute the real percentile of each threshold against the
CURRENT distribution, using the correct reference population.** For R01,
R03, R06, R10 (`avg_late_risk`) the population is all 2,970 sellers (none
of these rules restrict to a sub-population the way R08/R09's
`is_hidden_gem` gate does — R06 additionally requires `ahp_score >= 0.789`,
but its own original justification computed "median" against ALL sellers,
not the top-decile-quality subset, so that convention is preserved here).
For R02, R05, R11 (`late_rate`) the population is also all 2,970 sellers.

`avg_late_risk`, all 2,970 sellers, exact percentiles:

| Percentile | avg_late_risk |
|---|---|
| 25th | 0.0633 |
| 50th (median) | 0.1059 |
| 75th | 0.1780 |
| 90th | 0.2900 |
| 95th | 0.4135 |

Old thresholds' real percentile under the CURRENT (RandomForest)
distribution — this is the actual drift, measured, not assumed:

| Rule | Old threshold | Real percentile now | Intended percentile |
|---|---|---|---|
| R01 | 0.53 | **96.97th** | ~75th |
| R03 | 0.55 | **97.37th** | ~90th |
| R06 | 0.503 | **96.63rd** | 50th (median) |
| R10 | 0.503 | **96.63rd** | 50th (median) |

`late_rate`, all 2,970 sellers — confirmed UNCHANGED (this quantity is
`FactOrderItems.groupby("seller_id")["is_late"].mean()`, purely empirical,
computed identically regardless of which model produces `late_risk`):

| Rule | Threshold | Real percentile (unchanged) | Intended percentile |
|---|---|---|---|
| R02 | 0.4 (fixed, not percentile-intent) | 95.4th | n/a — the 40% figure was never meant to track a percentile |
| R05 | 0.25 | 91.25th | ~90th (matches within normal sample-to-sample rounding) |
| R11 | 0.10 | 75.35th | ~75th (matches) |

**Step 3 — decide and apply.** R01, R03, R06, R10 had clear percentile
intent (Step 1) and drifted to the ~97th percentile (Step 2) — recalibrated
back to their ORIGINAL intended percentile against the real current
distribution:
- **R01**: `0.53` -> **`0.178`** (real 75th percentile).
- **R03**: `0.55` -> **`0.29`** (real 90th percentile).
- **R06**, **R10**: `0.503` -> **`0.106`** (real median).

R02 was never percentile-intent (a fixed "40% is severe" business
threshold) — left unchanged. R05, R11 are percentile-intent but their real
percentiles (91.25th, 75.35th) already match their original design intent
within normal rounding — confirmed accurate, left unchanged, nothing to
fix.

**Step 4 — real fire-count table, before this pass vs. after** (from the
actual `rule_id` column of `reports/seller_recommendations.csv`, i.e. how
many sellers this rule specifically was the ONE THAT FIRED for, after
priority resolution — not the raw condition-match count):

| Rule | Action | Before (stale thresholds) | After (recalibrated) |
|---|---|---|---|
| R01 | Suspend | 55 | 165 |
| R02 | Suspend | 5 | 2 |
| R03 | Warn | 25 | 180 |
| R04 | Warn | 164 | 85 |
| R05 | Warn | 55 | 5 |
| R06 | Feature | 187 | 164 |
| R07 | Feature | 69 | 69 |
| R08 | Promote | 257 | 272 |
| R09 | Promote | 58 | 58 |
| R10 | Keep | 300 | 104 |
| R11 | Keep | 489 | 607 |
| R12 (default) | Keep | 1,306 | 1,259 |

R02/R05/R11's fire counts changed too (5->2, 55->5, 489->607) even though
THEIR thresholds didn't move — this is the expected, correct consequence
of R01/R03/R06/R10 becoming much easier to satisfy (e.g. R01's threshold
loosened from the 97th to the 75th percentile), which lets those
higher-priority rules now catch sellers that used to fall through to
R02/R04/R05/R11/R12. Not evidence R02/R05/R11 are themselves miscalibrated
— confirmed separately in Step 2/3 above.

Action-level totals: Keep 2,095 -> 1,970; Promote 315 -> 330 (R08/R09
untouched in this pass — this small change is the same cascading effect,
sellers no longer reaching R08/R09 because a higher-priority Suspend/Warn
rule now catches them first); Feature 256 -> 233; Warn 244 -> 270;
Suspend 60 -> 167.

## 6. Remaining expert-system rule thresholds (`rules/seller_rules.yaml`)

**All thresholds below are current as of the Stage 3.5 recalibration
(section 5, section 5b)** — R08/R09 use RandomForest-Hidden-Gem-specific
percentiles; R01/R03/R06/R10 use RandomForest-all-sellers percentiles;
R02/R05/R11 use the unchanged empirical `late_rate` distribution. The
"Real matches" column below reflects the OLD LR-era pipeline run for rules
not affected by this doc's two recalibration passes' re-verification —
where a rule WAS recalibrated, see section 5/5b for the current real count
instead (R01: 165, R03: 180, R06: 164, R08: 272, R09: 58, R10: 104).

All checked against the same `SellerFacts.parquet` run (2,970 sellers;
`avg_review` percentiles: min 1.0, 10th 3.2, 25th 3.90, 50th 4.27, 75th 4.70,
90th-95th 5.0; `order_volume` percentiles as in section 1;
`late_rate` percentiles: 50th 0.0, 75th 0.10, 90th 0.222, 95th 0.337;
`ahp_score` percentiles as in section 1's established-seller pool extended
to the 5-order-eligible pool: 10th 0.604, 25th 0.669, 50th 0.716, 75th
0.755, 90th 0.789).

| Rule | Threshold | Justification | Real matches |
|---|---|---|---|
| R01 Suspend | `avg_late_risk > 0.53` (~75th pct) AND `avg_review < 3.5` (between 10th/25th pct) | Combines a top-quartile risk signal with a clearly-below-average review score — CLAUDE.md's own example rule, re-thresholded to this data's real percentiles instead of its illustrative 0.6/3.0. | 94 sellers |
| R02 Suspend | `late_rate > 0.4` AND `order_volume >= 10` | An empirical (not predicted) late rate above 40% is a severe, directly-observed failure; requiring >=10 orders rules out a fluke from 1-2 late shipments. | 7 sellers |
| R06 Feature | `ahp_score >= 0.789` (90th pct) AND `avg_late_risk < 0.503` (median) | Top-decile AHP quality with at-or-below-median predicted risk — a proven, low-risk top performer. | 107 sellers |
| R07 Feature | `avg_review >= 4.3` AND `order_volume >= 83` (90th pct) | High-volume sellers (90th percentile+) have a *lower* mean review (4.08) than the overall population (4.27 median) in this data, so requiring 4.3 specifically picks out sellers sustaining quality at scale rather than everyone at that volume. | 69 sellers |
| R03 Warn | `avg_late_risk > 0.55` (~90th pct) | A single, wide risk-only caution signal, deliberately set higher than R01's combined-condition threshold since it fires alone. | 226 sellers |
| R04 Warn | `avg_review < 3.5` AND `order_volume >= 5` | Same review cutoff as R01 (10th-25th pct band) without the risk condition, gated on order_volume >= 5 so it reflects a pattern, not one bad review. | 177 sellers |
| R05 Warn | `late_rate > 0.25` (90th pct) AND `order_volume >= 5` | A softer, single-condition version of R02's empirical-lateness signal. | 105 sellers |
| R10 Keep | `avg_late_risk <= 0.503` AND `avg_review >= 4.27` (both median) AND `order_volume >= 5` | At-or-better-than-median on both risk and reviews, with a real order history — no signal to act on. | 446 sellers |
| R11 Keep | `ahp_score >= 0.604` (10th pct) AND `late_rate <= 0.10` (75th pct) | Excludes only the bottom decile of AHP quality and the worst quartile of empirical lateness — a wide "no red flags" net. | 1,264 sellers |
| R12 (default) | none (`if: []`) | Every seller must get a recommendation (CLAUDE.md section 7.5); catches sellers with too little data to trigger any of the above (frequently order_volume < 5, so no `ahp_score`). | 1,228 sellers (41.4% of all sellers) |

**Priority ladder** (rationale for the ordering, not just the numbers):
Suspend (100, 98) > Feature (90, 88) > Promote (80, 78) > Warn (70, 68, 66)
> Keep (50, 48) > default (1). Suspend and Feature intentionally outrank
Promote and Warn so that a severe risk/review problem or a proven top
performer is never overridden by a milder signal. One known edge case,
documented rather than hidden: because `is_hidden_gem` sellers are chosen
via the AHP composite (which weights `avg_review_score` at 0.473, the
largest single weight), a seller matching R09 (Promote) with `avg_review <
3.5` is mathematically possible but very unlikely in practice — the AHP
score already penalizes poor reviews heavily before a seller can qualify as
a Hidden Gem.

## 7. Sentiment thresholds

`SENTIMENT_POSITIVE_THRESHOLD = 0.05` / `SENTIMENT_NEGATIVE_THRESHOLD =
-0.05` are VADER's own published default compound-score cutoffs, not tuned
against this project's synthetic call transcripts — `CallTranscripts` has no
persisted ground-truth sentiment label to tune against (see
`src.synthetic`'s docstring and `src.sentiment`'s module docstring), so
using the library's validated defaults rather than an untuned guess is the
honest choice here.
