# Knowledge Engineering: Threshold Justifications

Every threshold below was checked against the real output of `python run_all.py`
on this project's actual `data/processed/` tables (2,970 sellers total, 110,189
fact rows), not invented. Percentiles are computed from `SellerFacts.parquet`
(written by `src.discovery.build_seller_facts`) and `FactOrderItems.parquet`
unless stated otherwise. If the pipeline is rerun on a different data
snapshot, these numbers should be recomputed and this file updated —
CLAUDE.md section 1 forbids quoting a number that wasn't actually measured.

**How to read this file.** Sections 5 and 5b, 8, 8b and 9 are the history of each recalibration
pass and quote the values of their day (for example R01 at 0.178 or 0.416); they are labelled
as such and are not current. The current rule thresholds, with their fire counts, are in
section 6 and `reports/RESULTS.md`; the latest change (a minimum of 5 items sold for Warn and
Suspend) is section 10. The rule file `rules/seller_rules.yaml` is the only authority for a
threshold's value.

## 1. AHP established-sellers threshold: `AHP_MIN_ORDERS = 30`

Per-seller order counts across all 2,970 sellers:

| Percentile | order_volume |
|---|---|
| 25th | 2 |
| 50th (median) | 8 |
| 75th | 26 |
| 90th | 83 |
| 95th | 151 |

30 items sold sits at the **77.9th percentile** of all sellers — i.e. it selects
the top ~22.9% (681 of 2,970) as "established" for the main AHP ranking.
This was chosen to keep the established-seller ranking restricted to sellers
with enough of a track record that an average-review/on-time-rate estimate
is meaningful, while still leaving a majority of the marketplace (the other
77%) for the lower-threshold discovery pass. 30 items sold is also comfortably
above the point (order_volume >= ~10-15) where an empirical rate like
`on_time_rate` stops being dominated by single-order noise.

Resulting AHP run: 681 established sellers, weights `avg_review_score=0.473,
on_time_rate=0.284, avg_price=0.073, order_volume=0.170` (from the pairwise
matrix in `config.AHP_PAIRWISE_MATRIX`), consistency ratio **CR = 0.0189**,
well under the `AHP_CR_THRESHOLD = 0.10` flag.

## 2. Discovery AHP-eligibility threshold: `DISCOVERY_MIN_ORDERS = 5`

5 items sold sits at the **41.9th percentile** — it makes 1,864 of 2,970 sellers
(62.8%) eligible for a quality (AHP) score, versus 22.9% at the 30-item
established-seller threshold. This is the whole point of the discovery
feature (CLAUDE.md section 7.7.1): a seller needs *some* items sold for an
average review score and on-time rate to mean anything at all, but 5 is low
enough that small/newer sellers — the ones a "sort by volume" view would
bury — are actually included, rather than only re-running the same
established-seller list at a different name.

## 3. Discovery confidence cutoff: `DISCOVERY_CONFIDENCE_MIN_ORDERS = 15`

15 items sold sits at the **66.1st percentile** of all sellers. Among the 1,864
AHP-eligible (>=5 items sold) sellers specifically, the median items-sold count is
**18** — meaning the 15-item confidence cutoff sits just below the middle
of the eligible population, splitting it into **818 "low" confidence rows
(43.9%) and 1,046 "normal" (56.1%)** on the last real run. This threshold
was picked so "low confidence" genuinely means "below-typical items sold
for an eligible seller" (not an arbitrary round number): a seller just past
the 5-item eligibility floor is treated with reduced confidence until they
clear roughly the eligible population's own midpoint.

## 4. Discovery quadrant thresholds: median split (both axes)

`config.DISCOVERY_QUALITY_THRESHOLD` / `DISCOVERY_POPULARITY_THRESHOLD` are
both `None` by default, meaning `compute_quadrants` splits each axis at its
own median among AHP-eligible sellers — the simplest split that guarantees
roughly balanced quadrants without hand-picking a magic number for either
axis. At Stage 4 this produced Star 454, Hidden Gem 478, Overrated 480,
Overlooked-Low-Quality 452. **Current numbers (Stage 5, after the order-level empirical-Bayes shrinkage of
section 8): Star 495, Hidden Gem 437, Overrated 439, Overlooked-Low-Quality 493** — a near-even four-way split,
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
wider-ranging signal, but it flipped which rule was
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

## 6. Rule thresholds (`rules/seller_rules.yaml`)

The rule file is the single source of every threshold; this table is generated from it and
from the pipeline outputs, and `tests/test_results.py` fails if a condition here differs from the
file. Fire counts come from `reports/seller_recommendations.csv` ("fires": the rule that gave the
seller's final recommendation) and from running every rule against `SellerFacts.parquet`
("matches": all conditions true, whatever the priority). The same counts are in `reports/RESULTS.md`.

Each percentile-intent threshold was set at the percentile of the **reference population** named
in the third column, not always of all sellers. R01 and R03 use the 1,864 sellers with at least 5
items sold because that is the group they apply to since the Stage 8 minimum-evidence rule
(section 10). R06 and R10 keep the median over all 2,970 sellers (R10 is gated at 5 items sold
but its median was set before the gate and was deliberately left).

Distributions the thresholds were checked against (2,970 sellers unless stated): `avg_review`
(2,965 sellers with a reviewed order; one review per distinct order) minimum 1.0, 10th 3.33, 25th
3.97, median 4.28, 75th 4.67, 90th 5.0; `late_rate` median 0, 75th 0.10, 90th 0.222, 95th 0.337;
`avg_late_risk` over all sellers 25th 0.277, median 0.339, 75th 0.416, 90th 0.501 and over the
1,864 sellers with at least 5 items sold 75th 0.412, 90th 0.476; `ahp_score` (1,864 sellers with
a score, after the Stage 5 order-level shrinkage, section 8) 10th 0.549, 25th 0.601, median 0.644,
75th 0.681, 90th 0.711; `order_volume` (items sold) median 8, 75th 26, 90th 83.

| Rule | Conditions | Reference population | Matches | Fires | Why this level |
|---|---|---|---|---|---|
| R01 Suspend | `avg_late_risk > 0.412 and avg_review < 3.5 and order_volume >= 5` | sellers with >= 5 items sold | 70 | 70 | Top quartile of predicted risk among sellers with at least 5 items sold, together with a clearly below-average review (between the 10th and 25th percentile). Risk alone is not enough to suspend. |
| R02 Suspend | `late_rate > 0.4 and order_volume >= 10` | sellers with >= 10 items sold | 7 | 6 | An observed (not predicted) late rate above 40% is a severe failure; 10 or more items sold rules out a fluke from one or two late shipments. A fixed severity level, not a percentile. |
| R06 Feature | `ahp_score >= 0.711 and avg_late_risk < 0.339` | all sellers | 137 | 137 | Top decile of AHP quality with at-or-below-median predicted risk: a proven, low-risk top performer. The median is taken over all sellers. |
| R07 Feature | `avg_review >= 4.3 and order_volume >= 83` | sellers with >= 83 items sold | 78 | 37 | High volume (90th percentile of items sold) sustained alongside a high average review; 4.3 is a little above the median seller. |
| R09 Promote | `is_hidden_gem == True and avg_late_risk < 0.252` | hidden gems | 111 | 104 | Hidden gem in roughly the lowest-risk quartile of hidden gems: the high-confidence Promote tier. |
| R08 Promote | `is_hidden_gem == True and avg_late_risk < 0.3` | hidden gems | 222 | 106 | Hidden gem with predicted risk below the original literal cut-off of 0.3: the broader, lower-confidence Promote tier. R09 is a strict subset of it. |
| R03 Warn | `avg_late_risk > 0.476 and order_volume >= 5` | sellers with >= 5 items sold | 184 | 144 | Top 10% of predicted risk among sellers with at least 5 items sold: on its own a caution signal, set above R01 because it fires without a review condition. |
| R04 Warn | `avg_review < 3.5 and order_volume >= 5` | sellers with >= 5 items sold | 137 | 66 | Average review below 3.5 with at least 5 items sold: a pattern, not one bad review. |
| R05 Warn | `late_rate > 0.25 and order_volume >= 5` | sellers with >= 5 items sold | 105 | 31 | More than a quarter of a seller's own items arrived late (about the 90th percentile), with at least 5 items sold. |
| R10 Keep | `avg_late_risk <= 0.339 and avg_review >= 4.28 and order_volume >= 5` | sellers with >= 5 items sold | 485 | 170 | At-or-better-than-median predicted risk (all sellers) and reviews (median 4.28) with a real history of 5 or more items: nothing to act on. |
| R11 Keep | `ahp_score >= 0.549 and late_rate <= 0.1` | all sellers | 1,288 | 701 | AHP quality above the bottom decile and an observed late rate at or below the 75th percentile: a wide no-red-flags net. |
| R12 Keep | `(always)` | all sellers | 2,970 | 1,398 | Default, so every seller gets a recommendation. Catches every seller with fewer than 5 items sold, plus sellers no stronger rule matched. |

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

## 8. Empirical-Bayes shrinkage of the AHP quality score (Stage 5, order level)

**Why.** Inside the Hidden Gem quadrant, items sold and AHP score correlated at -0.19
(Spearman): the more a seller sold, the *lower* it scored, because a handful of perfect
reviews or on-time deliveries gives a small seller an extreme raw average. The two rate-like
criteria (average review, on-time rate) are therefore pulled toward the marketplace mean
before scoring: `shrunk = (n * seller_value + k * marketplace_mean) / (n + k)`, i.e. weight
`n / (n + k)` on the seller's own record. Price and items sold are unchanged;
`compute_weights` and the pairwise matrix are untouched.

**The unit of evidence is the order, not the item.** A review belongs to an order and so does
lateness, so every item of an order repeats the same review and the same late flag.
Verified on the real data: the 110,189 items form 97,811 (seller, order) pairs (96,470
distinct orders; 1,275 orders span two or more sellers and count once for each); in none of
the 97,811 pairs do the items disagree on review or lateness; 8.9% of pairs contain more
than one item. Counting items therefore overstates how much independent evidence a seller
has. The first version of this stage (item level, k = 11 and 14) did exactly that, and its
`n_reviewed` counted reviewed *items*; it is superseded. Now:
- `n_reviewed` = distinct reviewed orders per seller (used as n for the review);
  `n_orders` = distinct delivered orders per seller (used as n for the on-time rate);
- `avg_review_raw` = mean of one review per order, `on_time_rate_raw` = 1 minus the share of
  late orders (raw, unshrunk, kept as their own columns in `ahp_ranking.csv`);
- eligibility is unchanged and still defined by **items sold**: >= 5 for the Discovery pool,
  >= 30 for Choice, so the pools and the quadrant rules keep their definition.

`SellerFacts.avg_review` (used by R01/R04/R07/R10 and the Implementation page) was at first
still the item-level raw mean (it differed from the order-level `avg_review_raw` by 0.058 on
average, 749 sellers by more than 0.05). That inconsistency is closed in Stage 5b (section 8b
below): the expert fact is now the same per-order mean Choice shows.

**How k is chosen: method-of-moments empirical Bayes, from the data.** For a criterion with
within-seller variance `s2` (spread of one order's value around its seller's own true value)
and between-seller variance `t2` (spread of the sellers' TRUE values), `k = s2 / t2`. The
observed variance of seller averages is `t2` plus sampling noise, so
`t2 = observed variance - mean(s2 / n)`. For reviews `s2` is the pooled order-level variance
of review scores (weighted by n - 1); for on-time rate it is the binomial `p(1 - p)` at the
marketplace on-time rate. Pool: the 1,864 sellers with at least `DISCOVERY_MIN_ORDERS = 5`
items sold. Real values (`reports/ahp_prior_strength.csv`, written by `python -m src.ahp`):

| Criterion | Sellers | Within variance | Observed variance | Sampling noise | Between variance | k estimate | k used |
|---|---|---|---|---|---|---|---|
| Average review (one review per order) | 1,860 | 1.6332 | 0.23422 | 0.14256 | 0.09165 | 17.82 | **18** |
| On-time rate (orders; p = 0.9198) | 1,864 | 0.07376 | 0.009380 | 0.006534 | 0.002847 | 25.91 | **26** |

(4 pool sellers have fewer than two reviewed orders and cannot inform the review variance.)
Rule: use the rounded estimate if it lies inside the 5 to 30 range tested in
`scripts/diagnostics/diagnostic_shrinkage.py`, otherwise `k = 10`. Both estimates are inside the range,
so `config.AHP_SHRINKAGE_K_REVIEW = 18` and `AHP_SHRINKAGE_K_ON_TIME = 26`. A test fails if
these constants stop matching the order-level estimate from the data.

*A note on the expected "about 22 and 28".* Those figures came from an earlier sensitivity
that restricted the pool to sellers with at least five distinct *orders* (1,766 sellers:
21.6 and 27.8). The stated pool here is items sold >= 5, which includes sellers with 5 items
in fewer orders; on that pool the estimates are 17.8 and 25.9.

**Sensitivities (one k per criterion is used for both Choice and Discovery so the two pages
share one definition of quality).**

| Pool | Review k | On-time k |
|---|---|---|
| Items sold >= 5 (adopted, order level) | 17.82 | 25.91 |
| Items sold >= 30, the Choice pool (681 sellers), order level | 25.65 | 41.82 |
| All sellers with data, order level | 22.32 | 13.65 |
| Items sold >= 5, item level (the superseded first version) | 10.63 | 14.42 |
| Items sold >= 30, item level (superseded) | 19.85 | 32.34 |

The Choice-pool on-time k (41.8) would be outside the 5 to 30 range and so, by the rule,
fall back to 10; it is reported as a sensitivity only. Larger sellers are more consistent
than small ones, so any single k is a compromise between the pools.

**Effect (real runs): committed state (no shrinkage), item-level shrinkage, order-level
shrinkage.**

| Quantity | No shrinkage | Item level | Order level |
|---|---|---|---|
| Star | 454 | 482 | 495 |
| Hidden Gem | 478 | 450 | 437 |
| Overrated | 480 | 452 | 439 |
| Overlooked-Low-Quality | 452 | 480 | 493 |
| Gems that stay / leave / enter, vs no shrinkage | | 450 / 28 / 0 | 420 / 58 / 17 |
| Gems normal / low confidence | 44 / 434 | 43 / 407 | 44 / 393 |
| Normal-confidence gems among the top 10 gems | 1 | 4 | 4 |
| Top 10 gems with a raw 5.0 average | 10 | 6 | 1 |
| Gems with a raw 5.0 average (all) | 54 | 53 | 50 |
| Spearman, items sold vs score inside the gem quadrant | -0.19 | +0.17 | +0.23 |
| Choice top 10: overlap with no-shrinkage ranking | | 7 of 10 | 5 of 10 |
| Keep / Promote / Feature / Warn / Suspend | 1,970 / 330 / 233 / 270 / 167 | 1,946 / 397 / 190 / 270 / 167 | 1,949 / 412 / 173 / 269 / 167 |

Choice top 10, order level (previous rank without shrinkage): 1, 2, 15, 8, 36, 26, 4, 10, 16, 68
(item level: 1, 2, 4, 3, 10, 8, 5, 15, 12, 16). The 4-in-10 normal-confidence result is
unchanged from the item-level version, but the perfect-average artefact is nearly gone from
the top of the gem list (1 of 10 instead of 6). Low/normal confidence for the whole pool is
unchanged (818 / 1,046) because it depends only on raw items sold.

**A side effect to know about.** Shrinkage compresses scores toward the middle, so the median
quality line that defines the quadrants fell from 0.716 to 0.644. All 17 sellers that
entered the Hidden Gem quadrant came from Overlooked-Low-Quality. (An earlier version of this
paragraph said they average 3.81 stars; that figure used the per-ITEM mean, which a single bad
review on a multi-item order drags down. Measured per reviewed order, as in section 8b, they
average 4.24, with 5 of 17 below the marketplace mean of 4.14, and 2.2% late deliveries.) The
58 that left averaged 4.43 per order and 9.9% late.

`ahp_score` percentiles over the 1,864 scored sellers (`SellerFacts`):

| Percentile | No shrinkage | Item level | Order level |
|---|---|---|---|
| 10th | 0.6036 | 0.5457 | 0.5491 |
| 25th | 0.6692 | 0.6088 | 0.6006 |
| 50th | 0.7157 | 0.6608 | 0.6443 |
| 75th | 0.7554 | 0.7019 | 0.6814 |
| 90th | 0.7892 | 0.7308 | 0.7110 |

**Rules reading `ahp_score` (the only rules recalibrated in this stage).** Re-derived at the
same percentile each time, intent taken from each rule's own justification (section 6):

| Rule | Condition on `ahp_score` | Intent | Original | Item level | Order level | Real percentile of order-level threshold |
|---|---|---|---|---|---|---|
| R06 Feature | `>=` | **Percentile** (90th, top decile) | 0.789 | 0.7308 | **0.711** | 90.0th (187 sellers at or above, 10.03%) |
| R11 Keep | `>=` | **Percentile** (10th, "above the bottom decile") | 0.604 | 0.546 | **0.549** | 10.0th (186 sellers below, 9.98%) |

Population: all sellers that have an `ahp_score` (1,864), the same convention as the original
thresholds. Under the order-level distribution the previous cutoffs would have been the
94.5th (0.7308) and 9.4th (0.546) percentiles, and the original ones the 99.9th (0.789) and
26.2nd (0.604). No other rule reads `ahp_score`. R02 is a fixed level (40% late) and was not
touched. The late-risk thresholds (R01, R03, R06, R10 on `avg_late_risk`; R08, R09) were
deliberately left as they were and will be recalibrated separately.

**R08/R09 against the new gem population (437 gems), late-risk thresholds unchanged.** R09
(`avg_late_risk < 0.05`) matches 114 of 437 (26.1%); R08 (`< 0.3`) matches 433 of 437
(99.1%). Before shrinkage: 119 of 478 (24.9%) and 472 of 478 (98.7%). Both are still strictly
between 0% and 100%, so the drift test passes, and R09 still sits at about the 25th
percentile of gem late risk (0.0492).

**Recommendations fired per rule (all 2,970 sellers; the Stage 5 state, before the Stage 5b
per-order `avg_review` fact, whose counts are in section 8b).**

| Rule | No shrinkage | Item level | Order level |
|---|---|---|---|
| R01 Suspend | 165 | 165 | 165 |
| R02 Suspend | 2 | 2 | 2 |
| R03 Warn | 180 | 180 | 181 |
| R04 Warn | 85 | 85 | 83 |
| R05 Warn | 5 | 5 | 5 |
| R06 Feature | 164 | 143 | 130 |
| R07 Feature | 69 | 47 | 43 |
| R08 Promote | 272 | 307 | 306 |
| R09 Promote | 58 | 90 | 106 |
| R10 Keep | 104 | 62 | 63 |
| R11 Keep | 607 | 614 | 601 |
| R12 Keep (default) | 1,259 | 1,270 | 1,285 |

Transitions, no shrinkage to order level (256 sellers changed action): Feature to Promote 120,
Keep to Feature 62, Promote to Keep 51, Keep to Promote 14, Feature to Keep 4, Promote to
Feature 2, Warn to Promote 2, Promote to Warn 1; Suspend is unchanged and Warn changes only
for those 3. As before, Feature to Promote is the main flow (tiny gems with perfect raw
averages used to reach the old R06 cutoff and were featured ahead of Promote), Keep to
Feature is shrinkage ranking steady high-volume sellers higher, and Promote to Keep is
mostly sellers that left the gem quadrant (58 left).

## 8b. One definition of "average review" (Stage 5b)

**Change.** `SellerFacts.avg_review`, the fact behind R01, R04, R07, R10 and the Implementation
page, is now the mean of one review per distinct order, computed for ALL 2,970 sellers from the
fact table (`build_seller_facts` uses `seller_order_table`; sellers with no reviewed order keep
NaN, as before: 5 sellers). It is the same number Choice shows as `avg_review_raw`, so Choice,
Discovery, Implementation and the rules share one definition. Items sold, late rate and
predicted late risk are unchanged. Quadrants, `ahp_ranking.csv` and the gem set are byte-identical
before and after (437 gems); only `SellerFacts`, the recommendations and their exports change.
The per-order mean differs from the per-item mean by 0.058 on average (749 sellers by more than
0.05), because one unhappy review on a multi-item order used to count once per item.

**Distribution of the fact** (2,965 sellers with a review): 10th / 25th / 50th / 75th / 90th
percentile were 3.20 / 3.90 / 4.27 / 4.70 / 5.00 per item and are 3.33 / 3.97 / 4.28 / 4.67 /
5.00 per order.

**Rules reading `avg_review`.** Intent is taken from each rule's own `because` text and
section 6; only percentile-intent rules are recalibrated.

| Rule | Condition | Intent | Percentile of the cutoff before / after | Decision |
|---|---|---|---|---|
| R01 Suspend | `avg_review < 3.5` | **Fixed level** (3.5 is the stated cutoff, CLAUDE.md's example rule re-thresholded; documented only as inside the 10th to 25th percentile band) | 12.85th / 11.60th | Left. Still inside the documented band |
| R04 Warn | `avg_review < 3.5` | **Fixed level**, "same review cutoff as R01" | 12.85th / 11.60th | Left |
| R07 Feature | `avg_review >= 4.3` | **Fixed level** in practice: the justification (section 6) picks 4.3 just above the overall median to isolate sellers sustaining quality at scale; its `because` text says "top quartile", but 4.3 was never near the 75th percentile (about the 51st). That mismatch pre-dates this change | 51.5th / 50.7th | Left (the fact change moved it by under 1 point); the text/number mismatch is flagged, not silently changed |
| R10 Keep | `avg_review >= 4.27` | **Percentile** (median: "at-or-above-median reviews") | 49.95th / 48.97th | Recalibrated to the new median: **4.28** (49.85th percentile below the cutoff; the exact median is 4.28125) |

`avg_late_risk` and `ahp_score` thresholds were not touched.

**Recommendations fired per rule (2,970 sellers).**

| Rule | Before the fact change | After the fact change (old thresholds) | After R10 recalibration |
|---|---|---|---|
| R01 Suspend | 165 | 152 | 152 |
| R02 Suspend | 2 | 5 | 5 |
| R03 Warn | 181 | 188 | 188 |
| R04 Warn | 83 | 58 | 58 |
| R05 Warn | 5 | 5 | 5 |
| R06 Feature | 130 | 130 | 130 |
| R07 Feature | 43 | 47 | 47 |
| R08 Promote | 306 | 306 | 306 |
| R09 Promote | 106 | 106 | 106 |
| R10 Keep | 63 | 67 | 63 |
| R11 Keep | 601 | 617 | 621 |
| R12 Keep (default) | 1,285 | 1,289 | 1,289 |

By action (Keep / Promote / Feature / Warn / Suspend): 1,949 / 412 / 173 / 269 / 167 before and
1,973 / 412 / 177 / 251 / 157 after; the R10 recalibration changes which Keep rule fires, not
any action. 77 sellers changed action (Warn to Keep 32, Suspend to Keep 9, Suspend to Warn 9,
Keep to Feature 8, Keep to Warn 7, Keep to Suspend 6, Feature to Keep 4, Warn to Suspend 2):
sellers whose bad record came from one multi-item order stop being penalised several times
over. R02 rises from 2 to 5 only because three sellers no longer fire the higher-priority R01
and fall through to R02.

**Do hidden gems really have good reviews? (read-only diagnostic; the gem definition is
unchanged).** Marketplace mean review per order: 4.14. Per-order average review and item-level
late rate:

| Group | Sellers | Mean / median review | Below marketplace mean | Late rate mean / median / 90th pct |
|---|---|---|---|---|
| All gems | 437 | 4.59 / 4.60 | 12 (2.7%) | 1.4% / 0.0% / 7.1% |
| Stayed gems | 420 | 4.61 / 4.62 | 7 (1.7%) | 1.4% / 0.0% / 7.1% |
| Entered the gem group (17) | 17 | 4.24 / 4.25 | 5 (29.4%) | 2.2% / 0.0% / 2.5% |
| Left the gem group (58) | 58 | 4.43 / 4.50 | 10 (17.2%) | 9.9% / 9.5% / 20.0% |
| All 1,864 AHP-eligible sellers | 1,864 | 4.17 | 753 (40.4%) | 8.1% mean |

Only 2.7% of gems (below the 5% bar) have a review below the marketplace mean, and none is
below 4.0 (lowest 4.07); the sellers that left the group were mainly those with poor delivery
(9.9% late against 1.4%). So no extra caveat was added to the Discovery page or the demo
script. The 17 entrants are the one place worth knowing: 5 of them sit just below the
marketplace mean on reviews and qualify on a near-perfect delivery record.


## 9. Order-grouped evaluation and out-of-fold late risk (Stage 6)

Sections 5 and 5b above describe the late-risk thresholds as they were calibrated against the
in-sample RandomForest scores; the thresholds and numbers in this section supersede them.

**Two problems, both measured before anything was changed**
(`scripts/diagnostics/diagnostic_group_leak.py`, `reports/diagnostics/diagnostic_group_leak.txt`).

1. *Sibling leakage.* Items of one order share `is_late` and most features (110,189 items =
   97,811 seller-order pairs = 96,470 orders), but the old split put individual item rows at
   random into train and test: 18.34% of the test items (16.75% of the test orders) had an
   item of the same order in train. Test ROC-AUC, random item split vs a split grouped by
   `order_id` (same seed, same sizes: 88,151 / 22,038 items, base rate 7.908% / 7.909% on both):

   | Model | Random item split | Grouped by order | Change |
   |---|---|---|---|
   | Logistic regression | 0.6347 | 0.6259 | -0.0088 |
   | Decision tree | 0.6738 | 0.6888 | +0.0150 |
   | Random forest (fully grown) | 0.7841 | 0.7274 | **-0.0567** |

   The forest was the model that profited from siblings; its lead over the decision tree fell
   from 0.110 to 0.039. The decision tree went up by 0.015 (noise at this size, or a lucky
   partition; the folds below show a spread of about 0.004).
2. *In-sample late risk.* The committed `late_risk` was the forest's prediction for every row,
   including the rows it trained on: AUC 1.0000 on its own training rows, 0.7841 on the rows it
   had not seen, 0.9724 over all rows. At seller level `avg_late_risk` correlated with the
   seller's actual late rate at Pearson 0.906 (Spearman 0.775) over all 2,970 sellers, 0.925
   (0.885) over the 1,864 with at least 5 items sold: largely the seller's own late rate echoed
   back, then fed to the Suspend, Warn, Feature, Promote and Keep rules.

**Fix.** `src/predict.py` now builds 5 `StratifiedGroupKFold` folds over `order_id` (shuffle,
seed 42) in one place, `grouped_stratified_folds`. Fold 0 is the single train/test definition
(metrics, permutation importance, sensitivity, persisted models); all five folds give
out-of-fold predictions (`out_of_fold_predictions`, one fresh model per fold and model, the
product-category encoding refitted on each fold's training rows, nothing persisted) that fill
`LatePredictions.parquet`. Features, seed and model configs are unchanged, except the forest's
leaf size (below). `reports/oof_metrics.json` carries the comparison:

| Model | Grouped test (fold 0) | Out-of-fold, 5 folds | Fold mean / std | AUC on its own training rows |
|---|---|---|---|---|
| Logistic regression | 0.6259 | 0.6241 | 0.6244 / 0.0070 | 0.6274 |
| Decision tree | 0.6888 | 0.6856 | 0.6856 / 0.0040 | 0.6937 |
| Random forest (leaf 20) | 0.7410 | 0.7367 | 0.7367 / 0.0041 | 0.9346 |

The previous in-sample column for the forest was 1.0000; the old random-fold cross-validation
that justified adopting the forest (0.7928 / 0.6870 / 0.6247 in `reports/diagnostics/diagnostic_rf_cv.txt`)
used item-level random folds and is superseded by this table. The forest still ranks first on
unseen orders (0.7367 vs 0.6856 vs 0.6241 out-of-fold), so no model was swapped, but its edge
over the decision tree is about 0.05, not 0.11.

**Forest leaf size** (`scripts/diagnostics/diagnostic_rf_leaf.py`, same grouped OOF protocol, committed
config except `min_samples_leaf`):

| `min_samples_leaf` | Pooled OOF AUC | Fold mean / std | vs leaf 1 | Fitted size |
|---|---|---|---|---|
| 1 (fully grown, before) | 0.7233 | 0.7233 / 0.0047 | | 421.2 MB |
| 5 | 0.7311 | 0.7310 / 0.0037 | +0.0077 | 193.3 MB (46%) |
| 20 | 0.7367 | 0.7367 / 0.0041 | +0.0133 | 67.7 MB (16%) |

The rule written beforehand was "adopt a larger leaf only if AUC is within 0.005 of the
baseline and the size is materially smaller". Larger leaves were *better* by more than 0.005,
not merely within it; I read the rule as a non-inferiority test (not worse by more than 0.005)
and adopted **20** (`config.PREDICT_RF_MIN_SAMPLES_LEAF`): highest AUC, smallest model. 20 was
the end of the grid, so a still larger leaf might do better; none was tried. The leaf was chosen
on the same folds it is reported on, so the +0.013 is slightly optimistic.

**Forward-looking check** (`scripts/diagnostics/diagnostic_temporal.py`): train on the earliest 80% of
orders (88,294 items, 2016-09-15 to 2018-05-26, late rate 8.58%), test on the latest 20%
(21,895 items, 2018-05-26 to 2018-08-29, late rate 5.20%). Forest at the adopted leaf size of 20.
The figures first written here (0.5210 and 0.4402) came from a fully grown forest; the script now
reproduces the numbers below and `reports/diagnostics/diagnostic_temporal.txt` holds them:

| Features | Logistic regression | Decision tree | Random forest | RF - LR | RF - DT |
|---|---|---|---|---|---|
| All 12 | 0.3975 | 0.5049 | 0.5178 | +0.1203 | +0.0129 |
| Without `order_month`, `day_of_week` | 0.4007 | 0.4188 | 0.4198 | +0.0191 | +0.0010 |

Plainly: **none of the models has useful skill on future months.** Every AUC is near or below
0.5 (below 0.5 means the ranking is, on average, inverted), the forest is nominally ahead of
the linear model, and level with the tree once the calendar inputs are removed (+0.001), but
that is the smallest of three weak results and not evidence of a usable forecast. The calendar inputs (`order_month` is the strongest feature on
unseen orders from the same period) do not carry forward. A caveat on the test period: its late
rate is lower (5.2%), possibly partly because orders delayed beyond the data extract never
appear as delivered; that was not verified here.

**Recalibrated rules (late-risk only).** After the fix the scores sit on a different scale
(median seller 0.34, not 0.11: with balanced class weights the forest's output is a ranking
score, not a probability) and are far less dispersed at the low end, so the old cutoffs landed
at the wrong percentiles. `avg_late_risk`, percentiles of the 2,970 sellers (all) and the 437
Hidden Gems:

| Percentile | All sellers, in-sample (before) | All sellers, out-of-fold (now) | Gems 25th / 50th / 75th / 90th before | Gems now |
|---|---|---|---|---|
| 10th | 0.0400 | 0.2250 | | |
| 25th | 0.0633 | 0.2767 | 0.0492 | 0.2519 |
| 50th | 0.1059 | 0.3391 | 0.0717 | 0.2996 |
| 75th | 0.1780 | 0.4164 | 0.0994 | 0.3534 |
| 90th | 0.2900 | 0.5013 | 0.1433 | 0.4079 |

Seller-level correlation with the actual late rate drops to Pearson 0.339 (Spearman 0.428) over
all sellers and 0.517 (0.583) for sellers with at least 5 items sold.

| Rule | Intent (from its own text and sections 5/5b) | Old cutoff, its percentile now | New cutoff | Matches now |
|---|---|---|---|---|
| R01 Suspend (`>`) | **Percentile**: 75th, top quartile, all sellers | 0.178, 2.5th (fires on 97.5%) | **0.416** | 746 of 2,970 (25.1%) |
| R03 Warn (`>`) | **Percentile**: 90th, all sellers | 0.29, 30.8th | **0.501** | 298 of 2,970 (10.0%) |
| R06 Feature (`<`) | **Percentile**: median, all sellers | 0.106, 0.07th | **0.339** | 1,484 of 2,970 (50.0%) |
| R10 Keep (`<=`) | **Percentile**: median, all sellers | 0.106, 0.07th | **0.339** | 1,484 of 2,970 (50.0%) |
| R09 Promote (`<`) | **Percentile**: about the 25th, Hidden Gems only | 0.05, 0.0th (fires on 0 gems) | **0.252** | 111 of 437 (25.4%) |
| R08 Promote (`<`) | **Fixed level**: CLAUDE.md's literal 0.3, gems only | 0.3, 50.8th among gems | 0.3 (unchanged) | 222 of 437 (50.8%) |
| R02, R05, R11 | Actual late rate (model-independent) | | untouched | |

R08 was kept because it fires on 50.8% of gems, neither 0% nor 100%, and 0.252 < 0.3 keeps R09
a strict subset (the high-confidence tier) of R08. R07 only had its `because` text corrected:
it said "top quartile" while 4.3 is about the median review; it now describes what it does
(a high review, at least 4.3, held at scale, at least 83 items sold), threshold unchanged.

**Recommendations fired per rule (2,970 sellers).**

| Rule | Before (in-sample risk, old cutoffs) | Out-of-fold risk, OLD cutoffs | Out-of-fold risk, new cutoffs |
|---|---|---|---|
| R01 Suspend | 152 | 334 | 137 |
| R02 Suspend | 5 | 5 | 6 |
| R03 Warn | 188 | 1,712 | 219 |
| R04 Warn | 58 | 4 | 69 |
| R05 Warn | 5 | 7 | 37 |
| R06 Feature | 130 | 1 | 137 |
| R07 Feature | 47 | 78 | 37 |
| R08 Promote | 306 | 222 | 106 |
| R09 Promote | 106 | 0 | 104 |
| R10 Keep | 63 | 0 | 170 |
| R11 Keep | 621 | 223 | 713 |
| R12 Keep (default) | 1,289 | 384 | 1,235 |

By action (Keep / Promote / Feature / Warn / Suspend): 1,973 / 412 / 177 / 251 / 157 before;
607 / 222 / 79 / 1,723 / 339 with the out-of-fold scores and the old cutoffs (the old cutoffs
simply no longer fit the new scale); **2,118 / 210 / 174 / 325 / 143** after recalibration.
521 sellers changed action from before to after: Keep to Warn 136, Promote to Keep 197, Warn to
Keep 78, Suspend to Keep 28, Keep to Suspend 25, Suspend to Warn 19, Feature to Keep 14, Keep
to Feature 11, Warn to Suspend 8, Promote to Warn 5.

What changed and why, in plain terms. The Promote tier roughly halves (412 to 210, 197 sellers
fall back to Keep): the in-sample risk of a gem was near zero because the forest had memorised
the rarely-late sellers, so almost every gem cleared R08; the honest out-of-fold risk puts the
literal 0.3 cutoff at the median of the gems (50.8%). R05 rises from 5 to 37 because sellers
with a high actual late rate used to be caught first by R03 (their in-sample risk echoed that
rate) and now reach R05. The late-risk rules fire in more plausible proportions (R01 25%, R03
10%, R06 and R10 50% of all sellers by construction).

## 10. Minimum evidence for Warn and Suspend (Stage 8)

**Problem.** R01 (Suspend) and R03 (Warn) read `avg_late_risk` with no minimum amount of history,
so 70 of R01's 137 and 120 of R03's 219 final recommendations were sellers with fewer than 5 items
sold (median 4 and 3 items), and 104 of R03's 219 Warn sellers had never delivered late. Warning
or suspending a seller on one or two items is weak decision support. R02, R04, R05 and R10
already carried an items-sold gate (R02 at 10, the others at 5).

**Change.** R01 and R03 now also require `order_volume >= 5` (items sold), and their `because`
texts say so. Their risk thresholds are percentile-intent (R01 the 75th percentile, R03 the 90th),
so they were re-derived on the group the rules now apply to, the 1,864 sellers with at least 5
items sold: R01 0.416 -> 0.412 and R03 0.501 -> 0.476 (3 decimals, as in earlier passes). No other
rule, condition or priority changed.

**Sensitivity.** Keeping the all-seller thresholds (0.416 and 0.501) with the new gate would have
given R01 67 and R03 99 final recommendations instead of 70 and 144: Warn 205 instead of 241,
Suspend 73 instead of 76, Keep 2,308 instead of 2,269. The eligible-population thresholds were
adopted because that is the population the rules apply to; lowering R03 to 0.476 moves 39
sellers with at least 5 items from Keep to Warn and lowering R01 to 0.412 moves 3 from Warn to
Suspend, a direct consequence of that choice.

**Effect (before -> after).** Suspend 143 -> 76, Warn 325 -> 241, Keep 2,118 -> 2,269, Feature 174
and Promote 210 unchanged. 232 sellers changed action: Suspend to Keep 70, Warn to Keep 120, Keep
to Warn 39, Warn to Suspend 3. All 1,106 sellers with fewer than 5 items sold now get the default
rule R12 ("Keep, pending more data"); 190 of them were on R01 or R03 before. R03 sellers who never
delivered late fell from 104 to 10, and R01 sellers from 34 to 3. The quadrants, the 437 hidden
gems, the AHP ranking, the risk scores and the models are byte-identical to before.

**Guards.** `tests/test_expert.py` fails if any Warn or Suspend rule lacks an `order_volume >= 5`
(or higher) condition, if a seller below 5 items sold has a Warn or Suspend recommendation, or if
R01 and R03 stop sitting at the 75th and 90th percentile of the sellers they apply to.

**Wording of the reason texts (follow-up).** The `because` texts of R02, R04, R05 and R10 used to say
"orders" where the rule reads `order_volume` (items sold) or `late_rate` (the share of a seller's
items delivered late, taken over item rows). They now say "items" or "items sold". Only the text
changed: no condition, threshold, priority or action, so all 2,970 recommendations keep their rule
and action and only the 273 sellers on those four rules show new reason wording. The quote of the
old R05 text in the table of section 5b is history and stays as it was.
