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

**Known, disclosed limitation — NOT fixed in this pass:** R01, R03, R06,
and R10 also key off `avg_late_risk`, with thresholds (0.53, 0.55, 0.503)
calibrated against the old LR distribution (median ~0.5). Checked against
the table above, RF's real median is 0.106 — meaning R06's/R10's "median
risk" condition (`< 0.503` / `<= 0.503`) is no longer close to the median
at all (0.503 sits around the **96th-97th percentile** of the real RF
distribution), and R01's/R03's "top quartile/decile risk" conditions
(`> 0.53` / `> 0.55`) now correspond to roughly the **top 2-3%**, not
25%/10%. These rules still function (checked: R01 combo -> 55 real
sellers, R03 -> 78, R06 combo -> 187, R10 combo -> 815 — none are
vacuous), but their percentile framing in section 6 below is now
inaccurate and was not recalibrated as part of this change, since it was
out of scope for "recalibrate R08/R09." Flagged in `CLAUDE.md` section 18
as follow-up work.

## 6. Remaining expert-system rule thresholds (`rules/seller_rules.yaml`)

**Note on staleness:** the `avg_late_risk`-dependent rows below (R01, R03,
R06, R10) were computed against the Stage 2/3 LR model's distribution and
were NOT recalibrated when Stage 3.5 adopted RandomForest — see section 5's
"known, disclosed limitation" note for the real, current match counts
against RF's distribution (R01: 55, R03: 78, R06: 187, R10: 815, vs. the
stale numbers quoted below). The rules still fire and aren't vacuous, but
their percentile framing here is outdated.

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
